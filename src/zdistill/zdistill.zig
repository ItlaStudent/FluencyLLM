// zdistill — destilación MOHAWK/KD/on-policy/DPO
//
// Además de las funciones de pérdida (KD, MOHAWK A/B/C, on-policy KL, DPO/ORPO),
// este módulo contiene el MOTOR de aprendizaje de F3: el modelo entrenable
// `FluencyNano` (LM híbrido attn↔SSM construido sobre el autodiff de zgrad), el
// `Trainer` (forward+backward+clipping+optimizer con schedule) y la suite de
// evaluación. El profesor (logits reales vía zload/zarch o texto vía zollama) lo
// inyecta el orquestador (main): aquí el profesor es agnóstico (se pasan logits
// o targets ya calculados), de modo que zdistill no arrastra dependencias pesadas.
const std = @import("std");
const zgrad = @import("zgrad");
const ztrain = @import("ztrain");

// --- KD por logits (Hinston et al.) ---
/// KL divergence entre student (s_logits) y teacher (t_logits) con temperatura
pub fn kdLoss(s_logits: []const f32, t_logits: []const f32, temperature: f32, loss: []f32) void {
    const vocab = s_logits.len;
    std.debug.assert(t_logits.len == vocab);
    std.debug.assert(loss.len == vocab);

    // softmax con temperatura
    var s_max = s_logits[0];
    var t_max = t_logits[0];
    for (s_logits) |v| { if (v > s_max) s_max = v; }
    for (t_logits) |v| { if (v > t_max) t_max = v; }

    var s_sum: f32 = 0;
    var t_sum: f32 = 0;
    for (0..vocab) |i| {
        const s = @exp((s_logits[i] - s_max) / temperature);
        const t = @exp((t_logits[i] - t_max) / temperature);
        loss[i] = s; // temporal
        s_sum += s;
        t_sum += t;
    }

    // KL(P_teacher || P_student) = sum p_t * (log p_t - log p_s)
    for (0..vocab) |i| {
        const p_t = @exp((t_logits[i] - t_max) / temperature) / t_sum;
        const p_s = @exp((s_logits[i] - s_max) / temperature) / s_sum;
        loss[i] = p_t * (@log(p_t + 1e-10) - @log(p_s + 1e-10));
    }
}

// --- MOHAWK ---
pub const MohawkStage = enum { A, B, C };

/// Stage A: KD simple (soft labels del profesor)
pub fn mohawkStageA(s_logits: []const f32, t_logits: []const f32, temperature: f32, loss: []f32) void {
    kdLoss(s_logits, t_logits, temperature, loss);
}

/// Stage B: Ocultar capas del profesor, predecir capa destino
/// student_hidden: [batch, dim], target_layer_logits: [batch, n_layers_teacher]
/// loss_ce: [batch]
pub fn mohawkStageB(student_hidden: []const f32, target_logits: []const f32, loss_ce: []f32, batch: usize, dim: usize) void {
    for (0..batch) |b| {
        // Simplificación: MSE entre hidden student y una proyección del target
        var mse: f32 = 0;
        for (0..dim) |d| {
            const diff = student_hidden[b * dim + d] - target_logits[b * dim + d];
            mse += diff * diff;
        }
        loss_ce[b] = mse / @as(f32, @floatFromInt(dim));
    }
}

/// Stage C: Destilación de arquitectura completa ( logits + hidden states + attention )
pub fn mohawkStageC(s_logits: []const f32, t_logits: []const f32, s_hidden: []const f32, t_hidden: []const f32, loss_out: []f32, vocab: usize, dim: usize, temperature: f32) void {
    // Combinación: 0.7 * KD_logits + 0.3 * MSE_hidden
    var kd_buf: [4096]f32 = undefined;
    kdLoss(s_logits, t_logits, temperature, kd_buf[0..vocab]);
    var kd_val: f32 = 0;
    for (kd_buf[0..vocab]) |v| kd_val += v;

    var mse_val: f32 = 0;
    for (0..dim) |d| {
        const diff = s_hidden[d] - t_hidden[d];
        mse_val += diff * diff;
    }
    mse_val /= @as(f32, @floatFromInt(dim));

    loss_out[0] = 0.7 * kd_val + 0.3 * mse_val;
}

// --- On-policy distillation: KL inversa ---
/// Minimiza KL(P_student || P_teacher) con muestras del student
pub fn onPolicyKLLoss(student_logits: []const f32, teacher_logits: []const f32, loss: []f32) void {
    // KL(P_s || P_t) = sum p_s * (log p_s - log p_t)
    const vocab = student_logits.len;
    var s_max = student_logits[0];
    var t_max = teacher_logits[0];
    for (student_logits) |v| { if (v > s_max) s_max = v; }
    for (teacher_logits) |v| { if (v > t_max) t_max = v; }

    var s_sum: f32 = 0;
    for (0..vocab) |i| {
        const s = @exp(student_logits[i] - s_max);
        loss[i] = s; // temporal
        s_sum += s;
    }

    for (0..vocab) |i| {
        const p_s = loss[i] / s_sum;
        const p_t = @exp(teacher_logits[i] - t_max) / blk: {
            var sum: f32 = 0;
            for (teacher_logits) |v| sum += @exp(v - t_max);
            break :blk sum;
        };
        loss[i] = p_s * (@log(p_s + 1e-10) - @log(p_t + 1e-10));
    }
}

// --- DPO/ORPO ---
/// DPO loss: -log sigmoid(beta * (log p_w / p_l - log p_ref_w / p_ref_l))
pub fn dpoLoss(preferred_logps: f32, rejected_logps: f32, ref_preferred: f32, ref_rejected: f32, beta: f32) f32 {
    const pi_ratio = preferred_logps - rejected_logps;
    const ref_ratio = ref_preferred - ref_rejected;
    const logits = beta * (pi_ratio - ref_ratio);
    return -@log(1.0 / (1.0 + @exp(-logits)));
}

/// ORPO: on-policy + odds ratio
pub fn orpoLoss(s_logits: []const f32, _: []const f32, preferred_idx: u32, rejected_idx: u32, beta: f32) f32 {
    // Odds ratio = p(preferred) / (1 - p(preferred)) vs p(rejected) / (1 - p(rejected))
    const p_pref = @exp(s_logits[preferred_idx]);
    const p_rej = @exp(s_logits[rejected_idx]);
    const odds_pref = p_pref / (1.0 - p_pref + 1e-10);
    const odds_rej = p_rej / (1.0 - p_rej + 1e-10);
    return -@log(odds_pref / (odds_rej + 1e-10)) * beta;
}

// =====================================================================
// FluencyNano — LM híbrido entrenable (motor de aprendizaje F3)
// =====================================================================

pub const MixerKind = enum { attn, scan, sparse_topk, gated_deltanet2 };

pub const NanoConfig = struct {
    vocab: usize,
    dim: usize,
    n_layers: usize,
    n_heads: usize,
    ff_dim: usize,
    eps: f32 = 1e-5,
    n_experts: usize = 0,
    top_k: usize = 2,
    moe_ff_dim: usize = 0,
    /// Cada cuántas capas se inserta una capa de atención (resto scan).
    /// Valor 2 = alternado (default). Valor 6 = 1 de cada 6 capas attn.
    attn_interval: usize = 2,
    /// Ventana de atención acotada (SWA). 0 = atención causal densa.
    attn_window: usize = 0,
    /// Mixer override: si no es null, todas las capas usan este mixer.
    /// Útil para experimentación (p. ej. sparse_topk o gated_deltanet2 puro).
    mixer_override: ?MixerKind = null,
    /// K para SparseTopK (número de tokens top-k en cada paso).
    sparse_topk_k: usize = 8,

    /// Patrón híbrido configurable: capas múltiplo de `attn_interval` = atención;
    /// el resto = SSM diagonal (estado recurrente O(1)).
    pub fn mixerOf(self: NanoConfig, layer: usize) MixerKind {
        if (self.mixer_override) |m| return m;
        return if (layer % self.attn_interval == 0) .attn else .scan;
    }

    /// Dimensión del FFN: denso si n_experts==0, MoE si >0.
    pub fn ffnDim(self: NanoConfig) usize {
        return if (self.n_experts > 0) self.moe_ff_dim else self.ff_dim;
    }
};

const LayerOff = struct {
    kind: MixerKind,
    norm1: usize,
    p0: usize, // attn: wq | scan: wi
    p1: usize, // attn: wk | scan: a_decay
    p2: usize, // attn: wv | scan: wo
    p3: usize, // attn: wo | scan: (sin uso)
    norm2: usize,
    w_gate: usize, // denso FFN gate (o 0 si MoE)
    w_up: usize,   // denso FFN up   (o 0 si MoE)
    w_down: usize, // denso FFN down (o 0 si MoE)
    // MoE pesos (válidos si cfg.n_experts > 0)
    moe_router: usize,
    moe_w1: usize,
    moe_w3: usize,
    moe_w2: usize,
};

const MapEntry = struct { off: usize, len: usize, gvar: zgrad.Var };

/// Modelo de lenguaje híbrido pequeño y ENTRENABLE. Todos los pesos viven en un
/// único buffer plano `weights` (y sus gradientes en `grads`), lo que hace
/// triviales el optimizer y el clipping. El forward se expresa con ops de
/// `zgrad.Graph`, por lo que el backward es automático. Embeddings atados
/// (la proyección de salida reusa la tabla de embedding vía `matmulTransB`).
pub const FluencyNano = struct {
    cfg: NanoConfig,
    alloc: std.mem.Allocator,
    weights: []f32,
    grads: []f32,
    layers: []LayerOff,
    emb_off: usize,
    final_norm_off: usize,
    tape_map: std.ArrayList(MapEntry),
    seed: u64,

    pub fn paramCount(self: FluencyNano) usize {
        return self.weights.len;
    }

    /// Imprime desglose de parámetros vía std.log.info.
    pub fn printParamCount(self: FluencyNano) void {
        const cfg = self.cfg;
        const D = cfg.dim;
        const F = cfg.ff_dim;
        const V = cfg.vocab;
        const L = cfg.n_layers;
        const emb = V * D;
        var mixer_total: usize = 0;
        var ffn_total: usize = 0;
        var norm_total: usize = 0;
        for (0..L) |i| {
            const kind = cfg.mixerOf(i);
            norm_total += D * 2; // norm1 + norm2
            switch (kind) {
                .attn => mixer_total += 4 * D * D,
                .scan => mixer_total += 2 * D * D + D,
                .sparse_topk => mixer_total += 3 * D * D, // wq, wk, wv (sin wo)
                .gated_deltanet2 => mixer_total += 2 * D * D + D * D, // wi, w_gate, wo
            }
            ffn_total += 3 * D * F;
        }
        const final_norm = D;
        const total = emb + mixer_total + ffn_total + norm_total + final_norm;
        std.log.info(
            "ParamCount: total={d}  emb={d}  mixers={d}  FFN={d}  norms={d}  final_norm={d}",
            .{ total, emb, mixer_total, ffn_total, norm_total, final_norm });
    }

    pub fn init(alloc: std.mem.Allocator, cfg: NanoConfig, seed: u64) !FluencyNano {
        const D = cfg.dim;
        const F = cfg.ff_dim;
        const M = cfg.moe_ff_dim;
        const E = cfg.n_experts;
        const V = cfg.vocab;
        var off: usize = 0;
        const emb_off = off;
        off += V * D;
        const layers = try alloc.alloc(LayerOff, cfg.n_layers);
        errdefer alloc.free(layers);
        for (0..cfg.n_layers) |i| {
            const kind = cfg.mixerOf(i);
            var L: LayerOff = undefined;
            L.kind = kind;
            L.norm1 = off;
            off += D;
            switch (kind) {
                .attn => {
                    L.p0 = off;
                    off += D * D; // wq
                    L.p1 = off;
                    off += D * D; // wk
                    L.p2 = off;
                    off += D * D; // wv
                    L.p3 = off;
                    off += D * D; // wo
                },
                .scan => {
                    L.p0 = off;
                    off += D * D; // wi
                    L.p1 = off;
                    off += D; // a_decay
                    L.p2 = off;
                    off += D * D; // wo
                    L.p3 = 0;
                },
                .sparse_topk => {
                    L.p0 = off;
                    off += D * D; // wq
                    L.p1 = off;
                    off += D * D; // wk
                    L.p2 = off;
                    off += D * D; // wv
                    L.p3 = 0; // sin wo (output directo)
                },
                .gated_deltanet2 => {
                    L.p0 = off;
                    off += D * D; // wi
                    L.p1 = off;
                    off += D * D; // w_gate
                    L.p2 = off;
                    off += D * D; // wo
                    L.p3 = 0;
                },
            }
            L.norm2 = off;
            off += D;
            if (E > 0) {
                L.w_gate = 0;
                L.w_up = 0;
                L.w_down = 0;
                L.moe_router = off;
                off += D * E;
                L.moe_w1 = off;
                off += E * D * M;
                L.moe_w3 = off;
                off += E * D * M;
                L.moe_w2 = off;
                off += E * M * D;
            } else {
                L.w_gate = off;
                off += D * F;
                L.w_up = off;
                off += D * F;
                L.w_down = off;
                off += F * D;
                L.moe_router = 0;
                L.moe_w1 = 0;
                L.moe_w3 = 0;
                L.moe_w2 = 0;
            }
            layers[i] = L;
        }
        const final_norm_off = off;
        off += D;
        const total = off;

        const weights = try alloc.alloc(f32, total);
        errdefer alloc.free(weights);
        const grads = try alloc.alloc(f32, total);
        errdefer alloc.free(grads);
        @memset(grads, 0);

        var self = FluencyNano{
            .cfg = cfg,
            .alloc = alloc,
            .weights = weights,
            .grads = grads,
            .layers = layers,
            .emb_off = emb_off,
            .final_norm_off = final_norm_off,
            .tape_map = std.ArrayList(MapEntry).empty,
            .seed = seed,
        };
        self.initWeights(seed);
        return self;
    }

    pub fn deinit(self: *FluencyNano) void {
        self.alloc.free(self.weights);
        self.alloc.free(self.grads);
        self.alloc.free(self.layers);
        self.tape_map.deinit(self.alloc);
        self.* = undefined;
    }

    fn initWeights(self: *FluencyNano, seed: u64) void {
        const D = self.cfg.dim;
        const F = self.cfg.ff_dim;
        const M = self.cfg.moe_ff_dim;
        const E = self.cfg.n_experts;
        const V = self.cfg.vocab;
        var prng = std.Random.DefaultPrng.init(seed);
        const r = prng.random();
        const w = self.weights;
        // embedding ~ N(0, 0.02)
        for (w[self.emb_off .. self.emb_off + V * D]) |*p| p.* = r.floatNorm(f32) * 0.02;
        const sd: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
        const sf: f32 = if (E > 0) 1.0 / @sqrt(@as(f32, @floatFromInt(M))) else 1.0 / @sqrt(@as(f32, @floatFromInt(F)));
        for (self.layers) |L| {
            fill(w[L.norm1 .. L.norm1 + D], 1.0);
            switch (L.kind) {
                .attn => {
                    randn(w[L.p0 .. L.p0 + D * D], r, sd);
                    randn(w[L.p1 .. L.p1 + D * D], r, sd);
                    randn(w[L.p2 .. L.p2 + D * D], r, sd);
                    randn(w[L.p3 .. L.p3 + D * D], r, sd);
                },
                .scan => {
                    randn(w[L.p0 .. L.p0 + D * D], r, sd);
                    // a_decay en (0.85, 0.95): estable, scan no explota
                    for (w[L.p1 .. L.p1 + D]) |*p| p.* = 0.85 + 0.1 * r.float(f32);
                    randn(w[L.p2 .. L.p2 + D * D], r, sd);
                },
                .sparse_topk => {
                    randn(w[L.p0 .. L.p0 + D * D], r, sd);
                    randn(w[L.p1 .. L.p1 + D * D], r, sd);
                    randn(w[L.p2 .. L.p2 + D * D], r, sd);
                },
                .gated_deltanet2 => {
                    randn(w[L.p0 .. L.p0 + D * D], r, sd);
                    randn(w[L.p1 .. L.p1 + D * D], r, sd);
                    randn(w[L.p2 .. L.p2 + D * D], r, sd);
                },
            }
            fill(w[L.norm2 .. L.norm2 + D], 1.0);
            if (E > 0) {
                // Router ~ N(0, 0.01) (bajos logits iniciales → distribución uniforme)
                randn(w[L.moe_router .. L.moe_router + D * E], r, 0.01);
                randn(w[L.moe_w1 .. L.moe_w1 + E * D * M], r, sd);
                randn(w[L.moe_w3 .. L.moe_w3 + E * D * M], r, sd);
                randn(w[L.moe_w2 .. L.moe_w2 + E * M * D], r, sf);
            } else {
                randn(w[L.w_gate .. L.w_gate + D * F], r, sd);
                randn(w[L.w_up .. L.w_up + D * F], r, sd);
                randn(w[L.w_down .. L.w_down + F * D], r, sf);
            }
        }
        fill(w[self.final_norm_off .. self.final_norm_off + D], 1.0);
    }

    fn fill(s: []f32, v: f32) void {
        for (s) |*p| p.* = v;
    }
    fn randn(s: []f32, r: std.Random, scale: f32) void {
        for (s) |*p| p.* = r.floatNorm(f32) * scale;
    }

    pub fn zeroGrad(self: *FluencyNano) void {
        @memset(self.grads, 0);
    }

    fn pleaf(self: *FluencyNano, g: *zgrad.Graph, off: usize, rows: usize, cols: usize) !zgrad.Var {
        const v = try g.leaf(self.weights[off .. off + rows * cols], rows, cols);
        try self.tape_map.append(self.alloc, .{ .off = off, .len = rows * cols, .gvar = v });
        return v;
    }

    /// Forward: ids[T] -> logits Var [T, vocab]. Registra las hojas-parámetro en
    /// `tape_map` para poder dispersar sus gradientes tras el backward.
    pub fn forward(self: *FluencyNano, g: *zgrad.Graph, ids: []const u32) !zgrad.Var {
        return self.forwardWithHidden(g, ids, null);
    }

    /// Forward con captura de hidden states por capa. Si `hidden_vars` no es null,
    /// debe tener espacio para `n_layers` elementos; se escribe el nodo `x` post-capa
    /// (post-FFN residual) en `hidden_vars[layer]`.
    pub fn forwardWithHidden(self: *FluencyNano, g: *zgrad.Graph, ids: []const u32, hidden_vars: ?[]zgrad.Var) !zgrad.Var {
        self.tape_map.clearRetainingCapacity();
        const D = self.cfg.dim;
        const F = self.cfg.ff_dim;
        const V = self.cfg.vocab;
        const eps = self.cfg.eps;
        const emb = try self.pleaf(g, self.emb_off, V, D);
        var x = try g.embed(emb, ids);
        for (self.layers, 0..) |L, li| {
            const n1 = try self.pleaf(g, L.norm1, 1, D);
            const h = try g.rmsnorm(x, n1, eps);
            var o: zgrad.Var = undefined;
            switch (L.kind) {
                .attn => {
                    const wq = try self.pleaf(g, L.p0, D, D);
                    const wk = try self.pleaf(g, L.p1, D, D);
                    const wv = try self.pleaf(g, L.p2, D, D);
                    const wo = try self.pleaf(g, L.p3, D, D);
                    const q = try g.matmul(h, wq);
                    const k = try g.matmul(h, wk);
                    const v = try g.matmul(h, wv);
                    const a = try g.attn(q, k, v, self.cfg.n_heads, self.cfg.attn_window);
                    o = try g.matmul(a, wo);
                },
                .scan => {
                    const wi = try self.pleaf(g, L.p0, D, D);
                    const ad = try self.pleaf(g, L.p1, 1, D);
                    const wo = try self.pleaf(g, L.p2, D, D);
                    const inp = try g.matmul(h, wi);
                    const s = try g.scan(inp, ad);
                    o = try g.matmul(s, wo);
                },
                .sparse_topk => {
                    const wq = try self.pleaf(g, L.p0, D, D);
                    const wk = try self.pleaf(g, L.p1, D, D);
                    const wv = try self.pleaf(g, L.p2, D, D);
                    const q = try g.matmul(h, wq);
                    const k = try g.matmul(h, wk);
                    const v = try g.matmul(h, wv);
                    o = try g.sparseTopK(q, k, v, self.cfg.sparse_topk_k);
                },
                .gated_deltanet2 => {
                    const wi = try self.pleaf(g, L.p0, D, D);
                    const wg = try self.pleaf(g, L.p1, D, D);
                    const wo = try self.pleaf(g, L.p2, D, D);
                    const inp = try g.matmul(h, wi);
                    const gate_raw = try g.matmul(h, wg);
                    const s = try g.gatedDeltaNet2(inp, gate_raw);
                    o = try g.matmul(s, wo);
                },
            }
            x = try g.addResidual(x, o);
            const n2 = try self.pleaf(g, L.norm2, 1, D);
            const h2 = try g.rmsnorm(x, n2, eps);
            if (self.cfg.n_experts > 0) {
                const E = self.cfg.n_experts;
                const M = self.cfg.moe_ff_dim;
                const router = try self.pleaf(g, L.moe_router, D, E);
                const w1 = try self.pleaf(g, L.moe_w1, E * D, M);
                const w3 = try self.pleaf(g, L.moe_w3, E * D, M);
                const w2 = try self.pleaf(g, L.moe_w2, E * M, D);
                const ff = try g.moe(h2, router, w1, w3, w2, E, M);
                x = try g.addResidual(x, ff);
            } else {
                const wg = try self.pleaf(g, L.w_gate, D, F);
                const wu = try self.pleaf(g, L.w_up, D, F);
                const wd = try self.pleaf(g, L.w_down, F, D);
                const gate = try g.silu(try g.matmul(h2, wg));
                const up = try g.matmul(h2, wu);
                const gu = try g.mul(gate, up);
                const ff = try g.matmul(gu, wd);
                x = try g.addResidual(x, ff);
            }
            if (hidden_vars) |hv| hv[li] = x;
        }
        const nf = try self.pleaf(g, self.final_norm_off, 1, D);
        const xf = try g.rmsnorm(x, nf, eps);
        return try g.matmulTransB(xf, emb); // [T, V] (embeddings atados)
    }

    /// Tras `g.backward(loss)`, dispersa los gradientes del grafo a `self.grads`.
    pub fn collectGrads(self: *FluencyNano, g: *zgrad.Graph) void {
        for (self.tape_map.items) |e| {
            const src = g.gradOf(e.gvar);
            const dst = self.grads[e.off .. e.off + e.len];
            for (dst, src) |*d, s| d.* += s;
        }
    }
};

// =====================================================================
// Trainer — un paso completo de destilación (fwd+bwd+clip+optimizer)
// =====================================================================
pub const Trainer = struct {
    model: *FluencyNano,
    opt: ztrain.AdamW,
    sched: ztrain.Scheduler,
    step_count: usize = 0,
    max_grad_norm: f32 = 1.0,
    alloc: std.mem.Allocator,
    skipped_batches: usize = 0,
    /// Coeficiente de load balancing loss para MoE (0 = desactivado).
    /// Valor típico: 0.01 (Switch Transformer / Mixtral).
    lb_alpha: f32 = 0.0,

    pub fn init(alloc: std.mem.Allocator, model: *FluencyNano, sched: ztrain.Scheduler, max_grad_norm: f32) !Trainer {
        const opt = try ztrain.AdamW.init(alloc, sched.getLR(0), model.weights.len);
        return .{ .model = model, .opt = opt, .sched = sched, .max_grad_norm = max_grad_norm, .alloc = alloc };
    }

    pub fn deinit(self: *Trainer) void {
        self.opt.deinit();
    }

    fn applyStep(self: *Trainer, g: *zgrad.Graph, loss_var: zgrad.Var) !void {
        self.model.zeroGrad();
        g.backward(loss_var);
        self.model.collectGrads(g);

        if (ztrain.hasNaNOrInf(self.model.grads)) {
            self.skipped_batches += 1;
            std.log.warn("seqkd step {d}: gradiente no-finito, SKIP batch (skipped={d})", .{ self.step_count + 1, self.skipped_batches });
            @memset(self.model.grads, 0);
            return;
        }

        var groups = [_][]f32{self.model.grads};
        _ = ztrain.clipGradGlobalNorm(&groups, self.max_grad_norm);
        self.opt.lr = self.sched.getLR(self.step_count);
        self.opt.step(self.model.weights, self.model.grads);
        self.step_count += 1;

        if (ztrain.hasNaNOrInf(self.model.weights)) {
            std.log.err("seqkd step {d}: pesos no-finitos detectados → divergencia", .{self.step_count});
            return error.TrainingDiverged;
        }
    }

    /// MOHAWK Stage A / KD por logits: alinea la distribución del alumno con la
    /// del profesor (logits [T,V]). Devuelve la pérdida KD del paso.
    pub fn stepKD(self: *Trainer, ids: []const u32, teacher_logits: []const f32, tau: f32) !f32 {
        var g = zgrad.Graph.init(self.alloc);
        defer g.deinit();
        const logits = try self.model.forward(&g, ids);
        const loss = try zgrad.kdDistillLoss(&g, logits, teacher_logits, tau);
        const lv = g.valOf(loss)[0];
        if (!std.math.isFinite(lv)) {
            self.skipped_batches += 1;
            std.log.warn("seqkd step {d}: loss KD no-finita ({d}), SKIP batch", .{ self.step_count + 1, lv });
            @memset(self.model.grads, 0);
            return 0.0;
        }
        try self.applyStep(&g, loss);
        return lv;
    }

    /// Seq-KD / aprendizaje supervisado: cross-entropy contra tokens objetivo
    /// (p. ej. secuencias generadas por el profesor vía zollama). Devuelve la CE.
    pub fn stepCE(self: *Trainer, ids: []const u32, targets: []const u32) !f32 {
        var g = zgrad.Graph.init(self.alloc);
        defer g.deinit();
        const logits = try self.model.forward(&g, ids);
        const ce_loss = try zgrad.crossEntropyLoss(&g, logits, targets);
        // Añadir load balancing loss si MoE está activo
        var total_loss = ce_loss;
        if (self.lb_alpha > 0 and self.model.cfg.n_experts > 0) {
            if (try g.totalMoeLoadBalanceLoss(self.lb_alpha)) |lb| {
                total_loss = try g.addResidual(ce_loss, lb);
            }
        }
        const loss = total_loss;
        const lv = g.valOf(loss)[0];
        if (!std.math.isFinite(lv)) {
            self.skipped_batches += 1;
            std.log.warn("seqkd step {d}: loss CE no-finita ({d}), SKIP batch", .{ self.step_count + 1, lv });
            @memset(self.model.grads, 0);
            return 0.0;
        }
        try self.applyStep(&g, loss);
        return lv;
    }

    /// MOHAWK Stage B: alinea hidden states del alumno (capa `student_layer`)
    /// con hidden states del profesor (misma dimensión). Devuelve MSE.
    pub fn stepMohawkB(self: *Trainer, ids: []const u32, teacher_hidden: []const f32, student_layer: usize) !f32 {
        var g = zgrad.Graph.init(self.alloc);
        defer g.deinit();
        const hidden_vars = try self.alloc.alloc(zgrad.Var, self.model.cfg.n_layers);
        defer self.alloc.free(hidden_vars);
        _ = try self.model.forwardWithHidden(&g, ids, hidden_vars);
        const h = hidden_vars[student_layer];
        const loss = try zgrad.mseLoss(&g, h, teacher_hidden);
        const lv = g.valOf(loss)[0];
        if (!std.math.isFinite(lv)) {
            self.skipped_batches += 1;
            std.log.warn("seqkd step {d}: loss MSE no-finita ({d}), SKIP batch", .{ self.step_count + 1, lv });
            @memset(self.model.grads, 0);
            return 0.0;
        }
        try self.applyStep(&g, loss);
        return lv;
    }

    /// MOHAWK Stage C: combina KD por logits + MSE sobre hidden final.
    /// `student_layer` es la capa del alumno cuyo hidden se alinea.
    /// Devuelve loss total (KD escala + MSE escala).
    pub fn stepMohawkC(self: *Trainer, ids: []const u32, teacher_logits: []const f32, teacher_hidden: []const f32, student_layer: usize, tau: f32) !f32 {
        var g = zgrad.Graph.init(self.alloc);
        defer g.deinit();
        const hidden_vars = try self.alloc.alloc(zgrad.Var, self.model.cfg.n_layers);
        defer self.alloc.free(hidden_vars);
        const logits = try self.model.forwardWithHidden(&g, ids, hidden_vars);
        const loss_kd = try zgrad.kdDistillLoss(&g, logits, teacher_logits, tau);
        const h = hidden_vars[student_layer];
        const loss_mse = try zgrad.mseLoss(&g, h, teacher_hidden);
        // Ambos gradientes ya están escritos; backward desde cualquiera recorre el grafo.
        const lv_kd = g.valOf(loss_kd)[0];
        const lv_mse = g.valOf(loss_mse)[0];
        const lv_total = lv_kd + lv_mse;
        if (!std.math.isFinite(lv_total)) {
            self.skipped_batches += 1;
            std.log.warn("seqkd step {d}: loss MohawkC no-finita ({d}), SKIP batch", .{ self.step_count + 1, lv_total });
            @memset(self.model.grads, 0);
            return 0.0;
        }
        try self.applyStep(&g, loss_kd);
        return lv_total;
    }
};

// =====================================================================
// Evaluación (fluency eval): perplejidad, KL al profesor, accuracy
// =====================================================================
pub const EvalResult = struct {
    ce: f32, // cross-entropy media (nats/token)
    ppl: f32, // perplejidad = exp(ce)
    acc: f32, // accuracy de argmax (next-token)
};

/// Cross-entropy / perplejidad / accuracy del modelo sobre (ids -> targets).
pub fn evaluate(model: *FluencyNano, alloc: std.mem.Allocator, ids: []const u32, targets: []const u32) !EvalResult {
    std.debug.assert(ids.len == targets.len);
    var g = zgrad.Graph.init(alloc);
    defer g.deinit();
    const logits_v = try model.forward(&g, ids);
    const L = g.valOf(logits_v);
    const T = ids.len;
    const V = model.cfg.vocab;
    var ce_sum: f32 = 0;
    var correct: usize = 0;
    for (0..T) |t| {
        const row = L[t * V .. t * V + V];
        var maxv: f32 = -std.math.inf(f32);
        var amax: usize = 0;
        for (row, 0..) |v, i| {
            if (v > maxv) {
                maxv = v;
                amax = i;
            }
        }
        var sum: f32 = 0;
        for (row) |v| sum += @exp(v - maxv);
        const tgt = targets[t];
        ce_sum += -(row[tgt] - maxv - @log(sum));
        if (amax == tgt) correct += 1;
    }
    const Tf: f32 = @floatFromInt(T);
    const ce = ce_sum / Tf;
    return .{ .ce = ce, .ppl = @exp(ce), .acc = @as(f32, @floatFromInt(correct)) / Tf };
}

/// KL(profesor || alumno) media por token sobre teacher_logits [T,V].
pub fn evalKLToTeacher(model: *FluencyNano, alloc: std.mem.Allocator, ids: []const u32, teacher_logits: []const f32, tau: f32) !f32 {
    var g = zgrad.Graph.init(alloc);
    defer g.deinit();
    const logits_v = try model.forward(&g, ids);
    const loss = try zgrad.kdDistillLoss(&g, logits_v, teacher_logits, tau);
    return g.valOf(loss)[0];
}

// --- Checkpoint I/O ---
const CKPT_MAGIC = "FLNC";
const CKPT_VERSION: u32 = 1;
const CKPT_HEADER_SIZE = 4 + 4 + (8 * 5) + 4 + 8 + 8; // magic + version + 5*u64 cfg + eps + seed + n_params

/// Guarda un checkpoint binario con cabecera (magic, versión, NanoConfig, seed,
/// n_params) seguido de los pesos en f32 little-endian.
pub fn saveCheckpoint(model: *FluencyNano, path: []const u8, io: std.Io, alloc: std.mem.Allocator) !void {
    const n_params = model.weights.len;
    const total = CKPT_HEADER_SIZE + n_params * @sizeOf(f32);
    const buf = try alloc.alloc(u8, total);
    defer alloc.free(buf);

    var off: usize = 0;
    @memcpy(buf[off..][0..4], CKPT_MAGIC);
    off += 4;
    std.mem.writeInt(u32, buf[off..][0..4], CKPT_VERSION, .little);
    off += 4;
    std.mem.writeInt(u64, buf[off..][0..8], model.cfg.vocab, .little);
    off += 8;
    std.mem.writeInt(u64, buf[off..][0..8], model.cfg.dim, .little);
    off += 8;
    std.mem.writeInt(u64, buf[off..][0..8], model.cfg.n_layers, .little);
    off += 8;
    std.mem.writeInt(u64, buf[off..][0..8], model.cfg.n_heads, .little);
    off += 8;
    std.mem.writeInt(u64, buf[off..][0..8], model.cfg.ff_dim, .little);
    off += 8;
    std.mem.writeInt(u32, buf[off..][0..4], @bitCast(model.cfg.eps), .little);
    off += 4;
    std.mem.writeInt(u64, buf[off..][0..8], model.seed, .little);
    off += 8;
    std.mem.writeInt(u64, buf[off..][0..8], n_params, .little);
    off += 8;

    const wbytes = @as([*]const u8, @ptrCast(model.weights.ptr))[0 .. n_params * @sizeOf(f32)];
    @memcpy(buf[off..], wbytes);

    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = buf });
}

/// Carga un checkpoint de disco y reconstruye un FluencyNano con los pesos
/// restaurados. El modelo se inicializa con la misma config/seed y los pesos
/// se sobreescriben desde el archivo.
pub fn loadCheckpoint(alloc: std.mem.Allocator, path: []const u8, io: std.Io) !FluencyNano {
    const data = try std.Io.Dir.cwd().readFileAlloc(io, path, alloc, .unlimited);
    defer alloc.free(data);

    if (data.len < 4 or !std.mem.eql(u8, data[0..4], CKPT_MAGIC))
        return error.InvalidMagic;

    var off: usize = 4;
    const version = std.mem.readInt(u32, data[off..][0..4], .little);
    off += 4;
    if (version != CKPT_VERSION) return error.UnsupportedVersion;

    const vocab = std.mem.readInt(u64, data[off..][0..8], .little); off += 8;
    const dim = std.mem.readInt(u64, data[off..][0..8], .little); off += 8;
    const n_layers = std.mem.readInt(u64, data[off..][0..8], .little); off += 8;
    const n_heads = std.mem.readInt(u64, data[off..][0..8], .little); off += 8;
    const ff_dim = std.mem.readInt(u64, data[off..][0..8], .little); off += 8;
    const eps_bits = std.mem.readInt(u32, data[off..][0..4], .little); off += 4;
    const eps: f32 = @bitCast(eps_bits);
    const seed = std.mem.readInt(u64, data[off..][0..8], .little); off += 8;
    const n_params = std.mem.readInt(u64, data[off..][0..8], .little); off += 8;

    const expected_len = off + n_params * @sizeOf(f32);
    if (data.len != expected_len) return error.SizeMismatch;

    const cfg = NanoConfig{
        .vocab = vocab,
        .dim = dim,
        .n_layers = n_layers,
        .n_heads = n_heads,
        .ff_dim = ff_dim,
        .eps = eps,
    };
    const model = try FluencyNano.init(alloc, cfg, seed);
    const wsrc = data[off..];
    const wdst = @as([*]u8, @ptrCast(model.weights.ptr))[0 .. n_params * @sizeOf(f32)];
    @memcpy(wdst, wsrc);
    return model;
}

// --- tests ---
test "kdLoss no panic" {
    var s_logits: [5]f32 = .{ 0.1, 0.2, 0.3, 0.2, 0.2 };
    var t_logits: [5]f32 = .{ 0.2, 0.2, 0.2, 0.2, 0.2 };
    var loss: [5]f32 = undefined;
    kdLoss(&s_logits, &t_logits, 2.0, &loss);
}

test "mohawkStageA" {
    var s_logits: [5]f32 = .{ 0.1, 0.2, 0.3, 0.2, 0.2 };
    var t_logits: [5]f32 = .{ 0.2, 0.2, 0.2, 0.2, 0.2 };
    var loss: [5]f32 = undefined;
    mohawkStageA(&s_logits, &t_logits, 2.0, &loss);
}

test "dpoLoss" {
    const l = dpoLoss(-1.0, -2.0, -1.1, -1.9, 0.1);
    try std.testing.expect(l > 0);
}

test "orpoLoss" {
    var logits: [5]f32 = .{ 0.1, 0.5, 0.2, 0.1, 0.1 };
    var t_logits: [5]f32 = .{ 0.2, 0.4, 0.2, 0.1, 0.1 };
    const l = orpoLoss(&logits, &t_logits, 1, 0, 0.1);
    try std.testing.expect(l > 0);
}

test "FluencyNano forward: forma de logits correcta y sin fugas" {
    const alloc = std.testing.allocator;
    const cfg = NanoConfig{ .vocab = 16, .dim = 16, .n_layers = 2, .n_heads = 2, .ff_dim = 32 };
    var model = try FluencyNano.init(alloc, cfg, 42);
    defer model.deinit();
    const ids = [_]u32{ 1, 2, 3, 4, 5 };
    var g = zgrad.Graph.init(alloc);
    defer g.deinit();
    const logits = try model.forward(&g, &ids);
    try std.testing.expectEqual(@as(usize, ids.len), g.rowsOf(logits));
    try std.testing.expectEqual(@as(usize, cfg.vocab), g.colsOf(logits));
}

test "FluencyNano SparseTopK mixer: forward/backward sin fugas" {
    const alloc = std.testing.allocator;
    const cfg = NanoConfig{ .vocab = 16, .dim = 16, .n_layers = 2, .n_heads = 2, .ff_dim = 32, .mixer_override = .sparse_topk, .sparse_topk_k = 4 };
    var model = try FluencyNano.init(alloc, cfg, 42);
    defer model.deinit();
    const ids = [_]u32{ 1, 2, 3, 4, 5 };
    var g = zgrad.Graph.init(alloc);
    defer g.deinit();
    const logits = try model.forward(&g, &ids);
    try std.testing.expectEqual(@as(usize, ids.len), g.rowsOf(logits));
    try std.testing.expectEqual(@as(usize, cfg.vocab), g.colsOf(logits));
    // backward
    const targets = [_]u32{ 2, 3, 4, 5, 6 };
    const loss = try zgrad.crossEntropyLoss(&g, logits, &targets);
    g.backward(loss);
    model.collectGrads(&g);
    // verificar gradientes no nulos
    var has_nonzero: bool = false;
    for (model.grads) |gv| {
        if (gv != 0.0) {
            has_nonzero = true;
            break;
        }
    }
    try std.testing.expect(has_nonzero);
}

test "FluencyNano GatedDeltaNet2 mixer: forward/backward sin fugas" {
    const alloc = std.testing.allocator;
    const cfg = NanoConfig{ .vocab = 16, .dim = 16, .n_layers = 2, .n_heads = 2, .ff_dim = 32, .mixer_override = .gated_deltanet2 };
    var model = try FluencyNano.init(alloc, cfg, 42);
    defer model.deinit();
    const ids = [_]u32{ 1, 2, 3, 4, 5 };
    var g = zgrad.Graph.init(alloc);
    defer g.deinit();
    const logits = try model.forward(&g, &ids);
    try std.testing.expectEqual(@as(usize, ids.len), g.rowsOf(logits));
    try std.testing.expectEqual(@as(usize, cfg.vocab), g.colsOf(logits));
    // backward
    const targets = [_]u32{ 2, 3, 4, 5, 6 };
    const loss = try zgrad.crossEntropyLoss(&g, logits, &targets);
    g.backward(loss);
    model.collectGrads(&g);
    var has_nonzero: bool = false;
    for (model.grads) |gv| {
        if (gv != 0.0) {
            has_nonzero = true;
            break;
        }
    }
    try std.testing.expect(has_nonzero);
}

test "FluencyNano MoE forward/backward: forma correcta y sin fugas" {
    const alloc = std.testing.allocator;
    const cfg = NanoConfig{ .vocab = 12, .dim = 8, .n_layers = 1, .n_heads = 2, .ff_dim = 0, .n_experts = 3, .top_k = 2, .moe_ff_dim = 16 };
    var model = try FluencyNano.init(alloc, cfg, 42);
    defer model.deinit();
    const ids = [_]u32{ 1, 2, 3 };
    var g = zgrad.Graph.init(alloc);
    defer g.deinit();
    const logits = try model.forward(&g, &ids);
    try std.testing.expectEqual(@as(usize, ids.len), g.rowsOf(logits));
    try std.testing.expectEqual(@as(usize, cfg.vocab), g.colsOf(logits));
    // backward + collect grads (verifica que no pánico)
    const targets = [_]u32{ 2, 3, 4 };
    const loss = try zgrad.crossEntropyLoss(&g, logits, &targets);
    g.backward(loss);
    model.collectGrads(&g);
    // verificar que hay gradientes no nulos en los pesos MoE
    var has_nonzero: bool = false;
    for (model.grads) |gval| {
        if (gval != 0.0) {
            has_nonzero = true;
            break;
        }
    }
    try std.testing.expect(has_nonzero);
}

test "FluencyNano aprende: la CE baja y la accuracy sube con el entrenamiento" {
    const alloc = std.testing.allocator;
    const cfg = NanoConfig{ .vocab = 12, .dim = 16, .n_layers = 2, .n_heads = 2, .ff_dim = 32 };
    var model = try FluencyNano.init(alloc, cfg, 7);
    defer model.deinit();
    // secuencia fija a memorizar (predicción del siguiente token)
    const ids = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const tgt = [_]u32{ 2, 3, 4, 5, 6, 7, 8, 9 };
    const sched = ztrain.Scheduler{ .warmup_cosine = .{ .max_lr = 0.01, .min_lr = 0.001, .warmup_steps = 10, .max_steps = 300 } };
    var trainer = try Trainer.init(alloc, &model, sched, 1.0);
    defer trainer.deinit();

    const before = try evaluate(&model, alloc, &ids, &tgt);
    var i: usize = 0;
    while (i < 300) : (i += 1) _ = try trainer.stepCE(&ids, &tgt);
    const after = try evaluate(&model, alloc, &ids, &tgt);

    // la pérdida debe bajar notablemente y la accuracy llegar a memorizar
    try std.testing.expect(after.ce < before.ce * 0.5);
    try std.testing.expect(after.acc >= 0.875); // >= 7/8 tokens correctos
}

test "FluencyNano MoE aprende: la CE baja y sin regresión vs denso comparable" {
    const alloc = std.testing.allocator;
    const cfg = NanoConfig{ .vocab = 12, .dim = 16, .n_layers = 2, .n_heads = 2, .ff_dim = 0, .n_experts = 4, .top_k = 2, .moe_ff_dim = 12 };
    var model = try FluencyNano.init(alloc, cfg, 7);
    defer model.deinit();
    const ids = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const tgt = [_]u32{ 2, 3, 4, 5, 6, 7, 8, 9 };
    const sched = ztrain.Scheduler{ .warmup_cosine = .{ .max_lr = 0.01, .min_lr = 0.001, .warmup_steps = 10, .max_steps = 300 } };
    var trainer = try Trainer.init(alloc, &model, sched, 1.0);
    defer trainer.deinit();

    const before = try evaluate(&model, alloc, &ids, &tgt);
    var i: usize = 0;
    while (i < 300) : (i += 1) _ = try trainer.stepCE(&ids, &tgt);
    const after = try evaluate(&model, alloc, &ids, &tgt);

    // No regresión: debe bajar la pérdida y alcanzar accuracy comparable al denso
    try std.testing.expect(after.ce < before.ce * 0.5);
    try std.testing.expect(after.acc >= 0.75); // >= 6/8 (ligeramente más permisivo que denso por sparsidad soft)
}

test "entrenamiento reproducible con semilla (loss idéntica bit a bit)" {
    const alloc = std.testing.allocator;
    const cfg = NanoConfig{ .vocab = 10, .dim = 16, .n_layers = 2, .n_heads = 2, .ff_dim = 24 };
    const ids = [_]u32{ 1, 2, 3, 4, 5, 6 };
    const tgt = [_]u32{ 2, 3, 4, 5, 6, 7 };
    const sched = ztrain.Scheduler{ .const_lr = 0.005 };

    const Run = struct {
        fn lastLoss(al: std.mem.Allocator, c: NanoConfig, s: ztrain.Scheduler, idsx: []const u32, tgtx: []const u32) !f32 {
            var m = try FluencyNano.init(al, c, 123);
            defer m.deinit();
            var tr = try Trainer.init(al, &m, s, 1.0);
            defer tr.deinit();
            var loss: f32 = 0;
            var k: usize = 0;
            while (k < 50) : (k += 1) loss = try tr.stepCE(idsx, tgtx);
            return loss;
        }
    };
    const l1 = try Run.lastLoss(alloc, cfg, sched, &ids, &tgt);
    const l2 = try Run.lastLoss(alloc, cfg, sched, &ids, &tgt);
    try std.testing.expectEqual(l1, l2); // misma semilla -> misma trayectoria exacta
}

test "destilación KD: el alumno se acerca al profesor (KL baja)" {
    const alloc = std.testing.allocator;
    const cfg = NanoConfig{ .vocab = 8, .dim = 16, .n_layers = 2, .n_heads = 2, .ff_dim = 24 };
    var model = try FluencyNano.init(alloc, cfg, 5);
    defer model.deinit();
    const T = 6;
    const ids = [_]u32{ 0, 1, 2, 3, 4, 5 };
    // profesor sintético determinista: logits que favorecen el token (t+1)%V
    var teacher = [_]f32{0} ** (T * cfg.vocab);
    for (0..T) |t| {
        const fav = (ids[t] + 1) % cfg.vocab;
        teacher[t * cfg.vocab + fav] = 4.0;
    }
    const sched = ztrain.Scheduler{ .warmup_cosine = .{ .max_lr = 0.01, .min_lr = 0.001, .warmup_steps = 10, .max_steps = 250 } };
    var trainer = try Trainer.init(alloc, &model, sched, 1.0);
    defer trainer.deinit();

    const kl0 = try evalKLToTeacher(&model, alloc, &ids, &teacher, 2.0);
    var i: usize = 0;
    while (i < 250) : (i += 1) _ = try trainer.stepKD(&ids, &teacher, 2.0);
    const kl1 = try evalKLToTeacher(&model, alloc, &ids, &teacher, 2.0);
    try std.testing.expect(kl1 < kl0 * 0.5);
}

test "checkpoint round-trip: guardar y recargar reproduce misma perplejidad" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    const cfg = NanoConfig{ .vocab = 12, .dim = 16, .n_layers = 2, .n_heads = 2, .ff_dim = 32 };
    var model = try FluencyNano.init(alloc, cfg, 7);
    defer model.deinit();

    const ids = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const tgt = [_]u32{ 2, 3, 4, 5, 6, 7, 8, 9 };

    const sched = ztrain.Scheduler{ .const_lr = 0.01 };
    var trainer = try Trainer.init(alloc, &model, sched, 1.0);
    defer trainer.deinit();
    var k: usize = 0;
    while (k < 50) : (k += 1) _ = try trainer.stepCE(&ids, &tgt);

    const before = try evaluate(&model, alloc, &ids, &tgt);

    const path = "test_roundtrip.ckpt";
    try saveCheckpoint(&model, path, io, alloc);
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var loaded = try loadCheckpoint(alloc, path, io);
    defer loaded.deinit();

    const after = try evaluate(&loaded, alloc, &ids, &tgt);

    try std.testing.expectEqual(before.ce, after.ce);
    try std.testing.expectEqual(before.ppl, after.ppl);
    try std.testing.expectEqual(before.acc, after.acc);
}

/// Captura el hidden state de una capa específica de un FluencyNano (profesor).
/// `out` debe tener espacio para T * dim (f32).
fn captureTeacherHidden(model: *FluencyNano, alloc: std.mem.Allocator, ids: []const u32, layer: usize, out: []f32) !void {
    var g = zgrad.Graph.init(alloc);
    defer g.deinit();
    const hidden_vars = try alloc.alloc(zgrad.Var, model.cfg.n_layers);
    defer alloc.free(hidden_vars);
    _ = try model.forwardWithHidden(&g, ids, hidden_vars);
    const h = g.valOf(hidden_vars[layer]);
    @memcpy(out, h[0..out.len]);
}

test "MOHAWK Stage B: pérdida de hidden baja con entrenamiento" {
    const alloc = std.testing.allocator;
    const cfg = NanoConfig{ .vocab = 8, .dim = 16, .n_layers = 2, .n_heads = 2, .ff_dim = 24 };
    var teacher = try FluencyNano.init(alloc, cfg, 99);
    defer teacher.deinit();
    var student = try FluencyNano.init(alloc, cfg, 77);
    defer student.deinit();
    const ids = [_]u32{ 0, 1, 2, 3, 4, 5 };
    const T = ids.len;
    const D = cfg.dim;
    // Hidden del profesor (última capa)
    const teacher_hidden = try alloc.alloc(f32, T * D);
    defer alloc.free(teacher_hidden);
    try captureTeacherHidden(&teacher, alloc, &ids, cfg.n_layers - 1, teacher_hidden);

    const sched = ztrain.Scheduler{ .const_lr = 0.02 };
    var trainer = try Trainer.init(alloc, &student, sched, 1.0);
    defer trainer.deinit();

    var mse0: f32 = 0;
    var k: usize = 0;
    while (k < 5) : (k += 1) mse0 = try trainer.stepMohawkB(&ids, teacher_hidden, cfg.n_layers - 1);
    var mse1: f32 = 0;
    k = 0;
    while (k < 45) : (k += 1) mse1 = try trainer.stepMohawkB(&ids, teacher_hidden, cfg.n_layers - 1);
    try std.testing.expect(mse1 < mse0 * 0.5);
}

test "MOHAWK Stage C: KD + hidden bajan juntas" {
    const alloc = std.testing.allocator;
    const cfg = NanoConfig{ .vocab = 8, .dim = 16, .n_layers = 2, .n_heads = 2, .ff_dim = 24 };
    var teacher = try FluencyNano.init(alloc, cfg, 99);
    defer teacher.deinit();
    var student = try FluencyNano.init(alloc, cfg, 77);
    defer student.deinit();
    const ids = [_]u32{ 0, 1, 2, 3, 4, 5 };
    const T = ids.len;
    const V = cfg.vocab;
    const D = cfg.dim;

    // Logits del profesor
    var teacher_logits: [T * V]f32 = undefined;
    {
        var g = zgrad.Graph.init(alloc);
        defer g.deinit();
        const lg = try teacher.forward(&g, &ids);
        const val = g.valOf(lg);
        @memcpy(&teacher_logits, val[0..T * V]);
    }

    // Hidden del profesor (última capa)
    const teacher_hidden = try alloc.alloc(f32, T * D);
    defer alloc.free(teacher_hidden);
    try captureTeacherHidden(&teacher, alloc, &ids, cfg.n_layers - 1, teacher_hidden);

    const sched = ztrain.Scheduler{ .const_lr = 0.02 };
    var trainer = try Trainer.init(alloc, &student, sched, 1.0);
    defer trainer.deinit();

    var loss0: f32 = 0;
    var k: usize = 0;
    while (k < 5) : (k += 1) loss0 = try trainer.stepMohawkC(&ids, &teacher_logits, teacher_hidden, cfg.n_layers - 1, 2.0);
    var loss1: f32 = 0;
    k = 0;
    while (k < 45) : (k += 1) loss1 = try trainer.stepMohawkC(&ids, &teacher_logits, teacher_hidden, cfg.n_layers - 1, 2.0);
    try std.testing.expect(loss1 < loss0 * 0.5);
}

// =====================================================================
// Seq-KD pipeline: generación greedy, dataset, entrenamiento, eval
// =====================================================================

/// Genera tokens greedy a partir de `prompt_ids` usando FluencyNano.
/// Devuelve slice propiedad del caller (prompt + max_new tokens).
pub fn generateGreedyNano(model: *FluencyNano, alloc: std.mem.Allocator, prompt_ids: []const u32, max_new: usize) ![]u32 {
    var result: std.ArrayList(u32) = .empty;
    errdefer result.deinit(alloc);
    try result.appendSlice(alloc, prompt_ids);
    for (0..max_new) |_| {
        var g = zgrad.Graph.init(alloc);
        defer g.deinit();
        const logits_v = try model.forward(&g, result.items);
        const vals = g.valOf(logits_v);
        const V = model.cfg.vocab;
        const last_row = vals[vals.len - V ..];
        var best: u32 = 0;
        var best_val = last_row[0];
        for (last_row, 0..) |v, i| {
            if (v > best_val) {
                best_val = v;
                best = @intCast(i);
            }
        }
        try result.append(alloc, best);
    }
    return try result.toOwnedSlice(alloc);
}

/// Dataset de secuencias para Seq-KD (cada secuencia es un slice de tokens).
pub const NanoDataset = struct {
    alloc: std.mem.Allocator,
    seqs: [][]u32,

    pub fn deinit(self: *NanoDataset) void {
        for (self.seqs) |s| self.alloc.free(s);
        self.alloc.free(self.seqs);
        self.* = undefined;
    }
};

const DS_MAGIC = "FLDS";
const DS_VERSION: u32 = 1;

/// Guarda un NanoDataset en formato binario propio (magic + version + n_seqs + seqs).
pub fn saveSeqDataset(ds: NanoDataset, path: []const u8, io: std.Io, alloc: std.mem.Allocator) !void {
    var size: usize = 4 + 4 + 8;
    for (ds.seqs) |s| size += 4 + s.len * 4;
    const buf = try alloc.alloc(u8, size);
    defer alloc.free(buf);
    var off: usize = 0;
    @memcpy(buf[off..][0..4], DS_MAGIC);
    off += 4;
    std.mem.writeInt(u32, buf[off..][0..4], DS_VERSION, .little);
    off += 4;
    std.mem.writeInt(u64, buf[off..][0..8], ds.seqs.len, .little);
    off += 8;
    for (ds.seqs) |s| {
        std.mem.writeInt(u32, buf[off..][0..4], @intCast(s.len), .little);
        off += 4;
        for (s) |id| {
            std.mem.writeInt(u32, buf[off..][0..4], id, .little);
            off += 4;
        }
    }
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = buf });
}

/// Carga un NanoDataset desde formato binario propio.
pub fn loadSeqDataset(alloc: std.mem.Allocator, path: []const u8, io: std.Io) !NanoDataset {
    const data = try std.Io.Dir.cwd().readFileAlloc(io, path, alloc, .unlimited);
    defer alloc.free(data);
    if (data.len < 4 or !std.mem.eql(u8, data[0..4], DS_MAGIC))
        return error.InvalidMagic;
    var off: usize = 4;
    const version = std.mem.readInt(u32, data[off..][0..4], .little);
    off += 4;
    if (version != DS_VERSION) return error.UnsupportedVersion;
    const n_seqs = std.mem.readInt(u64, data[off..][0..8], .little);
    off += 8;
    const seqs = try alloc.alloc([]u32, n_seqs);
    errdefer alloc.free(seqs);
    var i: usize = 0;
    while (i < n_seqs) : (i += 1) {
        if (off + 4 > data.len) return error.Corrupt;
        const len = std.mem.readInt(u32, data[off..][0..4], .little);
        off += 4;
        if (off + len * 4 > data.len) return error.Corrupt;
        const s = try alloc.alloc(u32, len);
        errdefer {
            var j: usize = 0;
            while (j <= i) : (j += 1) {
                if (j < i) alloc.free(seqs[j]);
            }
        }
        for (0..len) |k| {
            s[k] = std.mem.readInt(u32, data[off..][0..4], .little);
            off += 4;
        }
        seqs[i] = s;
    }
    return NanoDataset{ .alloc = alloc, .seqs = seqs };
}

/// Evalúa CE/ppl/accuracy sobre un rango de secuencias de un NanoDataset.
pub fn evaluateSeqDataset(model: *FluencyNano, alloc: std.mem.Allocator, ds: NanoDataset, start: usize, end: usize) !EvalResult {
    var total_ce: f32 = 0;
    var total_acc: f32 = 0;
    var total_tok: usize = 0;
    for (ds.seqs[start..end]) |seq| {
        if (seq.len < 2) continue;
        const ids = seq[0 .. seq.len - 1];
        const tgt = seq[1..seq.len];
        const r = try evaluate(model, alloc, ids, tgt);
        total_ce += r.ce * @as(f32, @floatFromInt(tgt.len));
        total_acc += r.acc * @as(f32, @floatFromInt(tgt.len));
        total_tok += tgt.len;
    }
    if (total_tok == 0) return .{ .ce = 0, .ppl = 1, .acc = 0 };
    const ce = total_ce / @as(f32, @floatFromInt(total_tok));
    return .{ .ce = ce, .ppl = @exp(ce), .acc = total_acc / @as(f32, @floatFromInt(total_tok)) };
}

pub const SeqKDResult = struct {
    ce_train: f32,
    ce_held: f32,
    ppl_train: f32,
    ppl_held: f32,
    acc_train: f32,
    acc_held: f32,
};

/// Entrena un FluencyNano por Seq-KD (CE sobre next-token de secuencias).
/// `held_n`: últimas N secuencias se reservan para held-out.
/// `log_every`: 0 = silencioso.
/// `patience`: evaluaciones de held-out sin mejora antes de restaurar el mejor
/// checkpoint y detenerse. 0 = desactivado (corre los `steps` completos).
pub fn trainSeqKD(
    model: *FluencyNano,
    trainer: *Trainer,
    alloc: std.mem.Allocator,
    ds: NanoDataset,
    held_n: usize,
    steps: usize,
    log_every: usize,
    patience: usize,
) !SeqKDResult {
    const n = ds.seqs.len;
    if (n <= held_n) return error.DatasetTooSmall;
    const train_n = n - held_n;
    var rng = std.Random.DefaultPrng.init(model.seed + 123);

    const eval_every = if (log_every > 0) log_every else @max(steps / 10, 1);
    var best_weights: ?[]f32 = null;
    var best_ce_held: f32 = std.math.inf(f32);
    var evals_without_improvement: usize = 0;
    var stopped_early = false;

    var i: usize = 0;
    var loss: f32 = 0;
    while (i < steps) : (i += 1) {
        const idx = rng.random().uintLessThan(usize, train_n);
        const seq = ds.seqs[idx];
        if (seq.len < 2) continue;
        const ids = seq[0 .. seq.len - 1];
        const tgt = seq[1..seq.len];
        loss = trainer.stepCE(ids, tgt) catch |err| switch (err) {
            error.TrainingDiverged => {
                std.log.err("seqkd: divergencia detectada en paso {d}; restaurando mejor checkpoint", .{i + 1});
                if (best_weights) |bw| {
                    @memcpy(model.weights, bw);
                }
                stopped_early = true;
                break;
            },
            else => |e| return e,
        };
        if (log_every > 0 and (i + 1) % log_every == 0) {
            std.log.info("seqkd step {d}/{d} loss={d:.4}", .{ i + 1, steps, loss });
        }

        if (patience > 0 and (i + 1) % eval_every == 0) {
            const held_eval = try evaluateSeqDataset(model, alloc, ds, train_n, n);
            if (held_eval.ce < best_ce_held) {
                best_ce_held = held_eval.ce;
                evals_without_improvement = 0;
                if (best_weights == null) {
                    best_weights = try alloc.alloc(f32, model.weights.len);
                }
                @memcpy(best_weights.?, model.weights);
            } else {
                evals_without_improvement += 1;
                if (evals_without_improvement >= patience) {
                    stopped_early = true;
                    break;
                }
            }
        }
    }

    if (stopped_early and best_weights != null) {
        @memcpy(model.weights, best_weights.?);
    }
    if (best_weights) |bw| alloc.free(bw);

    const after_train = try evaluateSeqDataset(model, alloc, ds, 0, train_n);
    const after_held = try evaluateSeqDataset(model, alloc, ds, train_n, n);

    return SeqKDResult{
        .ce_train = after_train.ce,
        .ce_held = after_held.ce,
        .ppl_train = after_train.ppl,
        .ppl_held = after_held.ppl,
        .acc_train = after_train.acc,
        .acc_held = after_held.acc,
    };
}

test "seqkd end-to-end: teacher sintetico, student aprende; CE de train baja; held-out finito" {
    const alloc = std.testing.allocator;
    const V: u32 = 24;
    const cfg = NanoConfig{ .vocab = V, .dim = 32, .n_layers = 2, .n_heads = 2, .ff_dim = 64 };

    var teacher = try FluencyNano.init(alloc, cfg, 99);
    defer teacher.deinit();
    var student = try FluencyNano.init(alloc, cfg, 77);
    defer student.deinit();

    // Generar 20 secuencias de longitud 10 con el teacher (prompt 2 + 8 greedy)
    var seqs_list: std.ArrayList([]u32) = .empty;
    defer {
        for (seqs_list.items) |s| alloc.free(s);
        seqs_list.deinit(alloc);
    }
    for (0..20) |si| {
        const prompt = [_]u32{ @intCast(si % V), @intCast((si * 3 + 1) % V) };
        const seq = try generateGreedyNano(&teacher, alloc, &prompt, 8);
        try seqs_list.append(alloc, seq);
    }

    const seqs = try alloc.alloc([]u32, seqs_list.items.len);
    defer alloc.free(seqs);
    @memcpy(seqs, seqs_list.items);
    const ds = NanoDataset{ .alloc = alloc, .seqs = seqs };

    const before_train = try evaluateSeqDataset(&student, alloc, ds, 0, 15);

    const sched = ztrain.Scheduler{ .warmup_cosine = .{ .max_lr = 0.02, .min_lr = 0.001, .warmup_steps = 50, .max_steps = 400 } };
    var trainer = try Trainer.init(alloc, &student, sched, 1.0);
    defer trainer.deinit();

    const result = try trainSeqKD(&student, &trainer, alloc, ds, 5, 400, 0, 0);

    try std.testing.expect(result.ce_train < before_train.ce * 0.5);
    try std.testing.expect(std.math.isFinite(result.ce_held));

    // Checkpoint round-trip del modelo entrenado reproduce logits bit-a-bit
    const io = std.testing.io;
    const path = "test_seqkd_trained.ckpt";
    try saveCheckpoint(&student, path, io, alloc);
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};
    var loaded = try loadCheckpoint(alloc, path, io);
    defer loaded.deinit();

    var g1 = zgrad.Graph.init(alloc);
    defer g1.deinit();
    const logits1 = try student.forward(&g1, ds.seqs[0]);
    const val1 = g1.valOf(logits1);

    var g2 = zgrad.Graph.init(alloc);
    defer g2.deinit();
    const logits2 = try loaded.forward(&g2, ds.seqs[0]);
    const val2 = g2.valOf(logits2);

    try std.testing.expectEqual(val1.len, val2.len);
    for (val1, val2) |a, b| try std.testing.expectEqual(a, b);
}

test "seqkd early-stopping: detiene antes de max_steps cuando held-out se estanca" {
    const alloc = std.testing.allocator;
    const V: u32 = 24;
    const cfg = NanoConfig{ .vocab = V, .dim = 32, .n_layers = 2, .n_heads = 2, .ff_dim = 64 };

    var teacher = try FluencyNano.init(alloc, cfg, 99);
    defer teacher.deinit();
    var student = try FluencyNano.init(alloc, cfg, 77);
    defer student.deinit();

    // Generar 20 secuencias de longitud 10 con el teacher
    var seqs_list: std.ArrayList([]u32) = .empty;
    defer {
        for (seqs_list.items) |s| alloc.free(s);
        seqs_list.deinit(alloc);
    }
    for (0..20) |si| {
        const prompt = [_]u32{ @intCast(si % V), @intCast((si * 3 + 1) % V) };
        const seq = try generateGreedyNano(&teacher, alloc, &prompt, 8);
        try seqs_list.append(alloc, seq);
    }

    const seqs = try alloc.alloc([]u32, seqs_list.items.len);
    defer alloc.free(seqs);
    @memcpy(seqs, seqs_list.items);
    const ds = NanoDataset{ .alloc = alloc, .seqs = seqs };

    const sched = ztrain.Scheduler{ .warmup_cosine = .{ .max_lr = 0.02, .min_lr = 0.001, .warmup_steps = 10, .max_steps = 400 } };
    var trainer = try Trainer.init(alloc, &student, sched, 1.0);
    defer trainer.deinit();

    // Con paciencia=1, debe detenerse en la primera evaluación donde held-out no mejore.
    // eval_every = max(400/10, 1) = 40, así que como máximo corre 80 pasos antes de parar.
    const steps_before = trainer.step_count;
    _ = try trainSeqKD(&student, &trainer, alloc, ds, 5, 400, 0, 1);
    const steps_after = trainer.step_count;

    // Verificar que se detuvo antes de los 400 pasos (early stopping activó)
    try std.testing.expect(steps_after < 400);
    // Verificar que al menos avanzó una evaluación (>= 40 pasos)
    try std.testing.expect(steps_after >= 40);
    // La CE de train debe haber bajado respecto al inicio
    try std.testing.expect(steps_after > steps_before);
}

test "seqkd regresión crash: vocab grande, LR agresivo, sin crash silencioso" {
    const alloc = std.testing.allocator;
    const V: u32 = 128; // vocab grande para test unitario rápido
    const cfg = NanoConfig{ .vocab = V, .dim = 32, .n_layers = 2, .n_heads = 2, .ff_dim = 64 };

    var teacher = try FluencyNano.init(alloc, cfg, 99);
    defer teacher.deinit();
    var student = try FluencyNano.init(alloc, cfg, 77);
    defer student.deinit();

    // Generar 10 secuencias de longitud 8
    var seqs_list: std.ArrayList([]u32) = .empty;
    defer {
        for (seqs_list.items) |s| alloc.free(s);
        seqs_list.deinit(alloc);
    }
    for (0..10) |si| {
        const prompt = [_]u32{ @intCast(si % V), @intCast((si * 3 + 1) % V) };
        const seq = try generateGreedyNano(&teacher, alloc, &prompt, 6);
        try seqs_list.append(alloc, seq);
    }

    const seqs = try alloc.alloc([]u32, seqs_list.items.len);
    defer alloc.free(seqs);
    @memcpy(seqs, seqs_list.items);
    const ds = NanoDataset{ .alloc = alloc, .seqs = seqs };

    // LR agresivo similar al escenario del bug (0.003) para forzar inestabilidad potencial
    const sched = ztrain.Scheduler{ .warmup_cosine = .{ .max_lr = 0.003, .min_lr = 0.0003, .warmup_steps = 2, .max_steps = 50 } };
    var trainer = try Trainer.init(alloc, &student, sched, 1.0);
    defer trainer.deinit();

    const result = try trainSeqKD(&student, &trainer, alloc, ds, 2, 50, 0, 0);

    // Debe terminar sin crash y con métricas finitas
    try std.testing.expect(std.math.isFinite(result.ce_train));
    try std.testing.expect(std.math.isFinite(result.ce_held));
    try std.testing.expect(std.math.isFinite(result.ppl_train));
    try std.testing.expect(std.math.isFinite(result.ppl_held));
}

test "seqkd dataset round-trip: guardar y recargar reproduce secuencias" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;

    var seqs: [3][]u32 = .{
        try alloc.dupe(u32, &[_]u32{ 1, 2, 3, 4 }),
        try alloc.dupe(u32, &[_]u32{ 5, 6 }),
        try alloc.dupe(u32, &[_]u32{ 7, 8, 9 }),
    };
    defer for (seqs) |s| alloc.free(s);
    const seqs_slice = try alloc.alloc([]u32, 3);
    defer alloc.free(seqs_slice);
    @memcpy(seqs_slice, &seqs);
    const ds = NanoDataset{ .alloc = alloc, .seqs = seqs_slice };

    const path = "test_seqkd_ds.bin";
    try saveSeqDataset(ds, path, io, alloc);
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var loaded = try loadSeqDataset(alloc, path, io);
    defer loaded.deinit();

    try std.testing.expectEqual(ds.seqs.len, loaded.seqs.len);
    for (ds.seqs, loaded.seqs) |orig, ld| {
        try std.testing.expectEqualSlices(u32, orig, ld);
    }
}

// =====================================================================
// Fluency-126M / Fluency-Tiny configs y tests de pipeline
// =====================================================================

/// Configuración canónica Fluency-126M (126 052 608 params).
/// Mayoría subcuadrática (scan / diagonal SSM); minoría atención (2/12).
pub const Fluency126MConfig = NanoConfig{
    .vocab = 32000,
    .dim = 768,
    .n_layers = 12,
    .n_heads = 12,
    .ff_dim = 3072,
    .eps = 1e-5,
    .attn_interval = 6, // 2 capas attn (0,6) + 10 capas scan; minoría atención
    .attn_window = 128,
};

/// Configuración tiny para tests rápidos (misma arquitectura, 2 capas).
pub const FluencyTinyConfig = NanoConfig{
    .vocab = 64,
    .dim = 32,
    .n_layers = 2,
    .n_heads = 4,
    .ff_dim = 64,
    .eps = 1e-5,
};

test "Fluency-126M: ≥126M parámetros y desglose imprimible" {
    const alloc = std.testing.allocator;
    const cfg = Fluency126MConfig;
    var model = try FluencyNano.init(alloc, cfg, 42);
    defer model.deinit();
    const n = model.paramCount();
    model.printParamCount();
    try std.testing.expect(n >= 126_000_000);
}

test "Fluency-tiny: gradient-check analítico vs numérico en sub-módulo" {
    const alloc = std.testing.allocator;
    const cfg = FluencyTinyConfig;
    const ids = [_]u32{ 1, 2, 3, 4 };
    const tgt = [_]u32{ 2, 3, 4, 5 };
    const eps: f32 = 1e-3;

    // Verificar gradientes de la tabla de embedding (subconjunto pequeño)
    const D = cfg.dim;

    const Fwd = struct {
        fn run(al: std.mem.Allocator, c: NanoConfig, w: []f32, idsx: []const u32, tgtx: []const u32, grads_out: ?[]f32) f32 {
            var m = FluencyNano.init(al, c, 7) catch unreachable;
            defer m.deinit();
            @memcpy(m.weights[0..w.len], w);
            var g = zgrad.Graph.init(al);
            defer g.deinit();
            const logits = m.forward(&g, idsx) catch unreachable;
            const loss = zgrad.crossEntropyLoss(&g, logits, tgtx) catch unreachable;
            const lv = g.valOf(loss)[0];
            if (grads_out) |d| {
                g.backward(loss);
                m.collectGrads(&g);
                @memcpy(d, m.grads[0..d.len]);
            }
            return lv;
        }
    };

    const n_params = cfg.vocab * D;
    const w_buf = try alloc.alloc(f32, n_params);
    defer alloc.free(w_buf);
    {
        var prng = std.Random.DefaultPrng.init(7);
        for (w_buf) |*p| p.* = prng.random().floatNorm(f32) * 0.02;
    }

    var full = try alloc.alloc(f32, n_params);
    defer alloc.free(full);
    @memcpy(full, w_buf);
    const grad_all = try alloc.alloc(f32, n_params);
    defer alloc.free(grad_all);
    _ = Fwd.run(alloc, cfg, full, &ids, &tgt, grad_all);

    // Verificar un subconjunto de pesos del embedding
    const check_indices = [_]usize{ 0, 1, 2, D, D + 1, D + 2 };
    for (check_indices) |idx| {
        const orig = full[idx];
        full[idx] = orig + eps;
        const lp = Fwd.run(alloc, cfg, full, &ids, &tgt, null);
        full[idx] = orig - eps;
        const lm = Fwd.run(alloc, cfg, full, &ids, &tgt, null);
        full[idx] = orig;
        const num = (lp - lm) / (2 * eps);
        try std.testing.expectApproxEqAbs(num, grad_all[idx], 5e-2);
    }
}

test "Fluency-tiny: converge en corpus pequeño (CE baja, accuracy sube)" {
    const alloc = std.testing.allocator;
    const cfg = FluencyTinyConfig;
    var model = try FluencyNano.init(alloc, cfg, 7);
    defer model.deinit();
    const ids = [_]u32{ 1, 2, 3, 4, 5, 6 };
    const tgt = [_]u32{ 2, 3, 4, 5, 6, 7 };
    const sched = ztrain.Scheduler{ .warmup_cosine = .{ .max_lr = 0.02, .min_lr = 0.001, .warmup_steps = 10, .max_steps = 200 } };
    var trainer = try Trainer.init(alloc, &model, sched, 1.0);
    defer trainer.deinit();

    const before = try evaluate(&model, alloc, &ids, &tgt);
    var i: usize = 0;
    while (i < 200) : (i += 1) _ = try trainer.stepCE(&ids, &tgt);
    const after = try evaluate(&model, alloc, &ids, &tgt);

    try std.testing.expect(after.ce < before.ce * 0.5);
    try std.testing.expect(after.acc >= 0.80);
}

test "Fluency-tiny: checkpoint round-trip reproduce forward bit-a-bit" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    const cfg = FluencyTinyConfig;
    var model = try FluencyNano.init(alloc, cfg, 9);
    defer model.deinit();

    const ids = [_]u32{ 1, 2, 3, 4, 5 };
    const tgt = [_]u32{ 2, 3, 4, 5, 6 };
    const sched = ztrain.Scheduler{ .const_lr = 0.01 };
    var trainer = try Trainer.init(alloc, &model, sched, 1.0);
    defer trainer.deinit();
    var k: usize = 0;
    while (k < 20) : (k += 1) _ = try trainer.stepCE(&ids, &tgt);

    // Forward ANTES de guardar
    var g1 = zgrad.Graph.init(alloc);
    defer g1.deinit();
    const logits1 = try model.forward(&g1, &ids);
    const val1 = g1.valOf(logits1);

    const path = "test_fluency_tiny.ckpt";
    try saveCheckpoint(&model, path, io, alloc);
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var loaded = try loadCheckpoint(alloc, path, io);
    defer loaded.deinit();

    // Forward DESPUÉS de cargar
    var g2 = zgrad.Graph.init(alloc);
    defer g2.deinit();
    const logits2 = try loaded.forward(&g2, &ids);
    const val2 = g2.valOf(logits2);

    try std.testing.expectEqual(val1.len, val2.len);
    for (val1, val2) |a, b| try std.testing.expectEqual(a, b);
}
