// zarch — bloque híbrido + mixers + MoE
const std = @import("std");
const zkernel = @import("zkernel");

// F2: mixers subcuadráticos (reparados). El camino F1 sólo usa el Transformer;
// los mixers se exponen tras el trait `Mixer` para el bloque híbrido.
pub const mixer = @import("mixers/mixer.zig");
pub const swa = @import("mixers/swa.zig");
pub const mamba2 = @import("mixers/mamba2.zig");
pub const gated_deltanet = @import("mixers/gated_deltanet.zig");
pub const rwkv7 = @import("mixers/rwkv7.zig");
pub const hybrid = @import("mixers/hybrid.zig");
pub const grammar = @import("grammar.zig");

pub const Config = struct {
    dim: usize,
    hidden_dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    vocab_size: usize,
    seq_len: usize,
    theta: f32 = 10000.0,
    eps: f32 = 1e-5,
    // Convención de RoPE: true = NEOX (rota mitades i,i+d/2; Qwen2/Llama en GGUF),
    // false = GPT-J (pares adyacentes 2i,2i+1). Irrelevante en pos=0.
    rope_neox: bool = true,
    // MoE config (0 = single FFN)
    n_experts: usize = 0,
    top_k: usize = 2,
    moe_hidden_dim: usize = 0,
};

pub const QuantFormat = enum {
    f32,
    q4_0,
    q8_0,
    q4_k,
    q6_k,
};

/// Embedding table cuantizada (formato único, blob crudo GGUF [vocab, dim]).
pub const QuantEmbedding = struct {
    bytes: []const u8,
    fmt: QuantFormat,
};

/// Peso lineal cuantizado con formato y bytes POR CAPA. Necesario porque los
/// K-quant reales (p.ej. Qwen2.5 Q4_K_M) NO solo mezclan formatos entre tipos de
/// peso, sino también ENTRE CAPAS del mismo tipo (ffn_down/attn_v alternan Q4_K y
/// Q6_K según la capa), con tamaños de bloque distintos → no se pueden concatenar
/// con stride uniforme. `layers[l]` son los bytes crudos de la capa l y `fmts[l]`
/// su formato. Ambos slices tienen longitud n_layers.
pub const QuantWeight = struct {
    layers: []const []const u8,
    fmts: []const QuantFormat,
};

pub const Transformer = struct {
    config: Config,
    allocator: std.mem.Allocator,

    // Pesos planos (punteros a datos del GGUF)
    token_embedding_table: []const f32,
    token_embedding_table_q: ?QuantEmbedding = null,
    rms_att_weight: []const f32,
    rms_ffn_weight: []const f32,
    wq: []const f32,
    wk: []const f32,
    wv: []const f32,
    wo: []const f32,
    w1: []const f32,
    w2: []const f32,
    w3: []const f32,
    rms_final_weight: []const f32,
    wcls: ?[]const f32,
    // Bias QKV (Qwen2 los tiene; Llama no). null = sin bias.
    bq: ?[]const f32 = null,
    bk: ?[]const f32 = null,
    bv: ?[]const f32 = null,
    // MoE weights (null si single FFN)
    moe_router: ?[]const f32 = null,
    moe_w1: ?[]const f32 = null,
    moe_w2: ?[]const f32 = null,
    moe_w3: ?[]const f32 = null,

    // Pesos lineales cuantizados (layout GGUF original [out, in]) con formato y
    // bytes POR CAPA. Si no son null, se usan en lugar de los `[]const f32`.
    // El formato puede variar entre capas del mismo tipo (ver QuantWeight).
    wq_q: ?QuantWeight = null,
    wk_q: ?QuantWeight = null,
    wv_q: ?QuantWeight = null,
    wo_q: ?QuantWeight = null,
    w1_q: ?QuantWeight = null,
    w2_q: ?QuantWeight = null,
    w3_q: ?QuantWeight = null,

    // Estado durante forward
    x: []f32,
    xb: []f32,
    xb2: []f32,
    hb: []f32,
    hb2: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    att: []f32,
    logits: []f32,
    key_cache: []f32,
    value_cache: []f32,
    // MoE scratch (vacíos si single FFN)
    moe_hb: []f32 = &.{},
    moe_hb2: []f32 = &.{},
    moe_router_logits: []f32 = &.{},
    moe_expert_out: []f32 = &.{},
    // forwardBatch scratch (lazy-allocated, reused across calls)
    batch_cap: usize = 0,
    batch_x: ?[]f32 = null,
    batch_xb: ?[]f32 = null,
    batch_q: ?[]f32 = null,
    batch_k: ?[]f32 = null,
    batch_v: ?[]f32 = null,

    pub fn init(alloc: std.mem.Allocator, cfg: Config) !Transformer {
        const dim = cfg.dim;
        const hidden_dim = cfg.hidden_dim;
        const kv_dim = (cfg.dim * cfg.n_kv_heads) / cfg.n_heads;
        const moe_dim = if (cfg.moe_hidden_dim > 0) cfg.moe_hidden_dim else hidden_dim;
        const has_moe = cfg.n_experts > 0;
        return Transformer{
            .config = cfg,
            .allocator = alloc,
            .token_embedding_table = &.{},
            .rms_att_weight = &.{},
            .rms_ffn_weight = &.{},
            .wq = &.{},
            .wk = &.{},
            .wv = &.{},
            .wo = &.{},
            .w1 = &.{},
            .w2 = &.{},
            .w3 = &.{},
            .rms_final_weight = &.{},
            .wcls = null,
            .x = try alloc.alloc(f32, dim),
            .xb = try alloc.alloc(f32, dim),
            .xb2 = try alloc.alloc(f32, dim),
            .hb = try alloc.alloc(f32, hidden_dim),
            .hb2 = try alloc.alloc(f32, hidden_dim),
            .q = try alloc.alloc(f32, dim),
            .k = try alloc.alloc(f32, kv_dim),
            .v = try alloc.alloc(f32, kv_dim),
            .att = try alloc.alloc(f32, cfg.seq_len),
            .logits = try alloc.alloc(f32, cfg.vocab_size),
            .key_cache = try alloc.alloc(f32, cfg.n_layers * cfg.seq_len * kv_dim),
            .value_cache = try alloc.alloc(f32, cfg.n_layers * cfg.seq_len * kv_dim),
            .moe_hb = if (has_moe) try alloc.alloc(f32, moe_dim) else &.{},
            .moe_hb2 = if (has_moe) try alloc.alloc(f32, moe_dim) else &.{},
            .moe_router_logits = if (has_moe) try alloc.alloc(f32, cfg.n_experts) else &.{},
            .moe_expert_out = if (has_moe) try alloc.alloc(f32, dim) else &.{},
        };
    }

    pub fn deinit(self: *Transformer) void {
        self.allocator.free(self.x);
        self.allocator.free(self.xb);
        self.allocator.free(self.xb2);
        self.allocator.free(self.hb);
        self.allocator.free(self.hb2);
        self.allocator.free(self.q);
        self.allocator.free(self.k);
        self.allocator.free(self.v);
        self.allocator.free(self.att);
        self.allocator.free(self.logits);
        self.allocator.free(self.key_cache);
        self.allocator.free(self.value_cache);
        if (self.moe_hb.len > 0) self.allocator.free(self.moe_hb);
        if (self.moe_hb2.len > 0) self.allocator.free(self.moe_hb2);
        if (self.moe_router_logits.len > 0) self.allocator.free(self.moe_router_logits);
        if (self.moe_expert_out.len > 0) self.allocator.free(self.moe_expert_out);
        if (self.batch_x) |b| self.allocator.free(b);
        if (self.batch_xb) |b| self.allocator.free(b);
        if (self.batch_q) |b| self.allocator.free(b);
        if (self.batch_k) |b| self.allocator.free(b);
        if (self.batch_v) |b| self.allocator.free(b);
    }

    fn forwardMoE(self: *Transformer, xb: []const f32, layer: usize) void {
        const cfg = self.config;
        const dim = cfg.dim;
        const ne = cfg.n_experts;
        const top_k = cfg.top_k;
        const moe_dim = if (cfg.moe_hidden_dim > 0) cfg.moe_hidden_dim else cfg.hidden_dim;
        const router_w = self.moe_router.?;
        const w1 = self.moe_w1.?;
        const w2 = self.moe_w2.?;
        const w3 = self.moe_w3.?;

        // Router: gating logits = xb @ W_router  [dim * ne]
        zkernel.matmul(self.moe_router_logits, xb, router_w[layer * dim * ne ..][0..(dim * ne)], 1, dim, ne);

        // Softmax sobre logits del router
        var max_logit: f32 = self.moe_router_logits[0];
        for (self.moe_router_logits) |v| {
            if (v > max_logit) max_logit = v;
        }
        var sum_exp: f32 = 0;
        for (self.moe_router_logits) |*v| {
            v.* = @exp(v.* - max_logit);
            sum_exp += v.*;
        }
        const inv_sum = 1.0 / sum_exp;
        for (self.moe_router_logits) |*v| v.* *= inv_sum;

        // Top-k: encontrar los k expertos con mayor peso
        var top_indices: [4]usize = undefined;
        var top_weights: [4]f32 = undefined;
        const k = @min(top_k, ne);
        for (0..k) |i| {
            var best_idx: usize = 0;
            var best_val: f32 = -1;
            for (self.moe_router_logits, 0..) |v, ei| {
                var already_selected = false;
                for (0..i) |j| if (top_indices[j] == ei) { already_selected = true; break; };
                if (!already_selected and v > best_val) {
                    best_val = v;
                    best_idx = ei;
                }
            }
            top_indices[i] = best_idx;
            top_weights[i] = best_val;
        }
        // Normalizar pesos top-k
        var w_sum: f32 = 0;
        for (top_weights[0..k]) |w| w_sum += w;
        const inv_w_sum = if (w_sum > 0) 1.0 / w_sum else 1.0;
        for (top_weights[0..k]) |*w| w.* *= inv_w_sum;

        // Combinar salida de expertos
        @memset(self.moe_expert_out, 0);
        for (0..k) |ei| {
            const expert = top_indices[ei];
            const weight = top_weights[ei];
            const off = expert * dim * moe_dim;

            zkernel.matmul(self.moe_hb, xb, w1[off..][0..(dim * moe_dim)], 1, dim, moe_dim);
            zkernel.matmul(self.moe_hb2, xb, w3[off..][0..(dim * moe_dim)], 1, dim, moe_dim);
            for (0..moe_dim) |i| {
                self.moe_hb[i] = self.moe_hb[i] * (1.0 / (1.0 + @exp(-self.moe_hb[i]))) * self.moe_hb2[i];
            }
            zkernel.matmul(self.xb2, self.moe_hb, w2[expert * moe_dim * dim ..][0..(moe_dim * dim)], 1, moe_dim, dim);
            for (0..dim) |i| {
                self.moe_expert_out[i] += weight * self.xb2[i];
            }
        }
        // Residual
        for (0..dim) |i| self.x[i] += self.moe_expert_out[i];
    }

    // Helper: matmul vector × matriz cuantizada o f32.
    // weight_f32 layout [in,out]; el peso cuantizado layout GGUF [out,in].
    // `qw` es el peso cuantizado por-capa (o null → usa weight_f32); `layer`
    // selecciona la capa (su slice de bytes y su formato propios).
    fn matmulVec(self: *Transformer, out: []f32, weight_f32: ?[]const f32, qw: ?QuantWeight, layer: usize, N: usize, K: usize) void {
        self.matmulVecIn(out, self.xb, weight_f32, qw, layer, N, K);
    }

    // Igual que matmulVec pero con vector de entrada distinto a self.xb.
    fn matmulVecIn(_: *Transformer, out: []f32, inp: []const f32, weight_f32: ?[]const f32, qw: ?QuantWeight, layer: usize, N: usize, K: usize) void {
        if (qw) |q| {
            const wq = q.layers[layer];
            switch (q.fmts[layer]) {
                .f32 => unreachable, // un peso con bytes cuantizados nunca tiene fmt f32
                .q4_0 => zkernel.matmulVecQ4_0(out, wq, inp, N, K),
                .q8_0 => zkernel.matmulVecQ8_0(out, wq, inp, N, K),
                .q4_k => zkernel.matmulVecQ4_K(out, wq, inp, N, K),
                .q6_k => zkernel.matmulVecQ6_K(out, wq, inp, N, K),
            }
        } else {
            zkernel.matmul(out, inp, weight_f32.?, 1, K, N);
        }
    }

    // Matmul batch: out[M,N] = inp[M,K] @ weight[K,N] (f32) o weight cuantizado por fila.
    fn matmulBatch(_: *Transformer, out: []f32, inp: []const f32, weight_f32: ?[]const f32, qw: ?QuantWeight, layer: usize, M: usize, N: usize, K: usize) void {
        if (qw) |q| {
            const wq = q.layers[layer];
            switch (q.fmts[layer]) {
                .f32 => unreachable,
                .q4_0 => zkernel.matmulBatchQ4_0(out, wq, inp, M, N, K),
                .q8_0 => zkernel.matmulBatchQ8_0(out, wq, inp, M, N, K),
                .q4_k => zkernel.matmulBatchQ4_K(out, wq, inp, M, N, K),
                .q6_k => zkernel.matmulBatchQ6_K(out, wq, inp, M, N, K),
            }
        } else {
            for (0..M) |m| zkernel.matmul(out[m * N ..][0..N], inp[m * K ..][0..K], weight_f32.?, 1, K, N);
        }
    }

    pub fn forward(self: *Transformer, token: u32, pos: usize) []f32 {
        const cfg = self.config;
        const dim = cfg.dim;
        const kv_dim = (dim * cfg.n_kv_heads) / cfg.n_heads;
        const kv_mul = cfg.n_heads / cfg.n_kv_heads;
        const head_size = dim / cfg.n_heads;
        const hidden_dim = cfg.hidden_dim;

        // Embedding
        if (self.token_embedding_table_q) |emb_q| {
            switch (emb_q.fmt) {
                .f32 => unreachable,
                .q4_0 => zkernel.dequantRowQ4_0(emb_q.bytes, token, self.x, dim),
                .q8_0 => zkernel.dequantRowQ8_0(emb_q.bytes, token, self.x, dim),
                .q4_k => zkernel.dequantRowQ4_K(emb_q.bytes, token, self.x, dim),
                .q6_k => zkernel.dequantRowQ6_K(emb_q.bytes, token, self.x, dim),
            }
        } else {
            const content_row = self.token_embedding_table[token * dim ..][0..dim];
            @memcpy(self.x, content_row);
        }

        for (0..cfg.n_layers) |l| {
            // --- Attention ---
            zkernel.rmsnorm(self.xb, self.x, self.rms_att_weight[l * dim ..][0..dim], cfg.eps);

            // QKV
            self.matmulVec(self.q, if (self.wq_q == null) self.wq[l * dim * dim ..][0..(dim * dim)] else null, self.wq_q, l, dim, dim);
            self.matmulVec(self.k, if (self.wk_q == null) self.wk[l * kv_dim * dim ..][0..(kv_dim * dim)] else null, self.wk_q, l, kv_dim, dim);
            self.matmulVec(self.v, if (self.wv_q == null) self.wv[l * kv_dim * dim ..][0..(kv_dim * dim)] else null, self.wv_q, l, kv_dim, dim);

            // Bias QKV (Qwen2)
            if (self.bq) |bq| {
                for (0..dim) |i| self.q[i] += bq[l * dim + i];
            }
            if (self.bk) |bk| {
                for (0..kv_dim) |i| self.k[i] += bk[l * kv_dim + i];
            }
            if (self.bv) |bv| {
                for (0..kv_dim) |i| self.v[i] += bv[l * kv_dim + i];
            }

            // RoPE (NEOX para Qwen2/Llama-GGUF; GPT-J en otro caso).
            // Importante con GQA: rotar cada cabeza Q y cada cabeza KV UNA sola vez.
            for (0..cfg.n_heads) |h| {
                const q_h = self.q[h * head_size ..][0..head_size];
                if (cfg.rope_neox) zkernel.ropeVecNeox(q_h, pos, head_size, cfg.theta) else zkernel.ropeVecGptj(q_h, pos, head_size, cfg.theta);
            }
            for (0..cfg.n_kv_heads) |h| {
                const k_h = self.k[h * head_size ..][0..head_size];
                if (cfg.rope_neox) zkernel.ropeVecNeox(k_h, pos, head_size, cfg.theta) else zkernel.ropeVecGptj(k_h, pos, head_size, cfg.theta);
            }

            // KV cache
            const loff = l * cfg.seq_len * kv_dim;
            @memcpy(self.key_cache[loff + pos * kv_dim ..][0..kv_dim].ptr, self.k);
            @memcpy(self.value_cache[loff + pos * kv_dim ..][0..kv_dim].ptr, self.v);

            // Multi-head attention
            @memset(self.xb2, 0);
            for (0..cfg.n_heads) |h| {
                const q_head = self.q[h * head_size ..][0..head_size];
                var att_slice = self.att[0..(pos + 1)];
                for (0..pos + 1) |t| {
                    const k_cache = self.key_cache[loff + t * kv_dim + (h / kv_mul) * head_size ..][0..head_size];
                    var score: f32 = 0;
                    for (0..head_size) |i| score += q_head[i] * k_cache[i];
                    score /= @sqrt(@as(f32, @floatFromInt(head_size)));
                    att_slice[t] = score;
                }
                zkernel.softmax(att_slice, att_slice);
                const out_head = self.xb2[h * head_size ..][0..head_size];
                for (0..head_size) |i| out_head[i] = 0;
                for (0..pos + 1) |t| {
                    const v_cache = self.value_cache[loff + t * kv_dim + (h / kv_mul) * head_size ..][0..head_size];
                    const a = att_slice[t];
                    for (0..head_size) |i| out_head[i] += a * v_cache[i];
                }
            }

            // Projection + residual
            self.matmulVecIn(self.xb, self.xb2, if (self.wo_q == null) self.wo[l * dim * dim ..][0..(dim * dim)] else null, self.wo_q, l, dim, dim);
            for (0..dim) |i| self.x[i] += self.xb[i];

            // --- FFN / MoE ---
            zkernel.rmsnorm(self.xb, self.x, self.rms_ffn_weight[l * dim ..][0..dim], cfg.eps);
            if (cfg.n_experts > 0) {
                self.forwardMoE(self.xb, l);
            } else {
                self.matmulVec(self.hb, if (self.w1_q == null) self.w1[l * dim * hidden_dim ..][0..(dim * hidden_dim)] else null, self.w1_q, l, hidden_dim, dim);
                self.matmulVec(self.hb2, if (self.w3_q == null) self.w3[l * dim * hidden_dim ..][0..(dim * hidden_dim)] else null, self.w3_q, l, hidden_dim, dim);
                for (0..hidden_dim) |i| self.hb[i] = self.hb[i] * (1.0 / (1.0 + @exp(-self.hb[i]))) * self.hb2[i]; // SiLU * gate
                self.matmulVecIn(self.xb, self.hb, if (self.w2_q == null) self.w2[l * hidden_dim * dim ..][0..(hidden_dim * dim)] else null, self.w2_q, l, dim, hidden_dim);
                for (0..dim) |i| self.x[i] += self.xb[i];
            }
        }

        // Final norm + logits
        zkernel.rmsnorm(self.x, self.x, self.rms_final_weight, cfg.eps);
        if (self.wcls) |wcls| {
            // output.weight separado, cargado en layout [embd, vocab] (transpuesto).
            zkernel.matmul(self.logits, self.x, wcls, 1, dim, cfg.vocab_size);
        } else if (self.token_embedding_table_q) |emb_q| {
            // Embeddings atados cuantizados: reutilizar matmulVecQ* sobre el blob completo.
            switch (emb_q.fmt) {
                .f32 => unreachable,
                .q4_0 => zkernel.matmulVecQ4_0(self.logits, emb_q.bytes, self.x, cfg.vocab_size, dim),
                .q8_0 => zkernel.matmulVecQ8_0(self.logits, emb_q.bytes, self.x, cfg.vocab_size, dim),
                .q4_k => zkernel.matmulVecQ4_K(self.logits, emb_q.bytes, self.x, cfg.vocab_size, dim),
                .q6_k => zkernel.matmulVecQ6_K(self.logits, emb_q.bytes, self.x, cfg.vocab_size, dim),
            }
        } else {
            // Embeddings atados f32 (fallback).
            for (0..cfg.vocab_size) |vtok| {
                const row = self.token_embedding_table[vtok * dim ..][0..dim];
                var acc: f32 = 0;
                for (0..dim) |d| acc += self.x[d] * row[d];
                self.logits[vtok] = acc;
            }
        }
        return self.logits;
    }

    /// Igual que `forward`, pero si `hidden_out` no es null, guarda `self.x`
    /// después de cada capa (post-FFN residual) en `hidden_out[pos * n_layers * dim + l * dim ..]`.
    pub fn forwardWithHidden(self: *Transformer, token: u32, pos: usize, hidden_out: ?[]f32) []f32 {
        const cfg = self.config;
        const dim = cfg.dim;
        const kv_dim = (dim * cfg.n_kv_heads) / cfg.n_heads;
        const kv_mul = cfg.n_heads / cfg.n_kv_heads;
        const head_size = dim / cfg.n_heads;
        const hidden_dim = cfg.hidden_dim;

        if (self.token_embedding_table_q) |emb_q| {
            switch (emb_q.fmt) {
                .f32 => unreachable,
                .q4_0 => zkernel.dequantRowQ4_0(emb_q.bytes, token, self.x, dim),
                .q8_0 => zkernel.dequantRowQ8_0(emb_q.bytes, token, self.x, dim),
                .q4_k => zkernel.dequantRowQ4_K(emb_q.bytes, token, self.x, dim),
                .q6_k => zkernel.dequantRowQ6_K(emb_q.bytes, token, self.x, dim),
            }
        } else {
            const content_row = self.token_embedding_table[token * dim ..][0..dim];
            @memcpy(self.x, content_row);
        }

        for (0..cfg.n_layers) |l| {
            zkernel.rmsnorm(self.xb, self.x, self.rms_att_weight[l * dim ..][0..dim], cfg.eps);
            self.matmulVec(self.q, if (self.wq_q == null) self.wq[l * dim * dim ..][0..(dim * dim)] else null, self.wq_q, l, dim, dim);
            self.matmulVec(self.k, if (self.wk_q == null) self.wk[l * kv_dim * dim ..][0..(kv_dim * dim)] else null, self.wk_q, l, kv_dim, dim);
            self.matmulVec(self.v, if (self.wv_q == null) self.wv[l * kv_dim * dim ..][0..(kv_dim * dim)] else null, self.wv_q, l, kv_dim, dim);
            if (self.bq) |bq| { for (0..dim) |i| self.q[i] += bq[l * dim + i]; }
            if (self.bk) |bk| { for (0..kv_dim) |i| self.k[i] += bk[l * kv_dim + i]; }
            if (self.bv) |bv| { for (0..kv_dim) |i| self.v[i] += bv[l * kv_dim + i]; }
            for (0..cfg.n_heads) |h| {
                const q_h = self.q[h * head_size ..][0..head_size];
                if (cfg.rope_neox) zkernel.ropeVecNeox(q_h, pos, head_size, cfg.theta) else zkernel.ropeVecGptj(q_h, pos, head_size, cfg.theta);
            }
            for (0..cfg.n_kv_heads) |h| {
                const k_h = self.k[h * head_size ..][0..head_size];
                if (cfg.rope_neox) zkernel.ropeVecNeox(k_h, pos, head_size, cfg.theta) else zkernel.ropeVecGptj(k_h, pos, head_size, cfg.theta);
            }
            const loff = l * cfg.seq_len * kv_dim;
            @memcpy(self.key_cache[loff + pos * kv_dim ..][0..kv_dim].ptr, self.k);
            @memcpy(self.value_cache[loff + pos * kv_dim ..][0..kv_dim].ptr, self.v);
            @memset(self.xb2, 0);
            for (0..cfg.n_heads) |h| {
                const q_head = self.q[h * head_size ..][0..head_size];
                var att_slice = self.att[0..(pos + 1)];
                for (0..pos + 1) |t| {
                    const k_cache = self.key_cache[loff + t * kv_dim + (h / kv_mul) * head_size ..][0..head_size];
                    var score: f32 = 0;
                    for (0..head_size) |i| score += q_head[i] * k_cache[i];
                    score /= @sqrt(@as(f32, @floatFromInt(head_size)));
                    att_slice[t] = score;
                }
                zkernel.softmax(att_slice, att_slice);
                const out_head = self.xb2[h * head_size ..][0..head_size];
                for (0..head_size) |i| out_head[i] = 0;
                for (0..pos + 1) |t| {
                    const v_cache = self.value_cache[loff + t * kv_dim + (h / kv_mul) * head_size ..][0..head_size];
                    const a = att_slice[t];
                    for (0..head_size) |i| out_head[i] += a * v_cache[i];
                }
            }
            self.matmulVecIn(self.xb, self.xb2, if (self.wo_q == null) self.wo[l * dim * dim ..][0..(dim * dim)] else null, self.wo_q, l, dim, dim);
            for (0..dim) |i| self.x[i] += self.xb[i];
            zkernel.rmsnorm(self.xb, self.x, self.rms_ffn_weight[l * dim ..][0..dim], cfg.eps);
            if (cfg.n_experts > 0) {
                self.forwardMoE(self.xb, l);
            } else {
                self.matmulVec(self.hb, if (self.w1_q == null) self.w1[l * dim * hidden_dim ..][0..(dim * hidden_dim)] else null, self.w1_q, l, hidden_dim, dim);
                self.matmulVec(self.hb2, if (self.w3_q == null) self.w3[l * dim * hidden_dim ..][0..(dim * hidden_dim)] else null, self.w3_q, l, hidden_dim, dim);
                for (0..hidden_dim) |i| self.hb[i] = self.hb[i] * (1.0 / (1.0 + @exp(-self.hb[i]))) * self.hb2[i];
                self.matmulVecIn(self.xb, self.hb, if (self.w2_q == null) self.w2[l * hidden_dim * dim ..][0..(hidden_dim * dim)] else null, self.w2_q, l, dim, hidden_dim);
                for (0..dim) |i| self.x[i] += self.xb[i];
            }
            if (hidden_out) |hout| {
                const hoff = l * dim;
                @memcpy(hout[hoff..][0..dim], self.x);
            }
        }
        zkernel.rmsnorm(self.x, self.x, self.rms_final_weight, cfg.eps);
        if (self.wcls) |wcls| {
            zkernel.matmul(self.logits, self.x, wcls, 1, dim, cfg.vocab_size);
        } else if (self.token_embedding_table_q) |emb_q| {
            switch (emb_q.fmt) {
                .f32 => unreachable,
                .q4_0 => zkernel.matmulVecQ4_0(self.logits, emb_q.bytes, self.x, cfg.vocab_size, dim),
                .q8_0 => zkernel.matmulVecQ8_0(self.logits, emb_q.bytes, self.x, cfg.vocab_size, dim),
                .q4_k => zkernel.matmulVecQ4_K(self.logits, emb_q.bytes, self.x, cfg.vocab_size, dim),
                .q6_k => zkernel.matmulVecQ6_K(self.logits, emb_q.bytes, self.x, cfg.vocab_size, dim),
            }
        } else {
            for (0..cfg.vocab_size) |vtok| {
                const row = self.token_embedding_table[vtok * dim ..][0..dim];
                var acc: f32 = 0;
                for (0..dim) |d| acc += self.x[d] * row[d];
                self.logits[vtok] = acc;
            }
        }
        return self.logits;
    }

    /// Procesa una secuencia de `tokens` consecutivos empezando en `start_pos`.
    /// Escribe K vectores de logits (K = tokens.len) en `logits_out` (layout [K, vocab_size]).
    /// Actualiza el KV-cache con las K posiciones nuevas.
    /// Requiere: logits_out.len == tokens.len * self.config.vocab_size.
    pub fn forwardBatch(self: *Transformer, tokens: []const u32, start_pos: usize, logits_out: []f32) !void {
        const K = tokens.len;
        const cfg = self.config;
        const dim = cfg.dim;
        const kv_dim = (dim * cfg.n_kv_heads) / cfg.n_heads;
        const kv_mul = cfg.n_heads / cfg.n_kv_heads;
        const head_size = dim / cfg.n_heads;
        const hidden_dim = cfg.hidden_dim;
        const vocab = cfg.vocab_size;
        std.debug.assert(logits_out.len == K * vocab);

        if (self.batch_cap < K) {
            if (self.batch_x) |b| self.allocator.free(b);
            if (self.batch_xb) |b| self.allocator.free(b);
            if (self.batch_q) |b| self.allocator.free(b);
            if (self.batch_k) |b| self.allocator.free(b);
            if (self.batch_v) |b| self.allocator.free(b);
            self.batch_x = try self.allocator.alloc(f32, K * dim);
            self.batch_xb = try self.allocator.alloc(f32, K * dim);
            self.batch_q = try self.allocator.alloc(f32, K * dim);
            self.batch_k = try self.allocator.alloc(f32, K * kv_dim);
            self.batch_v = try self.allocator.alloc(f32, K * kv_dim);
            self.batch_cap = K;
        }
        const x_batch = self.batch_x.?;
        const xb_batch = self.batch_xb.?;
        const q_batch = self.batch_q.?;
        const k_batch = self.batch_k.?;
        const v_batch = self.batch_v.?;

        for (tokens, 0..) |tok, i| {
            if (self.token_embedding_table_q) |emb_q| {
                switch (emb_q.fmt) {
                    .f32 => unreachable,
                    .q4_0 => zkernel.dequantRowQ4_0(emb_q.bytes, tok, x_batch[i * dim ..][0..dim], dim),
                    .q8_0 => zkernel.dequantRowQ8_0(emb_q.bytes, tok, x_batch[i * dim ..][0..dim], dim),
                    .q4_k => zkernel.dequantRowQ4_K(emb_q.bytes, tok, x_batch[i * dim ..][0..dim], dim),
                    .q6_k => zkernel.dequantRowQ6_K(emb_q.bytes, tok, x_batch[i * dim ..][0..dim], dim),
                }
            } else {
                @memcpy(x_batch[i * dim ..][0..dim], self.token_embedding_table[tok * dim ..][0..dim]);
            }
        }

        for (0..cfg.n_layers) |l| {
            for (0..K) |i| {
                zkernel.rmsnorm(self.xb, x_batch[i * dim ..][0..dim], self.rms_att_weight[l * dim ..][0..dim], cfg.eps);
                @memcpy(xb_batch[i * dim ..][0..dim], self.xb);
            }

            self.matmulBatch(q_batch, xb_batch, if (self.wq_q == null) self.wq[l * dim * dim ..][0..(dim * dim)] else null, self.wq_q, l, K, dim, dim);
            self.matmulBatch(k_batch, xb_batch, if (self.wk_q == null) self.wk[l * kv_dim * dim ..][0..(kv_dim * dim)] else null, self.wk_q, l, K, kv_dim, dim);
            self.matmulBatch(v_batch, xb_batch, if (self.wv_q == null) self.wv[l * kv_dim * dim ..][0..(kv_dim * dim)] else null, self.wv_q, l, K, kv_dim, dim);

            if (self.bq) |bq| {
                for (0..K) |i| {
                    for (0..dim) |d| q_batch[i * dim + d] += bq[l * dim + d];
                }
            }
            if (self.bk) |bk| {
                for (0..K) |i| {
                    for (0..kv_dim) |d| k_batch[i * kv_dim + d] += bk[l * kv_dim + d];
                }
            }
            if (self.bv) |bv| {
                for (0..K) |i| {
                    for (0..kv_dim) |d| v_batch[i * kv_dim + d] += bv[l * kv_dim + d];
                }
            }

            for (0..K) |i| {
                const pos = start_pos + i;
                for (0..cfg.n_heads) |h| {
                    const q_h = q_batch[i * dim + h * head_size ..][0..head_size];
                    if (cfg.rope_neox) zkernel.ropeVecNeox(q_h, pos, head_size, cfg.theta) else zkernel.ropeVecGptj(q_h, pos, head_size, cfg.theta);
                }
                for (0..cfg.n_kv_heads) |h| {
                    const k_h = k_batch[i * kv_dim + h * head_size ..][0..head_size];
                    if (cfg.rope_neox) zkernel.ropeVecNeox(k_h, pos, head_size, cfg.theta) else zkernel.ropeVecGptj(k_h, pos, head_size, cfg.theta);
                }
            }

            const loff = l * cfg.seq_len * kv_dim;
            for (0..K) |i| {
                const pos = start_pos + i;
                @memcpy(self.key_cache[loff + pos * kv_dim ..][0..kv_dim].ptr, k_batch[i * kv_dim ..][0..kv_dim]);
                @memcpy(self.value_cache[loff + pos * kv_dim ..][0..kv_dim].ptr, v_batch[i * kv_dim ..][0..kv_dim]);
            }

            for (0..K) |i| {
                const pos = start_pos + i;
                @memset(self.xb2, 0);
                for (0..cfg.n_heads) |h| {
                    const q_head = q_batch[i * dim + h * head_size ..][0..head_size];
                    var att_slice = self.att[0..(pos + 1)];
                    for (0..pos + 1) |t| {
                        const k_cache = self.key_cache[loff + t * kv_dim + (h / kv_mul) * head_size ..][0..head_size];
                        var score: f32 = 0;
                        for (0..head_size) |j| score += q_head[j] * k_cache[j];
                        score /= @sqrt(@as(f32, @floatFromInt(head_size)));
                        att_slice[t] = score;
                    }
                    zkernel.softmax(att_slice, att_slice);
                    const out_head = self.xb2[h * head_size ..][0..head_size];
                    for (0..head_size) |j| out_head[j] = 0;
                    for (0..pos + 1) |t| {
                        const v_cache = self.value_cache[loff + t * kv_dim + (h / kv_mul) * head_size ..][0..head_size];
                        const a = att_slice[t];
                        for (0..head_size) |j| out_head[j] += a * v_cache[j];
                    }
                }
                self.matmulVecIn(self.xb, self.xb2, if (self.wo_q == null) self.wo[l * dim * dim ..][0..(dim * dim)] else null, self.wo_q, l, dim, dim);
                for (0..dim) |d| x_batch[i * dim + d] += self.xb[d];
            }

            for (0..K) |i| {
                zkernel.rmsnorm(self.xb, x_batch[i * dim ..][0..dim], self.rms_ffn_weight[l * dim ..][0..dim], cfg.eps);
                if (cfg.n_experts > 0) {
                    @memcpy(self.x[0..dim], x_batch[i * dim ..][0..dim]);
                    self.forwardMoE(self.xb, l);
                    @memcpy(x_batch[i * dim ..][0..dim], self.x[0..dim]);
                } else {
                    self.matmulVec(self.hb, if (self.w1_q == null) self.w1[l * dim * hidden_dim ..][0..(dim * hidden_dim)] else null, self.w1_q, l, hidden_dim, dim);
                    self.matmulVec(self.hb2, if (self.w3_q == null) self.w3[l * dim * hidden_dim ..][0..(dim * hidden_dim)] else null, self.w3_q, l, hidden_dim, dim);
                    for (0..hidden_dim) |j| self.hb[j] = self.hb[j] * (1.0 / (1.0 + @exp(-self.hb[j]))) * self.hb2[j];
                    self.matmulVecIn(self.xb, self.hb, if (self.w2_q == null) self.w2[l * hidden_dim * dim ..][0..(hidden_dim * dim)] else null, self.w2_q, l, dim, hidden_dim);
                    for (0..dim) |d| x_batch[i * dim + d] += self.xb[d];
                }
            }
        }

        for (0..K) |i| {
            zkernel.rmsnorm(self.x, x_batch[i * dim ..][0..dim], self.rms_final_weight, cfg.eps);
            if (self.wcls) |wcls| {
                zkernel.matmul(logits_out[i * vocab ..][0..vocab], self.x, wcls, 1, dim, vocab);
            } else if (self.token_embedding_table_q) |emb_q| {
                switch (emb_q.fmt) {
                    .f32 => unreachable,
                    .q4_0 => zkernel.matmulVecQ4_0(logits_out[i * vocab ..][0..vocab], emb_q.bytes, self.x, vocab, dim),
                    .q8_0 => zkernel.matmulVecQ8_0(logits_out[i * vocab ..][0..vocab], emb_q.bytes, self.x, vocab, dim),
                    .q4_k => zkernel.matmulVecQ4_K(logits_out[i * vocab ..][0..vocab], emb_q.bytes, self.x, vocab, dim),
                    .q6_k => zkernel.matmulVecQ6_K(logits_out[i * vocab ..][0..vocab], emb_q.bytes, self.x, vocab, dim),
                }
            } else {
                for (0..vocab) |vtok| {
                    const row = self.token_embedding_table[vtok * dim ..][0..dim];
                    var acc: f32 = 0;
                    for (0..dim) |d| acc += self.x[d] * row[d];
                    logits_out[i * vocab + vtok] = acc;
                }
            }
        }
    }

    pub fn generate(self: *Transformer, prompt_tokens: []const u32, max_tokens: usize, temperature: f32) !std.ArrayList(u32) {
        var out = std.ArrayList(u32).empty;
        var pos: usize = 0;
        // Prompt
        for (prompt_tokens) |tok| {
            _ = self.forward(tok, pos);
            pos += 1;
        }
        // Generación
        var next_tok = prompt_tokens[prompt_tokens.len - 1];
        while (pos < max_tokens) {
            const logits = self.forward(next_tok, pos);
            if (temperature == 0) {
                var best: u32 = 0;
                var best_val = logits[0];
                for (logits, 0..) |v, i| {
                    if (v > best_val) {
                        best_val = v;
                        best = @intCast(i);
                    }
                }
                next_tok = best;
            } else {
                // Sampling simple con temperatura
                var maxv = logits[0];
                for (logits) |v| {
                    if (v > maxv) maxv = v;
                }
                var sum: f32 = 0;
                for (logits) |*v| {
                    v.* = @exp((v.* - maxv) / temperature);
                    sum += v.*;
                }
                const inv = 1.0 / sum;
                for (logits) |*v| v.* *= inv;
                // Greedy por simplicidad
                var best: u32 = 0;
                var best_val = logits[0];
                for (logits, 0..) |v, i| {
                    if (v > best_val) {
                        best_val = v;
                        best = @intCast(i);
                    }
                }
                next_tok = best;
            }
            try out.append(self.allocator, next_tok);
            pos += 1;
        }
        return out;
    }
};

// Helper de test: alloca un buffer de pesos mutable con valores deterministas,
// lo registra para liberar, y lo devuelve (se asigna a un campo []const f32).
fn makeWeights(alloc: std.mem.Allocator, list: *std.ArrayList([]f32), n: usize, seed: f32) ![]f32 {
    const b = try alloc.alloc(f32, n);
    for (b, 0..) |*v, i| v.* = @sin(seed + @as(f32, @floatFromInt(i)) * 0.1) * 0.2;
    try list.append(alloc, b);
    return b;
}

test "Transformer forward F1: finitud, determinismo, bias QKV y embeddings atados" {
    const alloc = std.testing.allocator;
    // Config tipo-Qwen en miniatura: GQA (kv_mul=2), 2 capas, FFN simple, embeddings atados.
    const cfg = Config{
        .dim = 4,
        .hidden_dim = 8,
        .n_layers = 2,
        .n_heads = 2,
        .n_kv_heads = 1,
        .vocab_size = 8,
        .seq_len = 8,
        .n_experts = 0,
        .theta = 10000.0,
        .eps = 1e-5,
    };
    const dim = cfg.dim;
    const kv_dim = (dim * cfg.n_kv_heads) / cfg.n_heads; // 2
    const hidden = cfg.hidden_dim;
    const L = cfg.n_layers;
    const vocab = cfg.vocab_size;

    var tf = try Transformer.init(alloc, cfg);
    defer tf.deinit();

    var bufs: std.ArrayList([]f32) = .empty;
    defer {
        for (bufs.items) |b| alloc.free(b);
        bufs.deinit(alloc);
    }

    tf.token_embedding_table = try makeWeights(alloc, &bufs, vocab * dim, 1.0);
    tf.rms_att_weight = try makeWeights(alloc, &bufs, L * dim, 2.0);
    tf.rms_ffn_weight = try makeWeights(alloc, &bufs, L * dim, 3.0);
    tf.wq = try makeWeights(alloc, &bufs, L * dim * dim, 4.0);
    tf.wk = try makeWeights(alloc, &bufs, L * kv_dim * dim, 5.0);
    tf.wv = try makeWeights(alloc, &bufs, L * kv_dim * dim, 6.0);
    tf.wo = try makeWeights(alloc, &bufs, L * dim * dim, 7.0);
    tf.w1 = try makeWeights(alloc, &bufs, L * dim * hidden, 8.0);
    tf.w2 = try makeWeights(alloc, &bufs, L * hidden * dim, 9.0);
    tf.w3 = try makeWeights(alloc, &bufs, L * dim * hidden, 10.0);
    tf.rms_final_weight = try makeWeights(alloc, &bufs, dim, 11.0);
    // Bias QKV (como Qwen2). wcls = null -> embeddings atados.
    tf.bq = try makeWeights(alloc, &bufs, L * dim, 12.0);
    tf.bk = try makeWeights(alloc, &bufs, L * kv_dim, 13.0);
    tf.bv = try makeWeights(alloc, &bufs, L * kv_dim, 14.0);

    // 1) Finitud + longitud correcta (rama de embeddings atados).
    const l0 = tf.forward(3, 0);
    try std.testing.expectEqual(@as(usize, vocab), l0.len);
    var snap: [8]f32 = undefined;
    @memcpy(&snap, l0[0..vocab]);
    for (l0) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }

    // 2) Determinismo: forward(tok,0) reinicia x desde el embedding -> idempotente.
    const l0b = tf.forward(3, 0);
    for (0..vocab) |i| try std.testing.expectApproxEqAbs(snap[i], l0b[i], 1e-6);

    // 3) Sensibilidad a la entrada: otro token debe dar logits distintos.
    const l1 = tf.forward(5, 0);
    var differs = false;
    for (0..vocab) |i| {
        if (@abs(l1[i] - snap[i]) > 1e-6) differs = true;
    }
    try std.testing.expect(differs);

    // 4) El bias QKV realmente afecta la salida: al anularlo cambian los logits.
    tf.bq = null;
    tf.bk = null;
    tf.bv = null;
    const l_nobias = tf.forward(3, 0);
    var bias_matters = false;
    for (0..vocab) |i| {
        if (@abs(l_nobias[i] - snap[i]) > 1e-6) bias_matters = true;
    }
    try std.testing.expect(bias_matters);

    // 5) Multi-posición (KV cache + RoPE en pos>0 + atención sobre 2 tokens): sin NaN.
    _ = tf.forward(3, 0);
    const l_pos1 = tf.forward(5, 1);
    for (l_pos1) |v| try std.testing.expect(!std.math.isNan(v));
}

// Helper de test: envuelve un blob cuantizado de UNA capa en un QuantWeight.
fn qw1(alloc: std.mem.Allocator, blob: []u8, fmt: QuantFormat) !QuantWeight {
    const layers = try alloc.alloc([]const u8, 1);
    layers[0] = blob;
    const fmts = try alloc.alloc(QuantFormat, 1);
    fmts[0] = fmt;
    return .{ .layers = layers, .fmts = fmts };
}

fn freeQw(alloc: std.mem.Allocator, qw: QuantWeight) void {
    for (qw.layers) |l| alloc.free(@constCast(l));
    alloc.free(@constCast(qw.layers));
    alloc.free(@constCast(qw.fmts));
}

// Helper de test: QuantWeight de 2 capas con blobs y formatos por-capa.
fn qw2(alloc: std.mem.Allocator, b0: []u8, f0: QuantFormat, b1: []u8, f1: QuantFormat) !QuantWeight {
    const layers = try alloc.alloc([]const u8, 2);
    layers[0] = b0;
    layers[1] = b1;
    const fmts = try alloc.alloc(QuantFormat, 2);
    fmts[0] = f0;
    fmts[1] = f1;
    return .{ .layers = layers, .fmts = fmts };
}

// Helper de test: peso lineal de L capas, constante DISTINTO por capa (lossless en
// Q8_0 y Q4_0), registrado para liberación.
fn perLayerConst(alloc: std.mem.Allocator, list: *std.ArrayList([]f32), L: usize, per: usize, base: f32) ![]f32 {
    const buf = try alloc.alloc(f32, L * per);
    for (0..L) |l| {
        const v = base + 0.1 * @as(f32, @floatFromInt(l));
        for (0..per) |i| buf[l * per + i] = v;
    }
    try list.append(alloc, buf);
    return buf;
}

// Helper de test: cuantiza una matriz f32 [out, in] a Q8_0 (layout GGUF original).
fn quantizeQ8_0(alloc: std.mem.Allocator, weights: []const f32, out_f: usize, in_f: usize) ![]u8 {
    const blk = zkernel.Q8_0_BLOCK_SIZE; // 32
    const blk_bytes = zkernel.Q8_0_BLOCK_BYTES; // 34
    const n_blocks = (in_f + blk - 1) / blk;
    const total_bytes = out_f * n_blocks * blk_bytes;
    const buf = try alloc.alloc(u8, total_bytes);
    for (0..out_f) |o| {
        for (0..n_blocks) |b| {
            const base = o * n_blocks * blk_bytes + b * blk_bytes;
            const k_start = b * blk;
            const k_end = @min(k_start + blk, in_f);
            var max_abs: f32 = 0;
            for (k_start..k_end) |k| {
                const v = @abs(weights[o * in_f + k]);
                if (v > max_abs) max_abs = v;
            }
            const scale: f16 = if (max_abs > 0) @floatCast(max_abs / 127.0) else 0;
            std.mem.writeInt(u16, buf[base..][0..2], @bitCast(scale), .little);
            for (k_start..k_end) |k| {
                const qf = weights[o * in_f + k] / @as(f32, @floatCast(scale));
                const qi: i8 = @intFromFloat(@max(-127.0, @min(127.0, @round(qf))));
                buf[base + 2 + (k - k_start)] = @bitCast(qi);
            }
        }
    }
    return buf;
}

test "Transformer forward cuantizado Q8_0 coincide con f32 (tolerancia Q8)" {
    const alloc = std.testing.allocator;
    const cfg = Config{
        .dim = 32,
        .hidden_dim = 32,
        .n_layers = 1,
        .n_heads = 2,
        .n_kv_heads = 1,
        .vocab_size = 8,
        .seq_len = 8,
        .n_experts = 0,
        .theta = 10000.0,
        .eps = 1e-5,
    };
    const dim = cfg.dim;
    const kv_dim = (dim * cfg.n_kv_heads) / cfg.n_heads;
    const hidden = cfg.hidden_dim;
    const vocab = cfg.vocab_size;

    // --- Transformer f32 ---
    var tf_f32 = try Transformer.init(alloc, cfg);
    defer tf_f32.deinit();

    var wlist = std.ArrayList([]f32).empty;
    defer { for (wlist.items) |b| alloc.free(b); wlist.deinit(alloc); }

    tf_f32.token_embedding_table = try makeWeights(alloc, &wlist, vocab * dim, 1.0);
    tf_f32.rms_att_weight = try makeWeights(alloc, &wlist, dim, 2.0);
    tf_f32.rms_ffn_weight = try makeWeights(alloc, &wlist, dim, 3.0);
    tf_f32.rms_final_weight = try makeWeights(alloc, &wlist, dim, 4.0);
    tf_f32.wq = try makeWeights(alloc, &wlist, dim * dim, 5.0);
    tf_f32.wk = try makeWeights(alloc, &wlist, kv_dim * dim, 6.0);
    tf_f32.wv = try makeWeights(alloc, &wlist, kv_dim * dim, 7.0);
    tf_f32.wo = try makeWeights(alloc, &wlist, dim * dim, 8.0);
    tf_f32.w1 = try makeWeights(alloc, &wlist, dim * hidden, 9.0);
    tf_f32.w2 = try makeWeights(alloc, &wlist, hidden * dim, 10.0);
    tf_f32.w3 = try makeWeights(alloc, &wlist, dim * hidden, 11.0);
    tf_f32.bq = try makeWeights(alloc, &wlist, dim, 12.0);
    tf_f32.bk = try makeWeights(alloc, &wlist, kv_dim, 13.0);
    tf_f32.bv = try makeWeights(alloc, &wlist, kv_dim, 14.0);

    // Hacer todos los pesos no-negativos para evitar problemas de cuantización
    for (wlist.items) |b| {
        for (0..b.len) |i| b[i] = @abs(b[i]);
    }

    const logits_f32 = tf_f32.forward(3, 0);

    // --- Transformer Q8_0 ---
    var tf_q = try Transformer.init(alloc, cfg);
    defer tf_q.deinit();

    tf_q.token_embedding_table = tf_f32.token_embedding_table;
    tf_q.rms_att_weight = tf_f32.rms_att_weight;
    tf_q.rms_ffn_weight = tf_f32.rms_ffn_weight;
    tf_q.rms_final_weight = tf_f32.rms_final_weight;
    tf_q.bq = tf_f32.bq;
    tf_q.bk = tf_f32.bk;
    tf_q.bv = tf_f32.bv;

    // Cuantizar pesos lineales en layout GGUF original [out, in]
    // Para ello, primero transponemos los pesos de [in, out] a [out, in]
    const wq_t = try alloc.alloc(f32, dim * dim);
    defer alloc.free(wq_t);
    for (0..dim) |o| {
        for (0..dim) |i| {
            wq_t[o * dim + i] = tf_f32.wq[i * dim + o];
        }
    }
    tf_q.wq_q = try qw1(alloc, try quantizeQ8_0(alloc, wq_t, dim, dim), .q8_0);

    const wk_t = try alloc.alloc(f32, kv_dim * dim);
    defer alloc.free(wk_t);
    for (0..kv_dim) |o| {
        for (0..dim) |i| {
            wk_t[o * dim + i] = tf_f32.wk[i * kv_dim + o];
        }
    }
    tf_q.wk_q = try qw1(alloc, try quantizeQ8_0(alloc, wk_t, kv_dim, dim), .q8_0);

    const wv_t = try alloc.alloc(f32, kv_dim * dim);
    defer alloc.free(wv_t);
    for (0..kv_dim) |o| {
        for (0..dim) |i| {
            wv_t[o * dim + i] = tf_f32.wv[i * kv_dim + o];
        }
    }
    tf_q.wv_q = try qw1(alloc, try quantizeQ8_0(alloc, wv_t, kv_dim, dim), .q8_0);

    const wo_t = try alloc.alloc(f32, dim * dim);
    defer alloc.free(wo_t);
    for (0..dim) |o| {
        for (0..dim) |i| {
            wo_t[o * dim + i] = tf_f32.wo[i * dim + o];
        }
    }
    tf_q.wo_q = try qw1(alloc, try quantizeQ8_0(alloc, wo_t, dim, dim), .q8_0);

    const w1_t = try alloc.alloc(f32, hidden * dim);
    defer alloc.free(w1_t);
    for (0..hidden) |o| {
        for (0..dim) |i| {
            w1_t[o * dim + i] = tf_f32.w1[i * hidden + o];
        }
    }
    tf_q.w1_q = try qw1(alloc, try quantizeQ8_0(alloc, w1_t, hidden, dim), .q8_0);

    const w2_t = try alloc.alloc(f32, dim * hidden);
    defer alloc.free(w2_t);
    for (0..dim) |o| {
        for (0..hidden) |i| {
            w2_t[o * hidden + i] = tf_f32.w2[i * dim + o];
        }
    }
    tf_q.w2_q = try qw1(alloc, try quantizeQ8_0(alloc, w2_t, dim, hidden), .q8_0);

    const w3_t = try alloc.alloc(f32, hidden * dim);
    defer alloc.free(w3_t);
    for (0..hidden) |o| {
        for (0..dim) |i| {
            w3_t[o * dim + i] = tf_f32.w3[i * hidden + o];
        }
    }
    tf_q.w3_q = try qw1(alloc, try quantizeQ8_0(alloc, w3_t, hidden, dim), .q8_0);

    tf_q.wq = &.{}; tf_q.wk = &.{}; tf_q.wv = &.{};
    tf_q.wo = &.{}; tf_q.w1 = &.{}; tf_q.w2 = &.{}; tf_q.w3 = &.{};

    const logits_q = tf_q.forward(3, 0);

    // Comparar logits con tolerancia generosa para error de cuantización Q8
    for (logits_f32, logits_q) |f, q| {
        try std.testing.expect(@abs(f - q) < 0.5);
    }

    // Liberar buffers cuantizados
    freeQw(alloc, tf_q.wq_q.?);
    freeQw(alloc, tf_q.wk_q.?);
    freeQw(alloc, tf_q.wv_q.?);
    freeQw(alloc, tf_q.wo_q.?);
    freeQw(alloc, tf_q.w1_q.?);
    freeQw(alloc, tf_q.w2_q.?);
    freeQw(alloc, tf_q.w3_q.?);
}

// Helper de test: cuantiza una matriz f32 [out, in] a Q4_0 (esquema del repo:
// nibble 0..15 positivo, valor = nibble*scale; sólo pesos no-negativos).
fn quantizeQ4_0(alloc: std.mem.Allocator, weights: []const f32, out_f: usize, in_f: usize) ![]u8 {
    const blk = zkernel.Q4_0_BLOCK_SIZE; // 32
    const blk_bytes = zkernel.Q4_0_BLOCK_BYTES; // 18
    const n_blocks = (in_f + blk - 1) / blk;
    const buf = try alloc.alloc(u8, out_f * n_blocks * blk_bytes);
    @memset(buf, 0);
    for (0..out_f) |o| {
        for (0..n_blocks) |b| {
            const base = o * n_blocks * blk_bytes + b * blk_bytes;
            const k_start = b * blk;
            const k_end = @min(k_start + blk, in_f);
            var max_abs: f32 = 0;
            for (k_start..k_end) |k| {
                const v = @abs(weights[o * in_f + k]);
                if (v > max_abs) max_abs = v;
            }
            const scale: f16 = if (max_abs > 0) @floatCast(max_abs / 15.0) else 0;
            std.mem.writeInt(u16, buf[base..][0..2], @bitCast(scale), .little);
            for (k_start..k_end) |k| {
                const i = k - k_start;
                const qf = if (scale > 0) weights[o * in_f + k] / @as(f32, @floatCast(scale)) else 0;
                const nibble: u4 = @intFromFloat(@max(0.0, @min(15.0, @round(qf))));
                const byte_idx = base + 2 + i / 2;
                if (i & 1 == 0) {
                    buf[byte_idx] = (buf[byte_idx] & 0xF0) | nibble;
                } else {
                    buf[byte_idx] = (buf[byte_idx] & 0x0F) | (@as(u8, nibble) << 4);
                }
            }
        }
    }
    return buf;
}

test "Transformer forward formatos cuantizados MEZCLADOS (w1=Q4_0, resto=Q8_0) coincide con f32" {
    // Regresión del bug: el modelo real Qwen2.5 Q4_K_M mezcla Q4_K (attn/gate/up)
    // y Q6_K (ffn_down). El formato debe ser POR-TENSOR. Aquí se prueba el
    // enrutado con dos formatos coexistiendo en un mismo forward.
    const alloc = std.testing.allocator;
    const cfg = Config{
        .dim = 32,
        .hidden_dim = 32,
        .n_layers = 1,
        .n_heads = 2,
        .n_kv_heads = 1,
        .vocab_size = 8,
        .seq_len = 8,
        .n_experts = 0,
        .theta = 10000.0,
        .eps = 1e-5,
    };
    const dim = cfg.dim;
    const kv_dim = (dim * cfg.n_kv_heads) / cfg.n_heads;
    const hidden = cfg.hidden_dim;
    const vocab = cfg.vocab_size;

    var tf_f32 = try Transformer.init(alloc, cfg);
    defer tf_f32.deinit();

    var wlist = std.ArrayList([]f32).empty;
    defer { for (wlist.items) |b| alloc.free(b); wlist.deinit(alloc); }

    // Embeddings/norms/bias variados; pesos lineales CONSTANTES por matriz para
    // que tanto Q8_0 como Q4_0 sean lossless (sólo error del redondeo f16 de escala).
    tf_f32.token_embedding_table = try makeWeights(alloc, &wlist, vocab * dim, 1.0);
    tf_f32.rms_att_weight = try makeWeights(alloc, &wlist, dim, 2.0);
    tf_f32.rms_ffn_weight = try makeWeights(alloc, &wlist, dim, 3.0);
    tf_f32.rms_final_weight = try makeWeights(alloc, &wlist, dim, 4.0);
    tf_f32.bq = try makeWeights(alloc, &wlist, dim, 12.0);
    tf_f32.bk = try makeWeights(alloc, &wlist, kv_dim, 13.0);
    tf_f32.bv = try makeWeights(alloc, &wlist, kv_dim, 14.0);
    for (wlist.items) |b| { for (0..b.len) |i| b[i] = @abs(b[i]); }

    const constMat = struct {
        fn make(a: std.mem.Allocator, list: *std.ArrayList([]f32), n: usize, val: f32) ![]f32 {
            const buf = try a.alloc(f32, n);
            @memset(buf, val);
            try list.append(a, buf);
            return buf;
        }
    }.make;
    tf_f32.wq = try constMat(alloc, &wlist, dim * dim, 0.30);
    tf_f32.wk = try constMat(alloc, &wlist, kv_dim * dim, 0.25);
    tf_f32.wv = try constMat(alloc, &wlist, kv_dim * dim, 0.20);
    tf_f32.wo = try constMat(alloc, &wlist, dim * dim, 0.15);
    tf_f32.w1 = try constMat(alloc, &wlist, dim * hidden, 0.45); // -> Q4_0
    tf_f32.w2 = try constMat(alloc, &wlist, hidden * dim, 0.10);
    tf_f32.w3 = try constMat(alloc, &wlist, dim * hidden, 0.35);

    const logits_f32 = tf_f32.forward(3, 0);

    var tf_q = try Transformer.init(alloc, cfg);
    defer tf_q.deinit();
    tf_q.token_embedding_table = tf_f32.token_embedding_table;
    tf_q.rms_att_weight = tf_f32.rms_att_weight;
    tf_q.rms_ffn_weight = tf_f32.rms_ffn_weight;
    tf_q.rms_final_weight = tf_f32.rms_final_weight;
    tf_q.bq = tf_f32.bq;
    tf_q.bk = tf_f32.bk;
    tf_q.bv = tf_f32.bv;

    // Pesos constantes ⇒ el layout [out,in] es idéntico a [in,out]; cuantizar directo.
    tf_q.wq_q = try qw1(alloc, try quantizeQ8_0(alloc, tf_f32.wq, dim, dim), .q8_0);
    tf_q.wk_q = try qw1(alloc, try quantizeQ8_0(alloc, tf_f32.wk, kv_dim, dim), .q8_0);
    tf_q.wv_q = try qw1(alloc, try quantizeQ8_0(alloc, tf_f32.wv, kv_dim, dim), .q8_0);
    tf_q.wo_q = try qw1(alloc, try quantizeQ8_0(alloc, tf_f32.wo, dim, dim), .q8_0);
    tf_q.w1_q = try qw1(alloc, try quantizeQ4_0(alloc, tf_f32.w1, hidden, dim), .q4_0); // <- formato distinto
    tf_q.w2_q = try qw1(alloc, try quantizeQ8_0(alloc, tf_f32.w2, dim, hidden), .q8_0);
    tf_q.w3_q = try qw1(alloc, try quantizeQ8_0(alloc, tf_f32.w3, hidden, dim), .q8_0);
    tf_q.wq = &.{}; tf_q.wk = &.{}; tf_q.wv = &.{};
    tf_q.wo = &.{}; tf_q.w1 = &.{}; tf_q.w2 = &.{}; tf_q.w3 = &.{};

    const logits_q = tf_q.forward(3, 0);
    for (logits_f32, logits_q) |f, q| {
        try std.testing.expect(@abs(f - q) < 0.05);
    }

    freeQw(alloc, tf_q.wq_q.?);
    freeQw(alloc, tf_q.wk_q.?);
    freeQw(alloc, tf_q.wv_q.?);
    freeQw(alloc, tf_q.wo_q.?);
    freeQw(alloc, tf_q.w1_q.?);
    freeQw(alloc, tf_q.w2_q.?);
    freeQw(alloc, tf_q.w3_q.?);
}

test "Transformer forward cuantizado MULTI-CAPA (2 capas, pesos+formato distintos por capa) coincide con f32" {
    // Regresión del bug de capas: la primera versión solo cargaba/usaba 1 capa de
    // pesos cuantizados (pasaba el test de 1 capa, fallaba en el modelo real de 28).
    // Aquí cada capa tiene pesos DISTINTOS y w1 alterna formato por capa (q8_0/q4_0):
    // si el offset por capa o el formato por capa se rompen, los logits divergen.
    const alloc = std.testing.allocator;
    const cfg = Config{
        .dim = 32,
        .hidden_dim = 32,
        .n_layers = 2,
        .n_heads = 2,
        .n_kv_heads = 1,
        .vocab_size = 8,
        .seq_len = 8,
        .n_experts = 0,
        .theta = 10000.0,
        .eps = 1e-5,
    };
    const dim = cfg.dim;
    const kv_dim = (dim * cfg.n_kv_heads) / cfg.n_heads;
    const hidden = cfg.hidden_dim;
    const vocab = cfg.vocab_size;
    const L = cfg.n_layers;

    var tf_f32 = try Transformer.init(alloc, cfg);
    defer tf_f32.deinit();
    var wlist = std.ArrayList([]f32).empty;
    defer { for (wlist.items) |b| alloc.free(b); wlist.deinit(alloc); }

    tf_f32.token_embedding_table = try makeWeights(alloc, &wlist, vocab * dim, 1.0);
    tf_f32.rms_att_weight = try makeWeights(alloc, &wlist, L * dim, 2.0);
    tf_f32.rms_ffn_weight = try makeWeights(alloc, &wlist, L * dim, 3.0);
    tf_f32.rms_final_weight = try makeWeights(alloc, &wlist, dim, 4.0);
    tf_f32.bq = try makeWeights(alloc, &wlist, L * dim, 12.0);
    tf_f32.bk = try makeWeights(alloc, &wlist, L * kv_dim, 13.0);
    tf_f32.bv = try makeWeights(alloc, &wlist, L * kv_dim, 14.0);
    for (wlist.items) |b| { for (0..b.len) |i| b[i] = @abs(b[i]); }

    // Pesos lineales CONSTANTES distintos por capa (lossless en Q8_0 y Q4_0).
    tf_f32.wq = try perLayerConst(alloc, &wlist, L, dim * dim, 0.30);
    tf_f32.wk = try perLayerConst(alloc, &wlist, L, kv_dim * dim, 0.25);
    tf_f32.wv = try perLayerConst(alloc, &wlist, L, kv_dim * dim, 0.20);
    tf_f32.wo = try perLayerConst(alloc, &wlist, L, dim * dim, 0.15);
    tf_f32.w1 = try perLayerConst(alloc, &wlist, L, dim * hidden, 0.45);
    tf_f32.w2 = try perLayerConst(alloc, &wlist, L, hidden * dim, 0.10);
    tf_f32.w3 = try perLayerConst(alloc, &wlist, L, dim * hidden, 0.35);

    const logits_f32 = tf_f32.forward(3, 0);

    var tf_q = try Transformer.init(alloc, cfg);
    defer tf_q.deinit();
    tf_q.token_embedding_table = tf_f32.token_embedding_table;
    tf_q.rms_att_weight = tf_f32.rms_att_weight;
    tf_q.rms_ffn_weight = tf_f32.rms_ffn_weight;
    tf_q.rms_final_weight = tf_f32.rms_final_weight;
    tf_q.bq = tf_f32.bq;
    tf_q.bk = tf_f32.bk;
    tf_q.bv = tf_f32.bv;

    // Cuantiza cada capa por separado (pesos constantes ⇒ [out,in]=[in,out]).
    tf_q.wq_q = try qw2(alloc, try quantizeQ8_0(alloc, tf_f32.wq[0 .. dim * dim], dim, dim), .q8_0, try quantizeQ8_0(alloc, tf_f32.wq[dim * dim ..][0 .. dim * dim], dim, dim), .q8_0);
    tf_q.wk_q = try qw2(alloc, try quantizeQ8_0(alloc, tf_f32.wk[0 .. kv_dim * dim], kv_dim, dim), .q8_0, try quantizeQ8_0(alloc, tf_f32.wk[kv_dim * dim ..][0 .. kv_dim * dim], kv_dim, dim), .q8_0);
    tf_q.wv_q = try qw2(alloc, try quantizeQ8_0(alloc, tf_f32.wv[0 .. kv_dim * dim], kv_dim, dim), .q8_0, try quantizeQ8_0(alloc, tf_f32.wv[kv_dim * dim ..][0 .. kv_dim * dim], kv_dim, dim), .q8_0);
    tf_q.wo_q = try qw2(alloc, try quantizeQ8_0(alloc, tf_f32.wo[0 .. dim * dim], dim, dim), .q8_0, try quantizeQ8_0(alloc, tf_f32.wo[dim * dim ..][0 .. dim * dim], dim, dim), .q8_0);
    // w1: formato DISTINTO por capa (capa 0 = q8_0, capa 1 = q4_0).
    tf_q.w1_q = try qw2(alloc, try quantizeQ8_0(alloc, tf_f32.w1[0 .. hidden * dim], hidden, dim), .q8_0, try quantizeQ4_0(alloc, tf_f32.w1[hidden * dim ..][0 .. hidden * dim], hidden, dim), .q4_0);
    tf_q.w2_q = try qw2(alloc, try quantizeQ8_0(alloc, tf_f32.w2[0 .. dim * hidden], dim, hidden), .q8_0, try quantizeQ8_0(alloc, tf_f32.w2[dim * hidden ..][0 .. dim * hidden], dim, hidden), .q8_0);
    tf_q.w3_q = try qw2(alloc, try quantizeQ8_0(alloc, tf_f32.w3[0 .. hidden * dim], hidden, dim), .q8_0, try quantizeQ8_0(alloc, tf_f32.w3[hidden * dim ..][0 .. hidden * dim], hidden, dim), .q8_0);
    tf_q.wq = &.{}; tf_q.wk = &.{}; tf_q.wv = &.{};
    tf_q.wo = &.{}; tf_q.w1 = &.{}; tf_q.w2 = &.{}; tf_q.w3 = &.{ };

    const logits_q = tf_q.forward(3, 0);
    for (logits_f32, logits_q) |f, q| {
        try std.testing.expect(@abs(f - q) < 0.05);
    }

    freeQw(alloc, tf_q.wq_q.?);
    freeQw(alloc, tf_q.wk_q.?);
    freeQw(alloc, tf_q.wv_q.?);
    freeQw(alloc, tf_q.wo_q.?);
    freeQw(alloc, tf_q.w1_q.?);
    freeQw(alloc, tf_q.w2_q.?);
    freeQw(alloc, tf_q.w3_q.?);
}

test "forwardBatch K=1 coincide con forward" {
    const alloc = std.testing.allocator;
    const cfg = Config{ .dim = 8, .hidden_dim = 16, .n_layers = 1, .n_heads = 2, .n_kv_heads = 1, .vocab_size = 8, .seq_len = 16, .n_experts = 0, .theta = 10000.0, .eps = 1e-5 };
    const dim = cfg.dim; const kv_dim = (dim * cfg.n_kv_heads) / cfg.n_heads; const hidden = cfg.hidden_dim; const vocab = cfg.vocab_size; const L = cfg.n_layers;
    var tf_ref = try Transformer.init(alloc, cfg); defer tf_ref.deinit();
    var wlist = std.ArrayList([]f32).empty; defer { for (wlist.items) |b| alloc.free(b); wlist.deinit(alloc); }
    tf_ref.token_embedding_table = try makeWeights(alloc, &wlist, vocab * dim, 1.0);
    tf_ref.rms_att_weight = try makeWeights(alloc, &wlist, L * dim, 2.0);
    tf_ref.rms_ffn_weight = try makeWeights(alloc, &wlist, L * dim, 3.0);
    tf_ref.wq = try makeWeights(alloc, &wlist, L * dim * dim, 4.0);
    tf_ref.wk = try makeWeights(alloc, &wlist, L * kv_dim * dim, 5.0);
    tf_ref.wv = try makeWeights(alloc, &wlist, L * kv_dim * dim, 6.0);
    tf_ref.wo = try makeWeights(alloc, &wlist, L * dim * dim, 7.0);
    tf_ref.w1 = try makeWeights(alloc, &wlist, L * dim * hidden, 8.0);
    tf_ref.w2 = try makeWeights(alloc, &wlist, L * hidden * dim, 9.0);
    tf_ref.w3 = try makeWeights(alloc, &wlist, L * dim * hidden, 10.0);
    tf_ref.rms_final_weight = try makeWeights(alloc, &wlist, dim, 11.0);
    tf_ref.bq = try makeWeights(alloc, &wlist, L * dim, 12.0);
    tf_ref.bk = try makeWeights(alloc, &wlist, L * kv_dim, 13.0);
    tf_ref.bv = try makeWeights(alloc, &wlist, L * kv_dim, 14.0);
    const logits_ref = tf_ref.forward(3, 0);

    var tf_batch = try Transformer.init(alloc, cfg); defer tf_batch.deinit();
    tf_batch.token_embedding_table = tf_ref.token_embedding_table;
    tf_batch.rms_att_weight = tf_ref.rms_att_weight; tf_batch.rms_ffn_weight = tf_ref.rms_ffn_weight;
    tf_batch.wq = tf_ref.wq; tf_batch.wk = tf_ref.wk; tf_batch.wv = tf_ref.wv; tf_batch.wo = tf_ref.wo;
    tf_batch.w1 = tf_ref.w1; tf_batch.w2 = tf_ref.w2; tf_batch.w3 = tf_ref.w3;
    tf_batch.rms_final_weight = tf_ref.rms_final_weight;
    tf_batch.bq = tf_ref.bq; tf_batch.bk = tf_ref.bk; tf_batch.bv = tf_ref.bv;
    const logits_batch = try alloc.alloc(f32, vocab);
    defer alloc.free(logits_batch);
    try tf_batch.forwardBatch(&[_]u32{3}, 0, logits_batch);
    for (0..vocab) |v| {
        try std.testing.expectApproxEqAbs(logits_ref[v], logits_batch[v], 1e-4);
    }
}

test "forwardBatch K=2 coincide con 2xforward (oráculo)" {
    const alloc = std.testing.allocator;
    const cfg = Config{ .dim = 8, .hidden_dim = 16, .n_layers = 2, .n_heads = 2, .n_kv_heads = 1, .vocab_size = 8, .seq_len = 16, .n_experts = 0, .theta = 10000.0, .eps = 1e-5 };
    const dim = cfg.dim; const kv_dim = (dim * cfg.n_kv_heads) / cfg.n_heads; const hidden = cfg.hidden_dim; const vocab = cfg.vocab_size; const L = cfg.n_layers;

    var tf_ref = try Transformer.init(alloc, cfg); defer tf_ref.deinit();
    var wlist = std.ArrayList([]f32).empty; defer { for (wlist.items) |b| alloc.free(b); wlist.deinit(alloc); }
    tf_ref.token_embedding_table = try makeWeights(alloc, &wlist, vocab * dim, 1.0);
    tf_ref.rms_att_weight = try makeWeights(alloc, &wlist, L * dim, 2.0);
    tf_ref.rms_ffn_weight = try makeWeights(alloc, &wlist, L * dim, 3.0);
    tf_ref.wq = try makeWeights(alloc, &wlist, L * dim * dim, 4.0);
    tf_ref.wk = try makeWeights(alloc, &wlist, L * kv_dim * dim, 5.0);
    tf_ref.wv = try makeWeights(alloc, &wlist, L * kv_dim * dim, 6.0);
    tf_ref.wo = try makeWeights(alloc, &wlist, L * dim * dim, 7.0);
    tf_ref.w1 = try makeWeights(alloc, &wlist, L * dim * hidden, 8.0);
    tf_ref.w2 = try makeWeights(alloc, &wlist, L * hidden * dim, 9.0);
    tf_ref.w3 = try makeWeights(alloc, &wlist, L * dim * hidden, 10.0);
    tf_ref.rms_final_weight = try makeWeights(alloc, &wlist, dim, 11.0);
    tf_ref.bq = try makeWeights(alloc, &wlist, L * dim, 12.0);
    tf_ref.bk = try makeWeights(alloc, &wlist, L * kv_dim, 13.0);
    tf_ref.bv = try makeWeights(alloc, &wlist, L * kv_dim, 14.0);

    const tokens = &[_]u32{ 3, 5 };
    const logits_ref_0 = try alloc.dupe(f32, tf_ref.forward(tokens[0], 0));
    defer alloc.free(logits_ref_0);
    const logits_ref_1 = try alloc.dupe(f32, tf_ref.forward(tokens[1], 1));
    defer alloc.free(logits_ref_1);

    var tf_batch = try Transformer.init(alloc, cfg); defer tf_batch.deinit();
    tf_batch.token_embedding_table = tf_ref.token_embedding_table;
    tf_batch.rms_att_weight = tf_ref.rms_att_weight; tf_batch.rms_ffn_weight = tf_ref.rms_ffn_weight;
    tf_batch.wq = tf_ref.wq; tf_batch.wk = tf_ref.wk; tf_batch.wv = tf_ref.wv; tf_batch.wo = tf_ref.wo;
    tf_batch.w1 = tf_ref.w1; tf_batch.w2 = tf_ref.w2; tf_batch.w3 = tf_ref.w3;
    tf_batch.rms_final_weight = tf_ref.rms_final_weight;
    tf_batch.bq = tf_ref.bq; tf_batch.bk = tf_ref.bk; tf_batch.bv = tf_ref.bv;

    const logits_batch = try alloc.alloc(f32, tokens.len * vocab);
    defer alloc.free(logits_batch);
    try tf_batch.forwardBatch(tokens, 0, logits_batch);

    for (0..vocab) |v| try std.testing.expectApproxEqAbs(logits_ref_0[v], logits_batch[v], 1e-4);
    for (0..vocab) |v| try std.testing.expectApproxEqAbs(logits_ref_1[v], logits_batch[vocab + v], 1e-4);
}

test "embedding cuantizado Q8_0: forward idéntico a f32 (oráculo)" {
    const alloc = std.testing.allocator;
    const cfg = Config{ .dim = 32, .hidden_dim = 16, .n_layers = 1, .n_heads = 2, .n_kv_heads = 1, .vocab_size = 4, .seq_len = 8, .n_experts = 0, .theta = 10000.0, .eps = 1e-5 };
    const dim = cfg.dim;
    const vocab = cfg.vocab_size;
    const L = cfg.n_layers;
    const kv_dim = (dim * cfg.n_kv_heads) / cfg.n_heads;
    const hidden = cfg.hidden_dim;

    // Oráculo f32
    var tf_ref = try Transformer.init(alloc, cfg);
    defer tf_ref.deinit();
    var wlist = std.ArrayList([]f32).empty;
    defer { for (wlist.items) |b| alloc.free(b); wlist.deinit(alloc); }

    tf_ref.token_embedding_table = try makeWeights(alloc, &wlist, vocab * dim, 1.0);
    tf_ref.rms_att_weight = try makeWeights(alloc, &wlist, L * dim, 2.0);
    tf_ref.rms_ffn_weight = try makeWeights(alloc, &wlist, L * dim, 3.0);
    tf_ref.wq = try makeWeights(alloc, &wlist, L * dim * dim, 4.0);
    tf_ref.wk = try makeWeights(alloc, &wlist, L * kv_dim * dim, 5.0);
    tf_ref.wv = try makeWeights(alloc, &wlist, L * kv_dim * dim, 6.0);
    tf_ref.wo = try makeWeights(alloc, &wlist, L * dim * dim, 7.0);
    tf_ref.w1 = try makeWeights(alloc, &wlist, L * dim * hidden, 8.0);
    tf_ref.w2 = try makeWeights(alloc, &wlist, L * hidden * dim, 9.0);
    tf_ref.w3 = try makeWeights(alloc, &wlist, L * dim * hidden, 10.0);
    tf_ref.rms_final_weight = try makeWeights(alloc, &wlist, dim, 11.0);
    tf_ref.bq = try makeWeights(alloc, &wlist, L * dim, 12.0);
    tf_ref.bk = try makeWeights(alloc, &wlist, L * kv_dim, 13.0);
    tf_ref.bv = try makeWeights(alloc, &wlist, L * kv_dim, 14.0);

    const logits_ref = tf_ref.forward(2, 0);

    // Cuantizar embedding a Q8_0 manualmente (1 bloque por fila porque dim=32)
    const blk = zkernel.Q8_0_BLOCK_SIZE; // 32
    const blk_bytes = zkernel.Q8_0_BLOCK_BYTES; // 34
    const n_blocks = (dim + blk - 1) / blk; // 1
    var emb_quant = try alloc.alloc(u8, vocab * n_blocks * blk_bytes);
    defer alloc.free(emb_quant);
    for (0..vocab) |v| {
        const row = tf_ref.token_embedding_table[v * dim ..][0..dim];
        var max_abs: f32 = 0;
        for (row) |x| { const a = @abs(x); if (a > max_abs) max_abs = a; }
        const scale: f16 = @floatCast(max_abs / 127.0);
        const base = v * n_blocks * blk_bytes;
        std.mem.writeInt(u16, emb_quant[base..][0..2], @bitCast(scale), .little);
        for (0..dim) |i| {
            const qf = row[i] / @as(f32, @floatCast(scale));
            const clamped = @max(-127.0, @min(127.0, qf));
            const q: i8 = @intFromFloat(clamped);
            emb_quant[base + 2 + i] = @bitCast(q);
        }
    }

    var tf_q = try Transformer.init(alloc, cfg);
    defer tf_q.deinit();
    tf_q.token_embedding_table = &.{};
    tf_q.token_embedding_table_q = .{ .bytes = emb_quant, .fmt = .q8_0 };
    tf_q.rms_att_weight = tf_ref.rms_att_weight;
    tf_q.rms_ffn_weight = tf_ref.rms_ffn_weight;
    tf_q.wq = tf_ref.wq; tf_q.wk = tf_ref.wk; tf_q.wv = tf_ref.wv; tf_q.wo = tf_ref.wo;
    tf_q.w1 = tf_ref.w1; tf_q.w2 = tf_ref.w2; tf_q.w3 = tf_ref.w3;
    tf_q.rms_final_weight = tf_ref.rms_final_weight;
    tf_q.bq = tf_ref.bq; tf_q.bk = tf_ref.bk; tf_q.bv = tf_ref.bv;

    const logits_q = tf_q.forward(2, 0);

    for (0..vocab) |v| {
        try std.testing.expectApproxEqAbs(logits_ref[v], logits_q[v], 1e-3);
    }
}

test "Transformer MoE forward: finitud, determinismo, forwardBatch vs forward" {
    const alloc = std.testing.allocator;
    const cfg = Config{
        .dim = 8,
        .hidden_dim = 16,
        .n_layers = 1,
        .n_heads = 2,
        .n_kv_heads = 1,
        .vocab_size = 8,
        .seq_len = 16,
        .n_experts = 3,
        .top_k = 2,
        .moe_hidden_dim = 8,
        .theta = 10000.0,
        .eps = 1e-5,
    };
    const dim = cfg.dim;
    const kv_dim = (dim * cfg.n_kv_heads) / cfg.n_heads;
    const moe_dim = cfg.moe_hidden_dim;
    const ne = cfg.n_experts;
    const vocab = cfg.vocab_size;
    const L = cfg.n_layers;

    var tf = try Transformer.init(alloc, cfg);
    defer tf.deinit();

    var wlist = std.ArrayList([]f32).empty;
    defer { for (wlist.items) |b| alloc.free(b); wlist.deinit(alloc); }

    tf.token_embedding_table = try makeWeights(alloc, &wlist, vocab * dim, 1.0);
    tf.rms_att_weight = try makeWeights(alloc, &wlist, L * dim, 2.0);
    tf.rms_ffn_weight = try makeWeights(alloc, &wlist, L * dim, 3.0);
    tf.wq = try makeWeights(alloc, &wlist, L * dim * dim, 4.0);
    tf.wk = try makeWeights(alloc, &wlist, L * kv_dim * dim, 5.0);
    tf.wv = try makeWeights(alloc, &wlist, L * kv_dim * dim, 6.0);
    tf.wo = try makeWeights(alloc, &wlist, L * dim * dim, 7.0);
    tf.rms_final_weight = try makeWeights(alloc, &wlist, dim, 11.0);
    tf.bq = try makeWeights(alloc, &wlist, L * dim, 12.0);
    tf.bk = try makeWeights(alloc, &wlist, L * kv_dim, 13.0);
    tf.bv = try makeWeights(alloc, &wlist, L * kv_dim, 14.0);

    // Pesos MoE
    tf.moe_router = try makeWeights(alloc, &wlist, L * dim * ne, 15.0);
    tf.moe_w1 = try makeWeights(alloc, &wlist, ne * dim * moe_dim, 16.0);
    tf.moe_w2 = try makeWeights(alloc, &wlist, ne * moe_dim * dim, 17.0);
    tf.moe_w3 = try makeWeights(alloc, &wlist, ne * dim * moe_dim, 18.0);

    // 1) Finitud + no-NaN
    const l0 = tf.forward(3, 0);
    try std.testing.expectEqual(@as(usize, vocab), l0.len);
    for (l0) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }

    // 2) Determinismo
    var snap: [8]f32 = undefined;
    @memcpy(&snap, l0[0..vocab]);
    const l0b = tf.forward(3, 0);
    for (0..vocab) |i| try std.testing.expectApproxEqAbs(snap[i], l0b[i], 1e-6);

    // 3) forwardBatch K=1 coincide con forward
    var tf_batch = try Transformer.init(alloc, cfg);
    defer tf_batch.deinit();
    tf_batch.token_embedding_table = tf.token_embedding_table;
    tf_batch.rms_att_weight = tf.rms_att_weight;
    tf_batch.rms_ffn_weight = tf.rms_ffn_weight;
    tf_batch.wq = tf.wq; tf_batch.wk = tf.wk; tf_batch.wv = tf.wv; tf_batch.wo = tf.wo;
    tf_batch.rms_final_weight = tf.rms_final_weight;
    tf_batch.bq = tf.bq; tf_batch.bk = tf.bk; tf_batch.bv = tf.bv;
    tf_batch.moe_router = tf.moe_router;
    tf_batch.moe_w1 = tf.moe_w1;
    tf_batch.moe_w2 = tf.moe_w2;
    tf_batch.moe_w3 = tf.moe_w3;

    const logits_batch = try alloc.alloc(f32, vocab);
    defer alloc.free(logits_batch);
    try tf_batch.forwardBatch(&[_]u32{3}, 0, logits_batch);
    for (0..vocab) |v| try std.testing.expectApproxEqAbs(l0[v], logits_batch[v], 1e-4);

    // 4) forwardBatch K=2 coincide con 2xforward (oráculo)
    var tf_ref = try Transformer.init(alloc, cfg);
    defer tf_ref.deinit();
    tf_ref.token_embedding_table = tf.token_embedding_table;
    tf_ref.rms_att_weight = tf.rms_att_weight;
    tf_ref.rms_ffn_weight = tf.rms_ffn_weight;
    tf_ref.wq = tf.wq; tf_ref.wk = tf.wk; tf_ref.wv = tf.wv; tf_ref.wo = tf.wo;
    tf_ref.rms_final_weight = tf.rms_final_weight;
    tf_ref.bq = tf.bq; tf_ref.bk = tf.bk; tf_ref.bv = tf.bv;
    tf_ref.moe_router = tf.moe_router;
    tf_ref.moe_w1 = tf.moe_w1;
    tf_ref.moe_w2 = tf.moe_w2;
    tf_ref.moe_w3 = tf.moe_w3;

    const tokens = &[_]u32{ 3, 5 };
    const logits_ref_0 = try alloc.dupe(f32, tf_ref.forward(tokens[0], 0));
    defer alloc.free(logits_ref_0);
    const logits_ref_1 = try alloc.dupe(f32, tf_ref.forward(tokens[1], 1));
    defer alloc.free(logits_ref_1);

    var tf_batch2 = try Transformer.init(alloc, cfg);
    defer tf_batch2.deinit();
    tf_batch2.token_embedding_table = tf.token_embedding_table;
    tf_batch2.rms_att_weight = tf.rms_att_weight;
    tf_batch2.rms_ffn_weight = tf.rms_ffn_weight;
    tf_batch2.wq = tf.wq; tf_batch2.wk = tf.wk; tf_batch2.wv = tf.wv; tf_batch2.wo = tf.wo;
    tf_batch2.rms_final_weight = tf.rms_final_weight;
    tf_batch2.bq = tf.bq; tf_batch2.bk = tf.bk; tf_batch2.bv = tf.bv;
    tf_batch2.moe_router = tf.moe_router;
    tf_batch2.moe_w1 = tf.moe_w1;
    tf_batch2.moe_w2 = tf.moe_w2;
    tf_batch2.moe_w3 = tf.moe_w3;

    const logits_batch2 = try alloc.alloc(f32, tokens.len * vocab);
    defer alloc.free(logits_batch2);
    try tf_batch2.forwardBatch(tokens, 0, logits_batch2);

    for (0..vocab) |v| try std.testing.expectApproxEqAbs(logits_ref_0[v], logits_batch2[v], 1e-4);
    for (0..vocab) |v| try std.testing.expectApproxEqAbs(logits_ref_1[v], logits_batch2[vocab + v], 1e-4);
}
