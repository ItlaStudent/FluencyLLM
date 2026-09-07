// zgrad — autodiff reverse-mode
//
// Dos niveles:
//  1) `Tape` escalar (didáctico, tests de derivadas elementales).
//  2) `Graph` a nivel de TENSORES: el motor real de F3. Cada op guarda lo
//     necesario para su VJP (vector-Jacobian product) y `backward` recorre el
//     grafo en reversa acumulando gradientes. Cubre el op-set de un LM híbrido
//     (matmul, bias, residual, rmsnorm, silu, mul, embedding) MÁS los mixers
//     subcuadráticos con su backward: `scan` (SSM diagonal Mamba-2), `linAttn`
//     (atención lineal / delta) y `attn` (atención causal multi-cabeza).
//  Los VJP se validan por gradient-check numérico (diferencias finitas).
const std = @import("std");

// =====================================================================
// 1) Scalar autodiff (se conserva; tests de derivadas elementales)
// =====================================================================
pub const Value = struct {
    data: f32,
    grad: f32,
    idx: usize, // índice en el tape
};

pub const Op = enum { add, mul, relu, matmul };

pub const Node = struct {
    op: Op,
    inputs: [2]usize, // índices de inputs en el tape
    output: usize, // índice de output
};

pub const Tape = struct {
    values: std.ArrayList(Value),
    nodes: std.ArrayList(Node),
    alloc: std.mem.Allocator,

    pub fn init(alloc: std.mem.Allocator) Tape {
        return .{
            .values = std.ArrayList(Value).empty,
            .nodes = std.ArrayList(Node).empty,
            .alloc = alloc,
        };
    }

    pub fn deinit(self: *Tape) void {
        self.values.deinit(self.alloc);
        self.nodes.deinit(self.alloc);
    }

    pub fn scalar(self: *Tape, v: f32) !Value {
        const idx = self.values.items.len;
        try self.values.append(self.alloc, .{ .data = v, .grad = 0, .idx = idx });
        return self.values.items[idx];
    }

    pub fn add(self: *Tape, a: Value, b: Value) !Value {
        const out_v = a.data + b.data;
        const out = try self.scalar(out_v);
        try self.nodes.append(self.alloc, .{ .op = .add, .inputs = .{ a.idx, b.idx }, .output = out.idx });
        return out;
    }

    pub fn mul(self: *Tape, a: Value, b: Value) !Value {
        const out_v = a.data * b.data;
        const out = try self.scalar(out_v);
        try self.nodes.append(self.alloc, .{ .op = .mul, .inputs = .{ a.idx, b.idx }, .output = out.idx });
        return out;
    }

    pub fn relu(self: *Tape, a: Value) !Value {
        const out_v = if (a.data > 0) a.data else 0;
        const out = try self.scalar(out_v);
        try self.nodes.append(self.alloc, .{ .op = .relu, .inputs = .{ a.idx, a.idx }, .output = out.idx });
        return out;
    }

    pub fn backward(self: *Tape, output_idx: usize) void {
        self.values.items[output_idx].grad = 1.0;
        var i: usize = self.nodes.items.len;
        while (i > 0) {
            i -= 1;
            const node = self.nodes.items[i];
            const out = self.values.items[node.output];
            const a = self.values.items[node.inputs[0]];
            const b = self.values.items[node.inputs[1]];
            switch (node.op) {
                .add => {
                    self.values.items[node.inputs[0]].grad += out.grad;
                    self.values.items[node.inputs[1]].grad += out.grad;
                },
                .mul => {
                    self.values.items[node.inputs[0]].grad += out.grad * b.data;
                    self.values.items[node.inputs[1]].grad += out.grad * a.data;
                },
                .relu => {
                    self.values.items[node.inputs[0]].grad += if (a.data > 0) out.grad else 0;
                },
                .matmul => {},
            }
        }
    }

    pub fn zeroGrad(self: *Tape) void {
        for (self.values.items) |*v| v.grad = 0;
    }
};

// --- Gradientes de tensores planos (helpers sueltos, se conservan) ---
pub fn matmulGrad(dC: []const f32, A: []const f32, B: []const f32, dA: []f32, dB: []f32, M: usize, K: usize, N: usize) void {
    @memset(dA, 0);
    @memset(dB, 0);
    for (0..M) |m| {
        for (0..K) |k| {
            var acc: f32 = 0;
            for (0..N) |n| acc += dC[m * N + n] * B[k * N + n];
            dA[m * K + k] = acc;
        }
    }
    for (0..K) |k| {
        for (0..N) |n| {
            var acc: f32 = 0;
            for (0..M) |m| acc += A[m * K + k] * dC[m * N + n];
            dB[k * N + n] = acc;
        }
    }
}

/// Cross-entropy softmax gradient: dlogits = softmax(logits) - one_hot(target)
pub fn crossEntropyGrad(logits: []const f32, target: u32, dlogits: []f32) void {
    @memcpy(dlogits, logits);
    dlogits[target] -= 1.0;
}

/// MSE gradient: d = 2 * (pred - target) / N
pub fn mseGrad(pred: []const f32, target: []const f32, grad: []f32) void {
    const N: f32 = @floatFromInt(pred.len);
    for (pred, target, grad) |p, t, *g| {
        g.* = 2.0 * (p - t) / N;
    }
}

// =====================================================================
// 2) Tensor autodiff: `Graph` (el motor de F3)
// =====================================================================

/// Índice de un tensor dentro del grafo.
pub const Var = usize;

pub const GOp = enum {
    leaf, // hoja: parámetro o entrada (sin backward)
    matmul, // [M,K] @ [K,N] -> [M,N]
    matmul_tb, // [M,K] @ [N,K]^T -> [M,N] (proyección de salida atada al embedding)
    add_bias, // [M,N] + bias[1,N]
    add, // [M,N] + [M,N] (residual)
    rmsnorm, // por fila: x/sqrt(mean(x^2)+eps) * w
    silu, // x*sigmoid(x)
    mul, // elementwise [.,.] * [.,.]
    embed, // table[V,D], ids[M] -> [M,D]
    scan, // SSM diagonal causal: h_t=a*h_{t-1}+x_t
    lin_attn, // atención lineal causal (delta-like)
    attn, // atención causal multi-cabeza
    moe, // Mixture of Experts soft (diferenciable, entrenamiento)
    moe_lb_loss, // Load balancing loss auxiliar para MoE (Switch Transformer style)
    sparse_topk, // Atención lineal causal con sparsificación top-k
    gated_deltanet2, // Gated DeltaNet v2: SSM con gating y delta update
};

const GNode = struct {
    op: GOp,
    ins: []const Var, // inputs (0..3)
    out: Var,
    // datos guardados específicos de la op
    saved: ?[]f32 = null, // rms r por fila; probs de softmax; etc.
    ids: ?[]const u32 = null, // embed
    n_heads: usize = 1,
    head_dim: usize = 0,
    eps: f32 = 1e-5,
    n_experts: usize = 0,
    moe_dim: usize = 0,
    window_size: usize = 0,
};

/// Grafo de cómputo con autodiff reverse-mode a nivel de tensor.
/// Todas las activaciones (val/grad) viven en un arena que se libera de golpe
/// en `deinit`, así no hay fugas. Los parámetros del modelo son externos: se
/// inyectan como hojas con `leaf`, cuyo `.val` ALIAS-a el slice externo y cuyo
/// `.grad` es un buffer fresco del arena; tras `backward` se leen con `gradOf`.
pub const Graph = struct {
    arena: std.heap.ArenaAllocator,
    val: std.ArrayList([]f32),
    grad: std.ArrayList([]f32),
    rows: std.ArrayList(usize),
    cols: std.ArrayList(usize),
    nodes: std.ArrayList(GNode),

    pub fn init(backing: std.mem.Allocator) Graph {
        return .{
            .arena = std.heap.ArenaAllocator.init(backing),
            .val = std.ArrayList([]f32).empty,
            .grad = std.ArrayList([]f32).empty,
            .rows = std.ArrayList(usize).empty,
            .cols = std.ArrayList(usize).empty,
            .nodes = std.ArrayList(GNode).empty,
        };
    }

    pub fn deinit(self: *Graph) void {
        const backing = self.arena.child_allocator;
        self.val.deinit(backing);
        self.grad.deinit(backing);
        self.rows.deinit(backing);
        self.cols.deinit(backing);
        self.nodes.deinit(backing);
        self.arena.deinit();
    }

    fn ca(self: *Graph) std.mem.Allocator {
        return self.arena.child_allocator;
    }

    fn am(self: *Graph) std.mem.Allocator {
        return self.arena.allocator();
    }

    pub fn valOf(self: *Graph, v: Var) []f32 {
        return self.val.items[v];
    }
    pub fn gradOf(self: *Graph, v: Var) []f32 {
        return self.grad.items[v];
    }
    pub fn rowsOf(self: *Graph, v: Var) usize {
        return self.rows.items[v];
    }
    pub fn colsOf(self: *Graph, v: Var) usize {
        return self.cols.items[v];
    }

    fn push(self: *Graph, value: []f32, r: usize, c: usize) !Var {
        const id = self.val.items.len;
        const gr = try self.am().alloc(f32, r * c);
        @memset(gr, 0);
        try self.val.append(self.ca(), value);
        try self.grad.append(self.ca(), gr);
        try self.rows.append(self.ca(), r);
        try self.cols.append(self.ca(), c);
        return id;
    }

    /// Hoja que aliasa un slice externo (parámetro persistente del modelo).
    pub fn leaf(self: *Graph, data: []f32, r: usize, c: usize) !Var {
        std.debug.assert(data.len == r * c);
        const id = try self.push(data, r, c);
        try self.nodes.append(self.ca(), .{ .op = .leaf, .ins = &.{}, .out = id });
        return id;
    }

    fn newAct(self: *Graph, r: usize, c: usize) !Var {
        const value = try self.am().alloc(f32, r * c);
        @memset(value, 0);
        return try self.push(value, r, c);
    }

    fn insSlice(self: *Graph, items: []const Var) ![]const Var {
        const s = try self.am().alloc(Var, items.len);
        @memcpy(s, items);
        return s;
    }

    // ---- ops (forward) ----
    pub fn matmul(self: *Graph, av: Var, bv: Var) !Var {
        const M = self.rows.items[av];
        const K = self.cols.items[av];
        const N = self.cols.items[bv];
        std.debug.assert(self.rows.items[bv] == K);
        const out = try self.newAct(M, N);
        const A = self.val.items[av];
        const B = self.val.items[bv];
        const C = self.val.items[out];
        for (0..M) |m| {
            for (0..K) |k| {
                const aik = A[m * K + k];
                for (0..N) |n| C[m * N + n] += aik * B[k * N + n];
            }
        }
        try self.nodes.append(self.ca(), .{ .op = .matmul, .ins = try self.insSlice(&.{ av, bv }), .out = out });
        return out;
    }

    /// out[M,N] = A[M,K] @ B[N,K]^T, es decir out[m,n] = sum_k A[m,k]*B[n,k].
    /// Sirve para atar la proyección de salida al embedding (B = tabla [V,D]).
    pub fn matmulTransB(self: *Graph, av: Var, bv: Var) !Var {
        const M = self.rows.items[av];
        const K = self.cols.items[av];
        const N = self.rows.items[bv];
        std.debug.assert(self.cols.items[bv] == K);
        const out = try self.newAct(M, N);
        const A = self.val.items[av];
        const B = self.val.items[bv];
        const C = self.val.items[out];
        for (0..M) |m| {
            for (0..N) |n| {
                var acc: f32 = 0;
                for (0..K) |k| acc += A[m * K + k] * B[n * K + k];
                C[m * N + n] = acc;
            }
        }
        try self.nodes.append(self.ca(), .{ .op = .matmul_tb, .ins = try self.insSlice(&.{ av, bv }), .out = out });
        return out;
    }

    pub fn addBias(self: *Graph, av: Var, bias: Var) !Var {
        const M = self.rows.items[av];
        const N = self.cols.items[av];
        std.debug.assert(self.cols.items[bias] == N);
        const out = try self.newAct(M, N);
        const A = self.val.items[av];
        const Bs = self.val.items[bias];
        const O = self.val.items[out];
        for (0..M) |m| {
            for (0..N) |n| O[m * N + n] = A[m * N + n] + Bs[n];
        }
        try self.nodes.append(self.ca(), .{ .op = .add_bias, .ins = try self.insSlice(&.{ av, bias }), .out = out });
        return out;
    }

    pub fn addResidual(self: *Graph, av: Var, bv: Var) !Var {
        const M = self.rows.items[av];
        const N = self.cols.items[av];
        const out = try self.newAct(M, N);
        const A = self.val.items[av];
        const B = self.val.items[bv];
        const O = self.val.items[out];
        for (0..M * N) |i| O[i] = A[i] + B[i];
        try self.nodes.append(self.ca(), .{ .op = .add, .ins = try self.insSlice(&.{ av, bv }), .out = out });
        return out;
    }

    pub fn rmsnorm(self: *Graph, av: Var, w: Var, eps: f32) !Var {
        const M = self.rows.items[av];
        const D = self.cols.items[av];
        const out = try self.newAct(M, D);
        const A = self.val.items[av];
        const W = self.val.items[w];
        const O = self.val.items[out];
        const r_saved = try self.am().alloc(f32, M);
        for (0..M) |m| {
            var ss: f32 = 0;
            for (0..D) |d| ss += A[m * D + d] * A[m * D + d];
            const r = 1.0 / @sqrt(@max(ss / @as(f32, @floatFromInt(D)) + eps, eps));
            r_saved[m] = r;
            for (0..D) |d| O[m * D + d] = A[m * D + d] * r * W[d];
        }
        try self.nodes.append(self.ca(), .{ .op = .rmsnorm, .ins = try self.insSlice(&.{ av, w }), .out = out, .saved = r_saved, .eps = eps });
        return out;
    }

    pub fn silu(self: *Graph, av: Var) !Var {
        const M = self.rows.items[av];
        const N = self.cols.items[av];
        const out = try self.newAct(M, N);
        const A = self.val.items[av];
        const O = self.val.items[out];
        for (0..M * N) |i| {
            const x = A[i];
            if (x >= 0) {
                O[i] = x / (1.0 + @exp(-x));
            } else {
                const ex = @exp(x);
                O[i] = x * ex / (1.0 + ex);
            }
        }
        try self.nodes.append(self.ca(), .{ .op = .silu, .ins = try self.insSlice(&.{av}), .out = out });
        return out;
    }

    pub fn mul(self: *Graph, av: Var, bv: Var) !Var {
        const M = self.rows.items[av];
        const N = self.cols.items[av];
        const out = try self.newAct(M, N);
        const A = self.val.items[av];
        const B = self.val.items[bv];
        const O = self.val.items[out];
        for (0..M * N) |i| O[i] = A[i] * B[i];
        try self.nodes.append(self.ca(), .{ .op = .mul, .ins = try self.insSlice(&.{ av, bv }), .out = out });
        return out;
    }

    pub fn embed(self: *Graph, table: Var, ids: []const u32) !Var {
        const D = self.cols.items[table];
        const M = ids.len;
        const out = try self.newAct(M, D);
        const T = self.val.items[table];
        const O = self.val.items[out];
        for (0..M) |m| {
            const base = ids[m] * D;
            for (0..D) |d| O[m * D + d] = T[base + d];
        }
        try self.nodes.append(self.ca(), .{ .op = .embed, .ins = try self.insSlice(&.{table}), .out = out, .ids = ids });
        return out;
    }

    /// SSM diagonal causal: h_t = a ⊙ h_{t-1} + x_t,  out_t = h_t.
    /// x = av [T,D], a = decay [1,D] (valores en ~(0,1)). out [T,D] (=h).
    pub fn scan(self: *Graph, av: Var, a_decay: Var) !Var {
        const T = self.rows.items[av];
        const D = self.cols.items[av];
        std.debug.assert(self.cols.items[a_decay] == D);
        const out = try self.newAct(T, D);
        const X = self.val.items[av];
        const A = self.val.items[a_decay];
        const O = self.val.items[out];
        for (0..T) |t| {
            for (0..D) |e| {
                const prev = if (t == 0) 0.0 else O[(t - 1) * D + e];
                O[t * D + e] = A[e] * prev + X[t * D + e];
            }
        }
        try self.nodes.append(self.ca(), .{ .op = .scan, .ins = try self.insSlice(&.{ av, a_decay }), .out = out });
        return out;
    }

    /// Atención lineal causal (delta-like): o_t = sum_{t'<=t} (q_t·k_{t'}) v_{t'}.
    /// q,k,v = [T,D] cada uno. out [T,D].
    pub fn linAttn(self: *Graph, qv: Var, kv: Var, vv: Var) !Var {
        const T = self.rows.items[qv];
        const D = self.cols.items[qv];
        const out = try self.newAct(T, D);
        const Q = self.val.items[qv];
        const K = self.val.items[kv];
        const V = self.val.items[vv];
        const O = self.val.items[out];
        for (0..T) |t| {
            for (0..t + 1) |tp| {
                var s: f32 = 0;
                for (0..D) |j| s += Q[t * D + j] * K[tp * D + j];
                for (0..D) |i| O[t * D + i] += s * V[tp * D + i];
            }
        }
        try self.nodes.append(self.ca(), .{ .op = .lin_attn, .ins = try self.insSlice(&.{ qv, kv, vv }), .out = out });
        return out;
    }

    /// Atención causal multi-cabeza con softmax. q,k,v = [T, H*hd].
    /// Guarda las probabilidades P [H,T,T] para el backward.
    pub fn attn(self: *Graph, qv: Var, kv: Var, vv: Var, n_heads: usize, window_size: usize) !Var {
        const T = self.rows.items[qv];
        const HD = self.cols.items[qv];
        const hd = HD / n_heads;
        const out = try self.newAct(T, HD);
        const Q = self.val.items[qv];
        const K = self.val.items[kv];
        const V = self.val.items[vv];
        const O = self.val.items[out];
        const probs = try self.am().alloc(f32, n_heads * T * T);
        @memset(probs, 0);
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const scores = try self.ca().alloc(f32, T);
        defer self.ca().free(scores);
        for (0..n_heads) |h| {
            const off = h * hd;
            for (0..T) |i| {
                const j_start = if (window_size > 0 and i + 1 >= window_size) i + 1 - window_size else 0;
                // scores causales i>=j en ventana
                var maxv: f32 = -std.math.inf(f32);
                for (j_start..i + 1) |j| {
                    var s: f32 = 0;
                    for (0..hd) |d| s += Q[i * HD + off + d] * K[j * HD + off + d];
                    s *= scale;
                    scores[j] = s;
                    if (s > maxv) maxv = s;
                }
                var sum: f32 = 0;
                for (j_start..i + 1) |j| {
                    const e = @exp(scores[j] - maxv);
                    scores[j] = e;
                    sum += e;
                }
                if (sum <= 0) sum = 1e-20; // protección contra div/0 con logits adversariales
                const inv = 1.0 / sum;
                for (j_start..i + 1) |j| {
                    const p = scores[j] * inv;
                    probs[(h * T + i) * T + j] = p;
                    for (0..hd) |d| O[i * HD + off + d] += p * V[j * HD + off + d];
                }
            }
        }
        try self.nodes.append(self.ca(), .{ .op = .attn, .ins = try self.insSlice(&.{ qv, kv, vv }), .out = out, .saved = probs, .n_heads = n_heads, .head_dim = hd, .window_size = window_size });
        return out;
    }

    /// Mixture of Experts (soft, diferenciable) para entrenamiento.
    /// x [1, D], router [D, E], w1 [E*D, M], w3 [E*D, M], w2 [E*M, D].
    /// Layout: w1[e*D*M + d*M + m], w2[e*M*D + m*D + d].
    /// out [1, D]. Todos los expertos contribuyen ponderados por softmax(router).
    pub fn moe(self: *Graph, x: Var, router: Var, w1: Var, w3: Var, w2: Var, n_experts: usize, moe_dim: usize) !Var {
        const D = self.cols.items[x];
        const T = self.rows.items[x];
        std.debug.assert(self.rows.items[router] == D and self.cols.items[router] == n_experts);
        const E = n_experts;
        const M = moe_dim;
        const out = try self.newAct(T, D);
        const X = self.val.items[x];
        const R = self.val.items[router];
        const W1 = self.val.items[w1];
        const W3 = self.val.items[w3];
        const W2 = self.val.items[w2];
        const O = self.val.items[out];

        const block = 2 * E + 2 * E * M + E * D;
        const saved_len = T * block;
        const saved = try self.am().alloc(f32, saved_len);
        @memset(saved, 0);
        @memset(O, 0);
        const h1_buf = try self.am().alloc(f32, M);
        var router_logits = try self.am().alloc(f32, E);
        for (0..T) |t| {
            const x_off = t * D;
            // router_logits = x_t @ router
            for (0..E) |e| {
                var acc: f32 = 0;
                for (0..D) |d| acc += X[x_off + d] * R[d * E + e];
                router_logits[e] = acc;
            }
            // softmax
            var max_l = router_logits[0];
            for (1..E) |e| {
                if (router_logits[e] > max_l) {
                    max_l = router_logits[e];
                }
            }
            var sum_e: f32 = 0;
            for (0..E) |e| {
                router_logits[e] = @exp(router_logits[e] - max_l);
                sum_e += router_logits[e];
            }
            const inv_sum = 1.0 / sum_e;
            for (0..E) |e| router_logits[e] *= inv_sum;

            const s_off = t * block;
            for (0..E) |e| saved[s_off + e] = router_logits[e];
            for (0..E) |e| saved[s_off + E + e] = router_logits[e];

            for (0..E) |e| {
                const prob = router_logits[e];
                for (0..M) |m| {
                    var acc: f32 = 0;
                    for (0..D) |d| acc += X[x_off + d] * W1[e * D * M + d * M + m];
                    h1_buf[m] = acc;
                }
                for (0..M) |m| {
                    var acc: f32 = 0;
                    for (0..D) |d| acc += X[x_off + d] * W3[e * D * M + d * M + m];
                    saved[s_off + 2 * E + E * M + e * M + m] = acc; // h_gate
                }
                for (0..M) |m| {
                    const z = h1_buf[m];
                    h1_buf[m] = z / (1.0 + @exp(-z));
                    saved[s_off + 2 * E + e * M + m] = h1_buf[m]; // h_silu
                }
                for (0..M) |m| h1_buf[m] *= saved[s_off + 2 * E + E * M + e * M + m];
                for (0..D) |d| {
                    var acc: f32 = 0;
                    for (0..M) |m| acc += h1_buf[m] * W2[e * M * D + m * D + d];
                    saved[s_off + 2 * E + 2 * E * M + e * D + d] = acc; // o_exp
                    O[x_off + d] += prob * acc;
                }
            }
        }
        try self.nodes.append(self.ca(), .{ .op = .moe, .ins = try self.insSlice(&.{ x, router, w1, w3, w2 }), .out = out, .saved = saved, .n_experts = n_experts, .moe_dim = moe_dim });
        return out;
    }

    /// Load balancing loss auxiliar para MoE (estilo Switch Transformer / Mixtral).
    /// L_aux = alpha * E * sum_e (P_e^2), donde P_e = mean_t probs[t,e].
    /// Incentiva distribución uniforme de tokens entre expertos.
    /// `moe_out` debe ser la salida de una llamada a `moe()`.
    pub fn moeLoadBalanceLoss(self: *Graph, moe_out: Var, alpha: f32) !Var {
        // Encontrar el nodo moe que produce moe_out
        var moe_idx: ?usize = null;
        for (self.nodes.items, 0..) |n, i| {
            if (n.op == .moe and n.out == moe_out) {
                moe_idx = i;
                break;
            }
        }
        const idx = moe_idx orelse return error.MoeNodeNotFound;
        const node = &self.nodes.items[idx];
        const E = node.n_experts;
        const D = self.cols.items[node.ins[0]];
        const T = self.rows.items[node.ins[0]];
        const moe_saved = node.saved.?;
        const block = 2 * E + 2 * E * node.moe_dim + E * D;

        // Guardar probs [T*E] + alpha (último elemento)
        const lb_saved = try self.am().alloc(f32, T * E + 1);
        for (0..T) |t| {
            const s_off = t * block;
            for (0..E) |e| lb_saved[t * E + e] = moe_saved[s_off + e];
        }
        lb_saved[T * E] = alpha;

        // Forward: L_aux = alpha * E * sum_e (P_e^2)
        var loss: f32 = 0;
        for (0..E) |e| {
            var p_e: f32 = 0;
            for (0..T) |t| p_e += lb_saved[t * E + e];
            p_e /= @floatFromInt(T);
            loss += p_e * p_e;
        }
        loss *= alpha * @as(f32, @floatFromInt(E));

        const out = try self.newAct(1, 1);
        self.val.items[out][0] = loss;

        try self.nodes.append(self.ca(), .{
            .op = .moe_lb_loss,
            .ins = try self.insSlice(&.{ node.ins[0], node.ins[1] }), // x, router
            .out = out,
            .saved = lb_saved,
            .n_experts = E,
            .moe_dim = T, // reutilizamos moe_dim para guardar T
        });
        return out;
    }

    /// Calcula la load balancing loss total sumando sobre todos los nodos MoE
    /// del grafo. Devuelve null si no hay nodos MoE.
    pub fn totalMoeLoadBalanceLoss(self: *Graph, alpha: f32) !?Var {
        var total: ?Var = null;
        for (self.nodes.items) |n| {
            if (n.op == .moe) {
                const lb = try self.moeLoadBalanceLoss(n.out, alpha);
                if (total) |t| {
                    total = try self.addResidual(t, lb);
                } else {
                    total = lb;
                }
            }
        }
        return total;
    }

    /// SparseTopK: atención lineal causal con sparsificación top-k.
    /// Para cada token t, calcula scores s[t,t'] = q_t · k_{t'} para t' <= t,
    /// selecciona los top-k tokens con mayor score, y combina sus valores v.
    /// q,k,v = [T,D]. out [T,D]. k_val = número de tokens top-k.
    /// Complejidad: O(T * k * D) en vez de O(T^2 * D).
    pub fn sparseTopK(self: *Graph, qv: Var, kv: Var, vv: Var, k_val: usize) !Var {
        const T = self.rows.items[qv];
        const D = self.cols.items[qv];
        std.debug.assert(self.cols.items[kv] == D);
        std.debug.assert(self.cols.items[vv] == D);
        const out = try self.newAct(T, D);
        const Q = self.val.items[qv];
        const K = self.val.items[kv];
        const V = self.val.items[vv];
        const O = self.val.items[out];

        // Guardamos: scores [T*T] (para top-k selection), indices [T*k], q,k,v para backward
        const saved = try self.am().alloc(f32, T * D * 3); // q, k, v
        @memcpy(saved[0 .. T * D], Q);
        @memcpy(saved[T * D .. 2 * T * D], K);
        @memcpy(saved[2 * T * D .. 3 * T * D], V);

        const kk = @min(k_val, T);
        const idx_buf = try self.am().alloc(usize, T * kk);
        @memset(idx_buf, 0);

        for (0..T) |t| {
            // Calcular scores para t' <= t
            const max_prev = t + 1;
            var scores_buf: [256]f32 = undefined;
            const scores = if (max_prev <= 256) scores_buf[0..max_prev] else self.ca().alloc(f32, max_prev) catch return out;
            defer if (max_prev > 256) self.ca().free(scores);

            for (0..max_prev) |tp| {
                var s: f32 = 0;
                for (0..D) |d| s += Q[t * D + d] * K[tp * D + d];
                scores[tp] = s;
            }

            // Top-k selection (selection sort parcial)
            const k_actual = @min(kk, max_prev);
            for (0..k_actual) |ki| {
                var best_idx = ki;
                var best_score = scores[ki];
                for (ki + 1..max_prev) |tp| {
                    if (scores[tp] > best_score) {
                        best_score = scores[tp];
                        best_idx = tp;
                    }
                }
                // Swap
                const tmp_s = scores[ki];
                scores[ki] = scores[best_idx];
                scores[best_idx] = tmp_s;
                idx_buf[t * kk + ki] = best_idx;
            }

            // Softmax sobre los top-k scores
            var max_s = scores[0];
            for (1..k_actual) |ki| {
                if (scores[ki] > max_s) max_s = scores[ki];
            }
            var sum_e: f32 = 0;
            for (0..k_actual) |ki| {
                scores[ki] = @exp(scores[ki] - max_s);
                sum_e += scores[ki];
            }
            const inv_sum = 1.0 / sum_e;
            for (0..k_actual) |ki| scores[ki] *= inv_sum;

            // Output = sum of v[top_k_indices] * weights
            for (0..D) |d| {
                var acc: f32 = 0;
                for (0..k_actual) |ki| {
                    const tp = idx_buf[t * kk + ki];
                    acc += scores[ki] * V[tp * D + d];
                }
                O[t * D + d] = acc;
            }
        }

        try self.nodes.append(self.ca(), .{
            .op = .sparse_topk,
            .ins = try self.insSlice(&.{ qv, kv, vv }),
            .out = out,
            .saved = saved,
            .n_experts = k_val, // reutilizamos n_experts para k
            .moe_dim = idx_buf.len, // guardamos longitud del index buffer
        });
        return out;
    }

    /// GatedDeltaNet v2: SSM diagonal con gating y delta update.
    /// h_t = gate_t ⊙ h_{t-1} + (1 - gate_t) ⊙ x_t
    /// out_t = h_t
    /// x = av [T,D], gate = gv [T,D] (valores en (0,1) vía sigmoid). out [T,D].
    pub fn gatedDeltaNet2(self: *Graph, av: Var, gv: Var) !Var {
        const T = self.rows.items[av];
        const D = self.cols.items[av];
        std.debug.assert(self.rows.items[gv] == T);
        std.debug.assert(self.cols.items[gv] == D);
        const out = try self.newAct(T, D);
        const X = self.val.items[av];
        const G_raw = self.val.items[gv];
        const O = self.val.items[out];

        // Guardar: x [T*D], gate_sigmoid [T*D], h_states [T*D]
        const saved = try self.am().alloc(f32, T * D * 3);
        @memcpy(saved[0 .. T * D], X);
        // Compute sigmoid gates and save
        for (0..T * D) |i| {
            saved[T * D + i] = 1.0 / (1.0 + @exp(-G_raw[i]));
        }

        // Forward: h_t = gate_t * h_{t-1} + (1 - gate_t) * x_t
        var h: [128]f32 = undefined;
        const h_buf = if (D <= 128) h[0..D] else self.ca().alloc(f32, D) catch return out;
        defer if (D > 128) self.ca().free(h_buf);
        @memset(h_buf, 0);

        for (0..T) |t| {
            const x_off = t * D;
            const g_off = t * D;
            const h_off = t * D;
            for (0..D) |d| {
                const g = saved[T * D + g_off + d];
                h_buf[d] = g * h_buf[d] + (1.0 - g) * X[x_off + d];
                O[h_off + d] = h_buf[d];
                saved[2 * T * D + h_off + d] = h_buf[d]; // guardar h para backward
            }
        }

        try self.nodes.append(self.ca(), .{
            .op = .gated_deltanet2,
            .ins = try self.insSlice(&.{ av, gv }),
            .out = out,
            .saved = saved,
        });
        return out;
    }

    // ---- backward ----
    /// Propaga gradientes desde `loss_var` (debe ser escalar [1,1]) hacia atrás.
    pub fn backward(self: *Graph, loss_var: Var) void {
        std.debug.assert(self.rows.items[loss_var] == 1 and self.cols.items[loss_var] == 1);
        self.grad.items[loss_var][0] = 1.0;
        var ni: usize = self.nodes.items.len;
        while (ni > 0) {
            ni -= 1;
            const node = self.nodes.items[ni];
            switch (node.op) {
                .leaf => {},
                .matmul => self.bwdMatmul(node),
                .matmul_tb => self.bwdMatmulTb(node),
                .add_bias => self.bwdAddBias(node),
                .add => self.bwdAdd(node),
                .rmsnorm => self.bwdRmsnorm(node),
                .silu => self.bwdSilu(node),
                .mul => self.bwdMul(node),
                .embed => self.bwdEmbed(node),
                .scan => self.bwdScan(node),
                .lin_attn => self.bwdLinAttn(node),
                .attn => self.bwdAttn(node),
                .moe => self.bwdMoe(node),
                .moe_lb_loss => self.bwdMoeLbLoss(node),
                .sparse_topk => self.bwdSparseTopK(node),
                .gated_deltanet2 => self.bwdGatedDeltaNet2(node),
            }
        }
    }

    fn bwdMoe(self: *Graph, node: GNode) void {
        const x = node.ins[0];
        const router = node.ins[1];
        const w1 = node.ins[2];
        const w3 = node.ins[3];
        const w2 = node.ins[4];
        const E = node.n_experts;
        const M = node.moe_dim;
        const D = self.cols.items[x];
        const T = self.rows.items[x];
        const X = self.val.items[x];
        const R = self.val.items[router];
        const W1 = self.val.items[w1];
        const W3 = self.val.items[w3];
        const W2 = self.val.items[w2];
        const dO = self.grad.items[node.out];
        const dX = self.grad.items[x];
        const dR = self.grad.items[router];
        const dW1 = self.grad.items[w1];
        const dW3 = self.grad.items[w3];
        const dW2 = self.grad.items[w2];

        const saved = node.saved.?;
        const block = 2 * E + 2 * E * M + E * D;
        const d_probs = self.ca().alloc(f32, E) catch return;
        defer self.ca().free(d_probs);
        const d_logits = self.ca().alloc(f32, E) catch return;
        defer self.ca().free(d_logits);
        const d_x_router = self.ca().alloc(f32, D) catch return;
        defer self.ca().free(d_x_router);
        const d_h = self.ca().alloc(f32, M) catch return;
        defer self.ca().free(d_h);
        const d_xw1 = self.ca().alloc(f32, M) catch return;
        defer self.ca().free(d_xw1);
        const d_xw3 = self.ca().alloc(f32, M) catch return;
        defer self.ca().free(d_xw3);
        const d_x_e = self.ca().alloc(f32, D) catch return;
        defer self.ca().free(d_x_e);

        for (0..T) |t| {
            const x_off = t * D;
            const s_off = t * block;
            const probs = saved[s_off..s_off + E];
            const h_silu = saved[s_off + 2*E..s_off + 2*E + E*M];
            const h_gate = saved[s_off + 2*E + E*M..s_off + 2*E + 2*E*M];
            const o_exp = saved[s_off + 2*E + 2*E*M..s_off + 2*E + 2*E*M + E*D];
            const dO_t = dO[x_off..x_off + D];

            // d_probs[e] = sum_d dO[d] * o_exp[e,d]
            for (0..E) |e| {
                var acc: f32 = 0;
                for (0..D) |d| acc += dO_t[d] * o_exp[e*D + d];
                d_probs[e] = acc;
            }
            // softmax backward
            var dot: f32 = 0;
            for (0..E) |e| dot += probs[e] * d_probs[e];
            for (0..E) |e| d_logits[e] = probs[e] * (d_probs[e] - dot);

            // d_router += x_t^T @ d_logits
            for (0..D) |d| {
                for (0..E) |e| {
                    dR[d*E + e] += X[x_off + d] * d_logits[e];
                }
            }
            // d_x from router
            for (0..D) |d| {
                var acc: f32 = 0;
                for (0..E) |e| acc += d_logits[e] * R[d*E + e];
                d_x_router[d] = acc;
            }
            for (0..D) |d| dX[x_off + d] += d_x_router[d];

            for (0..E) |e| {
                const prob = probs[e];
                // d_w2 += h^T @ dy_e
                for (0..D) |d| {
                    const dy_e = prob * dO_t[d];
                    for (0..M) |m| {
                        const h = h_silu[e*M + m] * h_gate[e*M + m];
                        dW2[e*M*D + m*D + d] += h * dy_e;
                    }
                }
                // d_h = dy_e @ w2[e]^T
                for (0..M) |m| {
                    var acc: f32 = 0;
                    for (0..D) |d| acc += prob * dO_t[d] * W2[e*M*D + m*D + d];
                    d_h[m] = acc;
                }
                // d_a, d_b, silu'
                for (0..M) |m| {
                    const a = h_silu[e*M + m];
                    const b = h_gate[e*M + m];
                    const d_a = d_h[m] * b;
                    d_xw3[m] = d_h[m] * a;
                    var z: f32 = 0;
                    for (0..D) |d| z += X[x_off + d] * W1[e*D*M + d*M + m];
                    const sig = 1.0 / (1.0 + @exp(-z));
                    const ds = sig * (1.0 + z * (1.0 - sig));
                    d_xw1[m] = d_a * ds;
                }
                // d_w1 += x_t^T @ d_xw1, d_w3 += x_t^T @ d_xw3
                for (0..M) |m| {
                    for (0..D) |d| {
                        dW1[e*D*M + d*M + m] += X[x_off + d] * d_xw1[m];
                        dW3[e*D*M + d*M + m] += X[x_off + d] * d_xw3[m];
                    }
                }
                // d_x de este experto: X sólo afecta a la salida vía h[m] = silu(X·W1)·(X·W3),
                // por tanto SÓLO hay caminos W1 y W3. NO existe camino X→salida vía W2 (W2
                // multiplica h, no X); el efecto de W2 sobre d_x ya está contenido en d_h →
                // d_xw1/d_xw3. (Un término `d_h·W2` aquí sería un gradiente espurio — bug
                // detectado por gradient-check de todos los inputs.)
                @memset(d_x_e, 0);
                for (0..D) |d| {
                    var acc: f32 = 0;
                    for (0..M) |m| acc += d_xw1[m] * W1[e*D*M + d*M + m];
                    d_x_e[d] += acc;
                }
                for (0..D) |d| {
                    var acc: f32 = 0;
                    for (0..M) |m| acc += d_xw3[m] * W3[e*D*M + d*M + m];
                    d_x_e[d] += acc;
                }
                for (0..D) |d| dX[x_off + d] += d_x_e[d];
            }
        }
    }

    fn bwdMoeLbLoss(self: *Graph, node: GNode) void {
        const x = node.ins[0];
        const router = node.ins[1];
        const E = node.n_experts;
        const T = node.moe_dim; // reutilizamos moe_dim para T
        const D = self.cols.items[x];
        const X = self.val.items[x];
        const dR = self.grad.items[router];
        const saved = node.saved.?;
        const probs = saved[0 .. T * E];
        const alpha = saved[T * E];

        // dL/d_out = 1 (escalar), ya fijado por backward()
        const dL = self.grad.items[node.out][0];

        // P_e = mean_t probs[t,e]
        // L = alpha * E * sum_e P_e^2
        // dL/d_probs[t,e] = alpha * E * 2 * P_e * (1/T)
        // Through softmax backward: dL/d_logits[t,e] = probs[t,e] * (g_e - sum_j probs[t,j] * g_j)
        // where g_e = dL/d_probs[t,e] (same for all t since P_e is a mean)

        // Calcular P_e
        var p_e_buf: [64]f32 = undefined;
        const p_e = if (E <= 64) p_e_buf[0..E] else self.ca().alloc(f32, E) catch return;
        defer if (E > 64) self.ca().free(p_e);
        for (0..E) |e| {
            var s: f32 = 0;
            for (0..T) |t| s += probs[t * E + e];
            p_e[e] = s / @as(f32, @floatFromInt(T));
        }

        // g_e = dL/d_probs[t,e] = alpha * E * 2 * P_e * (1/T) * dL
        const coeff = alpha * @as(f32, @floatFromInt(E)) * 2.0 / @as(f32, @floatFromInt(T)) * dL;
        var g_buf: [64]f32 = undefined;
        const g = if (E <= 64) g_buf[0..E] else self.ca().alloc(f32, E) catch return;
        defer if (E > 64) self.ca().free(g);
        for (0..E) |e| g[e] = coeff * p_e[e];

        const d_logits = self.ca().alloc(f32, E) catch return;
        defer self.ca().free(d_logits);

        for (0..T) |t| {
            const x_off = t * D;
            const t_probs = probs[t * E .. t * E + E];
            // softmax backward: d_logits[e] = probs[e] * (g[e] - sum_j probs[j]*g[j])
            var dot: f32 = 0;
            for (0..E) |e| dot += t_probs[e] * g[e];
            for (0..E) |e| d_logits[e] = t_probs[e] * (g[e] - dot);
            // d_router += x_t^T @ d_logits
            for (0..D) |d| {
                for (0..E) |e| {
                    dR[d * E + e] += X[x_off + d] * d_logits[e];
                }
            }
        }
    }

    fn bwdSparseTopK(self: *Graph, node: GNode) void {
        // SparseTopK backward: aproxima el gradiente asumiendo que los índices top-k
        // son fijos (straight-through estimator). El gradiente fluye a través del
        // softmax hacia q, k, v.
        const qv = node.ins[0];
        const kv = node.ins[1];
        const vv = node.ins[2];
        const T = self.rows.items[qv];
        const D = self.cols.items[qv];
        const kk = @min(node.n_experts, T);
        const saved = node.saved.?;
        const Q = saved[0 .. T * D];
        const K = saved[T * D .. 2 * T * D];
        const V = saved[2 * T * D .. 3 * T * D];
        const dO = self.grad.items[node.out];
        const dQ = self.grad.items[qv];
        const dK = self.grad.items[kv];
        const dV = self.grad.items[vv];

        // Recalcular scores e índices top-k (igual que forward)
        for (0..T) |t| {
            const max_prev = t + 1;
            var scores_buf: [256]f32 = undefined;
            const scores = if (max_prev <= 256) scores_buf[0..max_prev] else self.ca().alloc(f32, max_prev) catch return;
            defer if (max_prev > 256) self.ca().free(scores);

            for (0..max_prev) |tp| {
                var s: f32 = 0;
                for (0..D) |d| s += Q[t * D + d] * K[tp * D + d];
                scores[tp] = s;
            }

            // Top-k selection
            var idx_buf: [256]usize = undefined;
            const idx = if (kk <= 256) idx_buf[0..kk] else self.ca().alloc(usize, kk) catch return;
            defer if (kk > 256) self.ca().free(idx);
            const k_actual = @min(kk, max_prev);
            for (0..k_actual) |ki| {
                var best_idx = ki;
                var best_score = scores[ki];
                for (ki + 1..max_prev) |tp| {
                    if (scores[tp] > best_score) {
                        best_score = scores[tp];
                        best_idx = tp;
                    }
                }
                const tmp_s = scores[ki];
                scores[ki] = scores[best_idx];
                scores[best_idx] = tmp_s;
                idx[ki] = best_idx;
            }

            // Softmax
            var max_s = scores[0];
            for (1..k_actual) |ki| {
                if (scores[ki] > max_s) max_s = scores[ki];
            }
            var sum_e: f32 = 0;
            for (0..k_actual) |ki| {
                scores[ki] = @exp(scores[ki] - max_s);
                sum_e += scores[ki];
            }
            const inv_sum = 1.0 / sum_e;
            const weights = scores[0..k_actual];
            for (weights) |*w| w.* *= inv_sum;

            // dV[tp] += weights[ki] * dO[t]
            for (0..k_actual) |ki| {
                const tp = idx[ki];
                const w = weights[ki];
                for (0..D) |d| {
                    dV[tp * D + d] += w * dO[t * D + d];
                }
            }

            // d_softmax: dw[ki] = sum_d dO[t,d] * V[idx[ki],d]
            var dw_buf: [256]f32 = undefined;
            const dw = if (k_actual <= 256) dw_buf[0..k_actual] else self.ca().alloc(f32, k_actual) catch return;
            defer if (k_actual > 256) self.ca().free(dw);
            for (0..k_actual) |ki| {
                const tp = idx[ki];
                var acc: f32 = 0;
                for (0..D) |d| acc += dO[t * D + d] * V[tp * D + d];
                dw[ki] = acc;
            }

            // softmax backward: d_logits[ki] = w[ki] * (dw[ki] - sum_j w[j]*dw[j])
            var dot: f32 = 0;
            for (0..k_actual) |ki| dot += weights[ki] * dw[ki];
            var d_logits_buf: [256]f32 = undefined;
            const d_logits = if (k_actual <= 256) d_logits_buf[0..k_actual] else self.ca().alloc(f32, k_actual) catch return;
            defer if (k_actual > 256) self.ca().free(d_logits);
            for (0..k_actual) |ki| {
                d_logits[ki] = weights[ki] * (dw[ki] - dot);
            }

            // dQ[t] += sum_ki d_logits[ki] * K[idx[ki]]
            for (0..k_actual) |ki| {
                const tp = idx[ki];
                const dl = d_logits[ki];
                for (0..D) |d| {
                    dQ[t * D + d] += dl * K[tp * D + d];
                }
            }
            // dK[tp] += sum_ki d_logits[ki] * Q[t] (sólo para los tp seleccionados)
            for (0..k_actual) |ki| {
                const tp = idx[ki];
                const dl = d_logits[ki];
                for (0..D) |d| {
                    dK[tp * D + d] += dl * Q[t * D + d];
                }
            }
        }
    }

    fn bwdGatedDeltaNet2(self: *Graph, node: GNode) void {
        // h_t = g_t * h_{t-1} + (1 - g_t) * x_t
        // dh_{t-1} += g_t * dh_t
        // dg_t += dh_t * (h_{t-1} - x_t)
        // dx_t += (1 - g_t) * dh_t
        // sigmoid backward: dG_raw[t,d] = g * (1-g) * dg[t,d]
        const av = node.ins[0];
        const gv = node.ins[1];
        const T = self.rows.items[av];
        const D = self.cols.items[av];
        const saved = node.saved.?;
        const X = saved[0 .. T * D];
        const gates = saved[T * D .. 2 * T * D]; // sigmoid gates
        const h_states = saved[2 * T * D .. 3 * T * D];
        const dO = self.grad.items[node.out];
        const dX = self.grad.items[av];
        const dG = self.grad.items[gv];

        // Backward en orden inverso
        var dh_buf: [128]f32 = undefined;
        const dh = if (D <= 128) dh_buf[0..D] else self.ca().alloc(f32, D) catch return;
        defer if (D > 128) self.ca().free(dh);
        @memset(dh, 0);

        var t: usize = T;
        while (t > 0) {
            t -= 1;
            const off = t * D;
            for (0..D) |d| {
                dh[d] += dO[off + d];
                const g = gates[off + d];
                const h_prev = if (t > 0) h_states[(t - 1) * D + d] else 0;
                // dx_t += (1 - g) * dh
                dX[off + d] += (1.0 - g) * dh[d];
                // dg_t += dh * (h_prev - x_t)
                const dg_val = dh[d] * (h_prev - X[off + d]);
                // sigmoid backward: dG_raw = g * (1-g) * dg
                dG[off + d] += g * (1.0 - g) * dg_val;
            }
            // Propagar dh al paso anterior: dh_{t-1} += g * dh_t
            for (0..D) |d| {
                dh[d] = gates[off + d] * dh[d];
            }
        }
    }

    fn bwdMatmul(self: *Graph, node: GNode) void {
        const av = node.ins[0];
        const bv = node.ins[1];
        const M = self.rows.items[av];
        const K = self.cols.items[av];
        const N = self.cols.items[bv];
        const A = self.val.items[av];
        const B = self.val.items[bv];
        const dO = self.grad.items[node.out];
        const dA = self.grad.items[av];
        const dB = self.grad.items[bv];
        for (0..M) |m| {
            for (0..K) |k| {
                var acc: f32 = 0;
                for (0..N) |n| acc += dO[m * N + n] * B[k * N + n];
                dA[m * K + k] += acc;
            }
        }
        for (0..K) |k| {
            for (0..N) |n| {
                var acc: f32 = 0;
                for (0..M) |m| acc += A[m * K + k] * dO[m * N + n];
                dB[k * N + n] += acc;
            }
        }
    }

    fn bwdMatmulTb(self: *Graph, node: GNode) void {
        const av = node.ins[0];
        const bv = node.ins[1];
        const M = self.rows.items[av];
        const K = self.cols.items[av];
        const N = self.rows.items[bv];
        const A = self.val.items[av];
        const B = self.val.items[bv];
        const dO = self.grad.items[node.out];
        const dA = self.grad.items[av];
        const dB = self.grad.items[bv];
        // out[m,n] = sum_k A[m,k]*B[n,k]
        for (0..M) |m| {
            for (0..N) |n| {
                const g = dO[m * N + n];
                for (0..K) |k| {
                    dA[m * K + k] += g * B[n * K + k];
                    dB[n * K + k] += g * A[m * K + k];
                }
            }
        }
    }

    fn bwdAddBias(self: *Graph, node: GNode) void {
        const av = node.ins[0];
        const bias = node.ins[1];
        const M = self.rows.items[av];
        const N = self.cols.items[av];
        const dO = self.grad.items[node.out];
        const dA = self.grad.items[av];
        const dBias = self.grad.items[bias];
        for (0..M) |m| {
            for (0..N) |n| {
                dA[m * N + n] += dO[m * N + n];
                dBias[n] += dO[m * N + n];
            }
        }
    }

    fn bwdAdd(self: *Graph, node: GNode) void {
        const dO = self.grad.items[node.out];
        const dA = self.grad.items[node.ins[0]];
        const dB = self.grad.items[node.ins[1]];
        for (0..dO.len) |i| {
            dA[i] += dO[i];
            dB[i] += dO[i];
        }
    }

    fn bwdRmsnorm(self: *Graph, node: GNode) void {
        const av = node.ins[0];
        const w = node.ins[1];
        const M = self.rows.items[av];
        const D = self.cols.items[av];
        const A = self.val.items[av];
        const W = self.val.items[w];
        const dO = self.grad.items[node.out];
        const dA = self.grad.items[av];
        const dW = self.grad.items[w];
        const r_saved = node.saved.?;
        const Df: f32 = @floatFromInt(D);
        for (0..M) |m| {
            const r = r_saved[m];
            // sum_gx = sum_i (dO_i * w_i) * x_i
            var sum_gx: f32 = 0;
            for (0..D) |d| sum_gx += (dO[m * D + d] * W[d]) * A[m * D + d];
            for (0..D) |d| {
                const x = A[m * D + d];
                const g = dO[m * D + d] * W[d];
                dA[m * D + d] += r * g - (r * r * r * x / Df) * sum_gx;
                dW[d] += dO[m * D + d] * r * x;
            }
        }
    }

    fn bwdSilu(self: *Graph, node: GNode) void {
        const av = node.ins[0];
        const A = self.val.items[av];
        const dO = self.grad.items[node.out];
        const dA = self.grad.items[av];
        for (0..A.len) |i| {
            const x = A[i];
            const sig = if (x >= 0) 1.0 / (1.0 + @exp(-x)) else @exp(x) / (1.0 + @exp(x));
            const d = sig * (1.0 + x * (1.0 - sig));
            dA[i] += dO[i] * d;
        }
    }

    fn bwdMul(self: *Graph, node: GNode) void {
        const av = node.ins[0];
        const bv = node.ins[1];
        const A = self.val.items[av];
        const B = self.val.items[bv];
        const dO = self.grad.items[node.out];
        const dA = self.grad.items[av];
        const dB = self.grad.items[bv];
        for (0..A.len) |i| {
            dA[i] += dO[i] * B[i];
            dB[i] += dO[i] * A[i];
        }
    }

    fn bwdEmbed(self: *Graph, node: GNode) void {
        const table = node.ins[0];
        const D = self.cols.items[table];
        const ids = node.ids.?;
        const dO = self.grad.items[node.out];
        const dT = self.grad.items[table];
        for (0..ids.len) |m| {
            const base = ids[m] * D;
            for (0..D) |d| dT[base + d] += dO[m * D + d];
        }
    }

    fn bwdScan(self: *Graph, node: GNode) void {
        const av = node.ins[0];
        const a_decay = node.ins[1];
        const T = self.rows.items[av];
        const D = self.cols.items[av];
        const A = self.val.items[a_decay];
        const O = self.val.items[node.out]; // = h
        const dO = self.grad.items[node.out];
        const dX = self.grad.items[av];
        const dA = self.grad.items[a_decay];
        // G_t = dO_t + a * G_{t+1}; dx_t = G_t; da += G_t * h_{t-1}
        const gnext = self.ca().alloc(f32, D) catch return;
        defer self.ca().free(gnext);
        @memset(gnext, 0);
        var t: usize = T;
        while (t > 0) {
            t -= 1;
            for (0..D) |e| {
                const G = dO[t * D + e] + A[e] * gnext[e];
                dX[t * D + e] += G;
                const hprev = if (t == 0) 0.0 else O[(t - 1) * D + e];
                dA[e] += G * hprev;
                gnext[e] = G;
            }
        }
    }

    fn bwdLinAttn(self: *Graph, node: GNode) void {
        const qv = node.ins[0];
        const kv = node.ins[1];
        const vv = node.ins[2];
        const T = self.rows.items[qv];
        const D = self.cols.items[qv];
        const Q = self.val.items[qv];
        const K = self.val.items[kv];
        const V = self.val.items[vv];
        const dO = self.grad.items[node.out];
        const dQ = self.grad.items[qv];
        const dK = self.grad.items[kv];
        const dV = self.grad.items[vv];
        // o_t[i] = sum_{t'<=t} (q_t·k_{t'}) v_{t'}[i]
        for (0..T) |t| {
            for (0..t + 1) |tp| {
                var s: f32 = 0;
                for (0..D) |j| s += Q[t * D + j] * K[tp * D + j];
                // ds = sum_i dO_t[i] * v_{t'}[i]
                var ds: f32 = 0;
                for (0..D) |i| ds += dO[t * D + i] * V[tp * D + i];
                for (0..D) |i| {
                    dV[tp * D + i] += s * dO[t * D + i];
                    dQ[t * D + i] += ds * K[tp * D + i];
                    dK[tp * D + i] += ds * Q[t * D + i];
                }
            }
        }
    }

    fn bwdAttn(self: *Graph, node: GNode) void {
        const qv = node.ins[0];
        const kv = node.ins[1];
        const vv = node.ins[2];
        const T = self.rows.items[qv];
        const HD = self.cols.items[qv];
        const H = node.n_heads;
        const hd = node.head_dim;
        const window_size = node.window_size;
        const Q = self.val.items[qv];
        const K = self.val.items[kv];
        const V = self.val.items[vv];
        const probs = node.saved.?;
        const dO = self.grad.items[node.out];
        const dQ = self.grad.items[qv];
        const dK = self.grad.items[kv];
        const dV = self.grad.items[vv];
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const dP = self.ca().alloc(f32, T) catch return;
        defer self.ca().free(dP);
        const dS = self.ca().alloc(f32, T) catch return;
        defer self.ca().free(dS);
        for (0..H) |h| {
            const off = h * hd;
            for (0..T) |i| {
                const j_start = if (window_size > 0 and i + 1 >= window_size) i + 1 - window_size else 0;
                // dV y dP
                for (j_start..i + 1) |j| {
                    const p = probs[(h * T + i) * T + j];
                    var dp: f32 = 0;
                    for (0..hd) |d| {
                        dV[j * HD + off + d] += p * dO[i * HD + off + d];
                        dp += dO[i * HD + off + d] * V[j * HD + off + d];
                    }
                    dP[j] = dp;
                }
                // softmax backward: dS_j = p_j (dP_j - sum_k p_k dP_k)
                var dot: f32 = 0;
                for (j_start..i + 1) |j| dot += probs[(h * T + i) * T + j] * dP[j];
                for (j_start..i + 1) |j| {
                    const p = probs[(h * T + i) * T + j];
                    dS[j] = p * (dP[j] - dot);
                }
                // dQ_i, dK_j
                for (j_start..i + 1) |j| {
                    const sj = dS[j] * scale;
                    for (0..hd) |d| {
                        dQ[i * HD + off + d] += sj * K[j * HD + off + d];
                        dK[j * HD + off + d] += sj * Q[i * HD + off + d];
                    }
                }
            }
        }
    }
};

// =====================================================================
// Losses diferenciables que cierran el grafo (devuelven loss escalar y
// dejan el dgrad listo en la entrada `logits`).
// =====================================================================

/// Cross-entropy sobre [M,V] con targets [M]. Devuelve loss media y escribe
/// el gradiente (softmax - onehot)/M en grad(logits). Cierra el grafo: tras
/// llamarla, `backward(loss_var)` con el loss_var devuelto propaga todo.
pub fn crossEntropyLoss(g: *Graph, logits: Var, targets: []const u32) !Var {
    const M = g.rows.items[logits];
    const V = g.cols.items[logits];
    std.debug.assert(targets.len == M);
    const L = g.val.items[logits];
    const loss_v = try g.newAct(1, 1);
    var total: f32 = 0;
    const dL = g.grad.items[logits];
    const Mf: f32 = @floatFromInt(M);
    const row = try g.ca().alloc(f32, V);
    defer g.ca().free(row);
    for (0..M) |m| {
        var maxv: f32 = -std.math.inf(f32);
        for (0..V) |v| {
            if (L[m * V + v] > maxv) maxv = L[m * V + v];
        }
        var sum: f32 = 0;
        for (0..V) |v| {
            const e = @exp(L[m * V + v] - maxv);
            row[v] = e;
            sum += e;
        }
        if (sum <= 0) sum = 1e-20;
        const log_sum = @log(sum);
        const tgt = targets[m];
        total += -(L[m * V + tgt] - maxv - log_sum);
        const inv = 1.0 / sum;
        for (0..V) |v| {
            const p = row[v] * inv;
            dL[m * V + v] = (p - (if (v == tgt) @as(f32, 1.0) else 0.0)) / Mf;
        }
    }
    g.val.items[loss_v][0] = total / Mf;
    // marca: el gradiente de logits ya está puesto; el nodo loss no tiene op,
    // pero backward arranca poniendo grad(loss)=1 y NO toca logits (ya escrito).
    // Para que backward propague desde logits, lo registramos como hoja-puente:
    try g.nodes.append(g.ca(), .{ .op = .leaf, .ins = &.{}, .out = loss_v });
    return loss_v;
}

/// KD por distilación: KL(softmax(teacher/τ) || softmax(student/τ)) sobre [M,V].
/// Devuelve loss escalar y deja el gradiente respecto a `student` (logits) en
/// grad(student), listo para backward. Escala por τ² (convención Hinton).
pub fn kdDistillLoss(g: *Graph, student: Var, teacher_logits: []const f32, tau: f32) !Var {
    const M = g.rows.items[student];
    const V = g.cols.items[student];
    std.debug.assert(teacher_logits.len == M * V);
    const S = g.val.items[student];
    const dS = g.grad.items[student];
    const loss_v = try g.newAct(1, 1);
    var total: f32 = 0;
    const Mf: f32 = @floatFromInt(M);
    const ps = try g.ca().alloc(f32, V);
    defer g.ca().free(ps);
    const pt = try g.ca().alloc(f32, V);
    defer g.ca().free(pt);
    for (0..M) |m| {
        // softmax teacher/τ
        var tmax: f32 = -std.math.inf(f32);
        var smax: f32 = -std.math.inf(f32);
        for (0..V) |v| {
            if (teacher_logits[m * V + v] > tmax) tmax = teacher_logits[m * V + v];
            if (S[m * V + v] > smax) smax = S[m * V + v];
        }
        var tsum: f32 = 0;
        var ssum: f32 = 0;
        for (0..V) |v| {
            const te = @exp((teacher_logits[m * V + v] - tmax) / tau);
            const se = @exp((S[m * V + v] - smax) / tau);
            pt[v] = te;
            ps[v] = se;
            tsum += te;
            ssum += se;
        }
        if (tsum <= 0) tsum = 1e-20;
        if (ssum <= 0) ssum = 1e-20;
        const tinv = 1.0 / tsum;
        const sinv = 1.0 / ssum;
        for (0..V) |v| {
            const p_t = pt[v] * tinv;
            const p_s = ps[v] * sinv;
            total += p_t * (@log(p_t + 1e-20) - @log(p_s + 1e-20));
            // d/d student_v de KL(pt||ps) = (p_s - p_t)/τ ; promediado por M y *τ²
            dS[m * V + v] = (p_s - p_t) * tau / Mf;
        }
    }
    g.val.items[loss_v][0] = (total / Mf) * tau * tau;
    try g.nodes.append(g.ca(), .{ .op = .leaf, .ins = &.{}, .out = loss_v });
    return loss_v;
}

/// MSE entre un nodo del grafo `hidden` [M,N] y un target fijo `target` [M,N].
/// Escribe el gradiente dL/dhidden en `grad[hidden]` y devuelve un nodo leaf
/// dummy con el valor del MSE. El backward propaga desde hidden hacia los
/// parámetros anteriores automáticamente.
pub fn mseLoss(g: *Graph, hidden: Var, target: []const f32) !Var {
    const M = g.rows.items[hidden];
    const N = g.cols.items[hidden];
    std.debug.assert(target.len == M * N);
    const H = g.val.items[hidden];
    const dH = g.grad.items[hidden];
    const loss_v = try g.newAct(1, 1);
    var total: f32 = 0;
    const Nf: f32 = @floatFromInt(M * N);
    for (0..M * N) |i| {
        const diff = H[i] - target[i];
        total += diff * diff;
        dH[i] = 2.0 * diff / Nf;
    }
    g.val.items[loss_v][0] = total / Nf;
    try g.nodes.append(g.ca(), .{ .op = .leaf, .ins = &.{}, .out = loss_v });
    return loss_v;
}

// =====================================================================
// tests
// =====================================================================
test "scalar autodiff mul+add" {
    const alloc = std.testing.allocator;
    var tape = Tape.init(alloc);
    defer tape.deinit();
    const x = try tape.scalar(2.0);
    const y = try tape.scalar(3.0);
    const z = try tape.add(x, y);
    const w = try tape.mul(z, x);
    tape.backward(w.idx);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), tape.values.items[w.idx].data, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), tape.values.items[x.idx].grad, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), tape.values.items[y.idx].grad, 1e-5);
}

test "matmulGrad" {
    const alloc = std.testing.allocator;
    const M, const K, const N = .{ 2, 3, 4 };
    const A = try alloc.alloc(f32, M * K);
    const B = try alloc.alloc(f32, K * N);
    const dC = try alloc.alloc(f32, M * N);
    const dA = try alloc.alloc(f32, M * K);
    const dB = try alloc.alloc(f32, K * N);
    defer {
        alloc.free(A);
        alloc.free(B);
        alloc.free(dC);
        alloc.free(dA);
        alloc.free(dB);
    }
    for (A) |*p| p.* = 1.0;
    for (B) |*p| p.* = 1.0;
    for (dC) |*p| p.* = 1.0;
    matmulGrad(dC, A, B, dA, dB, M, K, N);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), dA[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), dB[0], 1e-5);
}

test "crossEntropyGrad" {
    var logits: [5]f32 = .{ 0.2, 0.2, 0.2, 0.2, 0.2 };
    var dlogits: [5]f32 = undefined;
    crossEntropyGrad(&logits, 2, &dlogits);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), dlogits[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2 - 1.0), dlogits[2], 1e-5);
}

test "mseGrad" {
    const pred = [_]f32{ 1.0, 2.0, 3.0 };
    const target = [_]f32{ 1.5, 2.5, 3.5 };
    var grad: [3]f32 = undefined;
    mseGrad(&pred, &target, &grad);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 * (-0.5) / 3.0), grad[0], 1e-5);
}

// ---- gradient-checks numéricos del motor Graph ----
// Verifican que cada VJP coincide con la derivada por diferencias finitas.

const gc = struct {
    // suma ponderada determinista del output (proxy de loss escalar)
    fn weightedSumLoss(g: *Graph, out: Var, wts: []const f32) !Var {
        const O = g.val.items[out];
        const loss_v = try g.newAct(1, 1);
        var s: f32 = 0;
        const dO = g.grad.items[out];
        for (0..O.len) |i| {
            s += O[i] * wts[i];
            dO[i] = wts[i];
        }
        g.val.items[loss_v][0] = s;
        try g.nodes.append(g.ca(), .{ .op = .leaf, .ins = &.{}, .out = loss_v });
        return loss_v;
    }
};

test "gradcheck matmul + addBias + rmsnorm + silu encadenados" {
    const alloc = std.testing.allocator;
    const M, const K, const N = .{ 3, 4, 5 };
    var A = [_]f32{0} ** (M * K);
    var W = [_]f32{0} ** (K * N);
    var bias = [_]f32{0} ** N;
    var rw = [_]f32{0} ** N;
    var rng = std.Random.DefaultPrng.init(123);
    for (&A) |*p| p.* = rng.random().float(f32) - 0.5;
    for (&W) |*p| p.* = rng.random().float(f32) - 0.5;
    for (&bias) |*p| p.* = rng.random().float(f32) - 0.5;
    for (&rw) |*p| p.* = rng.random().float(f32) + 0.5;
    var wts = [_]f32{0} ** (M * N);
    for (&wts) |*p| p.* = rng.random().float(f32) - 0.5;

    const Fwd = struct {
        fn run(al: std.mem.Allocator, a: []f32, w: []f32, b: []f32, r: []f32, ww: []const f32, dA_out: ?[]f32) f32 {
            var g = Graph.init(al);
            defer g.deinit();
            const av = g.leaf(a, M, K) catch unreachable;
            const wv = g.leaf(w, K, N) catch unreachable;
            const bv = g.leaf(b, 1, N) catch unreachable;
            const rv = g.leaf(r, 1, N) catch unreachable;
            const mm = g.matmul(av, wv) catch unreachable;
            const ab = g.addBias(mm, bv) catch unreachable;
            const nn = g.rmsnorm(ab, rv, 1e-5) catch unreachable;
            const si = g.silu(nn) catch unreachable;
            const loss = gc.weightedSumLoss(&g, si, ww) catch unreachable;
            const lv = g.val.items[loss][0];
            if (dA_out) |d| {
                g.backward(loss);
                @memcpy(d, g.gradOf(av));
            }
            return lv;
        }
    };
    var dA_ana = [_]f32{0} ** (M * K);
    _ = Fwd.run(alloc, &A, &W, &bias, &rw, &wts, &dA_ana);
    // numérico sobre A
    const eps: f32 = 1e-3;
    for (0..M * K) |i| {
        const orig = A[i];
        A[i] = orig + eps;
        const lp = Fwd.run(alloc, &A, &W, &bias, &rw, &wts, null);
        A[i] = orig - eps;
        const lm = Fwd.run(alloc, &A, &W, &bias, &rw, &wts, null);
        A[i] = orig;
        const num = (lp - lm) / (2 * eps);
        try std.testing.expectApproxEqAbs(num, dA_ana[i], 2e-2);
    }
}

test "gradcheck matmulTransB (embeddings atados) backward vs numérico" {
    const alloc = std.testing.allocator;
    const M, const K, const N = .{ 3, 4, 5 };
    var A = [_]f32{0} ** (M * K);
    var B = [_]f32{0} ** (N * K);
    var rng = std.Random.DefaultPrng.init(31);
    for (&A) |*p| p.* = rng.random().float(f32) - 0.5;
    for (&B) |*p| p.* = rng.random().float(f32) - 0.5;
    var wts = [_]f32{0} ** (M * N);
    for (&wts) |*p| p.* = rng.random().float(f32) - 0.5;

    const Fwd = struct {
        fn run(al: std.mem.Allocator, a: []f32, b: []f32, ww: []const f32, dB: ?[]f32) f32 {
            var g = Graph.init(al);
            defer g.deinit();
            const av = g.leaf(a, M, K) catch unreachable;
            const bv = g.leaf(b, N, K) catch unreachable;
            const mm = g.matmulTransB(av, bv) catch unreachable;
            const loss = gc.weightedSumLoss(&g, mm, ww) catch unreachable;
            const lv = g.val.items[loss][0];
            if (dB) |d| {
                g.backward(loss);
                @memcpy(d, g.gradOf(bv));
            }
            return lv;
        }
    };
    var dB_ana = [_]f32{0} ** (N * K);
    _ = Fwd.run(alloc, &A, &B, &wts, &dB_ana);
    const eps: f32 = 1e-3;
    for (0..N * K) |i| {
        const o = B[i];
        B[i] = o + eps;
        const lp = Fwd.run(alloc, &A, &B, &wts, null);
        B[i] = o - eps;
        const lm = Fwd.run(alloc, &A, &B, &wts, null);
        B[i] = o;
        try std.testing.expectApproxEqAbs((lp - lm) / (2 * eps), dB_ana[i], 2e-2);
    }
}

test "gradcheck scan (SSM) backward vs numérico" {
    const alloc = std.testing.allocator;
    const T, const D = .{ 6, 3 };
    var X = [_]f32{0} ** (T * D);
    var a = [_]f32{ 0.5, 0.8, 0.3 };
    var rng = std.Random.DefaultPrng.init(7);
    for (&X) |*p| p.* = rng.random().float(f32) - 0.5;
    var wts = [_]f32{0} ** (T * D);
    for (&wts) |*p| p.* = rng.random().float(f32) - 0.5;

    const Fwd = struct {
        fn run(al: std.mem.Allocator, x: []f32, av: []f32, ww: []const f32, dX: ?[]f32, dA: ?[]f32) f32 {
            var g = Graph.init(al);
            defer g.deinit();
            const xv = g.leaf(x, T, D) catch unreachable;
            const dv = g.leaf(av, 1, D) catch unreachable;
            const sc = g.scan(xv, dv) catch unreachable;
            const loss = gc.weightedSumLoss(&g, sc, ww) catch unreachable;
            const lv = g.val.items[loss][0];
            if (dX) |d| {
                g.backward(loss);
                @memcpy(d, g.gradOf(xv));
                if (dA) |da| @memcpy(da, g.gradOf(dv));
            }
            return lv;
        }
    };
    var dX_ana = [_]f32{0} ** (T * D);
    var dA_ana = [_]f32{0} ** D;
    _ = Fwd.run(alloc, &X, &a, &wts, &dX_ana, &dA_ana);
    const eps: f32 = 1e-3;
    for (0..T * D) |i| {
        const o = X[i];
        X[i] = o + eps;
        const lp = Fwd.run(alloc, &X, &a, &wts, null, null);
        X[i] = o - eps;
        const lm = Fwd.run(alloc, &X, &a, &wts, null, null);
        X[i] = o;
        try std.testing.expectApproxEqAbs((lp - lm) / (2 * eps), dX_ana[i], 2e-2);
    }
    for (0..D) |i| {
        const o = a[i];
        a[i] = o + eps;
        const lp = Fwd.run(alloc, &X, &a, &wts, null, null);
        a[i] = o - eps;
        const lm = Fwd.run(alloc, &X, &a, &wts, null, null);
        a[i] = o;
        try std.testing.expectApproxEqAbs((lp - lm) / (2 * eps), dA_ana[i], 2e-2);
    }
}

test "gradcheck linAttn (delta) backward vs numérico" {
    const alloc = std.testing.allocator;
    const T, const D = .{ 5, 3 };
    var Q = [_]f32{0} ** (T * D);
    var K = [_]f32{0} ** (T * D);
    var Vv = [_]f32{0} ** (T * D);
    var rng = std.Random.DefaultPrng.init(11);
    for (&Q) |*p| p.* = (rng.random().float(f32) - 0.5) * 0.5;
    for (&K) |*p| p.* = (rng.random().float(f32) - 0.5) * 0.5;
    for (&Vv) |*p| p.* = (rng.random().float(f32) - 0.5) * 0.5;
    var wts = [_]f32{0} ** (T * D);
    for (&wts) |*p| p.* = rng.random().float(f32) - 0.5;

    const Fwd = struct {
        fn run(al: std.mem.Allocator, q: []f32, k: []f32, v: []f32, ww: []const f32, dQ: ?[]f32) f32 {
            var g = Graph.init(al);
            defer g.deinit();
            const qv = g.leaf(q, T, D) catch unreachable;
            const kv = g.leaf(k, T, D) catch unreachable;
            const vv = g.leaf(v, T, D) catch unreachable;
            const la = g.linAttn(qv, kv, vv) catch unreachable;
            const loss = gc.weightedSumLoss(&g, la, ww) catch unreachable;
            const lv = g.val.items[loss][0];
            if (dQ) |d| {
                g.backward(loss);
                @memcpy(d, g.gradOf(qv));
            }
            return lv;
        }
    };
    var dQ_ana = [_]f32{0} ** (T * D);
    _ = Fwd.run(alloc, &Q, &K, &Vv, &wts, &dQ_ana);
    const eps: f32 = 1e-3;
    for (0..T * D) |i| {
        const o = Q[i];
        Q[i] = o + eps;
        const lp = Fwd.run(alloc, &Q, &K, &Vv, &wts, null);
        Q[i] = o - eps;
        const lm = Fwd.run(alloc, &Q, &K, &Vv, &wts, null);
        Q[i] = o;
        try std.testing.expectApproxEqAbs((lp - lm) / (2 * eps), dQ_ana[i], 2e-2);
    }
}

test "gradcheck attn (atención causal multi-cabeza) backward vs numérico" {
    const alloc = std.testing.allocator;
    const T, const HD, const H = .{ 4, 4, 2 };
    var Q = [_]f32{0} ** (T * HD);
    var K = [_]f32{0} ** (T * HD);
    var Vv = [_]f32{0} ** (T * HD);
    var rng = std.Random.DefaultPrng.init(13);
    for (&Q) |*p| p.* = (rng.random().float(f32) - 0.5) * 0.5;
    for (&K) |*p| p.* = (rng.random().float(f32) - 0.5) * 0.5;
    for (&Vv) |*p| p.* = (rng.random().float(f32) - 0.5) * 0.5;
    var wts = [_]f32{0} ** (T * HD);
    for (&wts) |*p| p.* = rng.random().float(f32) - 0.5;

    const Fwd = struct {
        fn run(al: std.mem.Allocator, q: []f32, k: []f32, v: []f32, ww: []const f32, dK: ?[]f32) f32 {
            var g = Graph.init(al);
            defer g.deinit();
            const qv = g.leaf(q, T, HD) catch unreachable;
            const kv = g.leaf(k, T, HD) catch unreachable;
            const vv = g.leaf(v, T, HD) catch unreachable;
            const at = g.attn(qv, kv, vv, H, 0) catch unreachable;
            const loss = gc.weightedSumLoss(&g, at, ww) catch unreachable;
            const lv = g.val.items[loss][0];
            if (dK) |d| {
                g.backward(loss);
                @memcpy(d, g.gradOf(kv));
            }
            return lv;
        }
    };
    var dK_ana = [_]f32{0} ** (T * HD);
    _ = Fwd.run(alloc, &Q, &K, &Vv, &wts, &dK_ana);
    const eps: f32 = 1e-3;
    for (0..T * HD) |i| {
        const o = K[i];
        K[i] = o + eps;
        const lp = Fwd.run(alloc, &Q, &K, &Vv, &wts, null);
        K[i] = o - eps;
        const lm = Fwd.run(alloc, &Q, &K, &Vv, &wts, null);
        K[i] = o;
        try std.testing.expectApproxEqAbs((lp - lm) / (2 * eps), dK_ana[i], 2e-2);
    }
}

test "gradcheck attn SWA (window=2) backward vs numérico" {
    const alloc = std.testing.allocator;
    const T, const HD, const H, const W = .{ 5, 4, 2, 2 };
    var Q = [_]f32{0} ** (T * HD);
    var K = [_]f32{0} ** (T * HD);
    var Vv = [_]f32{0} ** (T * HD);
    var rng = std.Random.DefaultPrng.init(14);
    for (&Q) |*p| p.* = (rng.random().float(f32) - 0.5) * 0.5;
    for (&K) |*p| p.* = (rng.random().float(f32) - 0.5) * 0.5;
    for (&Vv) |*p| p.* = (rng.random().float(f32) - 0.5) * 0.5;
    var wts = [_]f32{0} ** (T * HD);
    for (&wts) |*p| p.* = rng.random().float(f32) - 0.5;

    const Fwd = struct {
        fn run(al: std.mem.Allocator, q: []f32, k: []f32, v: []f32, ww: []const f32, dQ: ?[]f32) f32 {
            var g = Graph.init(al);
            defer g.deinit();
            const qv = g.leaf(q, T, HD) catch unreachable;
            const kv = g.leaf(k, T, HD) catch unreachable;
            const vv = g.leaf(v, T, HD) catch unreachable;
            const at = g.attn(qv, kv, vv, H, W) catch unreachable;
            const loss = gc.weightedSumLoss(&g, at, ww) catch unreachable;
            const lv = g.val.items[loss][0];
            if (dQ) |d| {
                g.backward(loss);
                @memcpy(d, g.gradOf(qv));
            }
            return lv;
        }
    };
    var dQ_ana = [_]f32{0} ** (T * HD);
    _ = Fwd.run(alloc, &Q, &K, &Vv, &wts, &dQ_ana);
    const eps: f32 = 1e-3;
    for (0..T * HD) |i| {
        const o = Q[i];
        Q[i] = o + eps;
        const lp = Fwd.run(alloc, &Q, &K, &Vv, &wts, null);
        Q[i] = o - eps;
        const lm = Fwd.run(alloc, &Q, &K, &Vv, &wts, null);
        Q[i] = o;
        try std.testing.expectApproxEqAbs((lp - lm) / (2 * eps), dQ_ana[i], 2e-2);
    }
}

test "gradcheck embed backward vs numérico" {
    const alloc = std.testing.allocator;
    const Vocab, const D = .{ 6, 4 };
    var table = [_]f32{0} ** (Vocab * D);
    var rng = std.Random.DefaultPrng.init(17);
    for (&table) |*p| p.* = rng.random().float(f32) - 0.5;
    const ids = [_]u32{ 1, 3, 3, 0 };
    var wts = [_]f32{0} ** (ids.len * D);
    for (&wts) |*p| p.* = rng.random().float(f32) - 0.5;

    const Fwd = struct {
        fn run(al: std.mem.Allocator, t: []f32, idsx: []const u32, ww: []const f32, dT: ?[]f32) f32 {
            var g = Graph.init(al);
            defer g.deinit();
            const tv = g.leaf(t, Vocab, D) catch unreachable;
            const ev = g.embed(tv, idsx) catch unreachable;
            const loss = gc.weightedSumLoss(&g, ev, ww) catch unreachable;
            const lv = g.val.items[loss][0];
            if (dT) |d| {
                g.backward(loss);
                @memcpy(d, g.gradOf(tv));
            }
            return lv;
        }
    };
    var dT_ana = [_]f32{0} ** (Vocab * D);
    _ = Fwd.run(alloc, &table, &ids, &wts, &dT_ana);
    const eps: f32 = 1e-3;
    for (0..Vocab * D) |i| {
        const o = table[i];
        table[i] = o + eps;
        const lp = Fwd.run(alloc, &table, &ids, &wts, null);
        table[i] = o - eps;
        const lm = Fwd.run(alloc, &table, &ids, &wts, null);
        table[i] = o;
        try std.testing.expectApproxEqAbs((lp - lm) / (2 * eps), dT_ana[i], 2e-2);
    }
}

test "crossEntropyLoss: gradiente coincide con softmax-onehot" {
    const alloc = std.testing.allocator;
    const M, const V = .{ 2, 5 };
    var logits = [_]f32{0} ** (M * V);
    var rng = std.Random.DefaultPrng.init(19);
    for (&logits) |*p| p.* = rng.random().float(f32) - 0.5;
    const targets = [_]u32{ 2, 4 };
    var g = Graph.init(alloc);
    defer g.deinit();
    const lv = try g.leaf(&logits, M, V);
    const loss = try crossEntropyLoss(&g, lv, &targets);
    g.backward(loss);
    // comprobar contra dlogits = (softmax - onehot)/M
    const dgot = g.gradOf(lv);
    for (0..M) |m| {
        var maxv: f32 = -std.math.inf(f32);
        for (0..V) |v| if (logits[m * V + v] > maxv) {
            maxv = logits[m * V + v];
        };
        var sum: f32 = 0;
        for (0..V) |v| sum += @exp(logits[m * V + v] - maxv);
        for (0..V) |v| {
            const p = @exp(logits[m * V + v] - maxv) / sum;
            const expect = (p - (if (v == targets[m]) @as(f32, 1) else 0)) / @as(f32, M);
            try std.testing.expectApproxEqAbs(expect, dgot[m * V + v], 1e-5);
        }
    }
}

test "Graph no fuga memoria (arena)" {
    const alloc = std.testing.allocator;
    var g = Graph.init(alloc);
    defer g.deinit();
    var A = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var W = [_]f32{ 1, 0, 0, 1, 1, 1 };
    const av = try g.leaf(&A, 2, 3);
    const wv = try g.leaf(&W, 3, 2);
    const mm = try g.matmul(av, wv);
    const wts = [_]f32{ 1, 1, 1, 1 };
    const loss = try gc.weightedSumLoss(&g, mm, &wts);
    g.backward(loss);
    try std.testing.expect(g.gradOf(av).len == 6);
}

test "gradcheck moe (soft) backward vs numérico" {
    const alloc = std.testing.allocator;
    const D, const E, const M = .{ 4, 3, 5 };
    var x = [_]f32{0} ** D;
    var router = [_]f32{0} ** (D * E);
    var w1 = [_]f32{0} ** (E * D * M);
    var w3 = [_]f32{0} ** (E * D * M);
    var w2 = [_]f32{0} ** (E * M * D);
    var rng = std.Random.DefaultPrng.init(99);
    for (&x) |*p| p.* = rng.random().float(f32) - 0.5;
    for (&router) |*p| p.* = rng.random().float(f32) * 0.1;
    for (&w1) |*p| p.* = rng.random().float(f32) * 0.1;
    for (&w3) |*p| p.* = rng.random().float(f32) * 0.1;
    for (&w2) |*p| p.* = rng.random().float(f32) * 0.1;
    var wts = [_]f32{0} ** D;
    for (&wts) |*p| p.* = rng.random().float(f32) - 0.5;

    const Fwd = struct {
        fn run(al: std.mem.Allocator, xv: []f32, rv: []f32, w1v: []f32, w3v: []f32, w2v: []f32, ww: []const f32, dx_out: ?[]f32) f32 {
            var g = Graph.init(al);
            defer g.deinit();
            const xv_ = g.leaf(xv, 1, D) catch unreachable;
            const rv_ = g.leaf(rv, D, E) catch unreachable;
            const w1_ = g.leaf(w1v, E * D, M) catch unreachable;
            const w3_ = g.leaf(w3v, E * D, M) catch unreachable;
            const w2_ = g.leaf(w2v, E * M, D) catch unreachable;
            const out = g.moe(xv_, rv_, w1_, w3_, w2_, E, M) catch unreachable;
            const loss = gc.weightedSumLoss(&g, out, ww) catch unreachable;
            const lv = g.val.items[loss][0];
            if (dx_out) |d| {
                g.backward(loss);
                @memcpy(d, g.gradOf(xv_));
            }
            return lv;
        }
    };
    var dx_ana = [_]f32{0} ** D;
    _ = Fwd.run(alloc, &x, &router, &w1, &w3, &w2, &wts, &dx_ana);
    const eps: f32 = 1e-3;
    for (0..D) |i| {
        const orig = x[i];
        x[i] = orig + eps;
        const lp = Fwd.run(alloc, &x, &router, &w1, &w3, &w2, &wts, null);
        x[i] = orig - eps;
        const lm = Fwd.run(alloc, &x, &router, &w1, &w3, &w2, &wts, null);
        x[i] = orig;
        const num = (lp - lm) / (2 * eps);
        try std.testing.expectApproxEqAbs(num, dx_ana[i], 5e-2);
    }
}

test "gradcheck moe (soft) — TODOS los inputs (x, router, w1, w3, w2)" {
    const alloc = std.testing.allocator;
    const D, const E, const M = .{ 4, 3, 5 };
    var x = [_]f32{0} ** D;
    var router = [_]f32{0} ** (D * E);
    var w1 = [_]f32{0} ** (E * D * M);
    var w3 = [_]f32{0} ** (E * D * M);
    var w2 = [_]f32{0} ** (E * M * D);
    var rng = std.Random.DefaultPrng.init(7);
    for (&x) |*p| p.* = rng.random().float(f32) - 0.5;
    for (&router) |*p| p.* = rng.random().float(f32) * 0.2;
    for (&w1) |*p| p.* = rng.random().float(f32) * 0.2;
    for (&w3) |*p| p.* = rng.random().float(f32) * 0.2;
    for (&w2) |*p| p.* = rng.random().float(f32) * 0.2;
    var wts = [_]f32{0} ** D;
    for (&wts) |*p| p.* = rng.random().float(f32) - 0.5;

    // which: 0=x, 1=router, 2=w1, 3=w3, 4=w2. Devuelve grad analítico del array elegido.
    const Fwd = struct {
        fn run(al: std.mem.Allocator, xv: []f32, rv: []f32, w1v: []f32, w3v: []f32, w2v: []f32, ww: []const f32, which: u8, grad_out: ?[]f32) f32 {
            var g = Graph.init(al);
            defer g.deinit();
            const xv_ = g.leaf(xv, 1, D) catch unreachable;
            const rv_ = g.leaf(rv, D, E) catch unreachable;
            const w1_ = g.leaf(w1v, E * D, M) catch unreachable;
            const w3_ = g.leaf(w3v, E * D, M) catch unreachable;
            const w2_ = g.leaf(w2v, E * M, D) catch unreachable;
            const out = g.moe(xv_, rv_, w1_, w3_, w2_, E, M) catch unreachable;
            const loss = gc.weightedSumLoss(&g, out, ww) catch unreachable;
            const lv = g.val.items[loss][0];
            if (grad_out) |d| {
                g.backward(loss);
                const src = switch (which) {
                    0 => g.gradOf(xv_),
                    1 => g.gradOf(rv_),
                    2 => g.gradOf(w1_),
                    3 => g.gradOf(w3_),
                    else => g.gradOf(w2_),
                };
                @memcpy(d, src);
            }
            return lv;
        }
    };

    const eps: f32 = 1e-3;
    const Param = struct { which: u8, arr: []f32 };
    var ana_buf = [_]f32{0} ** (E * D * M); // mayor de los tamaños
    const params = [_]Param{
        .{ .which = 0, .arr = &x },
        .{ .which = 1, .arr = &router },
        .{ .which = 2, .arr = &w1 },
        .{ .which = 3, .arr = &w3 },
        .{ .which = 4, .arr = &w2 },
    };
    for (params) |p| {
        const ana = ana_buf[0..p.arr.len];
        _ = Fwd.run(alloc, &x, &router, &w1, &w3, &w2, &wts, p.which, ana);
        for (0..p.arr.len) |i| {
            const orig = p.arr[i];
            p.arr[i] = orig + eps;
            const lp = Fwd.run(alloc, &x, &router, &w1, &w3, &w2, &wts, p.which, null);
            p.arr[i] = orig - eps;
            const lm = Fwd.run(alloc, &x, &router, &w1, &w3, &w2, &wts, p.which, null);
            p.arr[i] = orig;
            const num = (lp - lm) / (2 * eps);
            try std.testing.expectApproxEqAbs(num, ana[i], 2e-2);
        }
    }
}

test "moe load balancing loss: valor correcto y gradiente vs numérico" {
    const alloc = std.testing.allocator;
    const D, const E, const M, const T = .{ 4, 3, 5, 2 };
    var x = [_]f32{0} ** (T * D);
    var router = [_]f32{0} ** (D * E);
    var w1 = [_]f32{0} ** (E * D * M);
    var w3 = [_]f32{0} ** (E * D * M);
    var w2 = [_]f32{0} ** (E * M * D);
    var rng = std.Random.DefaultPrng.init(77);
    for (&x) |*p| p.* = rng.random().float(f32) - 0.5;
    for (&router) |*p| p.* = rng.random().float(f32) * 0.3;
    for (&w1) |*p| p.* = rng.random().float(f32) * 0.1;
    for (&w3) |*p| p.* = rng.random().float(f32) * 0.1;
    for (&w2) |*p| p.* = rng.random().float(f32) * 0.1;
    const alpha: f32 = 0.01;

    const Fwd = struct {
        fn run(al: std.mem.Allocator, xv: []f32, rv: []f32, w1v: []f32, w3v: []f32, w2v: []f32, dr_out: ?[]f32) f32 {
            var g = Graph.init(al);
            defer g.deinit();
            const xv_ = g.leaf(xv, T, D) catch unreachable;
            const rv_ = g.leaf(rv, D, E) catch unreachable;
            const w1_ = g.leaf(w1v, E * D, M) catch unreachable;
            const w3_ = g.leaf(w3v, E * D, M) catch unreachable;
            const w2_ = g.leaf(w2v, E * M, D) catch unreachable;
            const out = g.moe(xv_, rv_, w1_, w3_, w2_, E, M) catch unreachable;
            const lb = g.moeLoadBalanceLoss(out, alpha) catch unreachable;
            const lv = g.val.items[lb][0];
            if (dr_out) |d| {
                g.backward(lb);
                @memcpy(d, g.gradOf(rv_));
            }
            return lv;
        }
    };

    // 1. El valor de loss es no-negativo y finito
    const lv = Fwd.run(alloc, &x, &router, &w1, &w3, &w2, null);
    try std.testing.expect(lv >= 0);
    try std.testing.expect(!std.math.isNan(lv));

    // 2. Gradiente analítico del router
    var dr_ana = [_]f32{0} ** (D * E);
    _ = Fwd.run(alloc, &x, &router, &w1, &w3, &w2, &dr_ana);

    // 3. Gradiente numérico del router (diferencias finitas)
    const eps: f32 = 1e-3;
    for (0..D * E) |i| {
        const orig = router[i];
        router[i] = orig + eps;
        const lp = Fwd.run(alloc, &x, &router, &w1, &w3, &w2, null);
        router[i] = orig - eps;
        const lm = Fwd.run(alloc, &x, &router, &w1, &w3, &w2, null);
        router[i] = orig;
        const num = (lp - lm) / (2 * eps);
        try std.testing.expectApproxEqAbs(num, dr_ana[i], 5e-2);
    }
}

test "moe load balancing loss: distribución uniforme tiene loss menor que concentrada" {
    const alloc = std.testing.allocator;
    const D, const E, const M, const T = .{ 4, 4, 8, 10 };
    var x = [_]f32{0} ** (T * D);
    var rng = std.Random.DefaultPrng.init(55);
    for (&x) |*p| p.* = rng.random().float(f32) - 0.5;

    // Router que produce distribución ~uniforme (logits cercanos a 0)
    var router_uniform = [_]f32{0} ** (D * E);
    for (&router_uniform) |*p| p.* = rng.random().float(f32) * 0.001;

    // Router que produce distribución concentrada (logits grandes para expert 0)
    var router_concentrated = [_]f32{0} ** (D * E);
    for (&router_concentrated) |*p| p.* = rng.random().float(f32) * 0.001;
    for (0..D) |d| router_concentrated[d * E + 0] = 5.0; // expert 0 domina

    var w1 = [_]f32{0} ** (E * D * M);
    var w3 = [_]f32{0} ** (E * D * M);
    var w2 = [_]f32{0} ** (E * M * D);
    for (&w1) |*p| p.* = 0.1;
    for (&w3) |*p| p.* = 0.1;
    for (&w2) |*p| p.* = 0.1;

    const alpha: f32 = 1.0; // sin escalar para comparar

    var g1 = Graph.init(alloc);
    defer g1.deinit();
    const x1 = try g1.leaf(&x, T, D);
    const r1 = try g1.leaf(&router_uniform, D, E);
    const w1_1 = try g1.leaf(&w1, E * D, M);
    const w3_1 = try g1.leaf(&w3, E * D, M);
    const w2_1 = try g1.leaf(&w2, E * M, D);
    const out1 = try g1.moe(x1, r1, w1_1, w3_1, w2_1, E, M);
    const lb1 = try g1.moeLoadBalanceLoss(out1, alpha);
    const loss_uniform = g1.val.items[lb1][0];

    var g2 = Graph.init(alloc);
    defer g2.deinit();
    const x2 = try g2.leaf(&x, T, D);
    const r2 = try g2.leaf(&router_concentrated, D, E);
    const w1_2 = try g2.leaf(&w1, E * D, M);
    const w3_2 = try g2.leaf(&w3, E * D, M);
    const w2_2 = try g2.leaf(&w2, E * M, D);
    const out2 = try g2.moe(x2, r2, w1_2, w3_2, w2_2, E, M);
    const lb2 = try g2.moeLoadBalanceLoss(out2, alpha);
    const loss_concentrated = g2.val.items[lb2][0];

    // Distribución uniforme: P_e ≈ 1/E → L = E * sum (1/E)^2 = E * E * (1/E^2) = 1
    // Distribución concentrada: P_0 ≈ 1, resto ≈ 0 → L = E * 1 = E
    try std.testing.expect(loss_concentrated > loss_uniform);
}

test "gradcheck matmul individual dA y dB" {
    const alloc = std.testing.allocator;
    const M, const K, const N = .{ 2, 3, 4 };
    var A = [_]f32{0} ** (M * K);
    var B = [_]f32{0} ** (K * N);
    var rng = std.Random.DefaultPrng.init(41);
    for (&A) |*p| p.* = rng.random().float(f32) - 0.5;
    for (&B) |*p| p.* = rng.random().float(f32) - 0.5;
    var wts = [_]f32{0} ** (M * N);
    for (&wts) |*p| p.* = rng.random().float(f32) - 0.5;

    const Fwd = struct {
        fn run(al: std.mem.Allocator, a: []f32, b: []f32, ww: []const f32, which: u8, grad_out: ?[]f32) f32 {
            var g = Graph.init(al);
            defer g.deinit();
            const av = g.leaf(a, M, K) catch unreachable;
            const bv = g.leaf(b, K, N) catch unreachable;
            const mm = g.matmul(av, bv) catch unreachable;
            const loss = gc.weightedSumLoss(&g, mm, ww) catch unreachable;
            const lv = g.val.items[loss][0];
            if (grad_out) |d| {
                g.backward(loss);
                const src = if (which == 0) g.gradOf(av) else g.gradOf(bv);
                @memcpy(d, src);
            }
            return lv;
        }
    };
    const eps: f32 = 1e-3;
    var ana_buf = [_]f32{0} ** (K * N); // mayor de los tamaños
    var params = [_]struct { which: u8, arr: []f32, len: usize }{
        .{ .which = 0, .arr = &A, .len = M * K },
        .{ .which = 1, .arr = &B, .len = K * N },
    };
    for (&params) |p| {
        const ana = ana_buf[0..p.len];
        _ = Fwd.run(alloc, &A, &B, &wts, p.which, ana);
        for (0..p.len) |i| {
            const o = p.arr[i];
            p.arr[i] = o + eps;
            const lp = Fwd.run(alloc, &A, &B, &wts, p.which, null);
            p.arr[i] = o - eps;
            const lm = Fwd.run(alloc, &A, &B, &wts, p.which, null);
            p.arr[i] = o;
            try std.testing.expectApproxEqAbs((lp - lm) / (2 * eps), ana[i], 2e-2);
        }
    }
}

test "gradcheck addResidual individual" {
    const alloc = std.testing.allocator;
    const M, const N = .{ 2, 3 };
    var A = [_]f32{0} ** (M * N);
    var B = [_]f32{0} ** (M * N);
    var rng = std.Random.DefaultPrng.init(42);
    for (&A) |*p| p.* = rng.random().float(f32) - 0.5;
    for (&B) |*p| p.* = rng.random().float(f32) - 0.5;
    var wts = [_]f32{0} ** (M * N);
    for (&wts) |*p| p.* = rng.random().float(f32) - 0.5;

    const Fwd = struct {
        fn run(al: std.mem.Allocator, a: []f32, b: []f32, ww: []const f32, which: u8, grad_out: ?[]f32) f32 {
            var g = Graph.init(al);
            defer g.deinit();
            const av = g.leaf(a, M, N) catch unreachable;
            const bv = g.leaf(b, M, N) catch unreachable;
            const mm = g.addResidual(av, bv) catch unreachable;
            const loss = gc.weightedSumLoss(&g, mm, ww) catch unreachable;
            const lv = g.val.items[loss][0];
            if (grad_out) |d| {
                g.backward(loss);
                const src = if (which == 0) g.gradOf(av) else g.gradOf(bv);
                @memcpy(d, src);
            }
            return lv;
        }
    };
    const eps: f32 = 1e-3;
    var ana = [_]f32{0} ** (M * N);
    var params = [_]struct { which: u8, arr: []f32 }{
        .{ .which = 0, .arr = &A },
        .{ .which = 1, .arr = &B },
    };
    for (&params) |p| {
        _ = Fwd.run(alloc, &A, &B, &wts, p.which, &ana);
        for (0..M * N) |i| {
            const o = p.arr[i];
            p.arr[i] = o + eps;
            const lp = Fwd.run(alloc, &A, &B, &wts, p.which, null);
            p.arr[i] = o - eps;
            const lm = Fwd.run(alloc, &A, &B, &wts, p.which, null);
            p.arr[i] = o;
            try std.testing.expectApproxEqAbs((lp - lm) / (2 * eps), ana[i], 2e-2);
        }
    }
}

test "gradcheck rmsnorm individual" {
    const alloc = std.testing.allocator;
    const M, const N = .{ 2, 3 };
    var A = [_]f32{0} ** (M * N);
    var W = [_]f32{0} ** N;
    var rng = std.Random.DefaultPrng.init(43);
    for (&A) |*p| p.* = rng.random().float(f32) - 0.5;
    for (&W) |*p| p.* = rng.random().float(f32) + 0.5;
    var wts = [_]f32{0} ** (M * N);
    for (&wts) |*p| p.* = rng.random().float(f32) - 0.5;

    const Fwd = struct {
        fn run(al: std.mem.Allocator, a: []f32, w: []f32, ww: []const f32, which: u8, grad_out: ?[]f32) f32 {
            var g = Graph.init(al);
            defer g.deinit();
            const av = g.leaf(a, M, N) catch unreachable;
            const wv = g.leaf(w, 1, N) catch unreachable;
            const mm = g.rmsnorm(av, wv, 1e-5) catch unreachable;
            const loss = gc.weightedSumLoss(&g, mm, ww) catch unreachable;
            const lv = g.val.items[loss][0];
            if (grad_out) |d| {
                g.backward(loss);
                const src = if (which == 0) g.gradOf(av) else g.gradOf(wv);
                @memcpy(d, src);
            }
            return lv;
        }
    };
    const eps: f32 = 1e-3;
    var ana_buf = [_]f32{0} ** (M * N);
    var params = [_]struct { which: u8, arr: []f32, len: usize }{
        .{ .which = 0, .arr = &A, .len = M * N },
        .{ .which = 1, .arr = &W, .len = N },
    };
    for (&params) |p| {
        const ana = ana_buf[0..p.len];
        _ = Fwd.run(alloc, &A, &W, &wts, p.which, ana);
        for (0..p.len) |i| {
            const o = p.arr[i];
            p.arr[i] = o + eps;
            const lp = Fwd.run(alloc, &A, &W, &wts, p.which, null);
            p.arr[i] = o - eps;
            const lm = Fwd.run(alloc, &A, &W, &wts, p.which, null);
            p.arr[i] = o;
            try std.testing.expectApproxEqAbs((lp - lm) / (2 * eps), ana[i], 2e-2);
        }
    }
}

test "gradcheck silu individual" {
    const alloc = std.testing.allocator;
    const M, const N = .{ 2, 3 };
    var A = [_]f32{0} ** (M * N);
    var rng = std.Random.DefaultPrng.init(44);
    for (&A) |*p| p.* = rng.random().float(f32) - 0.5;
    var wts = [_]f32{0} ** (M * N);
    for (&wts) |*p| p.* = rng.random().float(f32) - 0.5;

    const Fwd = struct {
        fn run(al: std.mem.Allocator, a: []f32, ww: []const f32, dA: ?[]f32) f32 {
            var g = Graph.init(al);
            defer g.deinit();
            const av = g.leaf(a, M, N) catch unreachable;
            const mm = g.silu(av) catch unreachable;
            const loss = gc.weightedSumLoss(&g, mm, ww) catch unreachable;
            const lv = g.val.items[loss][0];
            if (dA) |d| {
                g.backward(loss);
                @memcpy(d, g.gradOf(av));
            }
            return lv;
        }
    };
    var dA_ana = [_]f32{0} ** (M * N);
    _ = Fwd.run(alloc, &A, &wts, &dA_ana);
    const eps: f32 = 1e-3;
    for (0..M * N) |i| {
        const o = A[i];
        A[i] = o + eps;
        const lp = Fwd.run(alloc, &A, &wts, null);
        A[i] = o - eps;
        const lm = Fwd.run(alloc, &A, &wts, null);
        A[i] = o;
        try std.testing.expectApproxEqAbs((lp - lm) / (2 * eps), dA_ana[i], 2e-2);
    }
}

test "gradcheck mul individual" {
    const alloc = std.testing.allocator;
    const M, const N = .{ 2, 3 };
    var A = [_]f32{0} ** (M * N);
    var B = [_]f32{0} ** (M * N);
    var rng = std.Random.DefaultPrng.init(45);
    for (&A) |*p| p.* = rng.random().float(f32) - 0.5;
    for (&B) |*p| p.* = rng.random().float(f32) - 0.5;
    var wts = [_]f32{0} ** (M * N);
    for (&wts) |*p| p.* = rng.random().float(f32) - 0.5;

    const Fwd = struct {
        fn run(al: std.mem.Allocator, a: []f32, b: []f32, ww: []const f32, which: u8, grad_out: ?[]f32) f32 {
            var g = Graph.init(al);
            defer g.deinit();
            const av = g.leaf(a, M, N) catch unreachable;
            const bv = g.leaf(b, M, N) catch unreachable;
            const mm = g.mul(av, bv) catch unreachable;
            const loss = gc.weightedSumLoss(&g, mm, ww) catch unreachable;
            const lv = g.val.items[loss][0];
            if (grad_out) |d| {
                g.backward(loss);
                const src = if (which == 0) g.gradOf(av) else g.gradOf(bv);
                @memcpy(d, src);
            }
            return lv;
        }
    };
    const eps: f32 = 1e-3;
    var ana = [_]f32{0} ** (M * N);
    var params = [_]struct { which: u8, arr: []f32 }{
        .{ .which = 0, .arr = &A },
        .{ .which = 1, .arr = &B },
    };
    for (&params) |p| {
        _ = Fwd.run(alloc, &A, &B, &wts, p.which, &ana);
        for (0..M * N) |i| {
            const o = p.arr[i];
            p.arr[i] = o + eps;
            const lp = Fwd.run(alloc, &A, &B, &wts, p.which, null);
            p.arr[i] = o - eps;
            const lm = Fwd.run(alloc, &A, &B, &wts, p.which, null);
            p.arr[i] = o;
            try std.testing.expectApproxEqAbs((lp - lm) / (2 * eps), ana[i], 2e-2);
        }
    }
}

test "gradcheck crossEntropyLoss vs numérico" {
    const alloc = std.testing.allocator;
    const M, const V = .{ 2, 5 };
    var logits = [_]f32{0} ** (M * V);
    var rng = std.Random.DefaultPrng.init(46);
    for (&logits) |*p| p.* = rng.random().float(f32) - 0.5;
    const targets = [_]u32{ 2, 4 };

    const Fwd = struct {
        fn run(al: std.mem.Allocator, l: []f32, dL: ?[]f32) f32 {
            var g = Graph.init(al);
            defer g.deinit();
            const lv = g.leaf(l, M, V) catch unreachable;
            const loss = crossEntropyLoss(&g, lv, &targets) catch unreachable;
            const val = g.val.items[loss][0];
            if (dL) |d| {
                g.backward(loss);
                @memcpy(d, g.gradOf(lv));
            }
            return val;
        }
    };
    var dL_ana = [_]f32{0} ** (M * V);
    _ = Fwd.run(alloc, &logits, &dL_ana);
    const eps: f32 = 1e-3;
    for (0..M * V) |i| {
        const o = logits[i];
        logits[i] = o + eps;
        const lp = Fwd.run(alloc, &logits, null);
        logits[i] = o - eps;
        const lm = Fwd.run(alloc, &logits, null);
        logits[i] = o;
        try std.testing.expectApproxEqAbs((lp - lm) / (2 * eps), dL_ana[i], 2e-2);
    }
}

test "gradcheck kdDistillLoss vs numérico" {
    const alloc = std.testing.allocator;
    const M, const V = .{ 2, 5 };
    var student = [_]f32{0} ** (M * V);
    var rng = std.Random.DefaultPrng.init(47);
    const teacher_logits: [M * V]f32 = blk: {
        var tmp: [M * V]f32 = undefined;
        for (&tmp) |*p| p.* = rng.random().float(f32) - 0.5;
        break :blk tmp;
    };
    for (&student) |*p| p.* = rng.random().float(f32) - 0.5;
    const tau: f32 = 2.0;

    const Fwd = struct {
        fn run(al: std.mem.Allocator, s: []f32, tl: []const f32, dS: ?[]f32) f32 {
            var g = Graph.init(al);
            defer g.deinit();
            const sv = g.leaf(s, M, V) catch unreachable;
            const loss = kdDistillLoss(&g, sv, tl, tau) catch unreachable;
            const val = g.val.items[loss][0];
            if (dS) |d| {
                g.backward(loss);
                @memcpy(d, g.gradOf(sv));
            }
            return val;
        }
    };
    var dS_ana = [_]f32{0} ** (M * V);
    _ = Fwd.run(alloc, &student, &teacher_logits, &dS_ana);
    const eps: f32 = 1e-3;
    for (0..M * V) |i| {
        const o = student[i];
        student[i] = o + eps;
        const lp = Fwd.run(alloc, &student, &teacher_logits, null);
        student[i] = o - eps;
        const lm = Fwd.run(alloc, &student, &teacher_logits, null);
        student[i] = o;
        try std.testing.expectApproxEqAbs((lp - lm) / (2 * eps), dS_ana[i], 2e-2);
    }
}

// --- Micro-tests adversariales de estabilidad numérica ---

test "rmsnorm con input cero produce output finito" {
    const alloc = std.testing.allocator;
    var A = [_]f32{0.0, 0.0, 0.0};
    var W = [_]f32{1.0, 1.0, 1.0};
    var g = Graph.init(alloc);
    defer g.deinit();
    const av = try g.leaf(&A, 1, 3);
    const wv = try g.leaf(&W, 1, 3);
    const out = try g.rmsnorm(av, wv, 1e-5);
    const O = g.valOf(out);
    for (O) |v| try std.testing.expect(std.math.isFinite(v));
}

test "silu estable: x=+100 y x=-100 produce finito" {
    const alloc = std.testing.allocator;
    var A = [_]f32{ 100.0, -100.0, 0.0 };
    var g = Graph.init(alloc);
    defer g.deinit();
    const av = try g.leaf(&A, 1, 3);
    const out = try g.silu(av);
    const O = g.valOf(out);
    for (O) |v| try std.testing.expect(std.math.isFinite(v));
    // backward también debe ser finito
    const loss = try gc.weightedSumLoss(&g, out, &[_]f32{ 1.0, 1.0, 1.0 });
    g.backward(loss);
    for (g.gradOf(av)) |v| try std.testing.expect(std.math.isFinite(v));
}

test "crossEntropyLoss con logits adversariales (vocab grande) es finita" {
    const alloc = std.testing.allocator;
    const V: usize = 128;
    var logits = [_]f32{0} ** V;
    // Un logit muy grande, el resto normal
    logits[0] = 100.0;
    for (1..V) |i| logits[i] = @as(f32, @floatFromInt(i)) * 0.01;
    var g = Graph.init(alloc);
    defer g.deinit();
    const lv = try g.leaf(&logits, 1, V);
    const loss = try crossEntropyLoss(&g, lv, &[_]u32{0});
    const val = g.val.items[loss][0];
    try std.testing.expect(std.math.isFinite(val));
}

test "kdDistillLoss con logits adversariales (vocab grande) es finita" {
    const alloc = std.testing.allocator;
    const V: usize = 128;
    var student = [_]f32{0} ** V;
    var teacher = [_]f32{0} ** V;
    student[0] = 100.0;
    teacher[V - 1] = 100.0;
    for (1..V) |i| student[i] = @as(f32, @floatFromInt(i)) * 0.01;
    for (0..V - 1) |i| teacher[i] = @as(f32, @floatFromInt(i)) * 0.01;
    var g = Graph.init(alloc);
    defer g.deinit();
    const sv = try g.leaf(&student, 1, V);
    const loss = try kdDistillLoss(&g, sv, &teacher, 2.0);
    const val = g.val.items[loss][0];
    try std.testing.expect(std.math.isFinite(val));
}

test "attn softmax con scores adversariales no produce Inf" {
    const alloc = std.testing.allocator;
    const T, const HD, const H = .{ 3, 4, 2 };
    var Q = [_]f32{0} ** (T * HD);
    var K = [_]f32{0} ** (T * HD);
    var Vv = [_]f32{0} ** (T * HD);
    // Q muy grande para forzar exp overflow sin max-subtract (pero max-subtract ya está)
    Q[0] = 80.0;
    K[0] = 80.0;
    for (&Vv) |*p| p.* = 0.1;
    var g = Graph.init(alloc);
    defer g.deinit();
    const qv = try g.leaf(&Q, T, HD);
    const kv = try g.leaf(&K, T, HD);
    const vv = try g.leaf(&Vv, T, HD);
    const out = try g.attn(qv, kv, vv, H, 0);
    const O = g.valOf(out);
    for (O) |v| try std.testing.expect(std.math.isFinite(v));
}

test "gradcheck mseLoss vs numérico" {
    const alloc = std.testing.allocator;
    const M, const N = .{ 2, 3 };
    var hidden = [_]f32{0} ** (M * N);
    var rng = std.Random.DefaultPrng.init(48);
    const target: [M * N]f32 = blk: {
        var tmp: [M * N]f32 = undefined;
        for (&tmp) |*p| p.* = rng.random().float(f32) - 0.5;
        break :blk tmp;
    };
    for (&hidden) |*p| p.* = rng.random().float(f32) - 0.5;

    const Fwd = struct {
        fn run(al: std.mem.Allocator, h: []f32, tgt: []const f32, dH: ?[]f32) f32 {
            var g = Graph.init(al);
            defer g.deinit();
            const hv = g.leaf(h, M, N) catch unreachable;
            const loss = mseLoss(&g, hv, tgt) catch unreachable;
            const val = g.val.items[loss][0];
            if (dH) |d| {
                g.backward(loss);
                @memcpy(d, g.gradOf(hv));
            }
            return val;
        }
    };
    var dH_ana = [_]f32{0} ** (M * N);
    _ = Fwd.run(alloc, &hidden, &target, &dH_ana);
    const eps: f32 = 1e-3;
    for (0..M * N) |i| {
        const o = hidden[i];
        hidden[i] = o + eps;
        const lp = Fwd.run(alloc, &hidden, &target, null);
        hidden[i] = o - eps;
        const lm = Fwd.run(alloc, &hidden, &target, null);
        hidden[i] = o;
        try std.testing.expectApproxEqAbs((lp - lm) / (2 * eps), dH_ana[i], 2e-2);
    }
}
