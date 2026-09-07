// zkernel — kernels SIMD (@Vector + comptime)
const std = @import("std");

// SIMD vector width (floats)
pub const VecWidth = 8;
pub const Vec = @Vector(VecWidth, f32);

// --- matmul: C = A @ B, A[M,K], B[K,N], C[M,N] ---
pub fn matmulRef(C: []f32, A: []const f32, B: []const f32, M: usize, K: usize, N: usize) void {
    @memset(C, 0);
    for (0..M) |m| {
        for (0..K) |k| {
            const a = A[m * K + k];
            for (0..N) |n| {
                C[m * N + n] += a * B[k * N + n];
            }
        }
    }
}

pub fn matmul(C: []f32, A: []const f32, B: []const f32, M: usize, K: usize, N: usize) void {
    // Tiled + SIMD
    const tile_m = 32;
    const tile_n = 32;
    const tile_k = 32;
    @memset(C, 0);
    var mt: usize = 0;
    while (mt < M) : (mt += tile_m) {
        var nt: usize = 0;
        while (nt < N) : (nt += tile_n) {
            var kt: usize = 0;
            while (kt < K) : (kt += tile_k) {
                const m_end = @min(mt + tile_m, M);
                const n_end = @min(nt + tile_n, N);
                const k_end = @min(kt + tile_k, K);
                for (mt..m_end) |m| {
                    var n = nt;
                    while (n + VecWidth <= n_end) : (n += VecWidth) {
                        var cv: Vec = @as(*align(1) const Vec, @ptrCast(&C[m * N + n])).*;
                        for (kt..k_end) |k| {
                            const a = A[m * K + k];
                            const bv: Vec = @as(*align(1) const Vec, @ptrCast(&B[k * N + n])).*;
                            cv = cv + @as(Vec, @splat(a)) * bv;
                        }
                        @as(*align(1) Vec, @ptrCast(&C[m * N + n])).* = cv;
                    }
                    for (n..n_end) |n_scalar| {
                        var acc = C[m * N + n_scalar];
                        for (kt..k_end) |k| acc += A[m * K + k] * B[k * N + n_scalar];
                        C[m * N + n_scalar] = acc;
                    }
                }
            }
        }
    }
}

// --- rmsnorm: x_i / sqrt(mean(x^2) + eps) * weight_i ---
pub fn rmsnormRef(out: []f32, inp: []const f32, weight: []const f32, eps: f32) void {
    std.debug.assert(out.len == inp.len);
    std.debug.assert(out.len == weight.len);
    var sum: f32 = 0;
    for (inp) |v| sum += v * v;
    const mean = sum / @as(f32, @floatFromInt(inp.len));
    const scale = 1.0 / @sqrt(mean + eps);
    for (out, 0..) |*o, i| o.* = inp[i] * scale * weight[i];
}

pub fn rmsnorm(out: []f32, inp: []const f32, weight: []const f32, eps: f32) void {
    std.debug.assert(out.len == inp.len);
    std.debug.assert(out.len == weight.len);
    const len = inp.len;
    var sum: f32 = 0;
    var i: usize = 0;
    while (i + VecWidth <= len) : (i += VecWidth) {
        const v: Vec = @as(*align(1) const Vec, @ptrCast(&inp[i])).*;
        sum += @reduce(.Add, v * v);
    }
    for (i..len) |j| sum += inp[j] * inp[j];
    const mean = sum / @as(f32, @floatFromInt(len));
    const scale = 1.0 / @sqrt(mean + eps);
    const s: Vec = @splat(scale);
    i = 0;
    while (i + VecWidth <= len) : (i += VecWidth) {
        const iv: Vec = @as(*align(1) const Vec, @ptrCast(&inp[i])).*;
        const wv: Vec = @as(*align(1) const Vec, @ptrCast(&weight[i])).*;
        const ov = iv * s * wv;
        @as(*align(1) Vec, @ptrCast(&out[i])).* = ov;
    }
    for (i..len) |j| out[j] = inp[j] * scale * weight[j];
}

// --- softmax: x_i = exp(x_i - max) / sum(exp(x - max)) ---
pub fn softmaxRef(out: []f32, inp: []const f32) void {
    std.debug.assert(out.len == inp.len);
    var maxv = inp[0];
    for (inp) |v| {
        if (v > maxv) maxv = v;
    }
    var sum: f32 = 0;
    for (inp, 0..) |v, i| {
        const e = @exp(v - maxv);
        out[i] = e;
        sum += e;
    }
    const inv = 1.0 / sum;
    for (out) |*o| o.* *= inv;
}

pub fn softmax(out: []f32, inp: []const f32) void {
    std.debug.assert(out.len == inp.len);
    var maxv = inp[0];
    for (inp) |v| {
        if (v > maxv) maxv = v;
    }
    const len = inp.len;
    var sum: f32 = 0;
    for (0..len) |i| {
        const e = @exp(inp[i] - maxv);
        out[i] = e;
        sum += e;
    }
    const inv = 1.0 / sum;
    for (out) |*o| o.* *= inv;
}

// --- silu: x * sigmoid(x) = x / (1 + exp(-x)) ---
pub fn siluRef(out: []f32, inp: []const f32) void {
    std.debug.assert(out.len == inp.len);
    for (out, 0..) |*o, i| {
        const x = inp[i];
        o.* = x / (1.0 + @exp(-x));
    }
}

pub fn silu(out: []f32, inp: []const f32) void {
    std.debug.assert(out.len == inp.len);
    const len = inp.len;
    var i: usize = 0;
    while (i + VecWidth <= len) : (i += VecWidth) {
        const xv: Vec = @as(*align(1) const Vec, @ptrCast(&inp[i])).*;
        const ov = xv / (@as(Vec, @splat(1.0)) + @exp(-xv));
        @as(*align(1) Vec, @ptrCast(&out[i])).* = ov;
    }
    for (i..len) |j| {
        const x = inp[j];
        out[j] = x / (1.0 + @exp(-x));
    }
}

// --- rope: aplica Rotary Position Embedding a pares de canales ---
pub fn ropeRef(q: []f32, k: []f32, pos: usize, head_dim: usize, theta: f32) void {
    std.debug.assert(q.len == head_dim);
    std.debug.assert(k.len == head_dim);
    for (0..head_dim / 2) |i| {
        const inv_freq = 1.0 / std.math.pow(f32, theta, 2.0 * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(head_dim)));
        const angle = @as(f32, @floatFromInt(pos)) * inv_freq;
        const cos_a = @cos(angle);
        const sin_a = @sin(angle);
        const q0 = q[2 * i];
        const q1 = q[2 * i + 1];
        q[2 * i] = q0 * cos_a - q1 * sin_a;
        q[2 * i + 1] = q0 * sin_a + q1 * cos_a;
        const k0 = k[2 * i];
        const k1 = k[2 * i + 1];
        k[2 * i] = k0 * cos_a - k1 * sin_a;
        k[2 * i + 1] = k0 * sin_a + k1 * cos_a;
    }
}

pub fn rope(q: []f32, k: []f32, pos: usize, head_dim: usize, theta: f32) void {
    ropeRef(q, k, pos, head_dim, theta);
}

// --- rope NEOX: rota las MITADES (i, i+head_dim/2), no pares adyacentes ---
// Es la convención usada por GPT-NeoX y por los GGUF de Qwen2/Llama (rope_type NEOX
// en llama.cpp/ggml). Difiere de ropeRef (estilo GPT-J, pares 2i/2i+1). En pos=0
// ambas son la identidad; la diferencia sólo aparece en pos>0 (generación multi-token).
pub fn ropeNeoxRef(q: []f32, k: []f32, pos: usize, head_dim: usize, theta: f32) void {
    std.debug.assert(q.len == head_dim);
    std.debug.assert(k.len == head_dim);
    const half = head_dim / 2;
    for (0..half) |i| {
        const inv_freq = 1.0 / std.math.pow(f32, theta, 2.0 * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(head_dim)));
        const angle = @as(f32, @floatFromInt(pos)) * inv_freq;
        const cos_a = @cos(angle);
        const sin_a = @sin(angle);
        const q0 = q[i];
        const q1 = q[i + half];
        q[i] = q0 * cos_a - q1 * sin_a;
        q[i + half] = q0 * sin_a + q1 * cos_a;
        const k0 = k[i];
        const k1 = k[i + half];
        k[i] = k0 * cos_a - k1 * sin_a;
        k[i + half] = k0 * sin_a + k1 * cos_a;
    }
}

pub fn ropeNeox(q: []f32, k: []f32, pos: usize, head_dim: usize, theta: f32) void {
    ropeNeoxRef(q, k, pos, head_dim, theta);
}

// Variantes de UN SOLO vector. Necesarias con GQA: la K se comparte entre varias
// cabezas Q, así que hay que rotar cada cabeza Q y cada cabeza KV exactamente una vez
// (rotar el par (q,k) por cada cabeza Q rotaría la K kv_mul veces — bug).
pub fn ropeVecNeox(x: []f32, pos: usize, head_dim: usize, theta: f32) void {
    const half = head_dim / 2;
    for (0..half) |i| {
        const inv_freq = 1.0 / std.math.pow(f32, theta, 2.0 * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(head_dim)));
        const angle = @as(f32, @floatFromInt(pos)) * inv_freq;
        const cos_a = @cos(angle);
        const sin_a = @sin(angle);
        const x0 = x[i];
        const x1 = x[i + half];
        x[i] = x0 * cos_a - x1 * sin_a;
        x[i + half] = x0 * sin_a + x1 * cos_a;
    }
}

pub fn ropeVecGptj(x: []f32, pos: usize, head_dim: usize, theta: f32) void {
    for (0..head_dim / 2) |i| {
        const inv_freq = 1.0 / std.math.pow(f32, theta, 2.0 * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(head_dim)));
        const angle = @as(f32, @floatFromInt(pos)) * inv_freq;
        const cos_a = @cos(angle);
        const sin_a = @sin(angle);
        const x0 = x[2 * i];
        const x1 = x[2 * i + 1];
        x[2 * i] = x0 * cos_a - x1 * sin_a;
        x[2 * i + 1] = x0 * sin_a + x1 * cos_a;
    }
}

// --- tests ---
test "matmul vs ref" {
    const alloc = std.testing.allocator;
    const M, const K, const N = .{ 16, 24, 32 };
    const A = try alloc.alloc(f32, M * K);
    const B = try alloc.alloc(f32, K * N);
    const C_ref = try alloc.alloc(f32, M * N);
    const C_fast = try alloc.alloc(f32, M * N);
    defer { alloc.free(A); alloc.free(B); alloc.free(C_ref); alloc.free(C_fast); }
    var rng = std.Random.DefaultPrng.init(42);
    for (A) |*p| p.* = rng.random().float(f32) - 0.5;
    for (B) |*p| p.* = rng.random().float(f32) - 0.5;
    matmulRef(C_ref, A, B, M, K, N);
    matmul(C_fast, A, B, M, K, N);
    for (C_ref, C_fast) |r, f| try std.testing.expect(@abs(r - f) < 1e-4);
}

test "rmsnorm vs ref" {
    const alloc = std.testing.allocator;
    const len = 64;
    const inp = try alloc.alloc(f32, len);
    const weight = try alloc.alloc(f32, len);
    const out_ref = try alloc.alloc(f32, len);
    const out_fast = try alloc.alloc(f32, len);
    defer { alloc.free(inp); alloc.free(weight); alloc.free(out_ref); alloc.free(out_fast); }
    var rng = std.Random.DefaultPrng.init(42);
    for (inp) |*p| p.* = rng.random().float(f32) - 0.5;
    for (weight) |*p| p.* = rng.random().float(f32) - 0.5;
    rmsnormRef(out_ref, inp, weight, 1e-5);
    rmsnorm(out_fast, inp, weight, 1e-5);
    for (out_ref, out_fast) |r, f| try std.testing.expect(@abs(r - f) < 1e-5);
}

test "softmax vs ref" {
    const alloc = std.testing.allocator;
    const len = 128;
    const inp = try alloc.alloc(f32, len);
    const out_ref = try alloc.alloc(f32, len);
    const out_fast = try alloc.alloc(f32, len);
    defer { alloc.free(inp); alloc.free(out_ref); alloc.free(out_fast); }
    var rng = std.Random.DefaultPrng.init(42);
    for (inp) |*p| p.* = rng.random().float(f32);
    softmaxRef(out_ref, inp);
    softmax(out_fast, inp);
    for (out_ref, out_fast) |r, f| try std.testing.expect(@abs(r - f) < 1e-5);
}

test "silu vs ref" {
    const alloc = std.testing.allocator;
    const len = 128;
    const inp = try alloc.alloc(f32, len);
    const out_ref = try alloc.alloc(f32, len);
    const out_fast = try alloc.alloc(f32, len);
    defer { alloc.free(inp); alloc.free(out_ref); alloc.free(out_fast); }
    var rng = std.Random.DefaultPrng.init(42);
    for (inp) |*p| p.* = rng.random().float(f32);
    siluRef(out_ref, inp);
    silu(out_fast, inp);
    for (out_ref, out_fast) |r, f| try std.testing.expect(@abs(r - f) < 1e-5);
}

test "rope vs ref" {
    const alloc = std.testing.allocator;
    const head_dim = 64;
    const q_ref = try alloc.alloc(f32, head_dim);
    const k_ref = try alloc.alloc(f32, head_dim);
    const q_fast = try alloc.alloc(f32, head_dim);
    const k_fast = try alloc.alloc(f32, head_dim);
    defer { alloc.free(q_ref); alloc.free(k_ref); alloc.free(q_fast); alloc.free(k_fast); }
    var rng = std.Random.DefaultPrng.init(42);
    for (q_ref, q_fast) |*r, *f| { const v = rng.random().float(f32); r.* = v; f.* = v; }
    for (k_ref, k_fast) |*r, *f| { const v = rng.random().float(f32); r.* = v; f.* = v; }
    ropeRef(q_ref, k_ref, 7, head_dim, 10000.0);
    rope(q_fast, k_fast, 7, head_dim, 10000.0);
    for (q_ref, q_fast) |r, f| try std.testing.expect(@abs(r - f) < 1e-5);
    for (k_ref, k_fast) |r, f| try std.testing.expect(@abs(r - f) < 1e-5);
}

test "rope NEOX: identidad en pos=0, rotación correcta en pos>0" {
    // 1) En pos=0 la rotación es identidad (cos=1, sin=0).
    var q0 = [_]f32{ 1, 2, 3, 4 };
    var k0 = [_]f32{ 5, 6, 7, 8 };
    const q0_orig = q0;
    const k0_orig = k0;
    ropeNeox(&q0, &k0, 0, 4, 10000.0);
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(q0_orig[i], q0[i], 1e-6);
        try std.testing.expectApproxEqAbs(k0_orig[i], k0[i], 1e-6);
    }

    // 2) head_dim=4, half=2 -> rota (x0,x2) e (x1,x3). En pos=1, i=0: freq=1, angle=1.
    //    i=1: freq=theta^(-2/4)=1/sqrt(theta), angle pequeño.
    var q = [_]f32{ 1, 0, 0, 1 };
    var k = [_]f32{ 0, 0, 0, 0 };
    ropeNeox(&q, &k, 1, 4, 10000.0);
    const c0 = @cos(@as(f32, 1.0));
    const s0 = @sin(@as(f32, 1.0));
    // q[0]=1, q[2]=0 -> q[0]=cos, q[2]=sin
    try std.testing.expectApproxEqAbs(c0, q[0], 1e-5);
    try std.testing.expectApproxEqAbs(s0, q[2], 1e-5);
    const inv1 = 1.0 / std.math.pow(f32, 10000.0, 2.0 / 4.0);
    const c1 = @cos(inv1);
    const s1 = @sin(inv1);
    // q[1]=0, q[3]=1 -> q[1]=-sin, q[3]=cos
    try std.testing.expectApproxEqAbs(-s1, q[1], 1e-5);
    try std.testing.expectApproxEqAbs(c1, q[3], 1e-5);

    // 3) NEOX y GPT-J difieren en pos>0 (sanity: no son la misma transformación).
    var qa = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var ka = [_]f32{ 1, 1, 1, 1, 1, 1, 1, 1 };
    var qb = qa;
    var kb = ka;
    ropeNeox(&qa, &ka, 5, 8, 10000.0);
    ropeRef(&qb, &kb, 5, 8, 10000.0);
    var differs = false;
    for (0..8) |i| if (@abs(qa[i] - qb[i]) > 1e-4) {
        differs = true;
    };
    try std.testing.expect(differs);
}

// --- scan_chunkwise: scan paralelo para SSM (Mamba-2) ---
/// Realiza un scan exclusivo sobre h hidden states
/// state: [h x d] estado previo, inp: [L x h x d], out: [L x h x d]
pub fn scanChunkwiseRef(state: []const f32, inp: []const f32, out: []f32, L: usize, h: usize, d: usize) void {
    // scan de referencia secuencial
    for (0..L) |t| {
        for (0..h * d) |i| {
            const prev = if (t == 0) state[i] else out[(t - 1) * h * d + i];
            out[t * h * d + i] = prev * 0.99 + inp[t * h * d + i];
        }
    }
}

// Versión CHUNKWISE: procesa la secuencia por bloques temporales, llevando el
// estado en la frontera de cada chunk. Las hd canales son independientes (paralelizables).
// Debe coincidir token-a-token con la recurrencia `scanChunkwiseRef` (criterio E3).
pub fn scanChunkwise(state: []const f32, inp: []const f32, out: []f32, L: usize, h: usize, d: usize) void {
    const hd = h * d;
    const C = 8; // tamaño de chunk temporal
    var t0: usize = 0;
    while (t0 < L) : (t0 += C) {
        const t_end = @min(t0 + C, L);
        for (0..hd) |e| {
            var acc = if (t0 == 0) state[e] else out[(t0 - 1) * hd + e];
            var t = t0;
            while (t < t_end) : (t += 1) {
                acc = acc * 0.99 + inp[t * hd + e];
                out[t * hd + e] = acc;
            }
        }
    }
}

// --- linear_attn_chunkwise: atención lineal con acumulador ---
/// S_t = S_{t-1} + k_t * v_t^T
/// o_t = q_t * S_t
pub fn linearAttnChunkwiseRef(S: []f32, q: []const f32, k: []const f32, v: []const f32, out: []f32, d: usize) void {
    // Actualizar estado
    for (0..d) |i| {
        for (0..d) |j| {
            S[i * d + j] += k[i] * v[j];
        }
    }
    // Output
    for (0..d) |i| {
        var acc: f32 = 0;
        for (0..d) |j| acc += q[j] * S[j * d + i];
        out[i] = acc;
    }
}

pub fn linearAttnChunkwise(S: []f32, q: []const f32, k: []const f32, v: []const f32, out: []f32, d: usize) void {
    linearAttnChunkwiseRef(S, q, k, v, out, d);
}

// --- delta_wy_chunkwise: DeltaNet update ---
/// S_t = S_{t-1} + delta_t * beta_t^T
/// o_t = S_t * q_t
pub fn deltaWyChunkwiseRef(S: []f32, q: []const f32, delta: []const f32, beta: []const f32, out: []f32, d: usize) void {
    for (0..d) |i| {
        for (0..d) |j| {
            S[i * d + j] += delta[i] * beta[j];
        }
    }
    for (0..d) |i| {
        var acc: f32 = 0;
        for (0..d) |j| acc += S[i * d + j] * q[j];
        out[i] = acc;
    }
}

pub fn deltaWyChunkwise(S: []f32, q: []const f32, delta: []const f32, beta: []const f32, out: []f32, d: usize) void {
    deltaWyChunkwiseRef(S, q, delta, beta, out, d);
}

// --- helpers little-endian (replicados para autonomía de zkernel) ---
fn readU16LE(data: []const u8, offset: usize) u16 {
    return std.mem.readInt(u16, data[offset..][0..2], .little);
}

// --- Dequantización Q4_0 ---
// Block: [f16 scale][u4 x 32] = 18 bytes por 32 elementos
pub const Q4_0_BLOCK_SIZE = 32;
pub const Q4_0_BLOCK_BYTES = 18;

pub fn dequantQ4_0(block: []const u8, out: []f32) void {
    std.debug.assert(block.len >= Q4_0_BLOCK_BYTES);
    std.debug.assert(out.len >= Q4_0_BLOCK_SIZE);
    const scale_f16 = std.mem.readInt(u16, block[0..2], .little);
    const scale: f32 = @floatCast(@as(f16, @bitCast(scale_f16)));
    for (0..Q4_0_BLOCK_SIZE) |i| {
        const byte_idx = 2 + i / 2;
        // Q4_0 estándar: nibble unsigned 0-15, interpretado como signed (-8..7)
        const nibble_u: u8 = if (i & 1 == 0) (block[byte_idx] & 0x0F) else (block[byte_idx] >> 4);
        const nibble: i8 = @as(i8, @intCast(nibble_u)) - 8;
        out[i] = @as(f32, @floatFromInt(nibble)) * scale;
    }
}

/// Versión SIMD de dequantQ4_0: vectoriza la expansión u4→f32 y la multiplicación por escala.
fn dequantQ4_0_simd(block: []const u8, out: []f32) void {
    std.debug.assert(block.len >= Q4_0_BLOCK_BYTES);
    std.debug.assert(out.len >= Q4_0_BLOCK_SIZE);
    const scale_f16 = std.mem.readInt(u16, block[0..2], .little);
    const scale: f32 = @floatCast(@as(f16, @bitCast(scale_f16)));
    const s: Vec = @splat(scale);
    var i: usize = 0;
    while (i < Q4_0_BLOCK_SIZE) : (i += 8) {
        const b0 = block[2 + i / 2 + 0];
        const b1 = block[2 + i / 2 + 1];
        const b2 = block[2 + i / 2 + 2];
        const b3 = block[2 + i / 2 + 3];
        const vals: @Vector(8, u32) = .{
            @as(u32, b0 & 0x0F),
            @as(u32, b0 >> 4),
            @as(u32, b1 & 0x0F),
            @as(u32, b1 >> 4),
            @as(u32, b2 & 0x0F),
            @as(u32, b2 >> 4),
            @as(u32, b3 & 0x0F),
            @as(u32, b3 >> 4),
        };
        // Q4_0 estándar: restar 8 para obtener signed (-8..7)
        const offset: @Vector(8, i32) = @splat(8);
        const signed_vals = @as(@Vector(8, i32), @intCast(vals)) - offset;
        const vf = @as(Vec, @floatFromInt(signed_vals));
        const ov = vf * s;
        @as(*align(1) Vec, @ptrCast(&out[i])).* = ov;
    }
}

// --- Dequantización Q8_0 ---
// Block: [f16 scale][i8 x 32] = 34 bytes por 32 elementos
pub const Q8_0_BLOCK_SIZE = 32;
pub const Q8_0_BLOCK_BYTES = 34;

pub fn dequantQ8_0(block: []const u8, out: []f32) void {
    std.debug.assert(block.len >= Q8_0_BLOCK_BYTES);
    std.debug.assert(out.len >= Q8_0_BLOCK_SIZE);
    const scale_f16 = std.mem.readInt(u16, block[0..2], .little);
    const scale: f32 = @floatCast(@as(f16, @bitCast(scale_f16)));
    for (0..Q8_0_BLOCK_SIZE) |i| {
        const qi = @as(i8, @bitCast(block[2 + i]));
        out[i] = @as(f32, @floatFromInt(qi)) * scale;
    }
}

/// Versión SIMD de dequantQ8_0: vectoriza la expansión i8→f32 y la multiplicación por escala.
fn dequantQ8_0_simd(block: []const u8, out: []f32) void {
    std.debug.assert(block.len >= Q8_0_BLOCK_BYTES);
    std.debug.assert(out.len >= Q8_0_BLOCK_SIZE);
    const scale_f16 = std.mem.readInt(u16, block[0..2], .little);
    const scale: f32 = @floatCast(@as(f16, @bitCast(scale_f16)));
    const s: Vec = @splat(scale);
    var i: usize = 0;
    while (i + VecWidth <= Q8_0_BLOCK_SIZE) : (i += VecWidth) {
        const q_bytes: @Vector(VecWidth, i8) = @as(*align(1) const @Vector(VecWidth, i8), @ptrCast(&block[2 + i])).*;
        const qf = @as(Vec, @floatFromInt(q_bytes));
        const ov = qf * s;
        @as(*align(1) Vec, @ptrCast(&out[i])).* = ov;
    }
    for (i..Q8_0_BLOCK_SIZE) |j| {
        const qi = @as(i8, @bitCast(block[2 + j]));
        out[j] = @as(f32, @floatFromInt(qi)) * scale;
    }
}

// =====================================================================
// Cuantización nativa (f32 → formato cuantizado) para inferencia
// =====================================================================

/// Cuantiza un bloque de 32 valores f32 a formato Q8_0.
/// `src` debe tener al menos 32 elementos. `dst` debe tener al menos 34 bytes.
/// Escala = max(|x|) / 127. Valores = round(x / escala) clampados a [-128, 127].
pub fn quantizeQ8_0Block(src: []const f32, dst: []u8) void {
    std.debug.assert(src.len >= Q8_0_BLOCK_SIZE);
    std.debug.assert(dst.len >= Q8_0_BLOCK_BYTES);
    // Encontrar max abs
    var max_abs: f32 = 0;
    for (0..Q8_0_BLOCK_SIZE) |i| {
        const a = @abs(src[i]);
        if (a > max_abs) max_abs = a;
    }
    // Escala
    const scale: f32 = if (max_abs > 1e-12) max_abs / 127.0 else 0.0;
    const scale_f16: f16 = @floatCast(scale);
    std.mem.writeInt(u16, dst[0..2], @bitCast(scale_f16), .little);
    // Cuantizar
    const inv_scale: f32 = if (scale > 1e-12) 1.0 / scale else 0.0;
    for (0..Q8_0_BLOCK_SIZE) |i| {
        const qf = src[i] * inv_scale;
        const qi: i32 = @intFromFloat(@max(-128.0, @min(127.0, @round(qf))));
        dst[2 + i] = @bitCast(@as(i8, @intCast(qi)));
    }
}

/// Cuantiza un bloque de 32 valores f32 a formato Q4_0 (estándar signed).
/// `src` debe tener al menos 32 elementos. `dst` debe tener al menos 18 bytes.
/// Escala = max(|x|) / 7. Valores signed [-8, 7] empaquetados como unsigned [0, 15] (+8).
pub fn quantizeQ4_0Block(src: []const f32, dst: []u8) void {
    std.debug.assert(src.len >= Q4_0_BLOCK_SIZE);
    std.debug.assert(dst.len >= Q4_0_BLOCK_BYTES);
    var max_abs: f32 = 0;
    for (0..Q4_0_BLOCK_SIZE) |i| {
        const a = @abs(src[i]);
        if (a > max_abs) max_abs = a;
    }
    const scale: f32 = if (max_abs > 1e-12) max_abs / 7.0 else 0.0;
    const scale_f16: f16 = @floatCast(scale);
    std.mem.writeInt(u16, dst[0..2], @bitCast(scale_f16), .little);
    const inv_scale: f32 = if (scale > 1e-12) 1.0 / scale else 0.0;
    for (0..Q4_0_BLOCK_SIZE / 2) |i| {
        const qf0 = src[i * 2] * inv_scale;
        const qf1 = src[i * 2 + 1] * inv_scale;
        // Clamp a [-8, 7] (signed)
        const qi0: i32 = @intFromFloat(@max(-8.0, @min(7.0, @round(qf0))));
        const qi1: i32 = @intFromFloat(@max(-8.0, @min(7.0, @round(qf1))));
        // Empaquetar como unsigned: nibble = signed_val + 8 → [0, 15]
        const qu0: u8 = @intCast(qi0 + 8);
        const qu1: u8 = @intCast(qi1 + 8);
        dst[2 + i] = (qu0 & 0x0F) | ((qu1 & 0x0F) << 4);
    }
}

/// Cuantiza una matriz completa f32 [N, K] a Q8_0.
/// `src` debe tener N*K elementos f32. `dst` debe tener N*n_blocks*Q8_0_BLOCK_BYTES bytes.
/// Devuelve el número de bytes escritos.
pub fn quantizeQ8_0Matrix(dst: []u8, src: []const f32, N: usize, K: usize) usize {
    const blk = Q8_0_BLOCK_SIZE;
    const blk_bytes = Q8_0_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;
    for (0..N) |n| {
        for (0..n_blocks) |b| {
            const src_off = n * K + b * blk;
            const dst_off = (n * n_blocks + b) * blk_bytes;
            // Último bloque puede tener padding
            if (b * blk + blk <= K) {
                quantizeQ8_0Block(src[src_off..], dst[dst_off..]);
            } else {
                // Padding con ceros
                var pad_buf: [Q8_0_BLOCK_SIZE]f32 = undefined;
                const remaining = K - b * blk;
                @memcpy(pad_buf[0..remaining], src[src_off..src_off + remaining]);
                @memset(pad_buf[remaining..], 0);
                quantizeQ8_0Block(&pad_buf, dst[dst_off..]);
            }
        }
    }
    return N * n_blocks * blk_bytes;
}

/// Cuantiza una matriz completa f32 [N, K] a Q4_0.
/// `src` debe tener N*K elementos f32. `dst` debe tener N*n_blocks*Q4_0_BLOCK_BYTES bytes.
/// Devuelve el número de bytes escritos.
pub fn quantizeQ4_0Matrix(dst: []u8, src: []const f32, N: usize, K: usize) usize {
    const blk = Q4_0_BLOCK_SIZE;
    const blk_bytes = Q4_0_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;
    for (0..N) |n| {
        for (0..n_blocks) |b| {
            const src_off = n * K + b * blk;
            const dst_off = (n * n_blocks + b) * blk_bytes;
            if (b * blk + blk <= K) {
                quantizeQ4_0Block(src[src_off..], dst[dst_off..]);
            } else {
                var pad_buf: [Q4_0_BLOCK_SIZE]f32 = undefined;
                const remaining = K - b * blk;
                @memcpy(pad_buf[0..remaining], src[src_off..src_off + remaining]);
                @memset(pad_buf[remaining..], 0);
                quantizeQ4_0Block(&pad_buf, dst[dst_off..]);
            }
        }
    }
    return N * n_blocks * blk_bytes;
}

// --- Matmul cuantizado Q4_0: C[M,N] = A_quant[M,K] @ B[K,N] ---
/// A_quant es raw data en formato Q4_0, B y C son f32
pub fn matmulQ4_0(C: []f32, A_quant: []const u8, B: []const f32, M: usize, K: usize, N: usize) void {
    @memset(C, 0);
    const blk = Q4_0_BLOCK_SIZE;
    const blk_bytes = Q4_0_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;
    var deq_buf: [Q4_0_BLOCK_SIZE]f32 = undefined;

    for (0..M) |m| {
        for (0..K) |k| {
            const bk = k / blk;
            const ki = k % blk;
            const block_off = m * n_blocks * blk_bytes + bk * blk_bytes;
            if (ki == 0) {
                dequantQ4_0(A_quant[block_off..], &deq_buf);
            }
            const a_val = deq_buf[ki];
            for (0..N) |n| {
                C[m * N + n] += a_val * B[k * N + n];
            }
        }
    }
}

// --- Matmul cuantizado Q8_0 ---
pub fn matmulQ8_0(C: []f32, A_quant: []const u8, B: []const f32, M: usize, K: usize, N: usize) void {
    @memset(C, 0);
    const blk = Q8_0_BLOCK_SIZE;
    const blk_bytes = Q8_0_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;
    var deq_buf: [Q8_0_BLOCK_SIZE]f32 = undefined;

    for (0..M) |m| {
        for (0..K) |k| {
            const bk = k / blk;
            const ki = k % blk;
            const block_off = m * n_blocks * blk_bytes + bk * blk_bytes;
            if (ki == 0) {
                dequantQ8_0(A_quant[block_off..], &deq_buf);
            }
            const a_val = deq_buf[ki];
            for (0..N) |n| {
                C[m * N + n] += a_val * B[k * N + n];
            }
        }
    }
}

// --- Dequantización de bloque Q4_K (256 elems, 144 bytes) ---
fn getScaleMinK4(j: usize, q: []const u8) struct { d: u8, m: u8 } {
    if (j < 4) {
        return .{ .d = q[j] & 63, .m = q[j + 4] & 63 };
    } else {
        const d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        const m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        return .{ .d = d, .m = m };
    }
}

pub fn dequantQ4_K_block(src: []const u8, dst: []f32) void {
    std.debug.assert(src.len >= 144);
    std.debug.assert(dst.len >= 256);
    const d: f32 = @floatCast(@as(f16, @bitCast(readU16LE(src, 0))));
    const dmin: f32 = @floatCast(@as(f16, @bitCast(readU16LE(src, 2))));
    const scales = src[4..][0..12];
    const q = src[16..][0..128];
    var is: usize = 0;
    var qoff: usize = 0;
    var y: usize = 0;
    var j: usize = 0;
    while (j < 256) : (j += 64) {
        const sm0 = getScaleMinK4(is, scales);
        const d1 = d * @as(f32, @floatFromInt(sm0.d));
        const m1 = dmin * @as(f32, @floatFromInt(sm0.m));
        const sm1 = getScaleMinK4(is + 1, scales);
        const d2 = d * @as(f32, @floatFromInt(sm1.d));
        const m2 = dmin * @as(f32, @floatFromInt(sm1.m));
        for (0..32) |l| {
            dst[y] = d1 * @as(f32, @floatFromInt(q[qoff + l] & 0x0F)) - m1;
            y += 1;
        }
        for (0..32) |l| {
            dst[y] = d2 * @as(f32, @floatFromInt(q[qoff + l] >> 4)) - m2;
            y += 1;
        }
        qoff += 32;
        is += 2;
    }
}

/// Versión SIMD de dequantQ4_K: vectoriza la extracción de nibbles y la multiplicación.
fn dequantQ4_K_simd(src: []const u8, dst: []f32) void {
    std.debug.assert(src.len >= 144);
    std.debug.assert(dst.len >= 256);
    const d: f32 = @floatCast(@as(f16, @bitCast(readU16LE(src, 0))));
    const dmin: f32 = @floatCast(@as(f16, @bitCast(readU16LE(src, 2))));
    const scales = src[4..][0..12];
    const q = src[16..][0..128];
    var is: usize = 0;
    var qoff: usize = 0;
    var y: usize = 0;
    var j: usize = 0;
    const mask4: @Vector(VecWidth, u8) = @splat(0x0F);
    while (j < 256) : (j += 64) {
        const sm0 = getScaleMinK4(is, scales);
        const d1 = d * @as(f32, @floatFromInt(sm0.d));
        const m1 = dmin * @as(f32, @floatFromInt(sm0.m));
        const sm1 = getScaleMinK4(is + 1, scales);
        const d2 = d * @as(f32, @floatFromInt(sm1.d));
        const m2 = dmin * @as(f32, @floatFromInt(sm1.m));
        const dv1: Vec = @splat(d1);
        const mv1: Vec = @splat(m1);
        const dv2: Vec = @splat(d2);
        const mv2: Vec = @splat(m2);
        // Low nibbles: 32 values from 32 bytes, 8 at a time
        var l: usize = 0;
        while (l + VecWidth <= 32) : (l += VecWidth) {
            const qb: @Vector(VecWidth, u8) = @as(*align(1) const @Vector(VecWidth, u8), @ptrCast(&q[qoff + l])).*;
            const lo: @Vector(VecWidth, u8) = qb & mask4;
            const lof = @as(Vec, @floatFromInt(lo));
            @as(*align(1) Vec, @ptrCast(&dst[y])).* = dv1 * lof - mv1;
            y += VecWidth;
        }
        for (l..32) |r| {
            dst[y] = d1 * @as(f32, @floatFromInt(q[qoff + r] & 0x0F)) - m1;
            y += 1;
        }
        // High nibbles: 32 values from same 32 bytes, 8 at a time
        l = 0;
        while (l + VecWidth <= 32) : (l += VecWidth) {
            const qb: @Vector(VecWidth, u8) = @as(*align(1) const @Vector(VecWidth, u8), @ptrCast(&q[qoff + l])).*;
            const hi: @Vector(VecWidth, u8) = qb >> @as(@Vector(VecWidth, u3), @splat(4));
            const hif = @as(Vec, @floatFromInt(hi));
            @as(*align(1) Vec, @ptrCast(&dst[y])).* = dv2 * hif - mv2;
            y += VecWidth;
        }
        for (l..32) |r| {
            dst[y] = d2 * @as(f32, @floatFromInt(q[qoff + r] >> 4)) - m2;
            y += 1;
        }
        qoff += 32;
        is += 2;
    }
}

// --- Dequantización de bloque Q6_K (256 elems, 210 bytes) ---
pub fn dequantQ6_K_block(src: []const u8, dst: []f32) void {
    std.debug.assert(src.len >= 210);
    std.debug.assert(dst.len >= 256);
    const ql = src[0..128];
    const qh = src[128..192];
    const sc = src[192..208];
    const d: f32 = @floatCast(@as(f16, @bitCast(readU16LE(src, 208))));
    var n: usize = 0;
    var qlb: usize = 0;
    var qhb: usize = 0;
    var scb: usize = 0;
    while (n < 256) : (n += 128) {
        for (0..32) |l| {
            const is = l / 16;
            const lo0: i32 = @as(i32, ql[qlb + l] & 0x0F) | (@as(i32, (qh[qhb + l] >> 0) & 3) << 4);
            const lo1: i32 = @as(i32, ql[qlb + l + 32] & 0x0F) | (@as(i32, (qh[qhb + l] >> 2) & 3) << 4);
            const hi0: i32 = @as(i32, ql[qlb + l] >> 4) | (@as(i32, (qh[qhb + l] >> 4) & 3) << 4);
            const hi1: i32 = @as(i32, ql[qlb + l + 32] >> 4) | (@as(i32, (qh[qhb + l] >> 6) & 3) << 4);
            const q1 = lo0 - 32;
            const q2 = lo1 - 32;
            const q3 = hi0 - 32;
            const q4 = hi1 - 32;
            const s0: i8 = @bitCast(sc[scb + is + 0]);
            const s2: i8 = @bitCast(sc[scb + is + 2]);
            const s4: i8 = @bitCast(sc[scb + is + 4]);
            const s6: i8 = @bitCast(sc[scb + is + 6]);
            dst[n + l] = d * @as(f32, @floatFromInt(s0)) * @as(f32, @floatFromInt(q1));
            dst[n + l + 32] = d * @as(f32, @floatFromInt(s2)) * @as(f32, @floatFromInt(q2));
            dst[n + l + 64] = d * @as(f32, @floatFromInt(s4)) * @as(f32, @floatFromInt(q3));
            dst[n + l + 96] = d * @as(f32, @floatFromInt(s6)) * @as(f32, @floatFromInt(q4));
        }
        qlb += 64;
        qhb += 32;
        scb += 8;
    }
}

/// Versión SIMD de dequantQ6_K: vectoriza la combinación de ql+qh y la multiplicación.
fn dequantQ6_K_simd(src: []const u8, dst: []f32) void {
    std.debug.assert(src.len >= 210);
    std.debug.assert(dst.len >= 256);
    const ql = src[0..128];
    const qh = src[128..192];
    const sc = src[192..208];
    const d: f32 = @floatCast(@as(f16, @bitCast(readU16LE(src, 208))));
    const dv: Vec = @splat(d);
    const mask4: @Vector(VecWidth, u8) = @splat(0x0F);
    const mask2: @Vector(VecWidth, u8) = @splat(3);
    const off32: Vec = @splat(-32.0);
    var n: usize = 0;
    var qlb: usize = 0;
    var qhb: usize = 0;
    var scb: usize = 0;
    while (n < 256) : (n += 128) {
        var l: usize = 0;
        while (l + VecWidth <= 32) : (l += VecWidth) {
            const is = l / 16;
            const qlb_vec: @Vector(VecWidth, u8) = @as(*align(1) const @Vector(VecWidth, u8), @ptrCast(&ql[qlb + l])).*;
            const qhb_vec: @Vector(VecWidth, u8) = @as(*align(1) const @Vector(VecWidth, u8), @ptrCast(&qh[qhb + l])).*;
            const qlb2_vec: @Vector(VecWidth, u8) = @as(*align(1) const @Vector(VecWidth, u8), @ptrCast(&ql[qlb + l + 32])).*;
            // lo0 = (ql & 0x0F) | ((qh >> 0) & 3) << 4
            const lo0: @Vector(VecWidth, u8) = (qlb_vec & mask4) | ((qhb_vec >> @as(@Vector(VecWidth, u3), @splat(0))) & mask2) << @as(@Vector(VecWidth, u3), @splat(4));
            // lo1 = (ql+32 & 0x0F) | ((qh >> 2) & 3) << 4
            const lo1: @Vector(VecWidth, u8) = (qlb2_vec & mask4) | ((qhb_vec >> @as(@Vector(VecWidth, u3), @splat(2))) & mask2) << @as(@Vector(VecWidth, u3), @splat(4));
            // hi0 = (ql >> 4) | ((qh >> 4) & 3) << 4
            const hi0: @Vector(VecWidth, u8) = (qlb_vec >> @as(@Vector(VecWidth, u3), @splat(4))) | ((qhb_vec >> @as(@Vector(VecWidth, u3), @splat(4))) & mask2) << @as(@Vector(VecWidth, u3), @splat(4));
            // hi1 = (ql+32 >> 4) | ((qh >> 6) & 3) << 4
            const hi1: @Vector(VecWidth, u8) = (qlb2_vec >> @as(@Vector(VecWidth, u3), @splat(4))) | ((qhb_vec >> @as(@Vector(VecWidth, u3), @splat(6))) & mask2) << @as(@Vector(VecWidth, u3), @splat(4));
            const s0: i8 = @bitCast(sc[scb + is + 0]);
            const s2: i8 = @bitCast(sc[scb + is + 2]);
            const s4: i8 = @bitCast(sc[scb + is + 4]);
            const s6: i8 = @bitCast(sc[scb + is + 6]);
            const sv0: Vec = @splat(@as(f32, @floatFromInt(s0)));
            const sv2: Vec = @splat(@as(f32, @floatFromInt(s2)));
            const sv4: Vec = @splat(@as(f32, @floatFromInt(s4)));
            const sv6: Vec = @splat(@as(f32, @floatFromInt(s6)));
            const lo0f = @as(Vec, @floatFromInt(lo0)) + off32;
            const lo1f = @as(Vec, @floatFromInt(lo1)) + off32;
            const hi0f = @as(Vec, @floatFromInt(hi0)) + off32;
            const hi1f = @as(Vec, @floatFromInt(hi1)) + off32;
            @as(*align(1) Vec, @ptrCast(&dst[n + l])).* = dv * sv0 * lo0f;
            @as(*align(1) Vec, @ptrCast(&dst[n + l + 32])).* = dv * sv2 * lo1f;
            @as(*align(1) Vec, @ptrCast(&dst[n + l + 64])).* = dv * sv4 * hi0f;
            @as(*align(1) Vec, @ptrCast(&dst[n + l + 96])).* = dv * sv6 * hi1f;
        }
        for (l..32) |r| {
            const is = r / 16;
            const lo0: i32 = @as(i32, ql[qlb + r] & 0x0F) | (@as(i32, (qh[qhb + r] >> 0) & 3) << 4);
            const lo1: i32 = @as(i32, ql[qlb + r + 32] & 0x0F) | (@as(i32, (qh[qhb + r] >> 2) & 3) << 4);
            const hi0: i32 = @as(i32, ql[qlb + r] >> 4) | (@as(i32, (qh[qhb + r] >> 4) & 3) << 4);
            const hi1: i32 = @as(i32, ql[qlb + r + 32] >> 4) | (@as(i32, (qh[qhb + r] >> 6) & 3) << 4);
            const s0: i8 = @bitCast(sc[scb + is + 0]);
            const s2: i8 = @bitCast(sc[scb + is + 2]);
            const s4: i8 = @bitCast(sc[scb + is + 4]);
            const s6: i8 = @bitCast(sc[scb + is + 6]);
            dst[n + r] = d * @as(f32, @floatFromInt(s0)) * @as(f32, @floatFromInt(lo0 - 32));
            dst[n + r + 32] = d * @as(f32, @floatFromInt(s2)) * @as(f32, @floatFromInt(lo1 - 32));
            dst[n + r + 64] = d * @as(f32, @floatFromInt(s4)) * @as(f32, @floatFromInt(hi0 - 32));
            dst[n + r + 96] = d * @as(f32, @floatFromInt(s6)) * @as(f32, @floatFromInt(hi1 - 32));
        }
        qlb += 64;
        qhb += 32;
        scb += 8;
    }
}

// --- Matmul vector × matriz cuantizada: referencias escalares ---
pub fn matmulVecQ4_0Ref(out: []f32, weight_quant: []const u8, x: []const f32, N: usize, K: usize) void {
    const blk = Q4_0_BLOCK_SIZE;
    const blk_bytes = Q4_0_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;
    var deq_buf: [Q4_0_BLOCK_SIZE]f32 = undefined;
    for (0..N) |n| {
        const row_off = n * n_blocks * blk_bytes;
        var acc: f32 = 0;
        var k: usize = 0;
        var bk: usize = 0;
        while (k < K) : (k += blk) {
            dequantQ4_0(weight_quant[row_off + bk * blk_bytes ..], &deq_buf);
            const k_end = @min(k + blk, K);
            for (k..k_end) |ki| acc += deq_buf[ki - k] * x[ki];
            bk += 1;
        }
        out[n] = acc;
    }
}

pub fn matmulVecQ8_0Ref(out: []f32, weight_quant: []const u8, x: []const f32, N: usize, K: usize) void {
    const blk = Q8_0_BLOCK_SIZE;
    const blk_bytes = Q8_0_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;
    var deq_buf: [Q8_0_BLOCK_SIZE]f32 = undefined;
    for (0..N) |n| {
        const row_off = n * n_blocks * blk_bytes;
        var acc: f32 = 0;
        var k: usize = 0;
        var bk: usize = 0;
        while (k < K) : (k += blk) {
            dequantQ8_0(weight_quant[row_off + bk * blk_bytes ..], &deq_buf);
            const k_end = @min(k + blk, K);
            for (k..k_end) |ki| acc += deq_buf[ki - k] * x[ki];
            bk += 1;
        }
        out[n] = acc;
    }
}

pub fn matmulVecQ4_KRef(out: []f32, weight_quant: []const u8, x: []const f32, N: usize, K: usize) void {
    const blk = Q4_K_BLOCK_SIZE;
    const blk_bytes = Q4_K_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;
    var deq_buf: [Q4_K_BLOCK_SIZE]f32 = undefined;
    for (0..N) |n| {
        const row_off = n * n_blocks * blk_bytes;
        var acc: f32 = 0;
        var k: usize = 0;
        var bk: usize = 0;
        while (k < K) : (k += blk) {
            dequantQ4_K_block(weight_quant[row_off + bk * blk_bytes ..], &deq_buf);
            const k_end = @min(k + blk, K);
            for (k..k_end) |ki| acc += deq_buf[ki - k] * x[ki];
            bk += 1;
        }
        out[n] = acc;
    }
}

pub fn matmulVecQ6_KRef(out: []f32, weight_quant: []const u8, x: []const f32, N: usize, K: usize) void {
    const blk = Q6_K_BLOCK_SIZE;
    const blk_bytes = Q6_K_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;
    var deq_buf: [Q6_K_BLOCK_SIZE]f32 = undefined;
    for (0..N) |n| {
        const row_off = n * n_blocks * blk_bytes;
        var acc: f32 = 0;
        var k: usize = 0;
        var bk: usize = 0;
        while (k < K) : (k += blk) {
            dequantQ6_K_block(weight_quant[row_off + bk * blk_bytes ..], &deq_buf);
            const k_end = @min(k + blk, K);
            for (k..k_end) |ki| acc += deq_buf[ki - k] * x[ki];
            bk += 1;
        }
        out[n] = acc;
    }
}

// --- Helper SIMD dot-product por bloque ---
inline fn dotSimd(deq: []const f32, x: []const f32) f32 {
    const count = @min(deq.len, x.len);
    var acc_vec: Vec = @splat(0);
    var i: usize = 0;
    while (i + VecWidth <= count) : (i += VecWidth) {
        const deq_v: Vec = @as(*align(1) const Vec, @ptrCast(&deq[i])).*;
        const x_v: Vec = @as(*align(1) const Vec, @ptrCast(&x[i])).*;
        acc_vec += deq_v * x_v;
    }
    var acc: f32 = @reduce(.Add, acc_vec);
    for (i..count) |j| acc += deq[j] * x[j];
    return acc;
}

// --- Parallel helper: distribuye n ítems en ~num-cpu hilos ---
fn parallelForRows(n: usize, ctx: anytype, comptime func: fn (@TypeOf(ctx), usize) void) void {
    if (n == 0) return;
    if (@import("builtin").single_threaded) {
        var i: usize = 0;
        while (i < n) : (i += 1) func(ctx, i);
        return;
    }
    const serial_threshold = 256;
    const cpu_count = std.Thread.getCpuCount() catch 1;
    var workers = if (cpu_count == 0) 1 else cpu_count;
    if (n < serial_threshold or workers <= 1) {
        var i: usize = 0;
        while (i < n) : (i += 1) func(ctx, i);
        return;
    }
    if (workers > n) workers = n;
    const max_threads = 256;
    if (workers > max_threads) workers = max_threads;
    const chunk = (n + workers - 1) / workers;
    const Worker = struct {
        fn run(c: @TypeOf(ctx), start: usize, end: usize) void {
            var i = start;
            while (i < end) : (i += 1) func(c, i);
        }
    };
    var threads: [max_threads]?std.Thread = .{null} ** max_threads;
    var spawned: usize = 0;
    var start: usize = 0;
    var w: usize = 0;
    while (w < workers and start < n) : (w += 1) {
        const end = @min(start + chunk, n);
        const t = std.Thread.spawn(.{}, Worker.run, .{ ctx, start, end }) catch {
            Worker.run(ctx, start, end);
            start = end;
            continue;
        };
        threads[spawned] = t;
        spawned += 1;
        start = end;
    }
    if (start < n) Worker.run(ctx, start, n);
    var j: usize = 0;
    while (j < spawned) : (j += 1) {
        if (threads[j]) |t| t.join();
    }
}

// --- Matmul vector × matriz cuantizada: SIMD ---
/// C[n] = dot(x, dequantized_row_n(weight_quant))
/// weight_quant está en layout GGUF original: N filas × K columnas, bloques a lo largo de K.

fn matmulVecQ4_0_serial(out: []f32, weight_quant: []const u8, x: []const f32, N: usize, K: usize) void {
    const blk = Q4_0_BLOCK_SIZE;
    const blk_bytes = Q4_0_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;
    @memset(out[0..N], 0);
    var deq_buf: [Q4_0_BLOCK_SIZE]f32 = undefined;
    const tile_k: usize = 512;
    var k_base: usize = 0;
    while (k_base < K) : (k_base += tile_k) {
        const k_end = @min(k_base + tile_k, K);
        const bk_start = k_base / blk;
        for (0..N) |n| {
            const row_off = n * n_blocks * blk_bytes;
            var acc: f32 = 0;
            var bk = bk_start;
            var k = k_base;
            while (k < k_end) : (k += blk) {
                dequantQ4_0_simd(weight_quant[row_off + bk * blk_bytes ..], &deq_buf);
                const k_local_end = @min(k + blk, k_end);
                acc += dotSimd(&deq_buf, x[k..k_local_end]);
                bk += 1;
            }
            out[n] += acc;
        }
    }
}

fn matmulVecQ4_0_parallel(out: []f32, weight_quant: []const u8, x: []const f32, N: usize, K: usize) void {
    const blk = Q4_0_BLOCK_SIZE;
    const n_blocks = (K + blk - 1) / blk;
    @memset(out[0..N], 0);
    const Ctx = struct {
        out: []f32,
        weight_quant: []const u8,
        x: []const f32,
        K: usize,
        n_blocks: usize,
        fn run(self: @This(), n: usize) void {
            const row_off = n * self.n_blocks * Q4_0_BLOCK_BYTES;
            var deq_buf: [Q4_0_BLOCK_SIZE]f32 = undefined;
            const tile_k: usize = 512;
            var k_base: usize = 0;
            while (k_base < self.K) : (k_base += tile_k) {
                const k_end = @min(k_base + tile_k, self.K);
                const bk_start = k_base / Q4_0_BLOCK_SIZE;
                var acc: f32 = 0;
                var bk = bk_start;
                var k = k_base;
                while (k < k_end) : (k += Q4_0_BLOCK_SIZE) {
                    dequantQ4_0_simd(self.weight_quant[row_off + bk * Q4_0_BLOCK_BYTES ..], &deq_buf);
                    const k_local_end = @min(k + Q4_0_BLOCK_SIZE, k_end);
                    acc += dotSimd(&deq_buf, self.x[k..k_local_end]);
                    bk += 1;
                }
                self.out[n] += acc;
            }
        }
    };
    parallelForRows(N, Ctx{ .out = out, .weight_quant = weight_quant, .x = x, .K = K, .n_blocks = n_blocks }, Ctx.run);
}

pub fn matmulVecQ4_0(out: []f32, weight_quant: []const u8, x: []const f32, N: usize, K: usize) void {
    const threshold_n: usize = 256;
    const threshold_work: usize = 100_000;
    if (N >= threshold_n and N * K >= threshold_work) {
        matmulVecQ4_0_parallel(out, weight_quant, x, N, K);
    } else {
        matmulVecQ4_0_serial(out, weight_quant, x, N, K);
    }
}

fn matmulVecQ8_0_serial(out: []f32, weight_quant: []const u8, x: []const f32, N: usize, K: usize) void {
    const blk = Q8_0_BLOCK_SIZE;
    const blk_bytes = Q8_0_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;
    @memset(out[0..N], 0);
    var deq_buf: [Q8_0_BLOCK_SIZE]f32 = undefined;
    const tile_k: usize = 512;
    var k_base: usize = 0;
    while (k_base < K) : (k_base += tile_k) {
        const k_end = @min(k_base + tile_k, K);
        const bk_start = k_base / blk;
        for (0..N) |n| {
            const row_off = n * n_blocks * blk_bytes;
            var acc: f32 = 0;
            var bk = bk_start;
            var k = k_base;
            while (k < k_end) : (k += blk) {
                dequantQ8_0_simd(weight_quant[row_off + bk * blk_bytes ..], &deq_buf);
                const k_local_end = @min(k + blk, k_end);
                acc += dotSimd(&deq_buf, x[k..k_local_end]);
                bk += 1;
            }
            out[n] += acc;
        }
    }
}

fn matmulVecQ8_0_parallel(out: []f32, weight_quant: []const u8, x: []const f32, N: usize, K: usize) void {
    const blk = Q8_0_BLOCK_SIZE;
    const n_blocks = (K + blk - 1) / blk;
    @memset(out[0..N], 0);
    const Ctx = struct {
        out: []f32,
        weight_quant: []const u8,
        x: []const f32,
        K: usize,
        n_blocks: usize,
        fn run(self: @This(), n: usize) void {
            const row_off = n * self.n_blocks * Q8_0_BLOCK_BYTES;
            var deq_buf: [Q8_0_BLOCK_SIZE]f32 = undefined;
            const tile_k: usize = 512;
            var k_base: usize = 0;
            while (k_base < self.K) : (k_base += tile_k) {
                const k_end = @min(k_base + tile_k, self.K);
                const bk_start = k_base / Q8_0_BLOCK_SIZE;
                var acc: f32 = 0;
                var bk = bk_start;
                var k = k_base;
                while (k < k_end) : (k += Q8_0_BLOCK_SIZE) {
                    dequantQ8_0_simd(self.weight_quant[row_off + bk * Q8_0_BLOCK_BYTES ..], &deq_buf);
                    const k_local_end = @min(k + Q8_0_BLOCK_SIZE, k_end);
                    acc += dotSimd(&deq_buf, self.x[k..k_local_end]);
                    bk += 1;
                }
                self.out[n] += acc;
            }
        }
    };
    parallelForRows(N, Ctx{ .out = out, .weight_quant = weight_quant, .x = x, .K = K, .n_blocks = n_blocks }, Ctx.run);
}

pub fn matmulVecQ8_0(out: []f32, weight_quant: []const u8, x: []const f32, N: usize, K: usize) void {
    const threshold_n: usize = 256;
    const threshold_work: usize = 100_000;
    if (N >= threshold_n and N * K >= threshold_work) {
        matmulVecQ8_0_parallel(out, weight_quant, x, N, K);
    } else {
        matmulVecQ8_0_serial(out, weight_quant, x, N, K);
    }
}

pub const Q4_K_BLOCK_SIZE = 256;
pub const Q4_K_BLOCK_BYTES = 144;

fn matmulVecQ4_K_serial(out: []f32, weight_quant: []const u8, x: []const f32, N: usize, K: usize) void {
    const blk = Q4_K_BLOCK_SIZE;
    const blk_bytes = Q4_K_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;
    @memset(out[0..N], 0);
    var deq_buf: [Q4_K_BLOCK_SIZE]f32 = undefined;
    const tile_k: usize = 512;
    var k_base: usize = 0;
    while (k_base < K) : (k_base += tile_k) {
        const k_end = @min(k_base + tile_k, K);
        const bk_start = k_base / blk;
        for (0..N) |n| {
            const row_off = n * n_blocks * blk_bytes;
            var acc: f32 = 0;
            var bk = bk_start;
            var k = k_base;
            while (k < k_end) : (k += blk) {
                dequantQ4_K_block(weight_quant[row_off + bk * blk_bytes ..], &deq_buf);
                const k_local_end = @min(k + blk, k_end);
                acc += dotSimd(&deq_buf, x[k..k_local_end]);
                bk += 1;
            }
            out[n] += acc;
        }
    }
}

fn matmulVecQ4_K_parallel(out: []f32, weight_quant: []const u8, x: []const f32, N: usize, K: usize) void {
    const blk = Q4_K_BLOCK_SIZE;
    const n_blocks = (K + blk - 1) / blk;
    @memset(out[0..N], 0);
    const Ctx = struct {
        out: []f32,
        weight_quant: []const u8,
        x: []const f32,
        K: usize,
        n_blocks: usize,
        fn run(self: @This(), n: usize) void {
            const row_off = n * self.n_blocks * Q4_K_BLOCK_BYTES;
            var deq_buf: [Q4_K_BLOCK_SIZE]f32 = undefined;
            const tile_k: usize = 512;
            var k_base: usize = 0;
            while (k_base < self.K) : (k_base += tile_k) {
                const k_end = @min(k_base + tile_k, self.K);
                const bk_start = k_base / Q4_K_BLOCK_SIZE;
                var acc: f32 = 0;
                var bk = bk_start;
                var k = k_base;
                while (k < k_end) : (k += Q4_K_BLOCK_SIZE) {
                    dequantQ4_K_block(self.weight_quant[row_off + bk * Q4_K_BLOCK_BYTES ..], &deq_buf);
                    const k_local_end = @min(k + Q4_K_BLOCK_SIZE, k_end);
                    acc += dotSimd(&deq_buf, self.x[k..k_local_end]);
                    bk += 1;
                }
                self.out[n] += acc;
            }
        }
    };
    parallelForRows(N, Ctx{ .out = out, .weight_quant = weight_quant, .x = x, .K = K, .n_blocks = n_blocks }, Ctx.run);
}

pub fn matmulVecQ4_K(out: []f32, weight_quant: []const u8, x: []const f32, N: usize, K: usize) void {
    const threshold_n: usize = 256;
    const threshold_work: usize = 100_000;
    if (N >= threshold_n and N * K >= threshold_work) {
        matmulVecQ4_K_parallel(out, weight_quant, x, N, K);
    } else {
        matmulVecQ4_K_serial(out, weight_quant, x, N, K);
    }
}

pub const Q6_K_BLOCK_SIZE = 256;
pub const Q6_K_BLOCK_BYTES = 210;

fn matmulVecQ6_K_serial(out: []f32, weight_quant: []const u8, x: []const f32, N: usize, K: usize) void {
    const blk = Q6_K_BLOCK_SIZE;
    const blk_bytes = Q6_K_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;
    @memset(out[0..N], 0);
    var deq_buf: [Q6_K_BLOCK_SIZE]f32 = undefined;
    const tile_k: usize = 512;
    var k_base: usize = 0;
    while (k_base < K) : (k_base += tile_k) {
        const k_end = @min(k_base + tile_k, K);
        const bk_start = k_base / blk;
        for (0..N) |n| {
            const row_off = n * n_blocks * blk_bytes;
            var acc: f32 = 0;
            var bk = bk_start;
            var k = k_base;
            while (k < k_end) : (k += blk) {
                dequantQ6_K_block(weight_quant[row_off + bk * blk_bytes ..], &deq_buf);
                const k_local_end = @min(k + blk, k_end);
                acc += dotSimd(&deq_buf, x[k..k_local_end]);
                bk += 1;
            }
            out[n] += acc;
        }
    }
}

fn matmulVecQ6_K_parallel(out: []f32, weight_quant: []const u8, x: []const f32, N: usize, K: usize) void {
    const blk = Q6_K_BLOCK_SIZE;
    const n_blocks = (K + blk - 1) / blk;
    @memset(out[0..N], 0);
    const Ctx = struct {
        out: []f32,
        weight_quant: []const u8,
        x: []const f32,
        K: usize,
        n_blocks: usize,
        fn run(self: @This(), n: usize) void {
            const row_off = n * self.n_blocks * Q6_K_BLOCK_BYTES;
            var deq_buf: [Q6_K_BLOCK_SIZE]f32 = undefined;
            const tile_k: usize = 512;
            var k_base: usize = 0;
            while (k_base < self.K) : (k_base += tile_k) {
                const k_end = @min(k_base + tile_k, self.K);
                const bk_start = k_base / Q6_K_BLOCK_SIZE;
                var acc: f32 = 0;
                var bk = bk_start;
                var k = k_base;
                while (k < k_end) : (k += Q6_K_BLOCK_SIZE) {
                    dequantQ6_K_block(self.weight_quant[row_off + bk * Q6_K_BLOCK_BYTES ..], &deq_buf);
                    const k_local_end = @min(k + Q6_K_BLOCK_SIZE, k_end);
                    acc += dotSimd(&deq_buf, self.x[k..k_local_end]);
                    bk += 1;
                }
                self.out[n] += acc;
            }
        }
    };
    parallelForRows(N, Ctx{ .out = out, .weight_quant = weight_quant, .x = x, .K = K, .n_blocks = n_blocks }, Ctx.run);
}

/// Dequantiza una sola fila `row` (0-indexada) de un blob cuantizado Q4_0 con K columnas.
pub fn dequantRowQ4_0(weight_quant: []const u8, row: usize, out: []f32, K: usize) void {
    const blk = Q4_0_BLOCK_SIZE;
    const blk_bytes = Q4_0_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;
    const row_off = row * n_blocks * blk_bytes;
    var deq_buf: [Q4_0_BLOCK_SIZE]f32 = undefined;
    var k: usize = 0;
    var bk: usize = 0;
    while (k < K) : (k += blk) {
        dequantQ4_0_simd(weight_quant[row_off + bk * blk_bytes ..], &deq_buf);
        const k_end = @min(k + blk, K);
        for (k..k_end) |ki| out[ki] = deq_buf[ki - k];
        bk += 1;
    }
}

/// Dequantiza una sola fila de un blob cuantizado Q8_0.
pub fn dequantRowQ8_0(weight_quant: []const u8, row: usize, out: []f32, K: usize) void {
    const blk = Q8_0_BLOCK_SIZE;
    const blk_bytes = Q8_0_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;
    const row_off = row * n_blocks * blk_bytes;
    var deq_buf: [Q8_0_BLOCK_SIZE]f32 = undefined;
    var k: usize = 0;
    var bk: usize = 0;
    while (k < K) : (k += blk) {
        dequantQ8_0_simd(weight_quant[row_off + bk * blk_bytes ..], &deq_buf);
        const k_end = @min(k + blk, K);
        for (k..k_end) |ki| out[ki] = deq_buf[ki - k];
        bk += 1;
    }
}

/// Dequantiza una sola fila de un blob cuantizado Q4_K.
pub fn dequantRowQ4_K(weight_quant: []const u8, row: usize, out: []f32, K: usize) void {
    const blk = Q4_K_BLOCK_SIZE;
    const blk_bytes = Q4_K_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;
    const row_off = row * n_blocks * blk_bytes;
    var deq_buf: [Q4_K_BLOCK_SIZE]f32 = undefined;
    var k: usize = 0;
    var bk: usize = 0;
    while (k < K) : (k += blk) {
        dequantQ4_K_simd(weight_quant[row_off + bk * blk_bytes ..], &deq_buf);
        const k_end = @min(k + blk, K);
        for (k..k_end) |ki| out[ki] = deq_buf[ki - k];
        bk += 1;
    }
}

/// Dequantiza una sola fila de un blob cuantizado Q6_K.
pub fn dequantRowQ6_K(weight_quant: []const u8, row: usize, out: []f32, K: usize) void {
    const blk = Q6_K_BLOCK_SIZE;
    const blk_bytes = Q6_K_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;
    const row_off = row * n_blocks * blk_bytes;
    var deq_buf: [Q6_K_BLOCK_SIZE]f32 = undefined;
    var k: usize = 0;
    var bk: usize = 0;
    while (k < K) : (k += blk) {
        dequantQ6_K_simd(weight_quant[row_off + bk * blk_bytes ..], &deq_buf);
        const k_end = @min(k + blk, K);
        for (k..k_end) |ki| out[ki] = deq_buf[ki - k];
        bk += 1;
    }
}

pub fn matmulVecQ6_K(out: []f32, weight_quant: []const u8, x: []const f32, N: usize, K: usize) void {
    const threshold_n: usize = 256;
    const threshold_work: usize = 100_000;
    if (N >= threshold_n and N * K >= threshold_work) {
        matmulVecQ6_K_parallel(out, weight_quant, x, N, K);
    } else {
        matmulVecQ6_K_serial(out, weight_quant, x, N, K);
    }
}

/// Matmul batch cuantizado Q4_0: out[M,N] = inp[M,K] @ weight_quant[N,K]^T
/// Cada bloque del peso se dequantiza UNA vez y se aplica a los M tokens.
pub fn matmulBatchQ4_0(out: []f32, weight_quant: []const u8, inp: []const f32, M: usize, N: usize, K: usize) void {
    const blk = Q4_0_BLOCK_SIZE;
    const blk_bytes = Q4_0_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;
    @memset(out[0 .. M * N], 0);
    var deq_buf: [Q4_0_BLOCK_SIZE]f32 = undefined;
    for (0..N) |n| {
        const row_off = n * n_blocks * blk_bytes;
        var bk: usize = 0;
        var k: usize = 0;
        while (k < K) : (k += blk) {
            dequantQ4_0_simd(weight_quant[row_off + bk * blk_bytes ..], &deq_buf);
            const k_end = @min(k + blk, K);
            for (0..M) |m| {
                out[m * N + n] += dotSimd(deq_buf[0 .. (k_end - k)], inp[m * K + k .. m * K + k_end]);
            }
            bk += 1;
        }
    }
}

pub fn matmulBatchQ8_0(out: []f32, weight_quant: []const u8, inp: []const f32, M: usize, N: usize, K: usize) void {
    const blk = Q8_0_BLOCK_SIZE;
    const blk_bytes = Q8_0_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;
    @memset(out[0 .. M * N], 0);
    var deq_buf: [Q8_0_BLOCK_SIZE]f32 = undefined;
    for (0..N) |n| {
        const row_off = n * n_blocks * blk_bytes;
        var bk: usize = 0;
        var k: usize = 0;
        while (k < K) : (k += blk) {
            dequantQ8_0_simd(weight_quant[row_off + bk * blk_bytes ..], &deq_buf);
            const k_end = @min(k + blk, K);
            for (0..M) |m| {
                out[m * N + n] += dotSimd(deq_buf[0 .. (k_end - k)], inp[m * K + k .. m * K + k_end]);
            }
            bk += 1;
        }
    }
}

pub fn matmulBatchQ4_K(out: []f32, weight_quant: []const u8, inp: []const f32, M: usize, N: usize, K: usize) void {
    const blk = Q4_K_BLOCK_SIZE;
    const blk_bytes = Q4_K_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;
    @memset(out[0 .. M * N], 0);
    var deq_buf: [Q4_K_BLOCK_SIZE]f32 = undefined;
    for (0..N) |n| {
        const row_off = n * n_blocks * blk_bytes;
        var bk: usize = 0;
        var k: usize = 0;
        while (k < K) : (k += blk) {
            dequantQ4_K_block(weight_quant[row_off + bk * blk_bytes ..], &deq_buf);
            const k_end = @min(k + blk, K);
            for (0..M) |m| {
                out[m * N + n] += dotSimd(deq_buf[0 .. (k_end - k)], inp[m * K + k .. m * K + k_end]);
            }
            bk += 1;
        }
    }
}

pub fn matmulBatchQ6_K(out: []f32, weight_quant: []const u8, inp: []const f32, M: usize, N: usize, K: usize) void {
    const blk = Q6_K_BLOCK_SIZE;
    const blk_bytes = Q6_K_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;
    @memset(out[0 .. M * N], 0);
    var deq_buf: [Q6_K_BLOCK_SIZE]f32 = undefined;
    for (0..N) |n| {
        const row_off = n * n_blocks * blk_bytes;
        var bk: usize = 0;
        var k: usize = 0;
        while (k < K) : (k += blk) {
            dequantQ6_K_block(weight_quant[row_off + bk * blk_bytes ..], &deq_buf);
            const k_end = @min(k + blk, K);
            for (0..M) |m| {
                out[m * N + n] += dotSimd(deq_buf[0 .. (k_end - k)], inp[m * K + k .. m * K + k_end]);
            }
            bk += 1;
        }
    }
}

// --- tests de kernels especializados ---
test "scan_chunkwise vs ref" {
    const alloc = std.testing.allocator;
    const L, const h, const d = .{ 4, 8, 16 };
    const state = try alloc.alloc(f32, h * d);
    const inp = try alloc.alloc(f32, L * h * d);
    const out_ref = try alloc.alloc(f32, L * h * d);
    const out_fast = try alloc.alloc(f32, L * h * d);
    defer { alloc.free(state); alloc.free(inp); alloc.free(out_ref); alloc.free(out_fast); }
    var rng = std.Random.DefaultPrng.init(42);
    for (state) |*p| p.* = rng.random().float(f32);
    for (inp) |*p| p.* = rng.random().float(f32);
    scanChunkwiseRef(state, inp, out_ref, L, h, d);
    scanChunkwise(state, inp, out_fast, L, h, d);
    for (out_ref, out_fast) |r, f| try std.testing.expect(@abs(r - f) < 1e-5);
}

test "linear_attn_chunkwise vs ref" {
    const alloc = std.testing.allocator;
    const d = 16;
    const S_ref = try alloc.alloc(f32, d * d);
    const S_fast = try alloc.alloc(f32, d * d);
    const q = try alloc.alloc(f32, d);
    const k = try alloc.alloc(f32, d);
    const v = try alloc.alloc(f32, d);
    const out_ref = try alloc.alloc(f32, d);
    const out_fast = try alloc.alloc(f32, d);
    defer { alloc.free(S_ref); alloc.free(S_fast); alloc.free(q); alloc.free(k); alloc.free(v); alloc.free(out_ref); alloc.free(out_fast); }
    var rng = std.Random.DefaultPrng.init(42);
    for (S_ref, S_fast) |*r, *f| { const v_ = rng.random().float(f32); r.* = v_; f.* = v_; }
    for (q) |*p| p.* = rng.random().float(f32);
    for (k) |*p| p.* = rng.random().float(f32);
    for (v) |*p| p.* = rng.random().float(f32);
    linearAttnChunkwiseRef(S_ref, q, k, v, out_ref, d);
    linearAttnChunkwise(S_fast, q, k, v, out_fast, d);
    for (out_ref, out_fast) |r, f| try std.testing.expect(@abs(r - f) < 1e-5);
}

test "delta_wy_chunkwise vs ref" {
    const alloc = std.testing.allocator;
    const d = 16;
    const S_ref = try alloc.alloc(f32, d * d);
    const S_fast = try alloc.alloc(f32, d * d);
    const q = try alloc.alloc(f32, d);
    const delta = try alloc.alloc(f32, d);
    const beta = try alloc.alloc(f32, d);
    const out_ref = try alloc.alloc(f32, d);
    const out_fast = try alloc.alloc(f32, d);
    defer { alloc.free(S_ref); alloc.free(S_fast); alloc.free(q); alloc.free(delta); alloc.free(beta); alloc.free(out_ref); alloc.free(out_fast); }
    var rng = std.Random.DefaultPrng.init(42);
    for (S_ref, S_fast) |*r, *f| { const v = rng.random().float(f32); r.* = v; f.* = v; }
    for (q) |*p| p.* = rng.random().float(f32);
    for (delta) |*p| p.* = rng.random().float(f32);
    for (beta) |*p| p.* = rng.random().float(f32);
    deltaWyChunkwiseRef(S_ref, q, delta, beta, out_ref, d);
    deltaWyChunkwise(S_fast, q, delta, beta, out_fast, d);
    for (out_ref, out_fast) |r, f| try std.testing.expect(@abs(r - f) < 1e-5);
}

test "dequant Q4_0 roundtrip" {
    var block: [Q4_0_BLOCK_BYTES]u8 = undefined;
    const scale_f16: f16 = 0.5;
    std.mem.writeInt(u16, block[0..2], @bitCast(scale_f16), .little);
    for (0..Q4_0_BLOCK_SIZE) |i| {
        const byte_idx = 2 + i / 2;
        const nibble: u4 = @intCast(i % 16);
        if (i & 1 == 0) {
            block[byte_idx] = (block[byte_idx] & 0xF0) | nibble;
        } else {
            block[byte_idx] = (block[byte_idx] & 0x0F) | (@as(u8, nibble) << 4);
        }
    }
    var out: [Q4_0_BLOCK_SIZE]f32 = undefined;
    dequantQ4_0(&block, &out);
    // Q4_0 signed: nibble 0 → -8, nibble 8 → 0, nibble 9 → 1
    // out[i] = (nibble - 8) * scale
    try std.testing.expect(@abs(out[0] - (-8.0 * 0.5)) < 1e-4); // nibble 0 → -8
    try std.testing.expect(@abs(out[8] - (0.0 * 0.5)) < 1e-4);  // nibble 8 → 0
    try std.testing.expect(@abs(out[9] - (1.0 * 0.5)) < 1e-4);  // nibble 9 → 1
}

test "dequant Q8_0 roundtrip" {
    var block: [Q8_0_BLOCK_BYTES]u8 = undefined;
    const scale_f16: f16 = 2.0;
    std.mem.writeInt(u16, block[0..2], @bitCast(scale_f16), .little);
    for (0..Q8_0_BLOCK_SIZE) |i| {
        block[2 + i] = @intCast(i);
    }
    var out: [Q8_0_BLOCK_SIZE]f32 = undefined;
    dequantQ8_0(&block, &out);
    try std.testing.expectEqual(@as(f32, 0.0), out[0]);
    try std.testing.expect(@abs(out[1] - 2.0) < 1e-4);
    try std.testing.expect(@abs(out[2] - 4.0) < 1e-4);
}

test "quantize Q8_0 block roundtrip: f32 → Q8_0 → f32 aprox igual" {
    var src: [Q8_0_BLOCK_SIZE]f32 = undefined;
    var rng = std.Random.DefaultPrng.init(123);
    for (&src) |*p| p.* = (rng.random().float(f32) - 0.5) * 10.0;

    var block: [Q8_0_BLOCK_BYTES]u8 = undefined;
    quantizeQ8_0Block(&src, &block);

    var deq: [Q8_0_BLOCK_SIZE]f32 = undefined;
    dequantQ8_0(&block, &deq);

    // Q8_0 tiene ~8 bits de precisión → error relativo < 1/127 ≈ 0.8%
    for (src, deq) |s, d| {
        const abs_err = @abs(s - d);
        const max_abs = @max(@abs(s), 1e-4);
        const rel_err = abs_err / max_abs;
        // Permitir error absoluto pequeño O error relativo moderado
        try std.testing.expect(abs_err < 0.1 or rel_err < 0.02);
    }
}

test "quantize Q4_0 block roundtrip: f32 → Q4_0 → f32 aprox igual" {
    var src: [Q4_0_BLOCK_SIZE]f32 = undefined;
    var rng = std.Random.DefaultPrng.init(456);
    for (&src) |*p| p.* = (rng.random().float(f32) - 0.5) * 4.0;

    var block: [Q4_0_BLOCK_BYTES]u8 = undefined;
    quantizeQ4_0Block(&src, &block);

    var deq: [Q4_0_BLOCK_SIZE]f32 = undefined;
    dequantQ4_0(&block, &deq);

    // Q4_0 tiene ~4 bits de precisión → step = max_abs/7, error máx ≈ step/2
    for (src, deq) |s, d| {
        const abs_err = @abs(s - d);
        const max_abs = @max(@abs(s), 1e-4);
        const rel_err = abs_err / max_abs;
        // Q4_0: error relativo teórico hasta ~1/7 ≈ 14%, pero puede ser mayor
        // para valores cercanos a cero. Permitimos abs_err < 0.5 O rel_err < 0.3
        try std.testing.expect(abs_err < 0.5 or rel_err < 0.3);
    }
}

test "quantize Q8_0 matrix: shape correcto y roundtrip por fila" {
    const alloc = std.testing.allocator;
    const N, const K = .{ 4, 64 };
    const n_blocks = (K + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
    const total_bytes = N * n_blocks * Q8_0_BLOCK_BYTES;

    const src = try alloc.alloc(f32, N * K);
    defer alloc.free(src);
    var rng = std.Random.DefaultPrng.init(789);
    for (src) |*p| p.* = (rng.random().float(f32) - 0.5) * 5.0;

    const dst = try alloc.alloc(u8, total_bytes);
    defer alloc.free(dst);

    const written = quantizeQ8_0Matrix(dst, src, N, K);
    try std.testing.expectEqual(total_bytes, written);

    // Verificar roundtrip por fila
    const row = try alloc.alloc(f32, K);
    defer alloc.free(row);
    for (0..N) |n| {
        dequantRowQ8_0(dst, n, row, K);
        for (0..K) |k| {
            const abs_err = @abs(src[n * K + k] - row[k]);
            const max_abs = @max(@abs(src[n * K + k]), 1e-4);
            const rel_err = abs_err / max_abs;
            try std.testing.expect(abs_err < 0.1 or rel_err < 0.02);
        }
    }
}

test "matmulQ4_0 vs matmul f32" {
    const alloc = std.testing.allocator;
    const M, const K, const N = .{ 2, 32, 4 };
    const blk = Q4_0_BLOCK_SIZE;
    const blk_bytes = Q4_0_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;

    var A_quant = try alloc.alloc(u8, M * n_blocks * blk_bytes);
    defer alloc.free(A_quant);
    for (0..M) |m| {
        const scale_f16: f16 = 1.0;
        std.mem.writeInt(u16, A_quant[m * n_blocks * blk_bytes ..][0..2], @bitCast(scale_f16), .little);
        for (0..K) |i| {
            const byte_off = m * n_blocks * blk_bytes + 2 + i / 2;
            const nibble: u4 = @intCast(i % 15 + 1);
            if (i & 1 == 0) {
                A_quant[byte_off] = (A_quant[byte_off] & 0xF0) | nibble;
            } else {
                A_quant[byte_off] = (A_quant[byte_off] & 0x0F) | (@as(u8, nibble) << 4);
            }
        }
    }
    const B = try alloc.alloc(f32, K * N);
    defer alloc.free(B);
    for (B, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i + 1) % 5)) / 5.0;
    const C_fast = try alloc.alloc(f32, M * N);
    const C_ref = try alloc.alloc(f32, M * N);
    defer alloc.free(C_fast);
    defer alloc.free(C_ref);

    matmulQ4_0(C_fast, A_quant, B, M, K, N);

    const A_f32 = try alloc.alloc(f32, M * K);
    defer alloc.free(A_f32);
    for (0..M) |m| {
        dequantQ4_0(A_quant[m * n_blocks * blk_bytes..], A_f32[m * K ..][0..K]);
    }
    matmulRef(C_ref, A_f32, B, M, K, N);

    for (C_ref, C_fast) |r, f| try std.testing.expect(@abs(r - f) < 1e-4);
}

test "matmulVecQ4_0 vs manual dequant" {
    const alloc = std.testing.allocator;
    const N, const K = .{ 4, 32 };
    const blk = Q4_0_BLOCK_SIZE;
    const blk_bytes = Q4_0_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk; // = 1

    var weight_quant = try alloc.alloc(u8, N * n_blocks * blk_bytes);
    defer alloc.free(weight_quant);
    for (0..N) |n| {
        const scale_f16: f16 = 1.0;
        std.mem.writeInt(u16, weight_quant[n * n_blocks * blk_bytes ..][0..2], @bitCast(scale_f16), .little);
        for (0..K) |i| {
            const byte_off = n * n_blocks * blk_bytes + 2 + i / 2;
            const nibble: u4 = @intCast((i % 15) + 1);
            if (i & 1 == 0) {
                weight_quant[byte_off] = (weight_quant[byte_off] & 0xF0) | nibble;
            } else {
                weight_quant[byte_off] = (weight_quant[byte_off] & 0x0F) | (@as(u8, nibble) << 4);
            }
        }
    }

    const x = try alloc.alloc(f32, K);
    defer alloc.free(x);
    for (x, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i + 1) % 5)) / 5.0;

    const out_fast = try alloc.alloc(f32, N);
    defer alloc.free(out_fast);

    matmulVecQ4_0(out_fast, weight_quant, x, N, K);

    // Reference: dequantizar cada fila y hacer dot con x
    const out_ref = try alloc.alloc(f32, N);
    defer alloc.free(out_ref);
    var deq_buf: [Q4_0_BLOCK_SIZE]f32 = undefined;
    for (0..N) |n| {
        dequantQ4_0(weight_quant[n * n_blocks * blk_bytes ..], &deq_buf);
        var acc: f32 = 0;
        for (0..K) |k| acc += deq_buf[k] * x[k];
        out_ref[n] = acc;
    }

    for (out_ref, out_fast) |r, f| try std.testing.expect(@abs(r - f) < 1e-4);
}

test "matmulVecQ8_0 vs manual dequant" {
    const alloc = std.testing.allocator;
    const N, const K = .{ 4, 32 };
    const blk = Q8_0_BLOCK_SIZE;
    const blk_bytes = Q8_0_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;

    var weight_quant = try alloc.alloc(u8, N * n_blocks * blk_bytes);
    defer alloc.free(weight_quant);
    for (0..N) |n| {
        const scale_f16: f16 = 0.5;
        std.mem.writeInt(u16, weight_quant[n * n_blocks * blk_bytes ..][0..2], @bitCast(scale_f16), .little);
        for (0..K) |i| {
            weight_quant[n * n_blocks * blk_bytes + 2 + i] = @intCast(i % 50);
        }
    }

    const x = try alloc.alloc(f32, K);
    defer alloc.free(x);
    for (x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 3)) / 3.0;

    const out_fast = try alloc.alloc(f32, N);
    defer alloc.free(out_fast);

    matmulVecQ8_0(out_fast, weight_quant, x, N, K);

    const out_ref = try alloc.alloc(f32, N);
    defer alloc.free(out_ref);
    var deq_buf: [Q8_0_BLOCK_SIZE]f32 = undefined;
    for (0..N) |n| {
        dequantQ8_0(weight_quant[n * n_blocks * blk_bytes ..], &deq_buf);
        var acc: f32 = 0;
        for (0..K) |k| acc += deq_buf[k] * x[k];
        out_ref[n] = acc;
    }

    for (out_ref, out_fast) |r, f| try std.testing.expect(@abs(r - f) < 1e-4);
}

test "matmulVecQ4_K vs manual dequant" {
    const alloc = std.testing.allocator;
    const N, const K = .{ 3, 256 };
    const blk = Q4_K_BLOCK_SIZE; // 256
    const blk_bytes = Q4_K_BLOCK_BYTES; // 144
    const n_blocks = (K + blk - 1) / blk; // = 1

    // Crear datos cuantizados Q4_K por fila
    var weight_quant = try alloc.alloc(u8, N * n_blocks * blk_bytes);
    defer alloc.free(weight_quant);
    for (0..N) |n| {
        const base = n * n_blocks * blk_bytes;
        const scale_f16: f16 = 0.5;
        const dmin_f16: f16 = 0.1;
        std.mem.writeInt(u16, weight_quant[base..][0..2], @bitCast(scale_f16), .little);
        std.mem.writeInt(u16, weight_quant[base..][2..4], @bitCast(dmin_f16), .little);
        for (0..12) |i| weight_quant[base + 4 + i] = @intCast(i + 1);
        for (0..128) |i| weight_quant[base + 16 + i] = @intCast(i % 16);
    }

    const x = try alloc.alloc(f32, K);
    defer alloc.free(x);
    for (x, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i + 1) % 7)) / 7.0;

    const out_fast = try alloc.alloc(f32, N);
    defer alloc.free(out_fast);
    matmulVecQ4_K(out_fast, weight_quant, x, N, K);

    const out_ref = try alloc.alloc(f32, N);
    defer alloc.free(out_ref);
    var deq_buf: [Q4_K_BLOCK_SIZE]f32 = undefined;
    for (0..N) |n| {
        dequantQ4_K_block(weight_quant[n * n_blocks * blk_bytes ..], &deq_buf);
        var acc: f32 = 0;
        for (0..K) |k| acc += deq_buf[k] * x[k];
        out_ref[n] = acc;
    }

    for (out_ref, out_fast) |r, f| try std.testing.expect(@abs(r - f) < 1e-3);
}

test "matmulVecQ6_K vs manual dequant" {
    const alloc = std.testing.allocator;
    const N, const K = .{ 2, 256 };
    const blk = Q6_K_BLOCK_SIZE; // 256
    const blk_bytes = Q6_K_BLOCK_BYTES; // 210
    const n_blocks = (K + blk - 1) / blk;

    var weight_quant = try alloc.alloc(u8, N * n_blocks * blk_bytes);
    defer alloc.free(weight_quant);
    for (0..N) |n| {
        const base = n * n_blocks * blk_bytes;
        for (0..208) |i| weight_quant[base + i] = @intCast((i + n * 7) % 256);
        const scale_f16: f16 = 1.0;
        std.mem.writeInt(u16, weight_quant[base..][208..210], @bitCast(scale_f16), .little);
    }

    const x = try alloc.alloc(f32, K);
    defer alloc.free(x);
    for (x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 5)) / 5.0;

    const out_fast = try alloc.alloc(f32, N);
    defer alloc.free(out_fast);
    matmulVecQ6_K(out_fast, weight_quant, x, N, K);

    const out_ref = try alloc.alloc(f32, N);
    defer alloc.free(out_ref);
    var deq_buf: [Q6_K_BLOCK_SIZE]f32 = undefined;
    for (0..N) |n| {
        dequantQ6_K_block(weight_quant[n * n_blocks * blk_bytes ..], &deq_buf);
        var acc: f32 = 0;
        for (0..K) |k| acc += deq_buf[k] * x[k];
        out_ref[n] = acc;
    }

    for (out_ref, out_fast) |r, f| try std.testing.expect(@abs(r - f) < 5e-3);
}

test "matmulVecQ4_0 SIMD vs ref" {
    const alloc = std.testing.allocator;
    const N, const K = .{ 4, 64 };
    const blk = Q4_0_BLOCK_SIZE;
    const blk_bytes = Q4_0_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;

    var weight_quant = try alloc.alloc(u8, N * n_blocks * blk_bytes);
    defer alloc.free(weight_quant);
    for (0..N) |n| {
        for (0..n_blocks) |b| {
            const base = (n * n_blocks + b) * blk_bytes;
            const scale_f16: f16 = 1.0;
            std.mem.writeInt(u16, weight_quant[base..][0..2], @bitCast(scale_f16), .little);
            for (0..blk) |i| {
                const byte_off = base + 2 + i / 2;
                const nibble: u4 = @intCast((i % 15) + 1);
                if (i & 1 == 0) {
                    weight_quant[byte_off] = (weight_quant[byte_off] & 0xF0) | nibble;
                } else {
                    weight_quant[byte_off] = (weight_quant[byte_off] & 0x0F) | (@as(u8, nibble) << 4);
                }
            }
        }
    }
    const x = try alloc.alloc(f32, K);
    defer alloc.free(x);
    for (x, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i + 1) % 5)) / 5.0;

    const out_simd = try alloc.alloc(f32, N);
    const out_ref = try alloc.alloc(f32, N);
    defer alloc.free(out_simd);
    defer alloc.free(out_ref);

    matmulVecQ4_0(out_simd, weight_quant, x, N, K);
    matmulVecQ4_0Ref(out_ref, weight_quant, x, N, K);

    for (out_ref, out_simd) |r, f| try std.testing.expect(@abs(r - f) < 1e-4);
}

test "matmulVecQ8_0 SIMD vs ref" {
    const alloc = std.testing.allocator;
    const N, const K = .{ 4, 64 };
    const blk = Q8_0_BLOCK_SIZE;
    const blk_bytes = Q8_0_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;

    var weight_quant = try alloc.alloc(u8, N * n_blocks * blk_bytes);
    defer alloc.free(weight_quant);
    for (0..N) |n| {
        for (0..n_blocks) |b| {
            const base = (n * n_blocks + b) * blk_bytes;
            const scale_f16: f16 = 0.5;
            std.mem.writeInt(u16, weight_quant[base..][0..2], @bitCast(scale_f16), .little);
            for (0..blk) |i| {
                weight_quant[base + 2 + i] = @intCast(i % 50);
            }
        }
    }
    const x = try alloc.alloc(f32, K);
    defer alloc.free(x);
    for (x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 3)) / 3.0;

    const out_simd = try alloc.alloc(f32, N);
    const out_ref = try alloc.alloc(f32, N);
    defer alloc.free(out_simd);
    defer alloc.free(out_ref);

    matmulVecQ8_0(out_simd, weight_quant, x, N, K);
    matmulVecQ8_0Ref(out_ref, weight_quant, x, N, K);

    for (out_ref, out_simd) |r, f| try std.testing.expect(@abs(r - f) < 1e-4);
}

test "matmulVecQ4_K SIMD vs ref" {
    const alloc = std.testing.allocator;
    const N, const K = .{ 3, 512 };
    const blk = Q4_K_BLOCK_SIZE;
    const blk_bytes = Q4_K_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;

    var weight_quant = try alloc.alloc(u8, N * n_blocks * blk_bytes);
    defer alloc.free(weight_quant);
    for (0..N) |n| {
        for (0..n_blocks) |b| {
            const base = (n * n_blocks + b) * blk_bytes;
            const scale_f16: f16 = 0.5;
            const dmin_f16: f16 = 0.1;
            std.mem.writeInt(u16, weight_quant[base..][0..2], @bitCast(scale_f16), .little);
            std.mem.writeInt(u16, weight_quant[base..][2..4], @bitCast(dmin_f16), .little);
            for (0..12) |i| weight_quant[base + 4 + i] = @intCast(i + 1);
            for (0..128) |i| weight_quant[base + 16 + i] = @intCast(i % 16);
        }
    }
    const x = try alloc.alloc(f32, K);
    defer alloc.free(x);
    for (x, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i + 1) % 7)) / 7.0;

    const out_simd = try alloc.alloc(f32, N);
    const out_ref = try alloc.alloc(f32, N);
    defer alloc.free(out_simd);
    defer alloc.free(out_ref);

    matmulVecQ4_K(out_simd, weight_quant, x, N, K);
    matmulVecQ4_KRef(out_ref, weight_quant, x, N, K);

    for (out_ref, out_simd) |r, f| try std.testing.expect(@abs(r - f) < 5e-3);
}

test "matmulVecQ6_K SIMD vs ref" {
    const alloc = std.testing.allocator;
    const N, const K = .{ 2, 512 };
    const blk = Q6_K_BLOCK_SIZE;
    const blk_bytes = Q6_K_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;

    var weight_quant = try alloc.alloc(u8, N * n_blocks * blk_bytes);
    defer alloc.free(weight_quant);
    for (0..N) |n| {
        for (0..n_blocks) |b| {
            const base = (n * n_blocks + b) * blk_bytes;
            for (0..208) |i| weight_quant[base + i] = @intCast((i + n * 7 + b * 13) % 256);
            const scale_f16: f16 = 1.0;
            std.mem.writeInt(u16, weight_quant[base..][208..210], @bitCast(scale_f16), .little);
        }
    }
    const x = try alloc.alloc(f32, K);
    defer alloc.free(x);
    for (x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 5)) / 5.0;

    const out_simd = try alloc.alloc(f32, N);
    const out_ref = try alloc.alloc(f32, N);
    defer alloc.free(out_simd);
    defer alloc.free(out_ref);

    matmulVecQ6_K(out_simd, weight_quant, x, N, K);
    matmulVecQ6_KRef(out_ref, weight_quant, x, N, K);

    for (out_ref, out_simd) |r, f| try std.testing.expect(@abs(r - f) < 1e-2);
}

test "matmulQ8_0 vs matmul f32" {
    const alloc = std.testing.allocator;
    const M, const K, const N = .{ 2, 32, 4 };
    const blk = Q8_0_BLOCK_SIZE;
    const blk_bytes = Q8_0_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;

    var A_quant = try alloc.alloc(u8, M * n_blocks * blk_bytes);
    defer alloc.free(A_quant);
    for (0..M) |m| {
        const scale_f16: f16 = 0.5;
        std.mem.writeInt(u16, A_quant[m * n_blocks * blk_bytes ..][0..2], @bitCast(scale_f16), .little);
        for (0..K) |i| {
            A_quant[m * n_blocks * blk_bytes + 2 + i] = @intCast(i % 50);
        }
    }
    const B = try alloc.alloc(f32, K * N);
    defer alloc.free(B);
    for (B, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 3)) / 3.0;
    const C_fast = try alloc.alloc(f32, M * N);
    const C_ref = try alloc.alloc(f32, M * N);
    defer alloc.free(C_fast);
    defer alloc.free(C_ref);

    matmulQ8_0(C_fast, A_quant, B, M, K, N);

    const A_f32 = try alloc.alloc(f32, M * K);
    defer alloc.free(A_f32);
    for (0..M) |m| {
        dequantQ8_0(A_quant[m * n_blocks * blk_bytes..], A_f32[m * K ..][0..K]);
    }
    matmulRef(C_ref, A_f32, B, M, K, N);

    for (C_ref, C_fast) |r, f| try std.testing.expect(@abs(r - f) < 1e-4);
}

test "dequant Q8_0 SIMD vs ref" {
    var block: [Q8_0_BLOCK_BYTES]u8 = undefined;
    const scale_f16: f16 = 2.0;
    std.mem.writeInt(u16, block[0..2], @bitCast(scale_f16), .little);
    for (0..Q8_0_BLOCK_SIZE) |i| {
        block[2 + i] = @intCast((i + 7) % 50);
    }
    var out_ref: [Q8_0_BLOCK_SIZE]f32 = undefined;
    var out_simd: [Q8_0_BLOCK_SIZE]f32 = undefined;
    dequantQ8_0(&block, &out_ref);
    dequantQ8_0_simd(&block, &out_simd);
    for (out_ref, out_simd) |r, f| {
        try std.testing.expect(@abs(r - f) < 1e-6);
    }
}

test "dequant Q4_0 SIMD vs ref" {
    var block: [Q4_0_BLOCK_BYTES]u8 = undefined;
    const scale_f16: f16 = 0.75;
    std.mem.writeInt(u16, block[0..2], @bitCast(scale_f16), .little);
    for (0..Q4_0_BLOCK_SIZE) |i| {
        const byte_idx = 2 + i / 2;
        const nibble: u4 = @intCast(i % 16);
        if (i & 1 == 0) {
            block[byte_idx] = (block[byte_idx] & 0xF0) | nibble;
        } else {
            block[byte_idx] = (block[byte_idx] & 0x0F) | (@as(u8, nibble) << 4);
        }
    }
    var out_ref: [Q4_0_BLOCK_SIZE]f32 = undefined;
    var out_simd: [Q4_0_BLOCK_SIZE]f32 = undefined;
    dequantQ4_0(&block, &out_ref);
    dequantQ4_0_simd(&block, &out_simd);
    for (out_ref, out_simd) |r, f| {
        try std.testing.expect(@abs(r - f) < 1e-6);
    }
}

test "dequant Q4_K SIMD vs ref" {
    const alloc = std.testing.allocator;
    const blk_bytes = Q4_K_BLOCK_BYTES;
    var block = try alloc.alloc(u8, blk_bytes);
    defer alloc.free(block);
    const scale_f16: f16 = 0.5;
    const dmin_f16: f16 = 0.1;
    std.mem.writeInt(u16, block[0..2], @bitCast(scale_f16), .little);
    std.mem.writeInt(u16, block[2..4], @bitCast(dmin_f16), .little);
    for (0..12) |i| block[4 + i] = @intCast(i + 1);
    for (0..128) |i| block[16 + i] = @intCast((i + 3) % 16);

    var out_ref: [Q4_K_BLOCK_SIZE]f32 = undefined;
    var out_simd: [Q4_K_BLOCK_SIZE]f32 = undefined;
    dequantQ4_K_block(block, &out_ref);
    dequantQ4_K_simd(block, &out_simd);
    for (out_ref, out_simd) |r, f| {
        try std.testing.expect(@abs(r - f) < 1e-6);
    }
}

test "dequant Q6_K SIMD vs ref" {
    const alloc = std.testing.allocator;
    const blk_bytes = Q6_K_BLOCK_BYTES;
    var block = try alloc.alloc(u8, blk_bytes);
    defer alloc.free(block);
    for (0..208) |i| block[i] = @intCast((i * 7 + 3) % 256);
    const scale_f16: f16 = 1.0;
    std.mem.writeInt(u16, block[208..210], @bitCast(scale_f16), .little);

    var out_ref: [Q6_K_BLOCK_SIZE]f32 = undefined;
    var out_simd: [Q6_K_BLOCK_SIZE]f32 = undefined;
    dequantQ6_K_block(block, &out_ref);
    dequantQ6_K_simd(block, &out_simd);
    for (out_ref, out_simd) |r, f| {
        try std.testing.expect(@abs(r - f) < 1e-5);
    }
}

test "matmulVecQ4_0 parallel vs ref" {
    const alloc = std.testing.allocator;
    const N, const K = .{ 512, 256 };
    const blk = Q4_0_BLOCK_SIZE;
    const blk_bytes = Q4_0_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;

    var weight_quant = try alloc.alloc(u8, N * n_blocks * blk_bytes);
    defer alloc.free(weight_quant);
    for (0..N) |n| {
        for (0..n_blocks) |b| {
            const base = (n * n_blocks + b) * blk_bytes;
            const scale_f16: f16 = 1.0;
            std.mem.writeInt(u16, weight_quant[base..][0..2], @bitCast(scale_f16), .little);
            for (0..blk) |i| {
                const byte_off = base + 2 + i / 2;
                const nibble: u4 = @intCast((i % 15) + 1);
                if (i & 1 == 0) {
                    weight_quant[byte_off] = (weight_quant[byte_off] & 0xF0) | nibble;
                } else {
                    weight_quant[byte_off] = (weight_quant[byte_off] & 0x0F) | (@as(u8, nibble) << 4);
                }
            }
        }
    }
    const x = try alloc.alloc(f32, K);
    defer alloc.free(x);
    for (x, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i + 1) % 5)) / 5.0;

    const out_fast = try alloc.alloc(f32, N);
    const out_ref = try alloc.alloc(f32, N);
    defer alloc.free(out_fast);
    defer alloc.free(out_ref);

    matmulVecQ4_0(out_fast, weight_quant, x, N, K);
    matmulVecQ4_0Ref(out_ref, weight_quant, x, N, K);

    for (out_ref, out_fast) |r, f| try std.testing.expect(@abs(r - f) < 1e-4);
}

test "matmulVecQ8_0 parallel vs ref" {
    const alloc = std.testing.allocator;
    const N, const K = .{ 512, 256 };
    const blk = Q8_0_BLOCK_SIZE;
    const blk_bytes = Q8_0_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;

    var weight_quant = try alloc.alloc(u8, N * n_blocks * blk_bytes);
    defer alloc.free(weight_quant);
    for (0..N) |n| {
        for (0..n_blocks) |b| {
            const base = (n * n_blocks + b) * blk_bytes;
            const scale_f16: f16 = 0.5;
            std.mem.writeInt(u16, weight_quant[base..][0..2], @bitCast(scale_f16), .little);
            for (0..blk) |i| {
                weight_quant[base + 2 + i] = @intCast((i + n * 3 + b * 7) % 50);
            }
        }
    }
    const x = try alloc.alloc(f32, K);
    defer alloc.free(x);
    for (x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 3)) / 3.0;

    const out_fast = try alloc.alloc(f32, N);
    const out_ref = try alloc.alloc(f32, N);
    defer alloc.free(out_fast);
    defer alloc.free(out_ref);

    matmulVecQ8_0(out_fast, weight_quant, x, N, K);
    matmulVecQ8_0Ref(out_ref, weight_quant, x, N, K);

    for (out_ref, out_fast) |r, f| try std.testing.expect(@abs(r - f) < 5e-3);
}

test "matmulVecQ4_K parallel vs ref" {
    const alloc = std.testing.allocator;
    const N, const K = .{ 512, 512 };
    const blk = Q4_K_BLOCK_SIZE;
    const blk_bytes = Q4_K_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;

    var weight_quant = try alloc.alloc(u8, N * n_blocks * blk_bytes);
    defer alloc.free(weight_quant);
    for (0..N) |n| {
        for (0..n_blocks) |b| {
            const base = (n * n_blocks + b) * blk_bytes;
            const scale_f16: f16 = 0.5;
            const dmin_f16: f16 = 0.1;
            std.mem.writeInt(u16, weight_quant[base..][0..2], @bitCast(scale_f16), .little);
            std.mem.writeInt(u16, weight_quant[base..][2..4], @bitCast(dmin_f16), .little);
            for (0..12) |i| weight_quant[base + 4 + i] = @intCast(i + 1);
            for (0..128) |i| weight_quant[base + 16 + i] = @intCast((i + n + b) % 16);
        }
    }
    const x = try alloc.alloc(f32, K);
    defer alloc.free(x);
    for (x, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i + 1) % 7)) / 7.0;

    const out_fast = try alloc.alloc(f32, N);
    const out_ref = try alloc.alloc(f32, N);
    defer alloc.free(out_fast);
    defer alloc.free(out_ref);

    matmulVecQ4_K(out_fast, weight_quant, x, N, K);
    matmulVecQ4_KRef(out_ref, weight_quant, x, N, K);

    for (out_ref, out_fast) |r, f| try std.testing.expect(@abs(r - f) < 5e-3);
}

test "dequantRowQ4_0 vs manual dequant" {
    const alloc = std.testing.allocator;
    const N, const K = .{ 4, 64 };
    const blk = Q4_0_BLOCK_SIZE;
    const blk_bytes = Q4_0_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;

    var weight_quant = try alloc.alloc(u8, N * n_blocks * blk_bytes);
    defer alloc.free(weight_quant);
    for (0..N) |n| {
        for (0..n_blocks) |b| {
            const base = (n * n_blocks + b) * blk_bytes;
            const scale_f16: f16 = 1.0;
            std.mem.writeInt(u16, weight_quant[base..][0..2], @bitCast(scale_f16), .little);
            for (0..blk) |i| {
                const byte_off = base + 2 + i / 2;
                const nibble: u4 = @intCast((i % 15) + 1);
                if (i & 1 == 0) {
                    weight_quant[byte_off] = (weight_quant[byte_off] & 0xF0) | nibble;
                } else {
                    weight_quant[byte_off] = (weight_quant[byte_off] & 0x0F) | (@as(u8, nibble) << 4);
                }
            }
        }
    }

    const out_row = try alloc.alloc(f32, K);
    defer alloc.free(out_row);
    dequantRowQ4_0(weight_quant, 2, out_row, K);

    // Oráculo: dequantizar bloque a bloque la fila 2
    var deq_buf: [Q4_0_BLOCK_SIZE]f32 = undefined;
    var oracle = try alloc.alloc(f32, K);
    defer alloc.free(oracle);
    for (0..n_blocks) |b| {
        dequantQ4_0(weight_quant[(2 * n_blocks + b) * blk_bytes ..], &deq_buf);
        const k_start = b * blk;
        const k_end = @min(k_start + blk, K);
        for (k_start..k_end) |k| oracle[k] = deq_buf[k - k_start];
    }

    for (out_row, oracle) |r, o| try std.testing.expect(@abs(r - o) < 1e-6);
}

test "dequantRowQ8_0 vs manual dequant" {
    const alloc = std.testing.allocator;
    const N, const K = .{ 3, 64 };
    const blk = Q8_0_BLOCK_SIZE;
    const blk_bytes = Q8_0_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;

    var weight_quant = try alloc.alloc(u8, N * n_blocks * blk_bytes);
    defer alloc.free(weight_quant);
    for (0..N) |n| {
        for (0..n_blocks) |b| {
            const base = (n * n_blocks + b) * blk_bytes;
            const scale_f16: f16 = 0.5;
            std.mem.writeInt(u16, weight_quant[base..][0..2], @bitCast(scale_f16), .little);
            for (0..blk) |i| {
                weight_quant[base + 2 + i] = @intCast((i + n * 3 + b * 7) % 50);
            }
        }
    }

    const out_row = try alloc.alloc(f32, K);
    defer alloc.free(out_row);
    dequantRowQ8_0(weight_quant, 1, out_row, K);

    var deq_buf: [Q8_0_BLOCK_SIZE]f32 = undefined;
    var oracle = try alloc.alloc(f32, K);
    defer alloc.free(oracle);
    for (0..n_blocks) |b| {
        dequantQ8_0(weight_quant[(1 * n_blocks + b) * blk_bytes ..], &deq_buf);
        const k_start = b * blk;
        const k_end = @min(k_start + blk, K);
        for (k_start..k_end) |k| oracle[k] = deq_buf[k - k_start];
    }

    for (out_row, oracle) |r, o| try std.testing.expect(@abs(r - o) < 1e-6);
}

test "dequantRowQ4_K vs manual dequant" {
    const alloc = std.testing.allocator;
    const N, const K = .{ 3, 256 };
    const blk = Q4_K_BLOCK_SIZE;
    const blk_bytes = Q4_K_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;

    var weight_quant = try alloc.alloc(u8, N * n_blocks * blk_bytes);
    defer alloc.free(weight_quant);
    for (0..N) |n| {
        for (0..n_blocks) |b| {
            const base = (n * n_blocks + b) * blk_bytes;
            const scale_f16: f16 = 0.5;
            const dmin_f16: f16 = 0.1;
            std.mem.writeInt(u16, weight_quant[base..][0..2], @bitCast(scale_f16), .little);
            std.mem.writeInt(u16, weight_quant[base..][2..4], @bitCast(dmin_f16), .little);
            for (0..12) |i| weight_quant[base + 4 + i] = @intCast(i + 1);
            for (0..128) |i| weight_quant[base + 16 + i] = @intCast((i + n + b) % 16);
        }
    }

    const out_row = try alloc.alloc(f32, K);
    defer alloc.free(out_row);
    dequantRowQ4_K(weight_quant, 2, out_row, K);

    var deq_buf: [Q4_K_BLOCK_SIZE]f32 = undefined;
    var oracle = try alloc.alloc(f32, K);
    defer alloc.free(oracle);
    for (0..n_blocks) |b| {
        dequantQ4_K_block(weight_quant[(2 * n_blocks + b) * blk_bytes ..], &deq_buf);
        const k_start = b * blk;
        const k_end = @min(k_start + blk, K);
        for (k_start..k_end) |k| oracle[k] = deq_buf[k - k_start];
    }

    for (out_row, oracle) |r, o| try std.testing.expect(@abs(r - o) < 1e-6);
}

test "dequantRowQ6_K vs manual dequant" {
    const alloc = std.testing.allocator;
    const N, const K = .{ 2, 256 };
    const blk = Q6_K_BLOCK_SIZE;
    const blk_bytes = Q6_K_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;

    var weight_quant = try alloc.alloc(u8, N * n_blocks * blk_bytes);
    defer alloc.free(weight_quant);
    for (0..N) |n| {
        for (0..n_blocks) |b| {
            const base = (n * n_blocks + b) * blk_bytes;
            for (0..208) |i| weight_quant[base + i] = @intCast((i + n * 7 + b * 13) % 256);
            const scale_f16: f16 = 1.0;
            std.mem.writeInt(u16, weight_quant[base..][208..210], @bitCast(scale_f16), .little);
        }
    }

    const out_row = try alloc.alloc(f32, K);
    defer alloc.free(out_row);
    dequantRowQ6_K(weight_quant, 0, out_row, K);

    var deq_buf: [Q6_K_BLOCK_SIZE]f32 = undefined;
    var oracle = try alloc.alloc(f32, K);
    defer alloc.free(oracle);
    for (0..n_blocks) |b| {
        dequantQ6_K_block(weight_quant[(0 * n_blocks + b) * blk_bytes ..], &deq_buf);
        const k_start = b * blk;
        const k_end = @min(k_start + blk, K);
        for (k_start..k_end) |k| oracle[k] = deq_buf[k - k_start];
    }

    for (out_row, oracle) |r, o| try std.testing.expect(@abs(r - o) < 1e-6);
}

test "matmulBatchQ4_0 vs M×matmulVecQ4_0" {
    const alloc = std.testing.allocator;
    const N, const K, const M = .{ 64, 64, 3 };
    const blk = Q4_0_BLOCK_SIZE;
    const blk_bytes = Q4_0_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;

    var weight_quant = try alloc.alloc(u8, N * n_blocks * blk_bytes);
    defer alloc.free(weight_quant);
    for (0..N) |n| {
        for (0..n_blocks) |b| {
            const base = (n * n_blocks + b) * blk_bytes;
            const scale_f16: f16 = @floatCast(0.5 + @as(f32, @floatFromInt(n % 4)) * 0.1);
            std.mem.writeInt(u16, weight_quant[base..][0..2], @bitCast(scale_f16), .little);
            for (0..blk / 2) |i| weight_quant[base + 2 + i] = @intCast((i + n + b) % 256);
        }
    }

    var inp = try alloc.alloc(f32, M * K);
    defer alloc.free(inp);
    for (0..M * K) |i| inp[i] = @as(f32, @floatFromInt((i % 7) + 1)) * 0.1;

    const out_batch = try alloc.alloc(f32, M * N);
    defer alloc.free(out_batch);
    matmulBatchQ4_0(out_batch, weight_quant, inp, M, N, K);

    var out_ref = try alloc.alloc(f32, M * N);
    defer alloc.free(out_ref);
    for (0..M) |m| matmulVecQ4_0(out_ref[m * N ..][0..N], weight_quant, inp[m * K ..][0..K], N, K);

    for (out_batch, out_ref) |b, r| try std.testing.expect(@abs(b - r) < 1e-4);
}

test "matmulBatchQ8_0 vs M×matmulVecQ8_0" {
    const alloc = std.testing.allocator;
    const N, const K, const M = .{ 64, 64, 3 };
    const blk = Q8_0_BLOCK_SIZE;
    const blk_bytes = Q8_0_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;

    var weight_quant = try alloc.alloc(u8, N * n_blocks * blk_bytes);
    defer alloc.free(weight_quant);
    for (0..N) |n| {
        for (0..n_blocks) |b| {
            const base = (n * n_blocks + b) * blk_bytes;
            const scale_f16: f16 = @floatCast(0.5 + @as(f32, @floatFromInt(n % 4)) * 0.1);
            std.mem.writeInt(u16, weight_quant[base..][0..2], @bitCast(scale_f16), .little);
            for (0..blk) |i| weight_quant[base + 2 + i] = @intCast((i + n + b) % 256);
        }
    }

    var inp = try alloc.alloc(f32, M * K);
    defer alloc.free(inp);
    for (0..M * K) |i| inp[i] = @as(f32, @floatFromInt((i % 7) + 1)) * 0.1;

    const out_batch = try alloc.alloc(f32, M * N);
    defer alloc.free(out_batch);
    matmulBatchQ8_0(out_batch, weight_quant, inp, M, N, K);

    var out_ref = try alloc.alloc(f32, M * N);
    defer alloc.free(out_ref);
    for (0..M) |m| matmulVecQ8_0(out_ref[m * N ..][0..N], weight_quant, inp[m * K ..][0..K], N, K);

    for (out_batch, out_ref) |b, r| try std.testing.expect(@abs(b - r) < 1e-4);
}

test "matmulBatchQ4_K vs M×matmulVecQ4_K" {
    const alloc = std.testing.allocator;
    const N, const K, const M = .{ 32, 256, 3 };
    const blk = Q4_K_BLOCK_SIZE;
    const blk_bytes = Q4_K_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;

    var weight_quant = try alloc.alloc(u8, N * n_blocks * blk_bytes);
    defer alloc.free(weight_quant);
    for (0..N) |n| {
        for (0..n_blocks) |b| {
            const base = (n * n_blocks + b) * blk_bytes;
            const scale_f16: f16 = @floatCast(0.5 + @as(f32, @floatFromInt(n % 4)) * 0.1);
            const dmin_f16: f16 = 0.1;
            std.mem.writeInt(u16, weight_quant[base..][0..2], @bitCast(scale_f16), .little);
            std.mem.writeInt(u16, weight_quant[base..][2..4], @bitCast(dmin_f16), .little);
            for (0..12) |i| weight_quant[base + 4 + i] = @intCast((i + n + b) % 256);
            for (0..128) |i| weight_quant[base + 16 + i] = @intCast((i + n + b) % 16);
        }
    }

    var inp = try alloc.alloc(f32, M * K);
    defer alloc.free(inp);
    for (0..M * K) |i| inp[i] = @as(f32, @floatFromInt((i % 7) + 1)) * 0.1;

    const out_batch = try alloc.alloc(f32, M * N);
    defer alloc.free(out_batch);
    matmulBatchQ4_K(out_batch, weight_quant, inp, M, N, K);

    var out_ref = try alloc.alloc(f32, M * N);
    defer alloc.free(out_ref);
    for (0..M) |m| matmulVecQ4_K(out_ref[m * N ..][0..N], weight_quant, inp[m * K ..][0..K], N, K);

    for (out_batch, out_ref) |b, r| try std.testing.expect(@abs(b - r) < 5e-3);
}

test "matmulBatchQ6_K vs M×matmulVecQ6_K" {
    const alloc = std.testing.allocator;
    const N, const K, const M = .{ 32, 256, 3 };
    const blk = Q6_K_BLOCK_SIZE;
    const blk_bytes = Q6_K_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;

    var weight_quant = try alloc.alloc(u8, N * n_blocks * blk_bytes);
    defer alloc.free(weight_quant);
    for (0..N) |n| {
        for (0..n_blocks) |b| {
            const base = (n * n_blocks + b) * blk_bytes;
            for (0..208) |i| weight_quant[base + i] = @intCast((i + n * 7 + b * 13) % 256);
            const scale_f16: f16 = @floatCast(1.0 + @as(f32, @floatFromInt(n % 3)) * 0.1);
            std.mem.writeInt(u16, weight_quant[base..][208..210], @bitCast(scale_f16), .little);
        }
    }

    var inp = try alloc.alloc(f32, M * K);
    defer alloc.free(inp);
    for (0..M * K) |i| inp[i] = @as(f32, @floatFromInt((i % 7) + 1)) * 0.1;

    const out_batch = try alloc.alloc(f32, M * N);
    defer alloc.free(out_batch);
    matmulBatchQ6_K(out_batch, weight_quant, inp, M, N, K);

    var out_ref = try alloc.alloc(f32, M * N);
    defer alloc.free(out_ref);
    for (0..M) |m| matmulVecQ6_K(out_ref[m * N ..][0..N], weight_quant, inp[m * K ..][0..K], N, K);

    for (out_batch, out_ref) |b, r| try std.testing.expect(@abs(b - r) < 5e-2);
}

test "matmulVecQ6_K parallel vs ref" {
    const alloc = std.testing.allocator;
    const N, const K = .{ 512, 512 };
    const blk = Q6_K_BLOCK_SIZE;
    const blk_bytes = Q6_K_BLOCK_BYTES;
    const n_blocks = (K + blk - 1) / blk;

    var weight_quant = try alloc.alloc(u8, N * n_blocks * blk_bytes);
    defer alloc.free(weight_quant);
    for (0..N) |n| {
        for (0..n_blocks) |b| {
            const base = (n * n_blocks + b) * blk_bytes;
            for (0..208) |i| weight_quant[base + i] = @intCast((i + n * 7 + b * 13) % 256);
            const scale_f16: f16 = 1.0;
            std.mem.writeInt(u16, weight_quant[base..][208..210], @bitCast(scale_f16), .little);
        }
    }
    const x = try alloc.alloc(f32, K);
    defer alloc.free(x);
    for (x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 5)) / 5.0;

    const out_fast = try alloc.alloc(f32, N);
    const out_ref = try alloc.alloc(f32, N);
    defer alloc.free(out_fast);
    defer alloc.free(out_ref);

    matmulVecQ6_K(out_fast, weight_quant, x, N, K);
    matmulVecQ6_KRef(out_ref, weight_quant, x, N, K);

    for (out_ref, out_fast) |r, f| try std.testing.expect(@abs(r - f) < 5e-2);
}
