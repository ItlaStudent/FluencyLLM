// swa.zig — Sliding Window Attention (atención minoritaria del híbrido)
const std = @import("std");
const zkernel = @import("zkernel");

pub const State = struct {
    // KV cache para decodificación incremental
    k_cache: []f32,
    v_cache: []f32,
    cache_pos: usize,
    window_size: usize,
    n_heads: usize,
    head_dim: usize,

    pub fn init(alloc: std.mem.Allocator, dim: usize) !State {
        const n_heads = 4; // configurable
        const head_dim = dim / n_heads;
        const window_size = 256;
        const max_seq = 2048;
        return .{
            .k_cache = try alloc.alloc(f32, max_seq * dim),
            .v_cache = try alloc.alloc(f32, max_seq * dim),
            .cache_pos = 0,
            .window_size = window_size,
            .n_heads = n_heads,
            .head_dim = head_dim,
        };
    }

    pub fn deinit(self: *State, alloc: std.mem.Allocator) void {
        alloc.free(self.k_cache);
        alloc.free(self.v_cache);
    }
};

pub fn forward(state: *State, inp: []const f32, out: []f32, seq_len: usize, dim: usize, pos: usize) void {
    // SWA simplificada: para F2, implementación de referencia
    // full attention sobre la ventana [max(0, pos - window_size) .. pos]
    _ = seq_len;
    @memset(out, 0);
    const n_heads = state.n_heads;
    const head_dim = state.head_dim;

    // Almacenar KV
    for (0..dim) |d| {
        state.k_cache[pos * dim + d] = inp[d];
        state.v_cache[pos * dim + d] = inp[d];
    }
    state.cache_pos = pos + 1;

    // Attention por cabeza
    var att_buf: [2048]f32 = undefined;
    for (0..n_heads) |h| {
        const q_off = h * head_dim;
        const start = if (pos > state.window_size) pos - state.window_size else 0;
        const win_len = pos - start + 1;

        // Scores
        for (start..pos + 1) |t| {
            var score: f32 = 0;
            for (0..head_dim) |i| {
                score += inp[q_off + i] * state.k_cache[t * dim + q_off + i];
            }
            score /= @sqrt(@as(f32, @floatFromInt(head_dim)));
            att_buf[t - start] = score;
        }
        zkernel.softmax(att_buf[0..win_len], att_buf[0..win_len]);

        // Output
        for (0..head_dim) |i| {
            var acc: f32 = 0;
            for (start..pos + 1) |t| {
                acc += att_buf[t - start] * state.v_cache[t * dim + q_off + i];
            }
            out[q_off + i] = acc;
        }
    }
}

test "swa: ventana acotada y estabilidad en secuencia larga" {
    const alloc = std.testing.allocator;
    const D = 64;
    var s = try State.init(alloc, D);
    defer s.deinit(alloc);
    // La ventana es fija (no crece con la secuencia) — coste por token O(W).
    try std.testing.expectEqual(@as(usize, 256), s.window_size);

    var inp: [D]f32 = undefined;
    var out: [D]f32 = undefined;
    for (&inp) |*p| p.* = 0.2;
    // Procesa más tokens que la ventana: debe seguir finito y no nulo.
    for (0..400) |pos| forward(&s, &inp, &out, 1, D, pos);
    var nonzero = false;
    for (&out) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
        if (v != 0) nonzero = true;
    }
    try std.testing.expect(nonzero);
}
