const std = @import("std");
const zkernel = @import("zkernel");

pub const State = struct {
    dim: usize,
    S: []f32,
    W_q: []f32,
    W_k: []f32,
    W_v: []f32,
    W_gate: []f32,
    delta_bias: f32,

    pub fn init(alloc: std.mem.Allocator, dim: usize) !State {
        const self = State{
            .dim = dim,
            .S = try alloc.alloc(f32, dim * dim),
            .W_q = try alloc.alloc(f32, dim * dim),
            .W_k = try alloc.alloc(f32, dim * dim),
            .W_v = try alloc.alloc(f32, dim * dim),
            .W_gate = try alloc.alloc(f32, dim * dim),
            .delta_bias = 0.5,
        };
        @memset(self.S, 0);
        for (self.W_q, 0..) |*w, i| w.* = if (i % dim == i / dim) 1.0 else @as(f32, @floatFromInt(i % 7)) / 20.0;
        for (self.W_k, 0..) |*w, i| w.* = if (i % dim == i / dim) 1.0 else @as(f32, @floatFromInt(i % 5)) / 20.0;
        for (self.W_v, 0..) |*w, i| w.* = if (i % dim == i / dim) 0.5 else @as(f32, @floatFromInt(i % 3)) / 20.0;
        @memset(self.W_gate, 0);
        for (self.W_gate, 0..) |*w, i| w.* = if (i % dim == i / dim) 0.1 else 0;
        return self;
    }

    pub fn deinit(self: *State, alloc: std.mem.Allocator) void {
        alloc.free(self.S);
        alloc.free(self.W_q);
        alloc.free(self.W_k);
        alloc.free(self.W_v);
        alloc.free(self.W_gate);
    }
};

pub fn forward(state: *State, inp: []const f32, out: []f32, seq_len: usize, dim: usize, pos: usize) void {
    _ = seq_len;
    _ = pos;
    const d = dim;
    var scratch: [4096]f32 = undefined;

    // q = W_q @ inp, k = W_k @ inp
    zkernel.matmul(scratch[0..d], inp, state.W_q, 1, d, d);
    zkernel.matmul(scratch[d..][0..d], inp, state.W_k, 1, d, d);

    const q = scratch[0..d];
    const k = scratch[d..][0..d];

    // delta = sigmoid(gate @ inp + bias)
    zkernel.matmul(scratch[3*d..][0..d], inp, state.W_gate, 1, d, d);
    const gate = scratch[3*d..][0..d];
    for (0..d) |i| gate[i] = 1.0 / (1.0 + @exp(-(gate[i] + state.delta_bias)));

    // beta = k (simplificación: beta = tanh(k))
    for (0..d) |i| scratch[2*d + i] = std.math.tanh(k[i]);
    const beta = scratch[2*d..][0..d];

    // Usar deltaWyChunkwise: S += delta * beta^T, out = S * q
    zkernel.deltaWyChunkwise(state.S, q, gate, beta, out, d);
}

test "gated_deltanet forward accumulates state" {
    const alloc = std.testing.allocator;
    var s = try State.init(alloc, 64);
    defer s.deinit(alloc);
    var inp: [64]f32 = undefined;
    var out1: [64]f32 = undefined;
    var out2: [64]f32 = undefined;
    for (&inp) |*p| p.* = 0.3;
    forward(&s, &inp, &out1, 1, 64, 0);
    for (&inp) |*p| p.* = 0.5;
    forward(&s, &inp, &out2, 1, 64, 1);
    // Output should differ because state accumulated
    var diff: f32 = 0;
    for (&out1, &out2) |a, b| diff += @abs(a - b);
    try std.testing.expect(diff > 1e-6);
}

test "gated_deltanet forward not all zeros" {
    const alloc = std.testing.allocator;
    var s = try State.init(alloc, 64);
    defer s.deinit(alloc);
    var inp: [64]f32 = undefined;
    var out: [64]f32 = undefined;
    for (&inp) |*p| p.* = 0.7;
    forward(&s, &inp, &out, 1, 64, 0);
    var has_nonzero = false;
    for (&out) |v| {
        if (v != 0) has_nonzero = true;
    }
    try std.testing.expect(has_nonzero);
}
