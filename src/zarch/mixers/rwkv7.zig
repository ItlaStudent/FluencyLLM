const std = @import("std");
const zkernel = @import("zkernel");

pub const State = struct {
    dim: usize,
    xx: []f32,
    aa: []f32,
    bb: []f32,
    pp: []f32,
    W_k: []f32,
    W_v: []f32,
    W_r: []f32,
    W_g1: []f32,
    W_g2: []f32,
    time_decay: []f32,
    time_faaaa: []f32,

    pub fn init(alloc: std.mem.Allocator, dim: usize) !State {
        const self = State{
            .dim = dim,
            .xx = try alloc.alloc(f32, dim),
            .aa = try alloc.alloc(f32, dim),
            .bb = try alloc.alloc(f32, dim),
            .pp = try alloc.alloc(f32, dim),
            .W_k = try alloc.alloc(f32, dim * dim),
            .W_v = try alloc.alloc(f32, dim * dim),
            .W_r = try alloc.alloc(f32, dim * dim),
            .W_g1 = try alloc.alloc(f32, dim * dim),
            .W_g2 = try alloc.alloc(f32, dim * dim),
            .time_decay = try alloc.alloc(f32, dim),
            .time_faaaa = try alloc.alloc(f32, dim),
        };
        @memset(self.xx, 0);
        @memset(self.aa, 0);
        @memset(self.bb, 0);
        @memset(self.pp, 0);
        for (self.W_k, 0..) |*w, i| w.* = if (i % dim == i / dim) 1.0 else @as(f32, @floatFromInt(i % 5)) / 20.0;
        for (self.W_v, 0..) |*w, i| w.* = if (i % dim == i / dim) 1.0 else @as(f32, @floatFromInt(i % 3)) / 20.0;
        for (self.W_r, 0..) |*w, i| w.* = if (i % dim == i / dim) 1.0 else @as(f32, @floatFromInt(i % 7)) / 20.0;
        @memset(self.W_g1, 0);
        @memset(self.W_g2, 0);
        for (self.W_g1, 0..) |*w, i| w.* = if (i % dim == i / dim) 0.5 else 0;
        for (self.W_g2, 0..) |*w, i| w.* = if (i % dim == i / dim) 0.5 else 0;
        @memset(self.time_decay, -1.0);
        for (self.time_faaaa) |*t| t.* = 0.5;
        return self;
    }

    pub fn deinit(self: *State, alloc: std.mem.Allocator) void {
        alloc.free(self.xx);
        alloc.free(self.aa);
        alloc.free(self.bb);
        alloc.free(self.pp);
        alloc.free(self.W_k);
        alloc.free(self.W_v);
        alloc.free(self.W_r);
        alloc.free(self.W_g1);
        alloc.free(self.W_g2);
        alloc.free(self.time_decay);
        alloc.free(self.time_faaaa);
    }
};

pub fn forward(state: *State, inp: []const f32, out: []f32, seq_len: usize, dim: usize, pos: usize) void {
    _ = seq_len;
    _ = dim;
    const d = state.dim;
    var scratch: [4096]f32 = undefined;

    // r = W_r @ inp (receptividad)
    zkernel.matmul(scratch[0..d], inp, state.W_r, 1, d, d);
    for (0..d) |i| scratch[i] = std.math.tanh(scratch[i]);
    const r = scratch[0..d];

    // k = W_k @ inp
    zkernel.matmul(scratch[d..][0..d], inp, state.W_k, 1, d, d);
    for (0..d) |i| scratch[d + i] = std.math.tanh(scratch[d + i]);
    const k = scratch[d..][0..d];

    // v = W_v @ inp
    zkernel.matmul(scratch[2*d..][0..d], inp, state.W_v, 1, d, d);
    const v = scratch[2*d..][0..d];

    // Token shift: xx = previous inp
    const xx = state.xx;
    const aa = state.aa;
    const bb = state.bb;
    const pp = state.pp;

    for (0..d) |i| {
        const w = if (pos == 0) @exp(state.time_decay[i]) else @exp(state.time_decay[i]);
        const u = state.time_faaaa[i];

        // WKV computation (simplified per-channel recurrence)
        const p = @max(pp[i] + u, bb[i] + w);
        const e1 = @exp(pp[i] + u - p);
        const e2 = @exp(bb[i] + w - p);
        const wkv_num = e1 * aa[i] + e2 * v[i];
        const wkv_den = e1 + e2;

        // Output = (wkv_num / wkv_den) * r (receptividad)
        const wkv = if (wkv_den > 1e-10) wkv_num / wkv_den else 0;
        out[i] = wkv * r[i];

        // Update state
        const w2 = @exp(-@abs(pp[i] + state.time_decay[i]));
        aa[i] = aa[i] * w2 + v[i] * k[i];
        bb[i] = bb[i] * w2 + k[i];
        pp[i] = pp[i] + state.time_decay[i];
    }
    @memcpy(xx, inp);
}

test "rwkv7 forward produces output" {
    const alloc = std.testing.allocator;
    var s = try State.init(alloc, 64);
    defer s.deinit(alloc);
    var inp: [64]f32 = undefined;
    var out: [64]f32 = undefined;
    for (&inp) |*p| p.* = 0.5;
    forward(&s, &inp, &out, 1, 64, 0);
    var has_nonzero = false;
    for (&out) |v| {
        if (v != 0) has_nonzero = true;
    }
    try std.testing.expect(has_nonzero);
}

test "rwkv7 forward accumulative state" {
    const alloc = std.testing.allocator;
    var s = try State.init(alloc, 64);
    defer s.deinit(alloc);
    var inp: [64]f32 = undefined;
    var out1: [64]f32 = undefined;
    var out2: [64]f32 = undefined;
    for (&inp) |*p| p.* = 0.1;
    forward(&s, &inp, &out1, 1, 64, 0);
    forward(&s, &inp, &out2, 1, 64, 1);
    var diff: f32 = 0;
    for (&out1, &out2) |a, b| diff += @abs(a - b);
    try std.testing.expect(diff > 1e-6);
}
