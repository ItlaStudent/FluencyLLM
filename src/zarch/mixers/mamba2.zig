// mamba2.zig — SSM diagonal selectivo (familia Mamba-2 / SSD), 100% Zig.
//
// Mixer subcuadrático: el estado recurrente `h` es O(1) en el número de tokens
// (un escalar por canal), así que la decodificación usa memoria CONSTANTE — la
// identidad del proyecto (Art. II). Recurrencia por canal i:
//   a_i   = exp(-dt * exp(a_log_i))     (decaimiento en (0,1) -> estable)
//   h_i   = a_i * h_i + dt * b_i * x_i
//   y_i   = c_i * h_i
// Equivale a un scan lineal de primer orden (ver zkernel.scanChunkwise).
const std = @import("std");
const zkernel = @import("zkernel");

pub const State = struct {
    d: usize,
    h: []f32, // estado recurrente [d] — O(1) en tokens
    a_log: []f32, // [d] parámetros de decaimiento (log)
    b: []f32, // [d] ganancia de entrada
    c: []f32, // [d] ganancia de salida
    dt: f32,

    pub fn init(alloc: std.mem.Allocator, dim: usize) !State {
        if (dim == 0) return error.DimTooSmall;
        const self = State{
            .d = dim,
            .h = try alloc.alloc(f32, dim),
            .a_log = try alloc.alloc(f32, dim),
            .b = try alloc.alloc(f32, dim),
            .c = try alloc.alloc(f32, dim),
            .dt = 0.1,
        };
        @memset(self.h, 0);
        for (self.a_log, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 8)) / 8.0 - 1.0;
        for (self.b, 0..) |*v, i| v.* = 0.5 + @as(f32, @floatFromInt(i % 3)) / 10.0;
        for (self.c, 0..) |*v, i| v.* = 0.5 - @as(f32, @floatFromInt(i % 5)) / 20.0;
        return self;
    }

    pub fn deinit(self: *State, alloc: std.mem.Allocator) void {
        alloc.free(self.h);
        alloc.free(self.a_log);
        alloc.free(self.b);
        alloc.free(self.c);
    }

    /// Reinicia el estado recurrente (nueva secuencia).
    pub fn reset(self: *State) void {
        @memset(self.h, 0);
    }
};

/// Un paso recurrente: consume x_t, actualiza el estado y escribe y_t. O(d).
/// `pos` no se usa: el estado lo lleva `state.h` (decode O(1) memoria).
pub fn forward(state: *State, inp: []const f32, out: []f32, seq_len: usize, dim: usize, pos: usize) void {
    _ = seq_len;
    _ = dim;
    _ = pos;
    const d = state.d;
    const dt = state.dt;
    for (0..d) |i| {
        const a = @exp(-dt * @exp(state.a_log[i]));
        state.h[i] = a * state.h[i] + dt * state.b[i] * inp[i];
        out[i] = state.c[i] * state.h[i];
    }
}

test "mamba2: salida no nula y estado O(1) en tokens" {
    const alloc = std.testing.allocator;
    const D = 64;
    var s = try State.init(alloc, D);
    defer s.deinit(alloc);
    // El estado tiene tamaño D sin importar cuántos tokens procesemos.
    try std.testing.expectEqual(@as(usize, D), s.h.len);

    var inp: [D]f32 = undefined;
    var out: [D]f32 = undefined;
    for (&inp) |*p| p.* = 0.5;
    forward(&s, &inp, &out, 1, D, 0);
    try std.testing.expectEqual(@as(usize, D), s.h.len); // sigue siendo O(1)
    var nonzero = false;
    for (&out) |v| if (v != 0) {
        nonzero = true;
    };
    try std.testing.expect(nonzero);
}

test "mamba2: estabilidad en secuencia larga (sin NaN/Inf) y estado evoluciona" {
    const alloc = std.testing.allocator;
    const D = 32;
    var s = try State.init(alloc, D);
    defer s.deinit(alloc);
    var inp: [D]f32 = undefined;
    var out: [D]f32 = undefined;
    for (&inp) |*p| p.* = 1.0;

    forward(&s, &inp, &out, 1, D, 0);
    var first: [D]f32 = undefined;
    @memcpy(&first, &out);

    // 2000 pasos: el estado debe permanecer acotado (decaimiento < 1).
    for (0..2000) |t| forward(&s, &inp, &out, 1, D, t + 1);
    for (&out) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }
    // El estado evolucionó respecto al primer paso.
    var diff: f32 = 0;
    for (&first, &out) |a, b| diff += @abs(a - b);
    try std.testing.expect(diff > 1e-6);
}
