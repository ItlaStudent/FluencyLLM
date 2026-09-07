// hybrid.zig — bloque híbrido subcuadrático (F2).
//
// Compone una pila de capas con patrón configurable: mayoría de mixers
// recurrentes (Mamba-2, estado O(1) en tokens) + minoría de atención de ventana
// deslizante (SWA). Demuestra el "estado dual de inferencia": las capas SSM
// llevan estado recurrente de tamaño CONSTANTE respecto a los tokens generados,
// y `stepDecode` consume un token y emite uno usando memoria O(1) en esas capas.
const std = @import("std");
const mamba2 = @import("mamba2.zig");
const swa = @import("swa.zig");

pub const LayerKind = enum { mamba2, swa };

pub const Layer = union(LayerKind) {
    mamba2: mamba2.State,
    swa: swa.State,

    pub fn deinit(self: *Layer, alloc: std.mem.Allocator) void {
        switch (self.*) {
            .mamba2 => |*s| s.deinit(alloc),
            .swa => |*s| s.deinit(alloc),
        }
    }

    pub fn forward(self: *Layer, inp: []const f32, out: []f32, dim: usize, pos: usize) void {
        switch (self.*) {
            .mamba2 => |*s| mamba2.forward(s, inp, out, 1, dim, pos),
            .swa => |*s| swa.forward(s, inp, out, 1, dim, pos),
        }
    }

    /// Tamaño (en f32) del estado recurrente de la capa; 0 si no es recurrente.
    /// Es CONSTANTE respecto al número de tokens (clave del criterio E3).
    pub fn recurrentStateLen(self: Layer) usize {
        return switch (self) {
            .mamba2 => |s| s.h.len,
            .swa => 0,
        };
    }
};

pub const HybridBlock = struct {
    alloc: std.mem.Allocator,
    dim: usize,
    layers: []Layer,
    tmp: []f32,

    pub fn init(alloc: std.mem.Allocator, dim: usize, pattern: []const LayerKind) !HybridBlock {
        const layers = try alloc.alloc(Layer, pattern.len);
        errdefer alloc.free(layers);
        var made: usize = 0;
        errdefer for (layers[0..made]) |*l| l.deinit(alloc);
        for (pattern, 0..) |kind, i| {
            layers[i] = switch (kind) {
                .mamba2 => .{ .mamba2 = try mamba2.State.init(alloc, dim) },
                .swa => .{ .swa = try swa.State.init(alloc, dim) },
            };
            made = i + 1;
        }
        return .{
            .alloc = alloc,
            .dim = dim,
            .layers = layers,
            .tmp = try alloc.alloc(f32, dim),
        };
    }

    pub fn deinit(self: *HybridBlock) void {
        for (self.layers) |*l| l.deinit(self.alloc);
        self.alloc.free(self.layers);
        self.alloc.free(self.tmp);
    }

    /// Procesa UN token (en `pos`) por toda la pila con conexiones residuales,
    /// escribiendo el resultado en `out`. Memoria O(1) en tokens para capas SSM.
    pub fn stepDecode(self: *HybridBlock, inp: []const f32, out: []f32, pos: usize) void {
        if (inp.ptr != out.ptr) @memcpy(out, inp[0..self.dim]);
        for (self.layers) |*l| {
            l.forward(out, self.tmp, self.dim, pos);
            for (0..self.dim) |i| out[i] += self.tmp[i]; // residual
        }
    }

    /// Suma del estado recurrente de todas las capas (debe ser constante en tokens).
    pub fn recurrentStateTotal(self: HybridBlock) usize {
        var total: usize = 0;
        for (self.layers) |l| total += l.recurrentStateLen();
        return total;
    }
};

// --- test del criterio E3 ---
test "F2/E3: híbrido prefill+decode con memoria recurrente CONSTANTE y estable" {
    const alloc = std.testing.allocator;
    const D = 64;
    // Patrón: mayoría recurrente (Mamba-2) + minoría de atención (SWA), como el plan.
    const pattern = [_]LayerKind{ .mamba2, .swa, .mamba2, .mamba2 };
    var block = try HybridBlock.init(alloc, D, &pattern);
    defer block.deinit();

    var inp: [D]f32 = undefined;
    var out: [D]f32 = undefined;
    for (&inp) |*p| p.* = 0.3;

    // Estado recurrente esperado: 3 capas Mamba-2 * D.
    const expected_state = 3 * D;
    try std.testing.expectEqual(@as(usize, expected_state), block.recurrentStateTotal());

    // Prefill de 16 tokens.
    var pos: usize = 0;
    while (pos < 16) : (pos += 1) block.stepDecode(&inp, &out, pos);

    // El estado recurrente NO crece con los tokens (O(1) en tokens) — núcleo de E3.
    try std.testing.expectEqual(@as(usize, expected_state), block.recurrentStateTotal());

    // Decode de 64 tokens más: el estado recurrente sigue evolucionando con
    // memoria CONSTANTE y la salida permanece finita (decaimiento SSM < 1).
    while (pos < 80) : (pos += 1) {
        block.stepDecode(&inp, &out, pos);
        for (&out) |v| {
            try std.testing.expect(!std.math.isNan(v));
            try std.testing.expect(!std.math.isInf(v));
        }
    }
    try std.testing.expectEqual(@as(usize, expected_state), block.recurrentStateTotal());
}

test "F2: mixer recurrente vs atención — la atención no tiene estado recurrente" {
    const alloc = std.testing.allocator;
    const D = 32;
    var m = Layer{ .mamba2 = try mamba2.State.init(alloc, D) };
    defer m.deinit(alloc);
    var a = Layer{ .swa = try swa.State.init(alloc, D) };
    defer a.deinit(alloc);
    try std.testing.expectEqual(@as(usize, D), m.recurrentStateLen());
    try std.testing.expectEqual(@as(usize, 0), a.recurrentStateLen());
}
