// rng — PRNG con semilla para determinismo
const std = @import("std");

/// PRNG determinista basado en Xoshiro (std.Random.DefaultPrng).
/// La misma semilla produce siempre la misma secuencia.
pub const Rng = struct {
    state: std.Random.DefaultPrng,

    pub fn init(seed: u64) Rng {
        return .{ .state = std.Random.DefaultPrng.init(seed) };
    }

    /// Siguiente entero pseudoaleatorio de 64 bits.
    pub fn next(self: *Rng) u64 {
        return self.state.random().int(u64);
    }

    /// Float en [0, 1).
    pub fn float(self: *Rng) f32 {
        return self.state.random().float(f32);
    }

    /// Entero uniforme en [0, n). Devuelve 0 si n == 0.
    pub fn below(self: *Rng, n: usize) usize {
        if (n == 0) return 0;
        return self.state.random().uintLessThan(usize, n);
    }
};

// --- tests ---
test "rng same seed same sequence" {
    var a = Rng.init(12345);
    var b = Rng.init(12345);
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        try std.testing.expectEqual(a.next(), b.next());
    }
}

test "rng different seed different sequence" {
    var a = Rng.init(1);
    var b = Rng.init(2);
    var differs = false;
    var i: usize = 0;
    while (i < 32) : (i += 1) {
        if (a.next() != b.next()) {
            differs = true;
            break;
        }
    }
    try std.testing.expect(differs);
}

test "rng float range" {
    var r = Rng.init(99);
    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        const f = r.float();
        try std.testing.expect(f >= 0.0 and f < 1.0);
    }
}

test "rng below bound" {
    var r = Rng.init(7);
    try std.testing.expectEqual(@as(usize, 0), r.below(0));
    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        try std.testing.expect(r.below(10) < 10);
    }
}
