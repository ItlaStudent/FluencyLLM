// ztrain — bucle de entrenamiento + optimizadores
const std = @import("std");
const ztensor = @import("ztensor");

// --- SGD ---
pub const SGD = struct {
    lr: f32,
    momentum: f32,
    vel: ?[]f32 = null,
    allocator: ?std.mem.Allocator = null,

    pub fn init(alloc: std.mem.Allocator, lr: f32, momentum: f32, n_params: usize) !SGD {
        const vel = try alloc.alloc(f32, n_params);
        @memset(vel, 0);
        return .{ .lr = lr, .momentum = momentum, .vel = vel, .allocator = alloc };
    }

    pub fn deinit(self: *SGD) void {
        if (self.allocator) |a| a.free(self.vel.?);
    }

    pub fn step(self: *SGD, params: []f32, grads: []const f32) void {
        const vel = self.vel.?;
        for (params, grads, vel) |*p, g, *v| {
            v.* = self.momentum * v.* + g;
            p.* -= self.lr * v.*;
        }
    }
};

// --- AdamW ---
pub const AdamW = struct {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    m: []f32,
    v: []f32,
    t: u32,
    allocator: std.mem.Allocator,

    pub fn init(alloc: std.mem.Allocator, lr: f32, n_params: usize) !AdamW {
        const m = try alloc.alloc(f32, n_params);
        const v = try alloc.alloc(f32, n_params);
        @memset(m, 0);
        @memset(v, 0);
        return .{
            .lr = lr,
            .beta1 = 0.9,
            .beta2 = 0.999,
            .eps = 1e-8,
            .weight_decay = 0.01,
            .m = m,
            .v = v,
            .t = 0,
            .allocator = alloc,
        };
    }

    pub fn deinit(self: *AdamW) void {
        self.allocator.free(self.m);
        self.allocator.free(self.v);
    }

    pub fn step(self: *AdamW, params: []f32, grads: []const f32) void {
        self.t += 1;
        const t_f: f32 = @floatFromInt(self.t);
        const lr_t = self.lr * @sqrt(1.0 - std.math.pow(f32, self.beta2, t_f)) / (1.0 - std.math.pow(f32, self.beta1, t_f));
        for (params, grads, self.m, self.v) |*p, g, *m, *v| {
            m.* = self.beta1 * m.* + (1.0 - self.beta1) * g;
            v.* = self.beta2 * v.* + (1.0 - self.beta2) * g * g;
            p.* -= lr_t * m.* / (@sqrt(v.*) + self.eps) + self.lr * self.weight_decay * p.*;
        }
    }
};

// --- Muon ---
pub const Muon = struct {
    lr: f32,
    momentum: f32,
    m: []f32,
    allocator: std.mem.Allocator,

    pub fn init(alloc: std.mem.Allocator, lr: f32, n_params: usize) !Muon {
        const m = try alloc.alloc(f32, n_params);
        @memset(m, 0);
        return .{
            .lr = lr,
            .momentum = 0.95,
            .m = m,
            .allocator = alloc,
        };
    }

    pub fn deinit(self: *Muon) void {
        self.allocator.free(self.m);
    }

    /// Realiza un paso de optimización de Muon sobre matrices 2D
    /// params: [M, N], grads: [M, N]
    pub fn step(self: *Muon, params: []f32, grads: []const f32, M: usize, N: usize) !void {
        const beta = self.momentum;
        for (self.m, grads) |*m_val, g| {
            m_val.* = beta * m_val.* + (1.0 - beta) * g;
        }

        const total_size = M * N;
        const X = try self.allocator.alloc(f32, total_size);
        defer self.allocator.free(X);
        @memcpy(X, self.m);

        var f_norm_sq: f32 = 0;
        for (X) |x| f_norm_sq += x * x;
        const f_norm = @sqrt(f_norm_sq) + 1e-10;
        const inv_f_norm = 1.0 / f_norm;
        for (X) |*x| x.* *= inv_f_norm;

        const use_transposed = M < N;
        if (use_transposed) {
            const XXT = try self.allocator.alloc(f32, M * M);
            defer self.allocator.free(XXT);
            const Temp = try self.allocator.alloc(f32, total_size);
            defer self.allocator.free(Temp);

            var iter: usize = 0;
            while (iter < 5) : (iter += 1) {
                @memset(XXT, 0);
                for (0..M) |i| {
                    for (0..M) |j| {
                        var acc: f32 = 0;
                        for (0..N) |k| acc += X[i * N + k] * X[j * N + k];
                        XXT[i * M + j] = acc;
                    }
                }
                for (0..M) |i| {
                    for (0..N) |j| {
                        var acc: f32 = 0;
                        for (0..M) |k| acc += XXT[i * M + k] * X[k * N + j];
                        Temp[i * N + j] = 1.5 * X[i * N + j] - 0.5 * acc;
                    }
                }
                @memcpy(X, Temp);
            }
        } else {
            const XTX = try self.allocator.alloc(f32, N * N);
            defer self.allocator.free(XTX);
            const Temp = try self.allocator.alloc(f32, total_size);
            defer self.allocator.free(Temp);

            var iter: usize = 0;
            while (iter < 5) : (iter += 1) {
                @memset(XTX, 0);
                for (0..N) |i| {
                    for (0..N) |j| {
                        var acc: f32 = 0;
                        for (0..M) |k| acc += X[k * N + i] * X[k * N + j];
                        XTX[i * N + j] = acc;
                    }
                }
                for (0..M) |i| {
                    for (0..N) |j| {
                        var acc: f32 = 0;
                        for (0..N) |k| acc += X[i * N + k] * XTX[k * N + j];
                        Temp[i * N + j] = 1.5 * X[i * N + j] - 0.5 * acc;
                    }
                }
                @memcpy(X, Temp);
            }
        }

        for (params, X) |*p, x| {
            p.* -= self.lr * x;
        }
    }
};

// --- LR Scheduler ---
pub const Scheduler = union(enum) {
    const_lr: f32,
    cosine: struct { max_lr: f32, min_lr: f32, max_steps: usize },
    /// Calentamiento lineal hasta `max_lr` durante `warmup_steps`, luego coseno
    /// hasta `min_lr` en `max_steps`. Es el schedule estándar para destilar.
    warmup_cosine: struct { max_lr: f32, min_lr: f32, warmup_steps: usize, max_steps: usize },

    pub fn getLR(self: Scheduler, step_: usize) f32 {
        return switch (self) {
            .const_lr => |lr| lr,
            .cosine => |cfg| blk: {
                const progress = @min(@as(f32, @floatFromInt(step_)) / @as(f32, @floatFromInt(cfg.max_steps)), 1.0);
                break :blk cfg.min_lr + (cfg.max_lr - cfg.min_lr) * 0.5 * (1.0 + @cos(progress * std.math.pi));
            },
            .warmup_cosine => |cfg| blk: {
                if (step_ < cfg.warmup_steps) {
                    const w: f32 = @floatFromInt(cfg.warmup_steps);
                    break :blk cfg.max_lr * (@as(f32, @floatFromInt(step_ + 1)) / w);
                }
                const denom: f32 = @floatFromInt(@max(cfg.max_steps - cfg.warmup_steps, 1));
                const progress = @min(@as(f32, @floatFromInt(step_ - cfg.warmup_steps)) / denom, 1.0);
                break :blk cfg.min_lr + (cfg.max_lr - cfg.min_lr) * 0.5 * (1.0 + @cos(progress * std.math.pi));
            },
        };
    }
};

// --- Detección de no-finitos ---
/// Devuelve `true` si algún elemento es NaN o ±Inf.
pub fn hasNaNOrInf(buf: []const f32) bool {
    for (buf) |v| {
        if (!std.math.isFinite(v)) return true;
    }
    return false;
}

/// Cuenta elementos NaN o ±Inf.
pub fn countNonFinite(buf: []const f32) usize {
    var count: usize = 0;
    for (buf) |v| {
        if (!std.math.isFinite(v)) count += 1;
    }
    return count;
}

// --- Gradient clipping ---
/// Recorta los gradientes por norma L2 global sobre TODOS los grupos de
/// parámetros (lista de slices). Si la norma supera `max_norm`, escala todos
/// los gradientes por `max_norm/norm`. Devuelve la norma global previa al
/// recorte (útil para logging / detección de explosión). Imprescindible en
/// destilación para que el KD no diverja.
/// Si la norma es NaN/Inf, escala a 0 (evita propagar NaN a los pesos).
pub fn clipGradGlobalNorm(grads: []const []f32, max_norm: f32) f32 {
    var total: f32 = 0;
    for (grads) |g| {
        for (g) |v| total += v * v;
    }
    const norm = @sqrt(total);
    if (!std.math.isFinite(norm)) {
        // norma NaN/Inf → escala 0 para evitar corrupción de pesos
        for (grads) |g| {
            for (g) |*v| v.* = 0;
        }
        return norm;
    }
    if (norm > max_norm and norm > 1e-12) {
        const scale = max_norm / norm;
        for (grads) |g| {
            for (g) |*v| v.* *= scale;
        }
    }
    return norm;
}

// --- Acumulación de gradiente ---
/// Acumulador simple: suma `src` en `dst` (para micro-batches). El bucle de
/// entrenamiento escala la pérdida por 1/accum_steps y sólo aplica el optimizer
/// (y zera grads) cada `accum_steps` micro-pasos.
pub fn accumulateGrad(dst: []f32, src: []const f32) void {
    std.debug.assert(dst.len == src.len);
    for (dst, src) |*d, s| d.* += s;
}

// --- Pipeline de datos mmap ---
/// Stream de tokens respaldado por un archivo mapeado (u16 little-endian, el
/// formato típico de un .bin de tokens). Entrega ventanas (input, target) donde
/// target = input desplazado un token (predicción del siguiente token).
pub const TokenDataset = struct {
    bytes: []const u8, // vista cruda; tokens u16 LE
    mf: ?ztensor.MappedFile = null,
    n: usize, // nº de tokens

    /// Vista prestada sobre bytes (p. ej. tokens sintéticos en memoria).
    pub fn fromBytes(bytes: []const u8) TokenDataset {
        return .{ .bytes = bytes, .mf = null, .n = bytes.len / 2 };
    }

    /// Mapea un archivo .bin de tokens u16 desde disco.
    pub fn fromFile(alloc: std.mem.Allocator, io: std.Io, path: []const u8) !TokenDataset {
        const mf = try ztensor.mapFile(alloc, io, path);
        return .{ .bytes = mf.bytes, .mf = mf, .n = mf.bytes.len / 2 };
    }

    pub fn deinit(self: *TokenDataset) void {
        if (self.mf) |*m| m.deinit();
        self.* = undefined;
    }

    pub fn tokenAt(self: TokenDataset, i: usize) u32 {
        return std.mem.readInt(u16, self.bytes[i * 2 ..][0..2], .little);
    }

    /// Rellena `ids` (longitud = seq_len) con una ventana contigua aleatoria y
    /// `tgt` con la misma ventana desplazada +1 (siguiente token).
    pub fn sampleWindow(self: TokenDataset, rng: std.Random, ids: []u32, tgt: []u32) void {
        std.debug.assert(ids.len == tgt.len);
        const seq_len = ids.len;
        std.debug.assert(self.n > seq_len + 1);
        const start = rng.uintLessThan(usize, self.n - seq_len - 1);
        for (0..seq_len) |i| {
            ids[i] = self.tokenAt(start + i);
            tgt[i] = self.tokenAt(start + i + 1);
        }
    }
};

// --- Mixed Precision helper ---
/// Convierte tensor f32 a bf16 para forward, guarda f32 para grads
pub fn toBF16Forward(alloc: std.mem.Allocator, tensor: ztensor.Tensor) !ztensor.Tensor {
    std.debug.assert(tensor.dtype == .f32);
    const out = try ztensor.empty(alloc, .bf16, tensor.shape);
    const nelem = tensor.elemCount();
    const src = @as([*]f32, @ptrCast(@alignCast(tensor.data)))[0..nelem];
    const dst = @as([*]u16, @ptrCast(@alignCast(out.data)))[0..nelem];
    for (src, dst) |v, *d| d.* = ztensor.bf16.fromF32(v);
    return out;
}

/// Convierte tensor bf16 a f32 para backward
pub fn toF32Backward(alloc: std.mem.Allocator, tensor: ztensor.Tensor) !ztensor.Tensor {
    std.debug.assert(tensor.dtype == .bf16);
    const out = try ztensor.empty(alloc, .f32, tensor.shape);
    const nelem = tensor.elemCount();
    const src = @as([*]u16, @ptrCast(@alignCast(tensor.data)))[0..nelem];
    const dst = @as([*]f32, @ptrCast(@alignCast(out.data)))[0..nelem];
    for (src, dst) |v, *d| d.* = ztensor.bf16.toF32(v);
    return out;
}

// --- tests ---
test "SGD step" {
    const alloc = std.testing.allocator;
    var opt = try SGD.init(alloc, 0.1, 0.9, 3);
    defer opt.deinit();
    var params = [_]f32{ 1.0, 2.0, 3.0 };
    const grads = [_]f32{ 0.1, 0.2, 0.3 };
    opt.step(&params, &grads);
    try std.testing.expectApproxEqAbs(@as(f32, 0.99), params[0], 1e-5);
}

test "AdamW step" {
    const alloc = std.testing.allocator;
    var opt = try AdamW.init(alloc, 0.001, 3);
    defer opt.deinit();
    var params = [_]f32{ 1.0, 2.0, 3.0 };
    const grads = [_]f32{ 0.1, 0.2, 0.3 };
    opt.step(&params, &grads);
    // Verifica que params cambiaron
    try std.testing.expect(params[0] != 1.0);
}

test "cosine scheduler" {
    const s = Scheduler{ .cosine = .{ .max_lr = 1.0, .min_lr = 0.1, .max_steps = 100 } };
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), s.getLR(0), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.55), s.getLR(50), 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), s.getLR(100), 1e-5);
}

test "Muon step" {
    const alloc = std.testing.allocator;
    var opt = try Muon.init(alloc, 0.01, 4);
    defer opt.deinit();
    var params = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const grads = [_]f32{ 0.1, -0.2, 0.3, -0.4 };
    try opt.step(&params, &grads, 2, 2);
    // Verificar que los parámetros cambiaron
    try std.testing.expect(params[0] != 1.0);
    try std.testing.expect(params[1] != 2.0);
}

test "clipGradGlobalNorm recorta a la norma máxima" {
    var g1 = [_]f32{ 3.0, 0.0 };
    var g2 = [_]f32{ 4.0, 0.0 };
    const groups = [_][]f32{ &g1, &g2 };
    const norm = clipGradGlobalNorm(&groups, 1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), norm, 1e-5); // sqrt(9+16)
    // tras recortar, la norma global debe ser 1.0
    var total: f32 = 0;
    for (groups) |g| for (g) |v| {
        total += v * v;
    };
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), @sqrt(total), 1e-5);
    // sin recorte si ya es pequeña
    var g3 = [_]f32{ 0.1, 0.1 };
    const groups2 = [_][]f32{&g3};
    _ = clipGradGlobalNorm(&groups2, 10.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), g3[0], 1e-6);
}

test "hasNaNOrInf detecta NaN e Inf" {
    try std.testing.expect(hasNaNOrInf(&[_]f32{ 1.0, std.math.nan(f32), 2.0 }));
    try std.testing.expect(hasNaNOrInf(&[_]f32{ 1.0, std.math.inf(f32), 2.0 }));
    try std.testing.expect(hasNaNOrInf(&[_]f32{ 1.0, -std.math.inf(f32), 2.0 }));
    try std.testing.expect(!hasNaNOrInf(&[_]f32{ 1.0, 2.0, -3.0 }));
}

test "countNonFinite cuenta correctamente" {
    try std.testing.expectEqual(@as(usize, 1), countNonFinite(&[_]f32{ 1.0, std.math.nan(f32), 2.0 }));
    try std.testing.expectEqual(@as(usize, 2), countNonFinite(&[_]f32{ std.math.inf(f32), -std.math.inf(f32), 2.0 }));
    try std.testing.expectEqual(@as(usize, 0), countNonFinite(&[_]f32{ 1.0, 2.0, -3.0 }));
}

test "clipGradGlobalNorm con NaN/Inf escala a cero" {
    var g1 = [_]f32{ 1.0, std.math.nan(f32) };
    const groups = [_][]f32{&g1};
    const norm = clipGradGlobalNorm(&groups, 1.0);
    try std.testing.expect(!std.math.isFinite(norm));
    try std.testing.expectEqual(@as(f32, 0.0), g1[0]);
    try std.testing.expectEqual(@as(f32, 0.0), g1[1]);
}

test "warmup_cosine: sube en warmup y baja en coseno" {
    const s = Scheduler{ .warmup_cosine = .{ .max_lr = 1.0, .min_lr = 0.0, .warmup_steps = 10, .max_steps = 110 } };
    // dentro del warmup, crece linealmente
    try std.testing.expect(s.getLR(0) < s.getLR(5));
    try std.testing.expect(s.getLR(5) < s.getLR(9));
    // pico ~max_lr al final del warmup
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), s.getLR(9), 1e-5);
    // después decae
    try std.testing.expect(s.getLR(60) < 1.0);
    try std.testing.expect(s.getLR(110) < s.getLR(60));
}

test "TokenDataset entrega ventanas (input, target+1)" {
    // 8 tokens u16 LE: 10,11,12,13,14,15,16,17
    var bytes: [16]u8 = undefined;
    for (0..8) |i| std.mem.writeInt(u16, bytes[i * 2 ..][0..2], @intCast(10 + i), .little);
    var ds = TokenDataset.fromBytes(&bytes);
    defer ds.deinit();
    try std.testing.expectEqual(@as(usize, 8), ds.n);
    try std.testing.expectEqual(@as(u32, 13), ds.tokenAt(3));
    var ids: [3]u32 = undefined;
    var tgt: [3]u32 = undefined;
    var prng = std.Random.DefaultPrng.init(1);
    ds.sampleWindow(prng.random(), &ids, &tgt);
    // target es input desplazado +1
    for (0..3) |i| try std.testing.expectEqual(ids[i] + 1, tgt[i]);
}
