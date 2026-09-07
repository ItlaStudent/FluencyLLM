// parallel — parallelFor sobre std.Thread (std.Thread.Pool fue removido en 0.16)
const std = @import("std");

/// Ejecuta `func(ctx, i)` para cada i en [0, n), repartiendo el rango en
/// ~num-cpu trozos contiguos, cada uno en su propio hilo. Si n es pequeño o
/// el spawn falla, cae a ejecución serial. La función debe ser segura para
/// llamarse desde varios hilos sobre índices disjuntos.
pub fn parallelFor(n: usize, ctx: anytype, comptime func: fn (@TypeOf(ctx), usize) void) void {
    if (n == 0) return;

    const Ctx = @TypeOf(ctx);

    // Umbral por debajo del cual no compensa paralelizar.
    const serial_threshold = 1024;

    const cpu_count = std.Thread.getCpuCount() catch 1;
    var workers = if (cpu_count == 0) 1 else cpu_count;
    if (n < serial_threshold or workers <= 1) {
        runRange(Ctx, func, ctx, 0, n);
        return;
    }
    if (workers > n) workers = n;

    const Worker = struct {
        fn run(c: Ctx, start: usize, end: usize) void {
            var i = start;
            while (i < end) : (i += 1) func(c, i);
        }
    };

    // Buffer de hilos en stack (tope razonable).
    const max_threads = 256;
    if (workers > max_threads) workers = max_threads;

    var threads: [max_threads]?std.Thread = .{null} ** max_threads;
    const chunk = (n + workers - 1) / workers; // ceil

    var spawned: usize = 0;
    var start: usize = 0;
    var w: usize = 0;
    while (w < workers and start < n) : (w += 1) {
        const end = @min(start + chunk, n);
        const t = std.Thread.spawn(.{}, Worker.run, .{ ctx, start, end }) catch {
            // Spawn falló: ejecutar este trozo en serie aquí mismo.
            Worker.run(ctx, start, end);
            start = end;
            continue;
        };
        threads[spawned] = t;
        spawned += 1;
        start = end;
    }

    // Si quedó cola sin asignar por algún motivo, procesarla en serie.
    if (start < n) Worker.run(ctx, start, n);

    var j: usize = 0;
    while (j < spawned) : (j += 1) {
        if (threads[j]) |t| t.join();
    }
}

fn runRange(comptime Ctx: type, comptime func: fn (Ctx, usize) void, ctx: Ctx, start: usize, end: usize) void {
    var i = start;
    while (i < end) : (i += 1) func(ctx, i);
}

// =====================================================================
// ThreadPool — pool de hilos para paralelismo persistente (std.Thread.Pool
// fue removido en Zig 0.16). Reutiliza el buffer de hilos entre llamadas
// para evitar alloc/free repetidos. Los workers se spawn por llamada y
// se join al completar, evitando busy-waiting.
// =====================================================================

pub const ThreadPool = struct {
    alloc: std.mem.Allocator,
    max_workers: usize,
    thread_buf: []std.Thread,

    pub fn init(alloc: std.mem.Allocator, n_workers: usize) !ThreadPool {
        const cpu_count = std.Thread.getCpuCount() catch 1;
        const workers = if (n_workers == 0) cpu_count else @min(n_workers, cpu_count);
        return .{
            .alloc = alloc,
            .max_workers = workers,
            .thread_buf = try alloc.alloc(std.Thread, workers),
        };
    }

    pub fn deinit(self: *ThreadPool) void {
        self.alloc.free(self.thread_buf);
    }

    /// Ejecuta `func(ctx, i)` para cada i en [0, n) repartiendo entre los workers.
    /// Spawn workers por llamada, reutilizando el buffer de hilos pre-asignado.
    /// Bloquea hasta que todas las tareas completan.
    pub fn parallelFor(self: *ThreadPool, n: usize, ctx: anytype, comptime func: fn (@TypeOf(ctx), usize) void) void {
        if (n == 0) return;

        const Ctx = @TypeOf(ctx);
        const serial_threshold = 1024;
        if (n < serial_threshold or self.max_workers <= 1) {
            runRange(Ctx, func, ctx, 0, n);
            return;
        }

        const Worker = struct {
            fn run(c: Ctx, start: usize, end: usize) void {
                var i = start;
                while (i < end) : (i += 1) func(c, i);
            }
        };

        var workers = self.max_workers;
        if (workers > n) workers = n;
        if (workers > self.thread_buf.len) workers = self.thread_buf.len;

        const chunk = (n + workers - 1) / workers;
        var spawned: usize = 0;
        var start: usize = 0;

        while (start < n and spawned < workers) {
            const end = @min(start + chunk, n);
            self.thread_buf[spawned] = std.Thread.spawn(.{}, Worker.run, .{ ctx, start, end }) catch {
                // Spawn falló: ejecutar este trozo en serie
                Worker.run(ctx, start, end);
                start = end;
                continue;
            };
            spawned += 1;
            start = end;
        }

        // Procesar resto en serie si quedó cola
        if (start < n) Worker.run(ctx, start, n);

        // Join todos los hilos spawnados
        for (0..spawned) |i| {
            self.thread_buf[i].join();
        }
    }
};

// --- tests ---
const FillCtx = struct {
    data: []usize,
    fn fill(self: @This(), i: usize) void {
        self.data[i] = i * 2;
    }
};

test "parallelFor fills large array" {
    const alloc = std.testing.allocator;
    const n = 10000;
    const data = try alloc.alloc(usize, n);
    defer alloc.free(data);
    @memset(data, 0);

    const ctx = FillCtx{ .data = data };
    parallelFor(n, ctx, FillCtx.fill);

    for (data, 0..) |v, i| {
        try std.testing.expectEqual(i * 2, v);
    }
}

test "parallelFor small array serial path" {
    const alloc = std.testing.allocator;
    const n = 10;
    const data = try alloc.alloc(usize, n);
    defer alloc.free(data);
    @memset(data, 0);

    const ctx = FillCtx{ .data = data };
    parallelFor(n, ctx, FillCtx.fill);

    for (data, 0..) |v, i| {
        try std.testing.expectEqual(i * 2, v);
    }
}

test "parallelFor zero is noop" {
    var dummy: [1]usize = .{0};
    const ctx = FillCtx{ .data = &dummy };
    parallelFor(0, ctx, FillCtx.fill);
    try std.testing.expectEqual(@as(usize, 0), dummy[0]);
}

// --- ThreadPool tests ---

test "ThreadPool init/deinit sin fugas" {
    var pool = try ThreadPool.init(std.testing.allocator, 2);
    defer pool.deinit();
    try std.testing.expect(pool.thread_buf.len == 2);
}

test "ThreadPool parallelFor fills array" {
    const alloc = std.testing.allocator;
    var pool = try ThreadPool.init(alloc, 2);
    defer pool.deinit();

    const n = 500;
    const data = try alloc.alloc(usize, n);
    defer alloc.free(data);
    @memset(data, 0);

    const ctx = FillCtx{ .data = data };
    pool.parallelFor(n, ctx, FillCtx.fill);

    for (data, 0..) |v, i| {
        try std.testing.expectEqual(i * 2, v);
    }
}

test "ThreadPool parallelFor zero is noop" {
    var pool = try ThreadPool.init(std.testing.allocator, 2);
    defer pool.deinit();

    var dummy: [1]usize = .{0};
    const ctx = FillCtx{ .data = &dummy };
    pool.parallelFor(0, ctx, FillCtx.fill);
    try std.testing.expectEqual(@as(usize, 0), dummy[0]);
}
