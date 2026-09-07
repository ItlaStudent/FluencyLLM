const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    // Ejecutar `zig build test` como subproceso
    // Intenta primero `zig` en PATH; si no, usa el zig local descargado.
    const result = std.process.run(alloc, io, .{
        .argv = &.{ "zig", "build", "test" },
    }) catch |err| blk: {
        if (err == error.FileNotFound) {
            std.log.warn("`zig` no encontrado en PATH; intentando zig local...", .{});
            break :blk try std.process.run(alloc, io, .{
                .argv = &.{ "zig-0.16.0\\zig-x86_64-windows-0.16.0\\zig.exe", "build", "test" },
            });
        }
        return err;
    };
    defer {
        alloc.free(result.stdout);
        alloc.free(result.stderr);
    }

    if (result.term != .exited or result.term.exited != 0) {

        std.log.err("gate_quality FALLÓ: `zig build test` no pasó.\nstdout:\n{s}\nstderr:\n{s}", .{ result.stdout, result.stderr });
        std.process.exit(1);
    }

    std.log.info("gate_quality: OK — tests pasan sin leaks.", .{});
}
