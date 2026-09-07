// fluency_launcher — equivalente en Zig de fluency.cmd.
// Compila (si hace falta) y lanza el binario `fluency`.
// Uso:  zig run fluency_launcher.zig -- [args...]
//       (o compilar con: zig build-exe fluency_launcher.zig -OReleaseFast)

const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;
    const dir = std.Io.Dir.cwd();

    // Resolver rutas relativas al directorio de este script.
    // Asumimos que se ejecuta desde la raíz del repo.
    const zig_path = "zig-0.16.0/zig-x86_64-windows-0.16.0/zig.exe";
    const bin_path = "zig-out/bin/fluency.exe";

    // Verificar que el compilador Zig existe.
    dir.access(io, zig_path, .{}) catch {
        try io.stdout().print("No se encontro el compilador Zig en {s}.\n", .{zig_path});
        try io.stdout().print("Coloca el toolchain de Zig 0.16 en esa ruta y reintenta.\n", .{});
        std.process.exit(1);
    };

    // Si el binario no existe, compilarlo.
    dir.access(io, bin_path, .{}) catch {
        try io.stdout().print("Compilando por primera vez (ReleaseFast)...\n", .{});
        const build_args = [_][]const u8{ zig_path, "build", "-Doptimize=ReleaseFast" };
        var child = std.process.Child.init(build_args, alloc);
        child.cwd = ".";
        const term = child.spawnAndWait() catch |err| {
            try io.stdout().print("Fallo la compilacion: {s}\n", .{@errorName(err)});
            std.process.exit(1);
        };
        if (term != .exited or term.exited != 0) {
            try io.stdout().print("Fallo la compilacion (codigo {d}).\n", .{if (term == .exited) term.exited else -1});
            std.process.exit(1);
        }
    };

    // Recoger argumentos del usuario (todo lo después del nombre del programa).
    var arg_iter = init.args;
    _ = arg_iter.next(); // skip program name
    var user_args = std.ArrayList([]const u8).empty;
    defer user_args.deinit(alloc);

    while (arg_iter.next()) |a| {
        try user_args.append(alloc, a);
    }

    // Construir argv: [bin_path, user_args...]
    var argv = std.ArrayList([]const u8).empty;
    defer argv.deinit(alloc);
    try argv.append(alloc, bin_path);
    for (user_args.items) |a| {
        try argv.append(alloc, a);
    }

    // Si no hay argumentos, lanzar `chat` por defecto.
    if (argv.items.len == 1) {
        try argv.append(alloc, "chat");
    }

    var child = std.process.Child.init(argv.items, alloc);
    child.cwd = ".";
    const term = child.spawnAndWait() catch |err| {
        try io.stdout().print("Error al lanzar fluency: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
    if (term == .exited) {
        std.process.exit(term.exited);
    }
    std.process.exit(1);
}
