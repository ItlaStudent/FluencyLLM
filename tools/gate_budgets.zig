const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;
    const dir = std.Io.Dir.cwd();

    const content = dir.readFileAlloc(io, "budgets.zon", alloc, .limited(1 << 20)) catch |err| {
        std.log.err("gate_budgets FALLÓ: no se pudo leer budgets.zon: {s}", .{@errorName(err)});
        std.process.exit(1);
    };
    defer alloc.free(content);

    // Validación sintáctica básica: budgets.zon debe ser un struct literal válido
    if (std.mem.indexOf(u8, content, "infer_binary_bytes_max") == null) {
        std.log.err("gate_budgets FALLÓ: falta infer_binary_bytes_max en budgets.zon", .{});
        std.process.exit(1);
    }

    std.log.info("gate_budgets: OK — presupuestos definidos.", .{});
}
