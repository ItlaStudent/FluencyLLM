// preview.zig — vista previa NO interactiva de la TUI, usando el MOTOR REAL.
// Construye un Tui, compone dos frames (bienvenida vacía + conversación con
// Markdown/diff) y vuelca la rejilla de celdas como texto plano (sin ANSI).
//
// Uso:  zig build preview
const std = @import("std");
const engine = @import("ztui_engine");

fn dumpScreen(io: std.Io, stdout: std.Io.File, alloc: std.mem.Allocator, tui: *engine.Tui, title: []const u8) !void {
    const W = tui.screen.w;
    const H = tui.screen.h;
    var line = std.ArrayList(u8).empty;
    defer line.deinit(alloc);

    try stdout.writeStreamingAll(io, title);
    try stdout.writeStreamingAll(io, "\n");

    line.clearRetainingCapacity();
    try line.append(alloc, '+');
    for (0..W) |_| try line.append(alloc, '-');
    try line.appendSlice(alloc, "+\n");
    try stdout.writeStreamingAll(io, line.items);

    for (0..H) |y| {
        line.clearRetainingCapacity();
        try line.append(alloc, '|');
        for (0..W) |x| {
            const cell = tui.screen.cells[y * W + x];
            var ubuf: [4]u8 = undefined;
            const n = std.unicode.utf8Encode(cell.ch, &ubuf) catch blk: {
                ubuf[0] = '?';
                break :blk 1;
            };
            try line.appendSlice(alloc, ubuf[0..n]);
        }
        try line.appendSlice(alloc, "|\n");
        try stdout.writeStreamingAll(io, line.items);
    }
    line.clearRetainingCapacity();
    try line.append(alloc, '+');
    for (0..W) |_| try line.append(alloc, '-');
    try line.appendSlice(alloc, "+\n\n");
    try stdout.writeStreamingAll(io, line.items);
}

pub fn main() !void {
    const alloc = std.heap.page_allocator;
    var threaded = std.Io.Threaded.init(alloc, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const stdout = std.Io.File.stdout();

    // Frame 1: pantalla de bienvenida (transcript vacío).
    {
        var tui = try engine.Tui.init(alloc, io, 80, 22);
        defer tui.deinit();
        tui.header = "Fluency · Chat · sesión: nueva";
        try tui.compose();
        try dumpScreen(io, stdout, alloc, &tui, "=== Frame 1: bienvenida (sesión vacía) ===");
    }

    // Frame 2: conversación con Markdown, bloque de código y diff en tool-call.
    {
        var tui = try engine.Tui.init(alloc, io, 80, 26);
        defer tui.deinit();
        tui.header = "Fluency · Fluency-1.5b-v1 · sesión: refactor-auth";
        try tui.addEntry(.user, "refactoriza el login para usar el nuevo **AuthService**");
        try tui.addEntry(.assistant,
            \\# Plan
            \\Voy a migrar el login. Pasos:
            \\- leer `src/auth/login.ts`
            \\- usar **authService.login**
            \\```
            \\const u = await authService.login(creds)
            \\```
        );
        try tui.addEntry(.tool,
            \\fs_write src/auth/login.ts
            \\- const u = await db.users.find(...)
            \\+ const u = await authService.login(creds)
        );
        try tui.compose();
        try dumpScreen(io, stdout, alloc, &tui, "=== Frame 2: conversación (markdown + diff en tool-call) ===");
    }
}
