// ztui — TUI cell-based MVP (motor en engine.zig + helpers de sesión)
const std = @import("std");
const zagent = @import("zagent");

const engine = @import("ztui_engine");
pub const runChatLoop = engine.runChatLoop;
pub const Tui = engine.Tui;

fn sessionDir(alloc: std.mem.Allocator) ![]u8 {
    return try std.fs.path.join(alloc, &.{ ".fluency", "sessions" });
}

fn ensureSessionDir(io: std.Io, alloc: std.mem.Allocator) !void {
    const dir = try sessionDir(alloc);
    defer alloc.free(dir);
    _ = std.Io.Dir.cwd().createDirPath(io, dir) catch {};
}

fn sessionPath(alloc: std.mem.Allocator, name: []const u8) ![]u8 {
    const dir = try sessionDir(alloc);
    defer alloc.free(dir);
    return try std.fs.path.join(alloc, &.{ dir, name });
}

// =====================================================================
// Tests sustantivos (motor cell-based)
// =====================================================================

test "ztui Tui init/deinit compiles" {
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var tui = try Tui.init(std.testing.allocator, io, 80, 24);
    defer tui.deinit();
    try std.testing.expect(tui.screen.w == 80);
    try std.testing.expect(tui.screen.h == 24);
}

test "ztui Tui addEntry and paint" {
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var tui = try Tui.init(std.testing.allocator, io, 40, 10);
    defer tui.deinit();
    try tui.addEntry(.user, "hello world");
    try std.testing.expect(tui.transcript.entries.items.len == 1);
    // compose() rellena la rejilla SIN escribir a stdout (paint() escribiría ANSI
    // al canal IPC del test-runner y lo colgaría). Verificamos que compone bien.
    try tui.compose();
    try std.testing.expect(tui.screen.cells.len == 40 * 10);
}

test "ztui Tui appendStream" {
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var tui = try Tui.init(std.testing.allocator, io, 80, 24);
    defer tui.deinit();
    try tui.addEntry(.assistant, "hello");
    try tui.appendStream(" world");
    try std.testing.expect(std.mem.eql(u8, tui.transcript.entries.items[0].text.items, "hello world"));
}

test "ztui session path helper" {
    const alloc = std.testing.allocator;
    const path = try sessionPath(alloc, "demo");
    defer alloc.free(path);
    try std.testing.expect(std.mem.endsWith(u8, path, ".fluency\\sessions\\demo") or
        std.mem.endsWith(u8, path, ".fluency/sessions/demo"));
}
