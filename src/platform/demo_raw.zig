// demo_raw.zig — demo mínima de terminal crudo
// Uso: zig run src/platform/demo_raw.zig (desde el repo)
const std = @import("std");
const tty = @import("tty.zig");

pub fn main() !void {
    var threaded = std.Io.Threaded.init(std.heap.page_allocator, .{});
    const io = threaded.io();
    defer threaded.deinit();

    var t = tty.RawTty.init();
    var raw: bool = false;
    t.enable(io) catch {};
    raw = t.enabled;
    defer if (raw) t.deinit(io);

    const stdout = std.Io.File.stdout();
    if (raw) {
        try stdout.writeStreamingAll(io, "Raw mode ON. Press keys (q to quit):\r\n");
    } else {
        try stdout.writeStreamingAll(io, "Not a terminal (fallback buffered mode). Type 'q' to quit.\r\n");
    }

    while (true) {
        var keybuf: [1]u8 = undefined;
        var bufs = [1][]u8{&keybuf};
        const n = std.Io.File.readStreaming(std.Io.File.stdin(), io, &bufs) catch |err| switch (err) {
            error.EndOfStream => 0,
            else => return err,
        };
        if (n == 0) continue;
        const b = keybuf[0];
        if (b == 'q') break;

        var outbuf: [64]u8 = undefined;
        const msg = if (b == 0x1b)
            "ESC\r\n"
        else if (std.ascii.isPrint(b))
            try std.fmt.bufPrint(&outbuf, "{c}  0x{x:0>2}\r\n", .{ b, b })
        else
            try std.fmt.bufPrint(&outbuf, "?  0x{x:0>2}\r\n", .{b});
        try stdout.writeStreamingAll(io, msg);
    }

    try stdout.writeStreamingAll(io, "Bye!\r\n");
}
