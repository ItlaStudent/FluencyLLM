// screen.zig — motor cell-based para TUI (100% Zig, ANSI)
const std = @import("std");

pub const Color = struct { r: u8, g: u8, b: u8 };

pub const Style = struct {
    fg: Color = .{ .r = 255, .g = 255, .b = 255 },
    bg: Color = .{ .r = 0, .g = 0, .b = 0 },
    bold: bool = false,
};

pub const Cell = struct {
    ch: u21 = ' ',
    style: Style = .{},
};

pub const Rect = struct {
    x: u16,
    y: u16,
    w: u16,
    h: u16,
};

pub const Screen = struct {
    w: u16,
    h: u16,
    cells: []Cell,
    prev: []Cell,
    alloc: std.mem.Allocator,

    pub fn init(alloc: std.mem.Allocator, width: u16, height: u16) !Screen {
        const n = @as(usize, width) * height;
        const cells = try alloc.alloc(Cell, n);
        const prev = try alloc.alloc(Cell, n);
        @memset(cells, .{});
        @memset(prev, .{});
        return .{ .w = width, .h = height, .cells = cells, .prev = prev, .alloc = alloc };
    }

    pub fn deinit(self: *Screen) void {
        self.alloc.free(self.cells);
        self.alloc.free(self.prev);
    }

    pub fn clear(self: *Screen) void {
        @memset(self.cells, .{});
    }

    pub fn putCell(self: *Screen, x: u16, y: u16, cell: Cell) void {
        if (x >= self.w or y >= self.h) return;
        self.cells[@as(usize, y) * self.w + x] = cell;
    }

    pub fn putStr(self: *Screen, x: u16, y: u16, str: []const u8, style: Style) void {
        var cx = x;
        var cy = y;
        var i: usize = 0;
        while (i < str.len) {
            if (cy >= self.h) break;
            if (cx >= self.w) {
                cx = x;
                cy += 1;
                if (cy >= self.h) break;
            }
            const len = std.unicode.utf8ByteSequenceLength(str[i]) catch 1;
            if (i + len > str.len) break;
            const cp = std.unicode.utf8Decode(str[i .. i + len]) catch @as(u21, '?');
            self.putCell(cx, cy, .{ .ch = cp, .style = style });
            cx += 1;
            i += len;
        }
    }

    /// Fill rect with a single char/style.
    pub fn fill(self: *Screen, rect: Rect, ch: u21, style: Style) void {
        const x1 = @min(rect.x, self.w);
        const x2 = @min(rect.x + rect.w, self.w);
        const y1 = @min(rect.y, self.h);
        const y2 = @min(rect.y + rect.h, self.h);
        for (y1..y2) |y| {
            for (x1..x2) |x| {
                self.cells[y * self.w + x] = .{ .ch = ch, .style = style };
            }
        }
    }

    /// Render diff: only emit ANSI for changed cells.
    pub fn render(self: *Screen, stdout: std.Io.File, io: std.Io) !void {
        var esc_buf: [128]u8 = undefined;
        try stdout.writeStreamingAll(io, "\x1b[?25l"); // hide cursor

        for (0..self.h) |y| {
            for (0..self.w) |x| {
                const idx = y * self.w + x;
                const cell = self.cells[idx];
                const prev = self.prev[idx];
                if (cellEq(cell, prev)) continue;

                const pos = try std.fmt.bufPrint(&esc_buf, "\x1b[{};{}H", .{ y + 1, x + 1 });
                try stdout.writeStreamingAll(io, pos);

                const color = try std.fmt.bufPrint(&esc_buf, "\x1b[38;2;{};{};{};48;2;{};{};{}m", .{
                    cell.style.fg.r, cell.style.fg.g, cell.style.fg.b,
                    cell.style.bg.r, cell.style.bg.g, cell.style.bg.b,
                });
                try stdout.writeStreamingAll(io, color);
                if (cell.style.bold) try stdout.writeStreamingAll(io, "\x1b[1m");

                var ubuf: [4]u8 = undefined;
                const ulen = std.unicode.utf8Encode(@intCast(cell.ch), &ubuf) catch blk: {
                    ubuf[0] = '?';
                    break :blk 1;
                };
                try stdout.writeStreamingAll(io, ubuf[0..ulen]);
            }
        }

        try stdout.writeStreamingAll(io, "\x1b[0m\x1b[?25h"); // reset + show cursor
        @memcpy(self.prev, self.cells);
    }
};

fn cellEq(a: Cell, b: Cell) bool {
    return a.ch == b.ch and a.style.fg.r == b.style.fg.r and a.style.fg.g == b.style.fg.g and
        a.style.fg.b == b.style.fg.b and a.style.bg.r == b.style.bg.r and a.style.bg.g == b.style.bg.g and
        a.style.bg.b == b.style.bg.b and a.style.bold == b.style.bold;
}

// =====================================================================
// Theme & Layout
// =====================================================================

pub const ThemeKind = enum { dark, light };

pub const Theme = struct {
    header_fg: Color = .{ .r = 0, .g = 0, .b = 0 },
    header_bg: Color = .{ .r = 100, .g = 200, .b = 255 },
    transcript_fg: Color = .{ .r = 220, .g = 220, .b = 220 },
    transcript_bg: Color = .{ .r = 18, .g = 18, .b = 18 },
    input_fg: Color = .{ .r = 255, .g = 255, .b = 200 },
    input_bg: Color = .{ .r = 30, .g = 30, .b = 30 },
    status_fg: Color = .{ .r = 150, .g = 150, .b = 150 },
    status_bg: Color = .{ .r = 10, .g = 10, .b = 10 },
    user_label: Color = .{ .r = 100, .g = 255, .b = 150 },
    assistant_label: Color = .{ .r = 100, .g = 200, .b = 255 },
    tool_label: Color = .{ .r = 255, .g = 200, .b = 100 },
    diff_minus: Color = .{ .r = 255, .g = 100, .b = 100 },
    diff_plus: Color = .{ .r = 100, .g = 255, .b = 100 },
    // Markdown
    heading: Color = .{ .r = 255, .g = 220, .b = 120 },
    code_fg: Color = .{ .r = 180, .g = 230, .b = 180 },
    code_bg: Color = .{ .r = 40, .g = 40, .b = 40 },
    bullet: Color = .{ .r = 120, .g = 200, .b = 255 },
    // Syntax highlighting (code blocks)
    syntax_keyword: Color = .{ .r = 100, .g = 180, .b = 255 },
    syntax_string: Color = .{ .r = 200, .g = 160, .b = 100 },
    syntax_comment: Color = .{ .r = 110, .g = 110, .b = 130 },
    syntax_number: Color = .{ .r = 180, .g = 180, .b = 255 },
    // Selección/foco y overlay (paleta)
    sel_fg: Color = .{ .r = 0, .g = 0, .b = 0 },
    sel_bg: Color = .{ .r = 100, .g = 200, .b = 255 },
    overlay_fg: Color = .{ .r = 230, .g = 230, .b = 230 },
    overlay_bg: Color = .{ .r = 32, .g = 32, .b = 48 },

    pub fn dark() Theme {
        return .{};
    }

    pub fn light() Theme {
        return .{
            .header_fg = .{ .r = 255, .g = 255, .b = 255 },
            .header_bg = .{ .r = 30, .g = 110, .b = 200 },
            .transcript_fg = .{ .r = 30, .g = 30, .b = 30 },
            .transcript_bg = .{ .r = 248, .g = 248, .b = 248 },
            .input_fg = .{ .r = 20, .g = 20, .b = 20 },
            .input_bg = .{ .r = 230, .g = 230, .b = 230 },
            .status_fg = .{ .r = 90, .g = 90, .b = 90 },
            .status_bg = .{ .r = 215, .g = 215, .b = 215 },
            .user_label = .{ .r = 0, .g = 140, .b = 70 },
            .assistant_label = .{ .r = 20, .g = 90, .b = 200 },
            .tool_label = .{ .r = 170, .g = 110, .b = 0 },
            .diff_minus = .{ .r = 190, .g = 30, .b = 30 },
            .diff_plus = .{ .r = 20, .g = 140, .b = 20 },
            .heading = .{ .r = 150, .g = 90, .b = 0 },
            .code_fg = .{ .r = 40, .g = 90, .b = 40 },
            .code_bg = .{ .r = 228, .g = 228, .b = 228 },
            .bullet = .{ .r = 20, .g = 90, .b = 200 },
            .syntax_keyword = .{ .r = 20, .g = 80, .b = 200 },
            .syntax_string = .{ .r = 160, .g = 100, .b = 20 },
            .syntax_comment = .{ .r = 110, .g = 110, .b = 120 },
            .syntax_number = .{ .r = 80, .g = 60, .b = 180 },
            .sel_fg = .{ .r = 255, .g = 255, .b = 255 },
            .sel_bg = .{ .r = 30, .g = 110, .b = 200 },
            .overlay_fg = .{ .r = 20, .g = 20, .b = 20 },
            .overlay_bg = .{ .r = 220, .g = 224, .b = 235 },
        };
    }

    pub fn of(kind: ThemeKind) Theme {
        return switch (kind) {
            .dark => dark(),
            .light => light(),
        };
    }
};

pub const Layout = struct {
    header: Rect,
    transcript: Rect,
    input: Rect,
    status: Rect,

    pub fn calc(w: u16, h: u16) Layout {
        const header_h = @min(1, h);
        const status_h = @min(1, h);
        const input_h = @min(3, h - header_h - status_h);
        const trans_h = h - header_h - input_h - status_h;
        return .{
            .header = .{ .x = 0, .y = 0, .w = w, .h = header_h },
            .transcript = .{ .x = 0, .y = header_h, .w = w, .h = trans_h },
            .input = .{ .x = 0, .y = header_h + trans_h, .w = w, .h = input_h },
            .status = .{ .x = 0, .y = header_h + trans_h + input_h, .w = w, .h = status_h },
        };
    }
};

// =====================================================================
// Tests
// =====================================================================

test "Screen init/clear/deinit" {
    var s = try Screen.init(std.testing.allocator, 10, 5);
    defer s.deinit();
    try std.testing.expect(s.w == 10);
    try std.testing.expect(s.h == 5);
    s.clear();
    try std.testing.expect(s.cells[0].ch == ' ');
}

test "Screen putStr basic" {
    var s = try Screen.init(std.testing.allocator, 20, 3);
    defer s.deinit();
    s.putStr(0, 0, "hello", .{ .fg = .{ .r = 255, .g = 255, .b = 255 }, .bg = .{ .r = 0, .g = 0, .b = 0 } });
    try std.testing.expect(s.cells[0].ch == 'h');
    try std.testing.expect(s.cells[4].ch == 'o');
}

test "Screen fill rect" {
    var s = try Screen.init(std.testing.allocator, 10, 10);
    defer s.deinit();
    s.fill(.{ .x = 1, .y = 1, .w = 3, .h = 3 }, '#', .{});
    try std.testing.expect(s.cells[1 * 10 + 1].ch == '#');
    try std.testing.expect(s.cells[3 * 10 + 3].ch == '#');
    try std.testing.expect(s.cells[0].ch == ' ');
}

test "cellEq equality" {
    const a = Cell{ .ch = 'a', .style = .{ .fg = .{ .r = 1, .g = 2, .b = 3 }, .bg = .{ .r = 4, .g = 5, .b = 6 } } };
    const b = Cell{ .ch = 'a', .style = .{ .fg = .{ .r = 1, .g = 2, .b = 3 }, .bg = .{ .r = 4, .g = 5, .b = 6 } } };
    const c = Cell{ .ch = 'b', .style = .{ .fg = .{ .r = 1, .g = 2, .b = 3 }, .bg = .{ .r = 4, .g = 5, .b = 6 } } };
    try std.testing.expect(cellEq(a, b));
    try std.testing.expect(!cellEq(a, c));
}

test "Layout calc" {
    const l = Layout.calc(80, 24);
    try std.testing.expect(l.header.h == 1);
    try std.testing.expect(l.status.h == 1);
    try std.testing.expect(l.input.h == 3);
    try std.testing.expect(l.transcript.h == 19);
}
