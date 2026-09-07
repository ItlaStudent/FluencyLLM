// transcript.zig — historial de mensajes con scroll, word-wrap y render Markdown.
const std = @import("std");
const screen = @import("screen.zig");
const Screen = screen.Screen;
const Style = screen.Style;
const Rect = screen.Rect;
const Theme = screen.Theme;

pub const Role = enum { user, assistant, tool };

pub const Entry = struct {
    role: Role,
    text: std.ArrayList(u8),
    // true si está plegada (solo se muestra un resumen). Útil para tool-calls.
    folded: bool = false,

    pub fn deinit(self: *Entry, alloc: std.mem.Allocator) void {
        self.text.deinit(alloc);
    }
};

/// Tipo de línea ya "aplanada" para pintar (lleva el estilo a aplicar).
const LineKind = enum { label, normal, heading, bullet, code, fence, diff_plus, diff_minus, blank, fold };

const Line = struct {
    kind: LineKind,
    role: Role = .assistant, // para labels
    text: []const u8, // dup propiedad del Transcript.freeLines
    entry_index: usize = 0,
};

/// Historial con capacidad de scroll, word-wrap y Markdown.
pub const Transcript = struct {
    alloc: std.mem.Allocator,
    entries: std.ArrayList(Entry),
    scroll: usize = 0,

    pub fn init(alloc: std.mem.Allocator) Transcript {
        return .{ .alloc = alloc, .entries = std.ArrayList(Entry).empty };
    }

    pub fn deinit(self: *Transcript) void {
        for (self.entries.items) |*e| e.deinit(self.alloc);
        self.entries.deinit(self.alloc);
    }

    pub fn add(self: *Transcript, role: Role, text: []const u8) !void {
        var entry = Entry{ .role = role, .text = std.ArrayList(u8).empty };
        try entry.text.appendSlice(self.alloc, text);
        try self.entries.append(self.alloc, entry);
    }

    pub fn appendToLast(self: *Transcript, text: []const u8) !void {
        if (self.entries.items.len == 0) return;
        try self.entries.items[self.entries.items.len - 1].text.appendSlice(self.alloc, text);
    }

    pub fn scrollUp(self: *Transcript, lines: usize) void {
        self.scroll -|= lines;
    }

    pub fn scrollDown(self: *Transcript, lines: usize, total_lines: usize, visible: u16) void {
        const max_scroll = if (total_lines > visible) total_lines - visible else 0;
        self.scroll = @min(self.scroll + lines, max_scroll);
    }

    /// Word-wrap de una línea a ancho max_width. Las líneas resultantes se
    /// almacenan en `out` (el caller debe liberar con `freeLines`).
    pub fn wrapLine(self: *Transcript, line: []const u8, max_width: u16, out: *std.ArrayList([]const u8)) !void {
        if (line.len == 0) {
            try out.append(self.alloc, try self.alloc.dupe(u8, ""));
            return;
        }
        var start: usize = 0;
        while (start < line.len) {
            const end = @min(start + max_width, line.len);
            var cut = end;
            if (end < line.len) {
                var space_search: usize = end;
                while (space_search > start and line[space_search] != ' ') space_search -= 1;
                if (space_search > start) cut = space_search;
            }
            try out.append(self.alloc, try self.alloc.dupe(u8, line[start..cut]));
            start = if (cut < line.len and line[cut] == ' ') cut + 1 else cut;
        }
    }

    pub fn freeLines(self: *Transcript, lines: *std.ArrayList([]const u8)) void {
        for (lines.items) |ln| self.alloc.free(ln);
        lines.deinit(self.alloc);
    }

    fn freeRenderLines(self: *Transcript, lines: *std.ArrayList(Line)) void {
        for (lines.items) |ln| self.alloc.free(ln.text);
        lines.deinit(self.alloc);
    }

    /// Clasifica una línea de contenido (fuera de bloque de código) por su prefijo.
    fn classify(role: Role, raw: []const u8) LineKind {
        const t = std.mem.trimStart(u8,raw, " ");
        if (std.mem.startsWith(u8, t, "### ") or std.mem.startsWith(u8, t, "## ") or std.mem.startsWith(u8, t, "# ")) return .heading;
        // Los diffs solo se colorean dentro de tool-calls (en assistant, "- " es viñeta).
        if (role == .tool) {
            if (std.mem.startsWith(u8, t, "+ ") or std.mem.startsWith(u8, t, "+")) return .diff_plus;
            if (std.mem.startsWith(u8, t, "- ") or std.mem.startsWith(u8, t, "-")) return .diff_minus;
        }
        if (std.mem.startsWith(u8, t, "- ") or std.mem.startsWith(u8, t, "* ")) return .bullet;
        return .normal;
    }

    /// Construye la lista de líneas aplanadas (con estilos) de todas las entradas.
    fn buildLines(self: *Transcript, out: *std.ArrayList(Line), max_width: u16) !void {
        for (self.entries.items, 0..) |entry, ei| {
            try out.append(self.alloc, .{ .kind = .label, .role = entry.role, .text = try self.alloc.dupe(u8, roleLabel(entry.role)), .entry_index = ei });

            if (entry.folded) {
                var nlines: usize = 1;
                for (entry.text.items) |c| {
                    if (c == '\n') nlines += 1;
                }
                const msg = try std.fmt.allocPrint(self.alloc, "  … plegado ({d} líneas) — Espacio para desplegar", .{nlines});
                try out.append(self.alloc, .{ .kind = .fold, .role = entry.role, .text = msg, .entry_index = ei });
                try out.append(self.alloc, .{ .kind = .blank, .text = try self.alloc.dupe(u8, ""), .entry_index = ei });
                continue;
            }

            var in_code = false;
            var it = std.mem.splitScalar(u8, entry.text.items, '\n');
            while (it.next()) |raw_line| {
                const trimmed = std.mem.trimStart(u8,raw_line, " ");
                if (std.mem.startsWith(u8, trimmed, "```")) {
                    in_code = !in_code;
                    try out.append(self.alloc, .{ .kind = .fence, .role = entry.role, .text = try self.alloc.dupe(u8, "────────"), .entry_index = ei });
                    continue;
                }
                const kind: LineKind = if (in_code) .code else classify(entry.role, raw_line);
                // word-wrap conservando el tipo
                var wrapped = std.ArrayList([]const u8).empty;
                defer self.freeLines(&wrapped);
                try self.wrapLine(raw_line, if (max_width > 2) max_width - 2 else 1, &wrapped);
                for (wrapped.items) |wln| {
                    try out.append(self.alloc, .{ .kind = kind, .role = entry.role, .text = try self.alloc.dupe(u8, wln), .entry_index = ei });
                }
            }
            try out.append(self.alloc, .{ .kind = .blank, .text = try self.alloc.dupe(u8, ""), .entry_index = ei });
        }
    }

    /// Pinta texto con Markdown inline (**negrita**, `código`) celda a celda,
    /// empezando en (x0,y), sin pasar de max_x. Devuelve el x final.
    fn paintInline(s: *Screen, x0: u16, y: u16, max_x: u16, text: []const u8, base: Style, theme: Theme) void {
        var cx = x0;
        var bold = base.bold;
        var code = false;
        var i: usize = 0;
        while (i < text.len) {
            if (cx >= max_x) break;
            // marcadores
            if (i + 1 < text.len and text[i] == '*' and text[i + 1] == '*') {
                bold = !bold;
                i += 2;
                continue;
            }
            if (text[i] == '`') {
                code = !code;
                i += 1;
                continue;
            }
            const len = std.unicode.utf8ByteSequenceLength(text[i]) catch 1;
            if (i + len > text.len) break;
            const cp = std.unicode.utf8Decode(text[i .. i + len]) catch '?';
            var st = base;
            st.bold = bold;
            if (code) {
                st.fg = theme.code_fg;
                st.bg = theme.code_bg;
            }
            s.putCell(cx, y, .{ .ch = cp, .style = st });
            cx += 1;
            i += len;
        }
    }

    /// Cuántas líneas de render produce el transcript (para el cálculo de scroll).
    pub fn totalRenderLines(self: *Transcript, max_width: u16) !usize {
        var lines = std.ArrayList(Line).empty;
        defer self.freeRenderLines(&lines);
        try self.buildLines(&lines, max_width);
        return lines.items.len;
    }

    /// Dibuja el transcript dentro del rectángulo. `focus` resalta el label de
    /// la entrada enfocada (navegación/plegado).
/// Pinta una línea de código con resaltado básico de sintaxis (keywords, strings, comments, numbers).
/// Soporta Zig, Rust, C, JS/TS, Python — suficientemente genérico para la mayoría de bloques.
fn paintCodeLine(s: *Screen, x0: u16, y: u16, max_x: u16, text: []const u8, theme: Theme) void {
    const base = Style{ .fg = theme.code_fg, .bg = theme.code_bg };
    var i: usize = 0;
    var x = x0;
    while (i < text.len) {
        // Comment: // or # (until end of line)
        if (i + 1 < text.len and text[i] == '/' and text[i + 1] == '/') {
            s.putStr(x, y, text[i..], .{ .fg = theme.syntax_comment, .bg = theme.code_bg });
            return;
        }
        if (text[i] == '#' and (i == 0 or text[i - 1] == ' ' or text[i - 1] == '\t')) {
            s.putStr(x, y, text[i..], .{ .fg = theme.syntax_comment, .bg = theme.code_bg });
            return;
        }
        // String: "..." or '...'
        if (text[i] == '"' or text[i] == '\'') {
            const quote = text[i];
            const start = i;
            i += 1;
            while (i < text.len and text[i] != quote) {
                if (text[i] == '\\' and i + 1 < text.len) i += 1;
                i += 1;
            }
            if (i < text.len) i += 1; // closing quote
            const seg = text[start..i];
            const w = @min(seg.len, @as(usize, max_x -| x));
            s.putStr(x, y, seg[0..w], .{ .fg = theme.syntax_string, .bg = theme.code_bg });
            x += @intCast(w);
            continue;
        }
        // Number
        if (std.ascii.isDigit(text[i]) or (text[i] == '.' and i + 1 < text.len and std.ascii.isDigit(text[i + 1]))) {
            const start = i;
            while (i < text.len and (std.ascii.isDigit(text[i]) or text[i] == '.' or text[i] == 'x' or text[i] == 'X' or (text[i] >= 'a' and text[i] <= 'f') or (text[i] >= 'A' and text[i] <= 'F'))) i += 1;
            const seg = text[start..i];
            const w = @min(seg.len, @as(usize, max_x -| x));
            s.putStr(x, y, seg[0..w], .{ .fg = theme.syntax_number, .bg = theme.code_bg });
            x += @intCast(w);
            continue;
        }
        // Identifier / keyword
        if (std.ascii.isAlphabetic(text[i]) or text[i] == '_') {
            const start = i;
            while (i < text.len and (std.ascii.isAlphanumeric(text[i]) or text[i] == '_')) i += 1;
            const word = text[start..i];
            const is_kw = isKeyword(word);
            const st = if (is_kw) Style{ .fg = theme.syntax_keyword, .bg = theme.code_bg, .bold = true } else base;
            const w = @min(word.len, @as(usize, max_x -| x));
            s.putStr(x, y, word[0..w], st);
            x += @intCast(w);
            continue;
        }
        // Single char (operators, punctuation)
        if (x < max_x) {
            s.putCell(x, y, .{ .ch = @intCast(text[i]), .style = base });
            x += 1;
        }
        i += 1;
    }
}

/// Lista de keywords comunes a Zig/Rust/C/JS/TS/Python — suficientemente genérico.
fn isKeyword(word: []const u8) bool {
    const keywords = [_][]const u8{
        "fn",          "pub",       "const",     "var",       "let",     "mut",
        "if",          "else",      "while",     "for",       "return",  "break",
        "continue",    "switch",    "match",      "struct",   "enum",    "union",
        "interface",   "type",      "import",    "export",    "package", "module",
        "class",       "def",       "func",       "async",    "await",   "try",
        "catch",       "throw",     "raises",     "defer",    "errdefer","comptime",
        "inline",      "extern",    "export",     "volatile", "align",   "linksection",
        "callconv",    "packed",    "alignof",    "offsetof", "sizeof",  "test",
        "and",         "or",        "not",        "in",       "is",      "as",
        "with",        "from",      "lambda",     "yield",    "global",  "nonlocal",
        "True",        "False",     "None",       "nil",      "null",    "undefined",
        "void",        "bool",      "true",       "false",    "u8",      "u16",
        "u32",         "u64",       "i8",         "i16",      "i32",     "i64",
        "f32",         "f64",       "usize",      "isize",    "char",    "str",
        "int",         "float",     "double",     "string",   "self",    "Self",
        "this",        "super",     "init",       "deinit",   "new",     "delete",
        "malloc",      "free",      "print",      "println",  "if",      "elif",
        "else",        "endif",     "ifdef",      "ifndef",   "pragma",  "include",
        "using",       "namespace", "template",   "typename", "static",  "const",
        "constexpr",   "auto",      "go",         "chan",     "select",  "map",
        "make",        "range",     "type",       "struct",   "interface",
    };
    for (keywords) |kw| {
        if (std.mem.eql(u8, word, kw)) return true;
    }
    return false;
}

    pub fn paint(self: *Transcript, s: *Screen, rect: Rect, theme: Theme) !void {
        try self.paintFocus(s, rect, theme, null);
    }

    pub fn paintFocus(self: *Transcript, s: *Screen, rect: Rect, theme: Theme, focus: ?usize) !void {
        var lines = std.ArrayList(Line).empty;
        defer self.freeRenderLines(&lines);
        try self.buildLines(&lines, rect.w);

        const max_visible = @min(lines.items.len, rect.h);
        const start = if (self.scroll + max_visible > lines.items.len)
            (if (lines.items.len > max_visible) lines.items.len - max_visible else 0)
        else
            self.scroll;

        for (0..rect.h) |row| {
            const li = start + row;
            if (li >= lines.items.len) break;
            const line = lines.items[li];
            const y: u16 = @intCast(rect.y + row);
            const x0 = rect.x + 1;
            const max_x = rect.x + rect.w - 1;
            const focused = (focus != null and line.entry_index == focus.?);

            switch (line.kind) {
                .blank => {},
                .label => {
                    var st = Style{ .fg = labelColor(theme, line.role), .bg = theme.transcript_bg, .bold = true };
                    if (focused) {
                        st.fg = theme.sel_fg;
                        st.bg = theme.sel_bg;
                    }
                    s.putStr(x0, y, line.text, st);
                },
                .heading => {
                    paintInline(s, x0, y, max_x, line.text, .{ .fg = theme.heading, .bg = theme.transcript_bg, .bold = true }, theme);
                },
                .bullet => {
                    // sustituye el marcador "- "/"* " por "• "
                    const t = std.mem.trimStart(u8,line.text, " ");
                    s.putCell(x0, y, .{ .ch = '•', .style = .{ .fg = theme.bullet, .bg = theme.transcript_bg, .bold = true } });
                    const rest = if (t.len >= 2) t[2..] else t;
                    paintInline(s, x0 + 2, y, max_x, rest, .{ .fg = theme.transcript_fg, .bg = theme.transcript_bg }, theme);
                },
                .code => {
                    // resaltado de sintaxis básico (keywords, strings, comments, numbers)
                    paintCodeLine(s, x0, y, max_x, line.text, theme);
                },
                .fence => {
                    s.putStr(x0, y, line.text, .{ .fg = theme.code_fg, .bg = theme.transcript_bg });
                },
                .diff_plus => {
                    s.putStr(x0, y, line.text, .{ .fg = theme.diff_plus, .bg = theme.transcript_bg });
                },
                .diff_minus => {
                    s.putStr(x0, y, line.text, .{ .fg = theme.diff_minus, .bg = theme.transcript_bg });
                },
                .fold => {
                    s.putStr(x0, y, line.text, .{ .fg = theme.tool_label, .bg = theme.transcript_bg });
                },
                .normal => {
                    paintInline(s, x0, y, max_x, line.text, .{ .fg = theme.transcript_fg, .bg = theme.transcript_bg }, theme);
                },
            }
        }
    }
};

fn roleLabel(role: Role) []const u8 {
    return switch (role) {
        .user => "▸ tú",
        .assistant => "▾ fluency",
        .tool => "⚙ tool",
    };
}

fn labelColor(theme: Theme, role: Role) screen.Color {
    return switch (role) {
        .user => theme.user_label,
        .assistant => theme.assistant_label,
        .tool => theme.tool_label,
    };
}

// =====================================================================
// Tests
// =====================================================================

test "Transcript add and wrap" {
    var t = Transcript.init(std.testing.allocator);
    defer t.deinit();
    try t.add(.user, "hello world this is a long message");
    try std.testing.expect(t.entries.items.len == 1);
    try std.testing.expect(std.mem.eql(u8, t.entries.items[0].text.items, "hello world this is a long message"));

    var lines = std.ArrayList([]const u8).empty;
    defer t.freeLines(&lines);
    try t.wrapLine("hello world this is a long message", 10, &lines);
    try std.testing.expect(lines.items.len > 1);
}

test "Transcript appendToLast" {
    var t = Transcript.init(std.testing.allocator);
    defer t.deinit();
    try t.add(.assistant, "hello");
    try t.appendToLast(" world");
    try std.testing.expect(std.mem.eql(u8, t.entries.items[0].text.items, "hello world"));
}

test "Transcript scroll bounds" {
    var t = Transcript.init(std.testing.allocator);
    defer t.deinit();
    t.scroll = 5;
    t.scrollUp(3);
    try std.testing.expect(t.scroll == 2);
    t.scrollDown(100, 10, 5);
    try std.testing.expect(t.scroll == 5);
}

test "classify markdown vs diff" {
    try std.testing.expect(Transcript.classify(.assistant, "# Titulo") == .heading);
    try std.testing.expect(Transcript.classify(.assistant, "- item") == .bullet);
    try std.testing.expect(Transcript.classify(.assistant, "* item") == .bullet);
    // en assistant "- " es viñeta, NO diff
    try std.testing.expect(Transcript.classify(.assistant, "- borrado") == .bullet);
    // en tool "- "/"+ " son diff
    try std.testing.expect(Transcript.classify(.tool, "+ added") == .diff_plus);
    try std.testing.expect(Transcript.classify(.tool, "- removed") == .diff_minus);
    try std.testing.expect(Transcript.classify(.assistant, "texto normal") == .normal);
}

test "paintInline negrita y código no desbordan" {
    var s = try Screen.init(std.testing.allocator, 40, 3);
    defer s.deinit();
    const th = Theme.dark();
    Transcript.paintInline(&s, 0, 0, 40, "hola **mundo** y `code`", .{ .fg = th.transcript_fg, .bg = th.transcript_bg }, th);
    // 'h' al inicio
    try std.testing.expect(s.cells[0].ch == 'h');
    // los marcadores ** y ` no se imprimen como caracteres
    var has_star = false;
    for (s.cells) |c| {
        if (c.ch == '*') has_star = true;
    }
    try std.testing.expect(!has_star);
}

test "folded entry produce resumen corto" {
    var t = Transcript.init(std.testing.allocator);
    defer t.deinit();
    try t.add(.tool, "linea1\nlinea2\nlinea3");
    t.entries.items[0].folded = true;
    var lines = std.ArrayList(Line).empty;
    defer t.freeRenderLines(&lines);
    try t.buildLines(&lines, 60);
    // label + fold + blank = 3 (sin las 3 líneas de contenido)
    try std.testing.expect(lines.items.len == 3);
    var has_fold = false;
    for (lines.items) |ln| {
        if (ln.kind == .fold) has_fold = true;
    }
    try std.testing.expect(has_fold);
}

test "code fence marca líneas como code" {
    var t = Transcript.init(std.testing.allocator);
    defer t.deinit();
    try t.add(.assistant, "antes\n```\ncodigo()\n```\ndespues");
    var lines = std.ArrayList(Line).empty;
    defer t.freeRenderLines(&lines);
    try t.buildLines(&lines, 60);
    var has_code = false;
    var has_fence = false;
    for (lines.items) |ln| {
        if (ln.kind == .code) has_code = true;
        if (ln.kind == .fence) has_fence = true;
    }
    try std.testing.expect(has_code);
    try std.testing.expect(has_fence);
}
