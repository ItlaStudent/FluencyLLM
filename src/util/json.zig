// json — parser/serializador JSON propio (recursivo descendente)
const std = @import("std");

pub const Value = union(enum) {
    null,
    bool: bool,
    int: i64,
    float: f64,
    string: []const u8,
    array: []Value,
    object: []Member,

    /// Busca un miembro por clave en un objeto. null si no es objeto o no existe.
    pub fn get(self: Value, key: []const u8) ?Value {
        switch (self) {
            .object => |members| {
                for (members) |m| {
                    if (std.mem.eql(u8, m.key, key)) return m.value;
                }
                return null;
            },
            else => return null,
        }
    }

    /// Elemento de un array por índice. null si no es array o fuera de rango.
    pub fn at(self: Value, i: usize) ?Value {
        switch (self) {
            .array => |items| {
                if (i >= items.len) return null;
                return items[i];
            },
            else => return null,
        }
    }
};

pub const Member = struct {
    key: []const u8,
    value: Value,
};

pub const ParseError = error{
    UnexpectedEndOfInput,
    UnexpectedToken,
    InvalidNumber,
    InvalidString,
    InvalidEscape,
    InvalidLiteral,
    TrailingData,
    OutOfMemory,
};

/// Resultado del parseo. Toda la memoria (strings desescapadas, arrays,
/// objetos) vive en el arena interno: deinit() la libera de una vez.
pub const Parsed = struct {
    arena: *std.heap.ArenaAllocator,
    value: Value,

    pub fn deinit(self: *Parsed) void {
        const child = self.arena.child_allocator;
        self.arena.deinit();
        child.destroy(self.arena);
    }
};

const Parser = struct {
    text: []const u8,
    pos: usize,
    alloc: std.mem.Allocator, // arena allocator

    fn peek(self: *Parser) ?u8 {
        if (self.pos >= self.text.len) return null;
        return self.text[self.pos];
    }

    fn advance(self: *Parser) ?u8 {
        if (self.pos >= self.text.len) return null;
        const c = self.text[self.pos];
        self.pos += 1;
        return c;
    }

    fn skipWhitespace(self: *Parser) void {
        while (self.pos < self.text.len) {
            switch (self.text[self.pos]) {
                ' ', '\t', '\n', '\r' => self.pos += 1,
                else => break,
            }
        }
    }

    fn expect(self: *Parser, c: u8) ParseError!void {
        const got = self.advance() orelse return error.UnexpectedEndOfInput;
        if (got != c) return error.UnexpectedToken;
    }

    fn parseValue(self: *Parser) ParseError!Value {
        self.skipWhitespace();
        const c = self.peek() orelse return error.UnexpectedEndOfInput;
        return switch (c) {
            '{' => self.parseObject(),
            '[' => self.parseArray(),
            '"' => Value{ .string = try self.parseString() },
            't', 'f' => self.parseBool(),
            'n' => self.parseNull(),
            '-', '0'...'9' => self.parseNumber(),
            else => error.UnexpectedToken,
        };
    }

    fn parseObject(self: *Parser) ParseError!Value {
        try self.expect('{');
        var members: std.ArrayList(Member) = .empty;
        defer members.deinit(self.alloc);

        self.skipWhitespace();
        if (self.peek() == '}') {
            self.pos += 1;
            return Value{ .object = try members.toOwnedSlice(self.alloc) };
        }

        while (true) {
            self.skipWhitespace();
            if (self.peek() != '"') return error.UnexpectedToken;
            const key = try self.parseString();
            self.skipWhitespace();
            try self.expect(':');
            const value = try self.parseValue();
            try members.append(self.alloc, .{ .key = key, .value = value });

            self.skipWhitespace();
            const sep = self.advance() orelse return error.UnexpectedEndOfInput;
            if (sep == '}') break;
            if (sep != ',') return error.UnexpectedToken;
        }
        return Value{ .object = try members.toOwnedSlice(self.alloc) };
    }

    fn parseArray(self: *Parser) ParseError!Value {
        try self.expect('[');
        var items: std.ArrayList(Value) = .empty;
        defer items.deinit(self.alloc);

        self.skipWhitespace();
        if (self.peek() == ']') {
            self.pos += 1;
            return Value{ .array = try items.toOwnedSlice(self.alloc) };
        }

        while (true) {
            const value = try self.parseValue();
            try items.append(self.alloc, value);

            self.skipWhitespace();
            const sep = self.advance() orelse return error.UnexpectedEndOfInput;
            if (sep == ']') break;
            if (sep != ',') return error.UnexpectedToken;
        }
        return Value{ .array = try items.toOwnedSlice(self.alloc) };
    }

    fn parseString(self: *Parser) ParseError![]const u8 {
        try self.expect('"');
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(self.alloc);

        while (true) {
            const c = self.advance() orelse return error.InvalidString;
            switch (c) {
                '"' => break,
                '\\' => {
                    const esc = self.advance() orelse return error.InvalidEscape;
                    switch (esc) {
                        '"' => try out.append(self.alloc, '"'),
                        '\\' => try out.append(self.alloc, '\\'),
                        '/' => try out.append(self.alloc, '/'),
                        'b' => try out.append(self.alloc, 0x08),
                        'f' => try out.append(self.alloc, 0x0C),
                        'n' => try out.append(self.alloc, '\n'),
                        'r' => try out.append(self.alloc, '\r'),
                        't' => try out.append(self.alloc, '\t'),
                        'u' => try self.parseUnicodeEscape(&out),
                        else => return error.InvalidEscape,
                    }
                },
                else => {
                    if (c < 0x20) return error.InvalidString;
                    try out.append(self.alloc, c);
                },
            }
        }
        return out.toOwnedSlice(self.alloc);
    }

    fn parseHex4(self: *Parser) ParseError!u16 {
        var v: u16 = 0;
        var i: usize = 0;
        while (i < 4) : (i += 1) {
            const c = self.advance() orelse return error.InvalidEscape;
            const digit: u16 = switch (c) {
                '0'...'9' => c - '0',
                'a'...'f' => c - 'a' + 10,
                'A'...'F' => c - 'A' + 10,
                else => return error.InvalidEscape,
            };
            v = (v << 4) | digit;
        }
        return v;
    }

    fn parseUnicodeEscape(self: *Parser, out: *std.ArrayList(u8)) ParseError!void {
        const hi = try self.parseHex4();
        var codepoint: u21 = hi;
        // Pares sustitutos UTF-16.
        if (hi >= 0xD800 and hi <= 0xDBFF) {
            // Esperar \uXXXX bajo.
            const bs = self.advance() orelse return error.InvalidEscape;
            const u = self.advance() orelse return error.InvalidEscape;
            if (bs != '\\' or u != 'u') return error.InvalidEscape;
            const lo = try self.parseHex4();
            if (lo < 0xDC00 or lo > 0xDFFF) return error.InvalidEscape;
            codepoint = 0x10000 + ((@as(u21, hi - 0xD800) << 10) | @as(u21, lo - 0xDC00));
        } else if (hi >= 0xDC00 and hi <= 0xDFFF) {
            return error.InvalidEscape; // sustituto bajo aislado
        }
        var buf: [4]u8 = undefined;
        const n = std.unicode.utf8Encode(codepoint, &buf) catch return error.InvalidEscape;
        try out.appendSlice(self.alloc, buf[0..n]);
    }

    fn parseNumber(self: *Parser) ParseError!Value {
        const start = self.pos;
        var is_float = false;

        if (self.peek() == '-') self.pos += 1;
        while (self.peek()) |c| {
            switch (c) {
                '0'...'9' => self.pos += 1,
                '.', 'e', 'E', '+', '-' => {
                    is_float = true;
                    self.pos += 1;
                },
                else => break,
            }
        }
        const slice = self.text[start..self.pos];
        if (slice.len == 0) return error.InvalidNumber;

        if (is_float) {
            const f = std.fmt.parseFloat(f64, slice) catch return error.InvalidNumber;
            return Value{ .float = f };
        } else {
            const i = std.fmt.parseInt(i64, slice, 10) catch {
                // Demasiado grande para i64: tratar como float.
                const f = std.fmt.parseFloat(f64, slice) catch return error.InvalidNumber;
                return Value{ .float = f };
            };
            return Value{ .int = i };
        }
    }

    fn parseLiteral(self: *Parser, lit: []const u8) ParseError!void {
        if (self.pos + lit.len > self.text.len) return error.UnexpectedEndOfInput;
        if (!std.mem.eql(u8, self.text[self.pos .. self.pos + lit.len], lit))
            return error.InvalidLiteral;
        self.pos += lit.len;
    }

    fn parseBool(self: *Parser) ParseError!Value {
        if (self.peek() == 't') {
            try self.parseLiteral("true");
            return Value{ .bool = true };
        } else {
            try self.parseLiteral("false");
            return Value{ .bool = false };
        }
    }

    fn parseNull(self: *Parser) ParseError!Value {
        try self.parseLiteral("null");
        return Value{ .null = {} };
    }
};

/// Parsea `text` y devuelve un `Parsed`. Llamar `.deinit()` para liberar.
pub fn parse(alloc: std.mem.Allocator, text: []const u8) ParseError!Parsed {
    const arena = try alloc.create(std.heap.ArenaAllocator);
    errdefer alloc.destroy(arena);
    arena.* = std.heap.ArenaAllocator.init(alloc);
    errdefer arena.deinit();

    var p = Parser{ .text = text, .pos = 0, .alloc = arena.allocator() };
    const value = try p.parseValue();
    p.skipWhitespace();
    if (p.pos != text.len) return error.TrailingData;

    return Parsed{ .arena = arena, .value = value };
}

// --- serialización ---

/// Serializa `value` a un buffer JSON. El llamador libera el slice resultante.
pub fn stringify(alloc: std.mem.Allocator, value: Value) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(alloc);
    try writeValue(alloc, &out, value);
    return out.toOwnedSlice(alloc);
}

fn writeValue(alloc: std.mem.Allocator, out: *std.ArrayList(u8), value: Value) !void {
    switch (value) {
        .null => try out.appendSlice(alloc, "null"),
        .bool => |b| try out.appendSlice(alloc, if (b) "true" else "false"),
        .int => |i| {
            var buf: [32]u8 = undefined;
            const s = std.fmt.bufPrint(&buf, "{d}", .{i}) catch unreachable;
            try out.appendSlice(alloc, s);
        },
        .float => |f| {
            var buf: [64]u8 = undefined;
            const s = std.fmt.bufPrint(&buf, "{d}", .{f}) catch unreachable;
            try out.appendSlice(alloc, s);
        },
        .string => |s| try writeString(alloc, out, s),
        .array => |items| {
            try out.append(alloc, '[');
            for (items, 0..) |item, i| {
                if (i != 0) try out.append(alloc, ',');
                try writeValue(alloc, out, item);
            }
            try out.append(alloc, ']');
        },
        .object => |members| {
            try out.append(alloc, '{');
            for (members, 0..) |m, i| {
                if (i != 0) try out.append(alloc, ',');
                try writeString(alloc, out, m.key);
                try out.append(alloc, ':');
                try writeValue(alloc, out, m.value);
            }
            try out.append(alloc, '}');
        },
    }
}

fn writeString(alloc: std.mem.Allocator, out: *std.ArrayList(u8), s: []const u8) !void {
    try out.append(alloc, '"');
    for (s) |c| {
        switch (c) {
            '"' => try out.appendSlice(alloc, "\\\""),
            '\\' => try out.appendSlice(alloc, "\\\\"),
            '\n' => try out.appendSlice(alloc, "\\n"),
            '\r' => try out.appendSlice(alloc, "\\r"),
            '\t' => try out.appendSlice(alloc, "\\t"),
            0x08 => try out.appendSlice(alloc, "\\b"),
            0x0C => try out.appendSlice(alloc, "\\f"),
            else => {
                if (c < 0x20) {
                    var buf: [6]u8 = undefined;
                    const hexstr = std.fmt.bufPrint(&buf, "\\u{x:0>4}", .{c}) catch unreachable;
                    try out.appendSlice(alloc, hexstr);
                } else {
                    try out.append(alloc, c);
                }
            },
        }
    }
    try out.append(alloc, '"');
}

// --- tests ---
test "parse nested object and array" {
    const alloc = std.testing.allocator;
    const text =
        \\{
        \\  "name": "fluency",
        \\  "version": 16,
        \\  "ratio": 0.5,
        \\  "active": true,
        \\  "tags": ["zig", "llm", null],
        \\  "nested": { "a": 1, "b": [2, 3] }
        \\}
    ;
    var parsed = try parse(alloc, text);
    defer parsed.deinit();

    const root = parsed.value;
    try std.testing.expectEqualStrings("fluency", root.get("name").?.string);
    try std.testing.expectEqual(@as(i64, 16), root.get("version").?.int);
    try std.testing.expectEqual(@as(f64, 0.5), root.get("ratio").?.float);
    try std.testing.expectEqual(true, root.get("active").?.bool);

    const tags = root.get("tags").?;
    try std.testing.expectEqualStrings("zig", tags.at(0).?.string);
    try std.testing.expectEqualStrings("llm", tags.at(1).?.string);
    try std.testing.expect(tags.at(2).? == .null);

    const nested = root.get("nested").?;
    try std.testing.expectEqual(@as(i64, 1), nested.get("a").?.int);
    try std.testing.expectEqual(@as(i64, 3), nested.get("b").?.at(1).?.int);
}

test "parse string escapes" {
    const alloc = std.testing.allocator;
    var parsed = try parse(alloc, "\"line1\\nline2\\ttab\\\"quote\\\\slash\"");
    defer parsed.deinit();
    try std.testing.expectEqualStrings("line1\nline2\ttab\"quote\\slash", parsed.value.string);
}

test "parse unicode escape" {
    const alloc = std.testing.allocator;
    var parsed = try parse(alloc, "\"\\u00e9\\u0041\""); // é A
    defer parsed.deinit();
    try std.testing.expectEqualStrings("\u{00e9}A", parsed.value.string);
}

test "parse surrogate pair" {
    const alloc = std.testing.allocator;
    var parsed = try parse(alloc, "\"\\uD83D\\uDE00\""); // 😀
    defer parsed.deinit();
    try std.testing.expectEqualStrings("\u{1F600}", parsed.value.string);
}

test "stringify then parse roundtrip" {
    const alloc = std.testing.allocator;
    const original =
        \\{"id":42,"label":"hi\nthere","items":[1,2,3],"flag":false,"nothing":null}
    ;
    var parsed = try parse(alloc, original);
    defer parsed.deinit();

    const out = try stringify(alloc, parsed.value);
    defer alloc.free(out);

    var reparsed = try parse(alloc, out);
    defer reparsed.deinit();

    try std.testing.expectEqual(@as(i64, 42), reparsed.value.get("id").?.int);
    try std.testing.expectEqualStrings("hi\nthere", reparsed.value.get("label").?.string);
    try std.testing.expectEqual(@as(i64, 2), reparsed.value.get("items").?.at(1).?.int);
    try std.testing.expectEqual(false, reparsed.value.get("flag").?.bool);
    try std.testing.expect(reparsed.value.get("nothing").? == .null);
}

test "parse rejects trailing data" {
    const alloc = std.testing.allocator;
    try std.testing.expectError(error.TrailingData, parse(alloc, "1 2"));
}

test "parse empty containers" {
    const alloc = std.testing.allocator;
    var p1 = try parse(alloc, "{}");
    defer p1.deinit();
    try std.testing.expectEqual(@as(usize, 0), p1.value.object.len);

    var p2 = try parse(alloc, "[]");
    defer p2.deinit();
    try std.testing.expectEqual(@as(usize, 0), p2.value.array.len);
}

test "parse negative and float numbers" {
    const alloc = std.testing.allocator;
    var p = try parse(alloc, "[-5, 3.14, -2.5e3, 100]");
    defer p.deinit();
    try std.testing.expectEqual(@as(i64, -5), p.value.at(0).?.int);
    try std.testing.expectEqual(@as(f64, 3.14), p.value.at(1).?.float);
    try std.testing.expectEqual(@as(f64, -2500.0), p.value.at(2).?.float);
    try std.testing.expectEqual(@as(i64, 100), p.value.at(3).?.int);
}
