// bytes — lectura LE/BE sobre slices con cursor, helpers para parseo binario
const std = @import("std");

/// Lector con cursor sobre un slice de bytes. Lecturas con bounds-check
/// que devuelven error.EndOfStream al rebasar el final.
pub const Reader = struct {
    buf: []const u8,
    pos: usize = 0,

    pub fn init(b: []const u8) Reader {
        return .{ .buf = b, .pos = 0 };
    }

    /// Bytes restantes sin leer.
    pub fn remaining(self: *const Reader) usize {
        return self.buf.len - self.pos;
    }

    /// Lee un único byte.
    pub fn readU8(self: *Reader) !u8 {
        if (self.pos + 1 > self.buf.len) return error.EndOfStream;
        const v = self.buf[self.pos];
        self.pos += 1;
        return v;
    }

    /// Lee un entero de tipo T con el endianness dado.
    pub fn readInt(self: *Reader, comptime T: type, endian: std.builtin.Endian) !T {
        const n = @sizeOf(T);
        if (self.pos + n > self.buf.len) return error.EndOfStream;
        const v = std.mem.readInt(T, self.buf[self.pos..][0..n], endian);
        self.pos += n;
        return v;
    }

    /// Devuelve un sub-slice de n bytes y avanza el cursor.
    pub fn readBytes(self: *Reader, n: usize) ![]const u8 {
        if (self.pos + n > self.buf.len) return error.EndOfStream;
        const slice = self.buf[self.pos .. self.pos + n];
        self.pos += n;
        return slice;
    }

    /// Salta n bytes.
    pub fn skip(self: *Reader, n: usize) !void {
        if (self.pos + n > self.buf.len) return error.EndOfStream;
        self.pos += n;
    }
};

// --- tests ---
test "reader u16/u32/u64 LE" {
    const data = [_]u8{
        0x34, 0x12, // u16 LE = 0x1234
        0x78, 0x56, 0x34, 0x12, // u32 LE = 0x12345678
        0xF0, 0xDE, 0xBC, 0x9A, 0x78, 0x56, 0x34, 0x12, // u64 LE
    };
    var r = Reader.init(&data);
    try std.testing.expectEqual(@as(u16, 0x1234), try r.readInt(u16, .little));
    try std.testing.expectEqual(@as(u32, 0x12345678), try r.readInt(u32, .little));
    try std.testing.expectEqual(@as(u64, 0x123456789ABCDEF0), try r.readInt(u64, .little));
    try std.testing.expectEqual(@as(usize, 0), r.remaining());
}

test "reader u16/u32/u64 BE" {
    const data = [_]u8{
        0x12, 0x34, // u16 BE = 0x1234
        0x12, 0x34, 0x56, 0x78, // u32 BE
        0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, // u64 BE
    };
    var r = Reader.init(&data);
    try std.testing.expectEqual(@as(u16, 0x1234), try r.readInt(u16, .big));
    try std.testing.expectEqual(@as(u32, 0x12345678), try r.readInt(u32, .big));
    try std.testing.expectEqual(@as(u64, 0x123456789ABCDEF0), try r.readInt(u64, .big));
}

test "reader u8 and readBytes and skip" {
    const data = [_]u8{ 1, 2, 3, 4, 5, 6 };
    var r = Reader.init(&data);
    try std.testing.expectEqual(@as(u8, 1), try r.readU8());
    try r.skip(1);
    const b = try r.readBytes(3);
    try std.testing.expectEqualSlices(u8, &.{ 3, 4, 5 }, b);
    try std.testing.expectEqual(@as(u8, 6), try r.readU8());
}

test "reader end of stream" {
    const data = [_]u8{ 1, 2 };
    var r = Reader.init(&data);
    try std.testing.expectError(error.EndOfStream, r.readInt(u32, .little));
    _ = try r.readU8();
    _ = try r.readU8();
    try std.testing.expectError(error.EndOfStream, r.readU8());
    try std.testing.expectError(error.EndOfStream, r.readBytes(1));
    try std.testing.expectError(error.EndOfStream, r.skip(1));
}
