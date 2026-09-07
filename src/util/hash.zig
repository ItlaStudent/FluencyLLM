// hash — sha256 para GGUF/manifests y hash constitucional
const std = @import("std");

const Sha256 = std.crypto.hash.sha2.Sha256;

/// Calcula SHA-256 de `data` y devuelve el digest de 32 bytes.
pub fn sha256(data: []const u8) [32]u8 {
    var out: [32]u8 = undefined;
    Sha256.hash(data, &out, .{});
    return out;
}

/// Calcula SHA-256 de `data` y escribe su representación hex en minúsculas
/// (64 caracteres) en `out`.
pub fn sha256Hex(data: []const u8, out: *[64]u8) void {
    const digest = sha256(data);
    const hexchars = "0123456789abcdef";
    for (digest, 0..) |b, i| {
        out[i * 2] = hexchars[b >> 4];
        out[i * 2 + 1] = hexchars[b & 0x0f];
    }
}

// --- tests ---
test "sha256 known vector abc" {
    const digest = sha256("abc");
    const expected = [_]u8{
        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
        0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
        0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
        0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad,
    };
    try std.testing.expectEqualSlices(u8, &expected, &digest);
}

test "sha256 empty string" {
    const digest = sha256("");
    const expected = [_]u8{
        0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14,
        0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9, 0x24,
        0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c,
        0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52, 0xb8, 0x55,
    };
    try std.testing.expectEqualSlices(u8, &expected, &digest);
}

test "sha256Hex abc" {
    var hex: [64]u8 = undefined;
    sha256Hex("abc", &hex);
    try std.testing.expectEqualStrings(
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        &hex,
    );
}
