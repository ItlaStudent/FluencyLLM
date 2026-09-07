// crypto — ChaCha20-Poly1305 para sellar secretos (std.crypto)
const std = @import("std");

const ChaCha20Poly1305 = std.crypto.aead.chacha_poly.ChaCha20Poly1305;

pub const key_length = ChaCha20Poly1305.key_length; // 32
pub const nonce_length = ChaCha20Poly1305.nonce_length; // 12
pub const tag_length = ChaCha20Poly1305.tag_length; // 16

/// Sella (cifra + autentica) `plaintext` con datos asociados `ad`.
/// `out_ciphertext` debe tener la misma longitud que `plaintext`.
pub fn seal(
    key: [32]u8,
    nonce: [12]u8,
    plaintext: []const u8,
    ad: []const u8,
    out_ciphertext: []u8,
    out_tag: *[16]u8,
) void {
    std.debug.assert(out_ciphertext.len == plaintext.len);
    ChaCha20Poly1305.encrypt(out_ciphertext, out_tag, plaintext, ad, nonce, key);
}

/// Abre (descifra + verifica) `ciphertext`. Devuelve error.AuthenticationFailed
/// si el tag o los datos asociados no coinciden.
pub fn open(
    key: [32]u8,
    nonce: [12]u8,
    ciphertext: []const u8,
    tag: [16]u8,
    ad: []const u8,
    out_plaintext: []u8,
) !void {
    std.debug.assert(out_plaintext.len == ciphertext.len);
    try ChaCha20Poly1305.decrypt(out_plaintext, ciphertext, tag, ad, nonce, key);
}

// --- tests ---
test "seal then open roundtrip" {
    const key = [_]u8{0x42} ** 32;
    const nonce = [_]u8{0x07} ** 12;
    const plaintext = "secreto constitucional";
    const ad = "manifest-v1";

    var ct: [plaintext.len]u8 = undefined;
    var tag: [16]u8 = undefined;
    seal(key, nonce, plaintext, ad, &ct, &tag);

    var pt: [plaintext.len]u8 = undefined;
    try open(key, nonce, &ct, tag, ad, &pt);
    try std.testing.expectEqualSlices(u8, plaintext, &pt);
}

test "tampered tag fails" {
    const key = [_]u8{0x01} ** 32;
    const nonce = [_]u8{0x02} ** 12;
    const plaintext = "datos";
    const ad = "";

    var ct: [plaintext.len]u8 = undefined;
    var tag: [16]u8 = undefined;
    seal(key, nonce, plaintext, ad, &ct, &tag);

    tag[0] ^= 0xFF; // corromper tag
    var pt: [plaintext.len]u8 = undefined;
    try std.testing.expectError(error.AuthenticationFailed, open(key, nonce, &ct, tag, ad, &pt));
}

test "tampered ciphertext fails" {
    const key = [_]u8{0xAB} ** 32;
    const nonce = [_]u8{0xCD} ** 12;
    const plaintext = "mensaje largo de prueba";
    const ad = "header";

    var ct: [plaintext.len]u8 = undefined;
    var tag: [16]u8 = undefined;
    seal(key, nonce, plaintext, ad, &ct, &tag);

    ct[3] ^= 0x01;
    var pt: [plaintext.len]u8 = undefined;
    try std.testing.expectError(error.AuthenticationFailed, open(key, nonce, &ct, tag, ad, &pt));
}

test "wrong ad fails" {
    const key = [_]u8{0x11} ** 32;
    const nonce = [_]u8{0x22} ** 12;
    const plaintext = "x";

    var ct: [plaintext.len]u8 = undefined;
    var tag: [16]u8 = undefined;
    seal(key, nonce, plaintext, "good", &ct, &tag);

    var pt: [plaintext.len]u8 = undefined;
    try std.testing.expectError(error.AuthenticationFailed, open(key, nonce, &ct, tag, "bad", &pt));
}
