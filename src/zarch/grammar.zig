// grammar.zig — decodificación restringida (F2).
//
// Primitivas para forzar que la generación respete una gramática (p. ej. JSON
// válido para tool-calls): un autómata de pila que acepta JSON estructuralmente
// y un sampler que enmascara los tokens no permitidos. La integración completa
// json-schema -> gramática -> máscara de tokens se construye sobre esto.
const std = @import("std");

/// Autómata de pila que valida JSON byte a byte. `feed` devuelve false si el
/// byte rompe la estructura; `isComplete` indica un JSON balanceado y cerrado.
pub const BalancedJson = struct {
    stack: [256]u8 = undefined, // '{' o '['
    depth: usize = 0,
    in_string: bool = false,
    escape: bool = false,
    started: bool = false,
    done: bool = false,

    const Self = @This();

    pub fn feed(self: *Self, c: u8) bool {
        if (self.in_string) {
            if (self.escape) {
                self.escape = false;
                return true;
            }
            if (c == '\\') {
                self.escape = true;
                return true;
            }
            if (c == '"') {
                self.in_string = false;
                return true;
            }
            return true; // cualquier byte dentro de una cadena
        }
        switch (c) {
            ' ', '\t', '\n', '\r' => return true,
            '"' => {
                self.in_string = true;
                self.started = true;
                return true;
            },
            '{', '[' => {
                if (self.depth >= self.stack.len) return false;
                self.stack[self.depth] = c;
                self.depth += 1;
                self.started = true;
                return true;
            },
            '}' => {
                if (self.depth == 0 or self.stack[self.depth - 1] != '{') return false;
                self.depth -= 1;
                if (self.depth == 0) self.done = true;
                return true;
            },
            ']' => {
                if (self.depth == 0 or self.stack[self.depth - 1] != '[') return false;
                self.depth -= 1;
                if (self.depth == 0) self.done = true;
                return true;
            },
            ':', ',' => return self.depth > 0,
            '-', '+', '.', '0'...'9' => {
                self.started = true;
                return true;
            },
            // letras de true/false/null
            't', 'r', 'u', 'e', 'f', 'a', 'l', 's', 'n', 'E' => {
                self.started = true;
                return true;
            },
            else => return false,
        }
    }

    pub fn isComplete(self: Self) bool {
        return self.started and self.done and self.depth == 0 and !self.in_string;
    }
};

/// Argmax restringido a los tokens marcados como permitidos. null si ninguno.
pub fn maskedArgmax(logits: []const f32, allowed: []const bool) ?u32 {
    std.debug.assert(logits.len == allowed.len);
    var best: ?u32 = null;
    var best_val: f32 = -std.math.inf(f32);
    for (logits, 0..) |v, i| {
        if (allowed[i] and (best == null or v > best_val)) {
            best_val = v;
            best = @intCast(i);
        }
    }
    return best;
}

/// Argmax restringido por gramática: elige el token de mayor logit cuyos bytes
/// (decodificados por `decodeFn(ctx, id, buf) -> []const u8`) mantienen la
/// gramática válida, y AVANZA `grammar` con ese token. null si ninguno es válido.
/// Es la decodificación restringida que garantiza tool-calls JSON bien formados.
pub fn grammarArgmax(
    logits: []const f32,
    grammar: *BalancedJson,
    ctx: anytype,
    comptime decodeFn: fn (@TypeOf(ctx), u32, []u8) []const u8,
) ?u32 {
    var best: ?u32 = null;
    var best_val: f32 = -std.math.inf(f32);
    var buf: [64]u8 = undefined;
    for (logits, 0..) |v, i| {
        if (best != null and v <= best_val) continue; // no puede mejorar
        const bytes = decodeFn(ctx, @intCast(i), &buf);
        if (bytes.len == 0) continue;
        var g = grammar.*; // copia barata (pila de tamaño fijo)
        var ok = true;
        for (bytes) |b| {
            if (!g.feed(b)) {
                ok = false;
                break;
            }
        }
        if (ok) {
            best_val = v;
            best = @intCast(i);
        }
    }
    if (best) |id| {
        const bytes = decodeFn(ctx, id, &buf);
        for (bytes) |b| _ = grammar.feed(b);
    }
    return best;
}

// --- tests ---
test "gramática JSON acepta válido y rechaza desbalanceado" {
    var g = BalancedJson{};
    const valid = "{\"a\":[1,2],\"b\":\"x\\\"y\"}";
    for (valid) |c| try std.testing.expect(g.feed(c));
    try std.testing.expect(g.isComplete());

    // '}' sin abrir, o cierre cruzado.
    var g2 = BalancedJson{};
    try std.testing.expect(g2.feed('['));
    try std.testing.expect(!g2.feed('}')); // cierra '}' sobre '[' -> inválido
}

test "maskedArgmax respeta la máscara" {
    const logits = [_]f32{ 5.0, 1.0, 9.0, 2.0 };
    // El logit máximo (id 2) está prohibido; debe elegir el mejor permitido (id 0).
    const allowed = [_]bool{ true, false, false, true };
    try std.testing.expectEqual(@as(?u32, 0), maskedArgmax(&logits, &allowed));
    const none = [_]bool{ false, false, false, false };
    try std.testing.expectEqual(@as(?u32, null), maskedArgmax(&logits, &none));
}

test "grammarArgmax fuerza JSON válido aunque el logit máximo sea inválido" {
    const Ctx = struct {
        fn dec(_: @This(), id: u32, buf: []u8) []const u8 {
            const toks = [_][]const u8{ "{", "}", "\"", "abc", "[", "]" };
            const s = toks[id];
            @memcpy(buf[0..s.len], s);
            return buf[0..s.len];
        }
    };
    var g = BalancedJson{};
    // '}' (id1) tiene el logit máximo pero es inválido al inicio; '[' (id4) es el
    // mejor de los válidos ({, ", abc, [).
    const logits = [_]f32{ 1.0, 9.0, 0.5, 0.1, 2.0, 0.0 };
    const pick = grammarArgmax(&logits, &g, Ctx{}, Ctx.dec).?;
    try std.testing.expectEqual(@as(u32, 4), pick);
    try std.testing.expectEqual(@as(usize, 1), g.depth); // '[' avanzó la pila
}
