// ztok — tokenizador BPE byte-level (estilo GPT-2 / Qwen2), 100% Zig.
//
// Cubre el camino F1: cargar el vocabulario y los merges de un GGUF (vía
// addToken/addMerge desde main) y codificar/decodificar texto compatible con
// el modelo (Qwen2.5/Llama/Mistral usan BPE byte-level "gpt2").
//
// Pipeline de encode:
//   1. Pre-tokenización: parte el texto en "trozos" (reglas tipo Qwen2: palabras
//      con espacio inicial opcional, dígitos sueltos, signos, espacios).
//   2. Byte-level: cada byte del trozo se mapea a un carácter "visible" con la
//      tabla byte<->unicode de GPT-2 (p. ej. el espacio 0x20 -> 'Ġ').
//   3. BPE: se aplican los merges por rango (menor rango = mayor prioridad).
//   4. Lookup: cada símbolo final -> id del vocabulario.
//
// Decode invierte: id -> string byte-level -> bytes originales.
const std = @import("std");

pub const Tokenizer = struct {
    allocator: std.mem.Allocator,

    // Vocabulario en orden de id (id_to_token[i] es el string byte-level del id i).
    id_to_token: std.ArrayList([]u8),
    token_to_id: std.StringHashMap(u32),
    // Rango de cada merge. Clave = "izquierda derecha" (tal cual en el GGUF).
    merge_rank: std.StringHashMap(u32),

    // Tabla byte-level (GPT-2): byte -> codepoint -> UTF-8.
    byte_to_str: [256][4]u8,
    byte_to_len: [256]u8,
    cp_to_byte: std.AutoHashMap(u21, u8),

    bos_id: ?u32 = null,
    eos_id: ?u32 = null,

    pub fn init(alloc: std.mem.Allocator) !Tokenizer {
        var self = Tokenizer{
            .allocator = alloc,
            .id_to_token = .empty,
            .token_to_id = std.StringHashMap(u32).init(alloc),
            .merge_rank = std.StringHashMap(u32).init(alloc),
            .byte_to_str = undefined,
            .byte_to_len = undefined,
            .cp_to_byte = std.AutoHashMap(u21, u8).init(alloc),
        };
        try self.buildByteLevel();
        return self;
    }

    pub fn deinit(self: *Tokenizer) void {
        for (self.id_to_token.items) |t| self.allocator.free(t);
        self.id_to_token.deinit(self.allocator);
        self.token_to_id.deinit();
        var it = self.merge_rank.keyIterator();
        while (it.next()) |k| self.allocator.free(k.*);
        self.merge_rank.deinit();
        self.cp_to_byte.deinit();
    }

    // Construye la tabla byte<->unicode de GPT-2: los bytes "imprimibles" se
    // mapean a sí mismos; el resto a codepoints 256, 257, ... (en orden de byte).
    fn buildByteLevel(self: *Tokenizer) !void {
        var next_cp: u21 = 256;
        for (0..256) |bi| {
            const b: u8 = @intCast(bi);
            const printable = (b >= 0x21 and b <= 0x7E) or (b >= 0xA1 and b <= 0xAC) or (b >= 0xAE);
            const cp: u21 = if (printable) @as(u21, b) else blk: {
                const c = next_cp;
                next_cp += 1;
                break :blk c;
            };
            const len = std.unicode.utf8Encode(cp, self.byte_to_str[bi][0..]) catch unreachable;
            self.byte_to_len[bi] = @intCast(len);
            try self.cp_to_byte.put(cp, b);
        }
    }

    // Añade un token al vocabulario; el id es el orden de inserción.
    pub fn addToken(self: *Tokenizer, token: []const u8) !void {
        const id: u32 = @intCast(self.id_to_token.items.len);
        const dup = try self.allocator.dupe(u8, token);
        try self.id_to_token.append(self.allocator, dup);
        // Si hay duplicados en el vocab, conserva el primer id (como llama.cpp).
        if (!self.token_to_id.contains(dup)) try self.token_to_id.put(dup, id);
    }

    // Añade un merge "izq der"; su rango es el orden de inserción.
    pub fn addMerge(self: *Tokenizer, line: []const u8) !void {
        const rank: u32 = @intCast(self.merge_rank.count());
        const dup = try self.allocator.dupe(u8, line);
        const gop = try self.merge_rank.getOrPut(dup);
        if (gop.found_existing) {
            self.allocator.free(dup);
        } else {
            gop.value_ptr.* = rank;
        }
    }

    pub fn vocabSize(self: Tokenizer) usize {
        return self.id_to_token.items.len;
    }

    // --- clasificación de caracteres (aprox. de \p{L}, \p{N}) ---
    fn isLetter(c: u8) bool {
        return (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or c >= 0x80;
    }
    fn isDigit(c: u8) bool {
        return c >= '0' and c <= '9';
    }
    fn isSpace(c: u8) bool {
        return c == ' ' or c == '\t' or c == '\n' or c == '\r';
    }

    // Longitud de una contracción ('s 't 're 've 'm 'll 'd) en text[i..], o 0.
    fn contractionLen(text: []const u8, i: usize) usize {
        if (text[i] != '\'') return 0;
        const rest = text[i + 1 ..];
        const opts = [_][]const u8{ "re", "ve", "ll", "s", "t", "m", "d" };
        for (opts) |o| {
            if (rest.len >= o.len) {
                var match = true;
                for (o, 0..) |oc, k| {
                    if (std.ascii.toLower(rest[k]) != oc) {
                        match = false;
                        break;
                    }
                }
                if (match) return 1 + o.len;
            }
        }
        return 0;
    }

    // Pre-tokenización: devuelve el siguiente trozo [i..end) según reglas tipo Qwen2.
    fn nextChunk(text: []const u8, i: usize) usize {
        const c = text[i];
        const n = text.len;

        // 1) Contracciones.
        const cl = contractionLen(text, i);
        if (cl > 0) return i + cl;

        // 2) Letras con prefijo opcional de UN carácter no-letra/dígito/salto.
        if (isLetter(c)) {
            var j = i + 1;
            while (j < n and isLetter(text[j])) j += 1;
            return j;
        }
        if (c != '\r' and c != '\n' and !isDigit(c) and i + 1 < n and isLetter(text[i + 1])) {
            var j = i + 2;
            while (j < n and isLetter(text[j])) j += 1;
            return j;
        }

        // 3) Dígito suelto (\p{N} en Qwen2).
        if (isDigit(c)) return i + 1;

        // 4) Espacio opcional + signos + saltos de línea finales.
        const punct_start = if (c == ' ' and i + 1 < n and !isSpace(text[i + 1]) and !isLetter(text[i + 1]) and !isDigit(text[i + 1])) i + 1 else i;
        if (!isSpace(text[punct_start]) and !isLetter(text[punct_start]) and !isDigit(text[punct_start])) {
            var j = punct_start;
            while (j < n and !isSpace(text[j]) and !isLetter(text[j]) and !isDigit(text[j])) j += 1;
            while (j < n and (text[j] == '\r' or text[j] == '\n')) j += 1;
            return j;
        }

        // 5) Espacios en blanco: si tras la racha viene una palabra/símbolo que
        //    tomará un espacio inicial, dejamos el último espacio para ella.
        var j = i;
        while (j < n and isSpace(text[j])) j += 1;
        if (j < n and (isLetter(text[j]) or (!isDigit(text[j]) and !isSpace(text[j]))) and j - 1 > i) {
            return j - 1;
        }
        return j;
    }

    // Codifica `text` añadiendo los ids a `out`.
    pub fn encode(self: *Tokenizer, text: []const u8, out: *std.ArrayList(u32)) !void {
        if (text.len == 0) return;
        const alloc = self.allocator;

        // Buffers reutilizables (capacidad máxima por trozo).
        var word: std.ArrayList(u8) = .empty;
        defer word.deinit(alloc);
        var starts: std.ArrayList(usize) = .empty; // inicio de cada símbolo en `word`
        defer starts.deinit(alloc);
        var lens: std.ArrayList(usize) = .empty; // longitud de cada símbolo
        defer lens.deinit(alloc);
        var key: std.ArrayList(u8) = .empty; // scratch para claves de merge
        defer key.deinit(alloc);

        var i: usize = 0;
        while (i < text.len) {
            const end = nextChunk(text, i);
            const chunk = text[i..end];
            i = end;
            if (chunk.len == 0) {
                i += 1; // salvaguarda anti-bucle
                continue;
            }

            // Byte-level: construir `word` y los símbolos iniciales (1 por byte).
            word.clearRetainingCapacity();
            starts.clearRetainingCapacity();
            lens.clearRetainingCapacity();
            for (chunk) |b| {
                const start = word.items.len;
                const l = self.byte_to_len[b];
                try word.appendSlice(alloc, self.byte_to_str[b][0..l]);
                try starts.append(alloc, start);
                try lens.append(alloc, l);
            }

            // BPE: fusionar el par adyacente de menor rango hasta que no haya más.
            while (starts.items.len >= 2) {
                var best_rank: u32 = std.math.maxInt(u32);
                var best: ?usize = null;
                for (0..starts.items.len - 1) |s| {
                    const a = word.items[starts.items[s]..][0..lens.items[s]];
                    const b = word.items[starts.items[s + 1]..][0..lens.items[s + 1]];
                    key.clearRetainingCapacity();
                    try key.appendSlice(alloc, a);
                    try key.append(alloc, ' ');
                    try key.appendSlice(alloc, b);
                    if (self.merge_rank.get(key.items)) |r| {
                        if (r < best_rank) {
                            best_rank = r;
                            best = s;
                        }
                    }
                }
                const bi = best orelse break;
                // Fusionar bi y bi+1 (contiguos en `word`): basta extender la longitud.
                lens.items[bi] += lens.items[bi + 1];
                _ = starts.orderedRemove(bi + 1);
                _ = lens.orderedRemove(bi + 1);
            }

            // Lookup de cada símbolo final.
            for (0..starts.items.len) |s| {
                const sym = word.items[starts.items[s]..][0..lens.items[s]];
                if (self.token_to_id.get(sym)) |id| {
                    try out.append(alloc, id);
                } else {
                    // Fallback: descomponer en caracteres byte-level individuales
                    // (cada byte-char existe siempre como token en el vocab gpt2).
                    var off: usize = 0;
                    while (off < sym.len) {
                        const cp_len = std.unicode.utf8ByteSequenceLength(sym[off]) catch 1;
                        const piece = sym[off..][0..@min(cp_len, sym.len - off)];
                        if (self.token_to_id.get(piece)) |id| try out.append(alloc, id);
                        off += piece.len;
                    }
                }
            }
        }
    }

    // Decodifica UN token a bytes en `buf` (sin asignar). Para decodificación
    // restringida por gramática (probar candidatos sin tocar el heap).
    pub fn decodeToken(self: *Tokenizer, id: u32, buf: []u8) []const u8 {
        if (id >= self.id_to_token.items.len) return buf[0..0];
        const tok = self.id_to_token.items[id];
        var n: usize = 0;
        var off: usize = 0;
        while (off < tok.len) {
            const cp_len = std.unicode.utf8ByteSequenceLength(tok[off]) catch 1;
            const end = @min(off + cp_len, tok.len);
            const cp = std.unicode.utf8Decode(tok[off..end]) catch {
                off = end;
                continue;
            };
            if (self.cp_to_byte.get(cp)) |b| {
                if (n < buf.len) {
                    buf[n] = b;
                    n += 1;
                }
            } else {
                for (tok[off..end]) |bb| {
                    if (n < buf.len) {
                        buf[n] = bb;
                        n += 1;
                    }
                }
            }
            off = end;
        }
        return buf[0..n];
    }

    // Decodifica los ids a bytes, añadiéndolos a `out`.
    pub fn decode(self: *Tokenizer, ids: []const u32, out: *std.ArrayList(u8)) !void {
        const alloc = self.allocator;
        for (ids) |id| {
            if (id >= self.id_to_token.items.len) continue;
            const tok = self.id_to_token.items[id];
            // tok es byte-level: recorrer codepoints y mapear a byte original.
            var off: usize = 0;
            while (off < tok.len) {
                const cp_len = std.unicode.utf8ByteSequenceLength(tok[off]) catch 1;
                const end = @min(off + cp_len, tok.len);
                const cp = std.unicode.utf8Decode(tok[off..end]) catch {
                    off = end;
                    continue;
                };
                if (self.cp_to_byte.get(cp)) |b| {
                    try out.append(alloc, b);
                } else {
                    // Token especial (no byte-level, p. ej. "<|im_start|>"): copiar literal.
                    try out.appendSlice(alloc, tok[off..end]);
                }
                off = end;
            }
        }
    }
};

/// Construye un vocabulario byte-level simple: 256 tokens, uno por byte.
/// No registra merges, por lo que la tokenización es byte-a-byte (cada byte
/// se mapea a su representación byte-level en UTF-8). Útil para
/// demostraciones y tokenización sin depender de un GGUF.
pub fn buildByteVocabSimple(tok: *Tokenizer) !void {
    for (0..256) |bi| {
        const b: u8 = @intCast(bi);
        const l = tok.byte_to_len[b];
        try tok.addToken(tok.byte_to_str[b][0..l]);
    }
}

// --- tests ---

// Mini-vocab byte-level para validar el pipeline sin depender de un GGUF.
fn buildMini(tok: *Tokenizer) !void {
    // Caracteres byte-level necesarios: 'h','e','l','o', y 'Ġ' (espacio 0x20).
    const singles = [_][]const u8{ "h", "e", "l", "o", "Ġ", "w", "r", "d", "!", "1", "2" };
    for (singles) |s| try tok.addToken(s);
    // Algunos merges (orden = prioridad).
    try tok.addToken("he");
    try tok.addToken("ll");
    try tok.addToken("hell");
    try tok.addToken("hello");
    try tok.addToken("Ġw");
    try tok.addMerge("h e"); // -> "he"
    try tok.addMerge("l l"); // -> "ll"
    try tok.addMerge("he ll"); // -> "hell"
    try tok.addMerge("hell o"); // -> "hello"
    try tok.addMerge("Ġ w"); // -> "Ġw"
}

test "byte-level: espacio -> Ġ y roundtrip de bytes" {
    const alloc = std.testing.allocator;
    var tok = try Tokenizer.init(alloc);
    defer tok.deinit();
    // El espacio (0x20) no es imprimible en gpt2 -> codepoint 0x100 + n.
    try std.testing.expect(tok.byte_to_len[' '] == 2); // 'Ġ' es 2 bytes UTF-8
    // 'A' (0x41) es imprimible -> se mapea a sí mismo (1 byte).
    try std.testing.expectEqual(@as(u8, 1), tok.byte_to_len['A']);
    try std.testing.expectEqual(@as(u8, 'A'), tok.byte_to_str['A'][0]);
}

test "BPE encode aplica merges por rango" {
    const alloc = std.testing.allocator;
    var tok = try Tokenizer.init(alloc);
    defer tok.deinit();
    try buildMini(&tok);

    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(alloc);
    try tok.encode("hello", &ids);
    // "hello" debe fusionarse hasta un único token "hello".
    try std.testing.expectEqual(@as(usize, 1), ids.items.len);
    const hello_id = tok.token_to_id.get("hello").?;
    try std.testing.expectEqual(hello_id, ids.items[0]);
}

test "encode/decode roundtrip de texto" {
    const alloc = std.testing.allocator;
    var tok = try Tokenizer.init(alloc);
    defer tok.deinit();
    try buildMini(&tok);

    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(alloc);
    try tok.encode("hello world", &ids);

    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(alloc);
    try tok.decode(ids.items, &text);
    try std.testing.expectEqualStrings("hello world", text.items);
}

test "pre-tokenización: el espacio inicial se pega a la palabra" {
    const alloc = std.testing.allocator;
    var tok = try Tokenizer.init(alloc);
    defer tok.deinit();
    try buildMini(&tok);

    // "Ġw" existe como token (espacio+w). "hello world" -> [hello][Ġw][o][r][l][d]
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(alloc);
    try tok.encode("hello world", &ids);
    // El segundo token debe ser "Ġw" (espacio pegado a la w), no un espacio suelto.
    try std.testing.expectEqual(tok.token_to_id.get("Ġw").?, ids.items[1]);
}
