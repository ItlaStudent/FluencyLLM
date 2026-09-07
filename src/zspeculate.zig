// zspeculate — speculative decoding 100% Zig (prompt-lookup / n-gram)
// No requiere modelo borrador: busca repeticiones del propio contexto y
// propone continuation; el modelo verifica en una pasada.
//
// Invariante: la salida greedy con speculative ES EXACTAMENTE IGUAL a la
// greedy sin speculative (es un acelerador, no cambia la distribución).
const std = @import("std");
const zarch = @import("zarch");

/// Generador con speculative decoding por prompt-lookup.
pub const SpeculativeGenerator = struct {
    tf: *zarch.Transformer,

    // Parámetros tuneables (default razonables para textos repetitivos/código).
    ngram_len: usize = 4,
    max_draft: usize = 15,

    // Buffers reutilizables para forwardBatch (lazy-allocated en primera llamada).
    logits_buf: ?[]f32 = null,
    draft_buf: ?[]u32 = null,

    pub fn deinit(self: *SpeculativeGenerator, alloc: std.mem.Allocator) void {
        if (self.logits_buf) |b| alloc.free(b);
        if (self.draft_buf) |b| alloc.free(b);
    }

    /// Genera tokens usando speculative decoding. Escribe en `ids` (ya contiene
    /// el prompt) y emite bytes en `out`. `cap` es seq_len máxima.
    /// `tok` es el tokenizador (anytype para evitar dependencia cíclica).
    /// Devuelve tokens generados.
    pub fn generate(self: *SpeculativeGenerator, alloc: std.mem.Allocator, tok: anytype, ids: *std.ArrayList(u32), out: *std.ArrayList(u8), max_tokens: usize, cap: usize) !usize {
        var pos: usize = ids.items.len;
        var logits: []f32 = undefined;
        if (pos > 0) {
            logits = self.tf.forward(ids.items[pos - 1], pos - 1);
        } else {
            return error.NoPrompt;
        }
        var generated: usize = 0;
        const vocab = self.tf.config.vocab_size;

        outer: while (generated < max_tokens and pos < cap) {
            const draft = self.promptLookupDraft(ids.items);

            if (draft.len == 0) {
                const best = argmax(logits);
                if (tok.eos_id) |e| if (best == e) break :outer;
                try ids.append(alloc, best);
                const one = [_]u32{best};
                try tok.decode(one[0..], out);
                logits = self.tf.forward(best, pos);
                pos += 1;
                generated += 1;
                continue;
            }

            const accept_limit = @min(draft.len, max_tokens - generated);
            const best0 = argmax(logits);
            if (tok.eos_id) |e| if (best0 == e) break :outer;

            if (best0 != draft[0]) {
                try ids.append(alloc, best0);
                const one = [_]u32{best0};
                try tok.decode(one[0..], out);
                logits = self.tf.forward(best0, pos);
                pos += 1;
                generated += 1;
                continue;
            }

            // Copiar draft en buffer reutilizable (lazy-alloc)
            if (self.draft_buf == null or self.draft_buf.?.len < accept_limit) {
                if (self.draft_buf) |b| alloc.free(b);
                self.draft_buf = try alloc.alloc(u32, self.max_draft);
            }
            const draft_copy = self.draft_buf.?[0..accept_limit];
            @memcpy(draft_copy, draft[0..accept_limit]);

            // Buffer reutilizable para logits batch (lazy-alloc)
            const need_logits = self.max_draft * vocab;
            if (self.logits_buf == null or self.logits_buf.?.len < need_logits) {
                if (self.logits_buf) |b| alloc.free(b);
                self.logits_buf = try alloc.alloc(f32, need_logits);
            }
            const logits_batch = self.logits_buf.?[0..(accept_limit * vocab)];
            try self.tf.forwardBatch(draft_copy, pos, logits_batch);

            // Emitir best0 (ya verificado)
            try ids.append(alloc, best0);
            const one0 = [_]u32{best0};
            try tok.decode(one0[0..], out);
            pos += 1;
            generated += 1;

            var accepted: usize = 1;
            while (accepted < accept_limit and pos < cap) {
                const best = argmax(logits_batch[(accepted - 1) * vocab ..][0..vocab]);
                if (tok.eos_id) |e| if (best == e) break :outer;
                try ids.append(alloc, best);
                const one = [_]u32{best};
                try tok.decode(one[0..], out);
                pos += 1;
                generated += 1;
                if (best == draft_copy[accepted]) {
                    accepted += 1;
                } else {
                    break;
                }
            }

            if (accepted == accept_limit) {
                // Draft completamente aceptado: logits del último token ya están en el batch
                @memcpy(self.tf.logits[0..vocab], logits_batch[(accepted - 1) * vocab ..][0..vocab]);
                logits = self.tf.logits[0..vocab];
            } else {
                // Se rompió antes de aceptar todo: necesitamos forward del último token emitido
                logits = self.tf.forward(ids.items[pos - 1], pos - 1);
            }
        }
        return generated;
    }

    pub fn generateStream(self: *SpeculativeGenerator, alloc: std.mem.Allocator, tok: anytype, ids: *std.ArrayList(u32), max_tokens: usize, cap: usize, emit_ctx: *anyopaque, emit: EmitFn) !void {
        var pos: usize = ids.items.len;
        var logits: []f32 = undefined;
        if (pos > 0) {
            logits = self.tf.forward(ids.items[pos - 1], pos - 1);
        } else {
            return error.NoPrompt;
        }
        var generated: usize = 0;
        var dec_buf: [256]u8 = undefined;
        const vocab = self.tf.config.vocab_size;

        outer: while (generated < max_tokens and pos < cap) {
            const draft = self.promptLookupDraft(ids.items);

            if (draft.len == 0) {
                const best = argmax(logits);
                if (tok.eos_id) |e| if (best == e) break :outer;
                try ids.append(alloc, best);
                const piece = tok.decodeToken(best, &dec_buf);
                try emit(emit_ctx, piece);
                logits = self.tf.forward(best, pos);
                pos += 1;
                generated += 1;
                continue;
            }

            const accept_limit = @min(draft.len, max_tokens - generated);
            const best0 = argmax(logits);
            if (tok.eos_id) |e| if (best0 == e) break :outer;

            if (best0 != draft[0]) {
                try ids.append(alloc, best0);
                const piece = tok.decodeToken(best0, &dec_buf);
                try emit(emit_ctx, piece);
                logits = self.tf.forward(best0, pos);
                pos += 1;
                generated += 1;
                continue;
            }

            if (self.draft_buf == null or self.draft_buf.?.len < accept_limit) {
                if (self.draft_buf) |b| alloc.free(b);
                self.draft_buf = try alloc.alloc(u32, self.max_draft);
            }
            const draft_copy = self.draft_buf.?[0..accept_limit];
            @memcpy(draft_copy, draft[0..accept_limit]);

            const need_logits = self.max_draft * vocab;
            if (self.logits_buf == null or self.logits_buf.?.len < need_logits) {
                if (self.logits_buf) |b| alloc.free(b);
                self.logits_buf = try alloc.alloc(f32, need_logits);
            }
            const logits_batch = self.logits_buf.?[0..(accept_limit * vocab)];
            try self.tf.forwardBatch(draft_copy, pos, logits_batch);

            try ids.append(alloc, best0);
            const piece0 = tok.decodeToken(best0, &dec_buf);
            try emit(emit_ctx, piece0);
            pos += 1;
            generated += 1;

            var accepted: usize = 1;
            while (accepted < accept_limit and pos < cap) {
                const best = argmax(logits_batch[(accepted - 1) * vocab ..][0..vocab]);
                if (tok.eos_id) |e| if (best == e) break :outer;
                try ids.append(alloc, best);
                const piece = tok.decodeToken(best, &dec_buf);
                try emit(emit_ctx, piece);
                pos += 1;
                generated += 1;
                if (best == draft_copy[accepted]) {
                    accepted += 1;
                } else {
                    break;
                }
            }

            if (accepted == accept_limit) {
                @memcpy(self.tf.logits[0..vocab], logits_batch[(accepted - 1) * vocab ..][0..vocab]);
                logits = self.tf.logits[0..vocab];
            } else {
                logits = self.tf.forward(ids.items[pos - 1], pos - 1);
            }
        }
    }

    /// Busca un n-grama de longitud `ngram_len` en el contexto previo
    /// (excluyendo el final) que coincida con los últimos `ngram_len` tokens.
    /// Devuelve los tokens que siguen al match (hasta `max_draft`).
    fn promptLookupDraft(self: *SpeculativeGenerator, ids: []const u32) []const u32 {
        if (ids.len < self.ngram_len * 2) return &.{};
        const needle = ids[ids.len - self.ngram_len ..];
        // Buscar de derecha a izquierda (match más reciente), evitando overlap.
        const search_end = ids.len - self.ngram_len;
        var i: usize = search_end;
        while (i > 0) {
            i -= 1;
            if (matchNgram(ids, i, needle)) {
                const start = i + self.ngram_len;
                const limit = @min(start + self.max_draft, ids.len);
                if (start < limit) {
                    return ids[start..limit];
                }
                return &.{};
            }
            if (i == 0) break;
        }
        return &.{};
    }

    fn matchNgram(hay: []const u32, off: usize, needle: []const u32) bool {
        for (needle, 0..) |n, j| {
            if (hay[off + j] != n) return false;
        }
        return true;
    }
};

fn argmax(logits: []const f32) u32 {
    var best: u32 = 0;
    var best_val = logits[0];
    for (logits, 0..) |v, i| {
        if (v > best_val) {
            best_val = v;
            best = @intCast(i);
        }
    }
    return best;
}

pub const EmitFn = *const fn (ctx: *anyopaque, piece: []const u8) anyerror!void;

// --- tests ---

test "promptLookupDraft encuentra repeticion" {
    const ids = &[_]u32{ 0, 1, 0, 1 };
    var gen = SpeculativeGenerator{ .tf = undefined, .ngram_len = 2, .max_draft = 4 };
    const draft = gen.promptLookupDraft(ids);
    // needle = [0,1] (últimos 2). Match en pos=0. Draft = ids[2..] = [0,1].
    try std.testing.expectEqual(@as(usize, 2), draft.len);
    try std.testing.expectEqual(@as(u32, 0), draft[0]);
    try std.testing.expectEqual(@as(u32, 1), draft[1]);
}

test "promptLookupDraft evita overlap" {
    // Necesita >= ngram*2 tokens (guarda de la función). "aaaa" -> needle=[a,a] en ids[2..4];
    // match en i=1 -> draft = ids[3..4] = [a].
    const ids = &[_]u32{ 7, 7, 7, 7 };
    var gen = SpeculativeGenerator{ .tf = undefined, .ngram_len = 2, .max_draft = 4 };
    const draft = gen.promptLookupDraft(ids);
    try std.testing.expectEqual(@as(usize, 1), draft.len);
    try std.testing.expectEqual(@as(u32, 7), draft[0]);
}

test "promptLookupDraft sin repeticion devuelve vacio" {
    var gen = SpeculativeGenerator{ .tf = undefined, .ngram_len = 4, .max_draft = 8 };
    const ids = &[_]u32{ 1, 2, 3, 4, 5 };
    const draft = gen.promptLookupDraft(ids);
    try std.testing.expectEqual(@as(usize, 0), draft.len);
}
