// zmemory — motor de memoria híbrido propio en Zig ( embeddings + keyword match )
const std = @import("std");

pub const MemoryEntry = struct {
    text: []const u8,
    embedding: []const f32, // Opcional, puede estar vacío
};

pub const MemoryStore = struct {
    entries: std.ArrayList(MemoryEntry),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) MemoryStore {
        return .{
            .entries = std.ArrayList(MemoryEntry).empty,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MemoryStore) void {
        for (self.entries.items) |e| {
            self.allocator.free(e.text);
            self.allocator.free(e.embedding);
        }
        self.entries.deinit(self.allocator);
    }

    pub fn add(self: *MemoryStore, text: []const u8, embedding: []const f32) !void {
        const text_copy = try self.allocator.dupe(u8, text);
        const emb_copy = try self.allocator.dupe(f32, embedding);
        try self.entries.append(self.allocator, .{
            .text = text_copy,
            .embedding = emb_copy,
        });
    }

    /// Calcula la similitud coseno entre dos vectores f32
    pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
        if (a.len != b.len or a.len == 0) return 0;
        var dot: f32 = 0;
        var norm_a: f32 = 0;
        var norm_b: f32 = 0;
        for (a, b) |va, vb| {
            dot += va * vb;
            norm_a += va * va;
            norm_b += vb * vb;
        }
        if (norm_a == 0 or norm_b == 0) return 0;
        return dot / (@sqrt(norm_a) * @sqrt(norm_b));
    }

    /// Dimensión por defecto de los embeddings generados internamente.
    pub const EMBED_DIM: usize = 64;

    /// Genera un embedding determinista por "hashing trick" (feature hashing):
    /// cada token se mapea a un bucket con signo y se acumula; el vector se
    /// L2-normaliza. Es un embedding LÉXICO (no semántico) pero determinista,
    /// 100% Zig y sin dependencias — suficiente para que la similitud coseno
    /// aporte señal real a la búsqueda híbrida. El llamador libera el slice.
    pub fn embedText(allocator: std.mem.Allocator, text: []const u8, dim: usize) ![]f32 {
        const v = try allocator.alloc(f32, dim);
        @memset(v, 0);
        var it = std.mem.tokenizeAny(u8, text, " ,.!?\n\t");
        while (it.next()) |tok| {
            const h = std.hash.Wyhash.hash(0xF1, tok);
            const bucket: usize = @intCast(h % dim);
            const sign: f32 = if (((h >> 32) & 1) == 0) 1.0 else -1.0;
            v[bucket] += sign;
        }
        var norm: f32 = 0;
        for (v) |x| norm += x * x;
        if (norm > 0) {
            const inv = 1.0 / @sqrt(norm);
            for (v) |*x| x.* *= inv;
        }
        return v;
    }

    /// Frecuencia (case-insensitive) de un término dentro de un texto.
    fn termFreq(text: []const u8, term: []const u8) f32 {
        var tf: f32 = 0;
        var it = std.mem.tokenizeAny(u8, text, " ,.!?\n\t");
        while (it.next()) |t| {
            if (std.ascii.eqlIgnoreCase(t, term)) tf += 1;
        }
        return tf;
    }

    /// Añade una entrada generando su embedding automáticamente (hybrid real).
    pub fn addText(self: *MemoryStore, text: []const u8) !void {
        const emb = try embedText(self.allocator, text, EMBED_DIM);
        defer self.allocator.free(emb);
        try self.add(text, emb);
    }

    /// Búsqueda generando el embedding de la query internamente.
    pub fn searchText(self: MemoryStore, query: []const u8, top_k: usize, out_results: *std.ArrayList([]const u8)) !void {
        const qe = try embedText(self.allocator, query, EMBED_DIM);
        defer self.allocator.free(qe);
        try self.search(query, qe, top_k, out_results);
    }

    /// Búsqueda híbrida: 0.7 * coseno(embeddings) + 0.3 * BM25(léxico).
    /// BM25 real: IDF = log(1 + (N - df + 0.5)/(df + 0.5)) y normalización por
    /// longitud de documento (k1=1.5, b=0.75) sobre el corpus de la store.
    pub fn search(self: MemoryStore, query: []const u8, query_emb: []const f32, top_k: usize, out_results: *std.ArrayList([]const u8)) !void {
        const N = self.entries.items.len;
        if (N == 0) return;
        const a = self.allocator;
        const ScoreIndex = struct { index: usize, score: f32 };
        const k1: f32 = 1.5;
        const b: f32 = 0.75;
        const Nf: f32 = @floatFromInt(N);

        // Longitudes de documento y media (avgdl) para la normalización BM25.
        const dls = try a.alloc(f32, N);
        defer a.free(dls);
        var total_dl: f32 = 0;
        for (self.entries.items, 0..) |e, i| {
            var c: f32 = 0;
            var it = std.mem.tokenizeAny(u8, e.text, " ,.!?\n\t");
            while (it.next()) |_| c += 1;
            dls[i] = c;
            total_dl += c;
        }
        const avgdl = if (total_dl > 0) total_dl / Nf else 1.0;

        // Términos únicos de la query.
        var qterms = std.ArrayList([]const u8).empty;
        defer qterms.deinit(a);
        var qit = std.mem.tokenizeAny(u8, query, " ,.!?\n\t");
        outer: while (qit.next()) |t| {
            for (qterms.items) |ex| {
                if (std.ascii.eqlIgnoreCase(ex, t)) continue :outer;
            }
            try qterms.append(a, t);
        }
        // Document frequency por término (cuántos docs lo contienen).
        const dfs = try a.alloc(f32, qterms.items.len);
        defer a.free(dfs);
        for (qterms.items, 0..) |qt, qi| {
            var df: f32 = 0;
            for (self.entries.items) |e| {
                if (termFreq(e.text, qt) > 0) df += 1;
            }
            dfs[qi] = df;
        }

        var scores = std.ArrayList(ScoreIndex).empty;
        defer scores.deinit(a);
        for (self.entries.items, 0..) |e, idx| {
            var bm: f32 = 0;
            for (qterms.items, 0..) |qt, qi| {
                const tf = termFreq(e.text, qt);
                const df = dfs[qi];
                if (tf > 0 and df > 0) {
                    const idf = @log(1.0 + (Nf - df + 0.5) / (df + 0.5));
                    const denom = tf + k1 * (1.0 - b + b * dls[idx] / avgdl);
                    bm += idf * (tf * (k1 + 1.0)) / denom;
                }
            }
            const cos_sim = if (e.embedding.len > 0 and query_emb.len > 0)
                cosineSimilarity(e.embedding, query_emb)
            else
                0;
            const final_score = 0.7 * cos_sim + 0.3 * bm;
            try scores.append(a, .{ .index = idx, .score = final_score });
        }

        std.mem.sort(ScoreIndex, scores.items, {}, struct {
            pub fn lessThan(_: void, x: ScoreIndex, y: ScoreIndex) bool {
                return x.score > y.score;
            }
        }.lessThan);

        const limit = @min(top_k, scores.items.len);
        for (scores.items[0..limit]) |s| {
            try out_results.append(self.allocator, self.entries.items[s.index].text);
        }
    }

    // -----------------------------------------------------------------------
    // Persistencia binaria (formato ZMEM v1)
    // -----------------------------------------------------------------------
    const MAGIC = "ZMEM";
    const FORMAT_VERSION: u32 = 1;

    pub fn serialize(self: MemoryStore, allocator: std.mem.Allocator) ![]u8 {
        var buf = std.ArrayList(u8).empty;
        errdefer buf.deinit(allocator);

        try buf.appendSlice(allocator, MAGIC);

        var tmp: [4]u8 = undefined;
        std.mem.writeInt(u32, &tmp, FORMAT_VERSION, .little);
        try buf.appendSlice(allocator, &tmp);

        std.mem.writeInt(u32, &tmp, @intCast(self.entries.items.len), .little);
        try buf.appendSlice(allocator, &tmp);

        for (self.entries.items) |e| {
            std.mem.writeInt(u32, &tmp, @intCast(e.text.len), .little);
            try buf.appendSlice(allocator, &tmp);
            try buf.appendSlice(allocator, e.text);

            std.mem.writeInt(u32, &tmp, @intCast(e.embedding.len), .little);
            try buf.appendSlice(allocator, &tmp);
            if (e.embedding.len > 0) {
                const emb_bytes = @as([*]const u8, @ptrCast(e.embedding.ptr))[0 .. e.embedding.len * @sizeOf(f32)];
                try buf.appendSlice(allocator, emb_bytes);
            }
        }

        return buf.toOwnedSlice(allocator);
    }

    pub fn deserialize(self: *MemoryStore, data: []const u8) !void {
        if (data.len < 12 or !std.mem.eql(u8, data[0..4], MAGIC))
            return error.InvalidFormat;

        const version = std.mem.readInt(u32, data[4..8], .little);
        if (version != FORMAT_VERSION) return error.UnsupportedVersion;

        const n_entries = std.mem.readInt(u32, data[8..12], .little);

        // Limpiar entradas previas
        for (self.entries.items) |e| {
            self.allocator.free(e.text);
            self.allocator.free(e.embedding);
        }
        self.entries.clearRetainingCapacity();

        var off: usize = 12;
        var i: usize = 0;
        while (i < n_entries) : (i += 1) {
            if (off + 8 > data.len) return error.Truncated;
            const text_len = std.mem.readInt(u32, data[off..][0..4], .little);
            off += 4;
            if (off + text_len > data.len) return error.Truncated;
            const text = try self.allocator.dupe(u8, data[off..][0..text_len]);
            errdefer self.allocator.free(text);
            off += text_len;

            const emb_len = std.mem.readInt(u32, data[off..][0..4], .little);
            off += 4;
            const emb_bytes_len = emb_len * @sizeOf(f32);
            if (off + emb_bytes_len > data.len) return error.Truncated;
            const emb = try self.allocator.alloc(f32, emb_len);
            errdefer self.allocator.free(emb);
            if (emb_len > 0) {
                @memcpy(std.mem.sliceAsBytes(emb), data[off..][0..emb_bytes_len]);
            }
            off += emb_bytes_len;

            try self.entries.append(self.allocator, .{ .text = text, .embedding = emb });
        }
    }

    pub fn saveToFile(self: MemoryStore, io: std.Io, path: []const u8, scratch: std.mem.Allocator) !void {
        const buf = try self.serialize(scratch);
        defer scratch.free(buf);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = buf });
    }

    pub fn loadFromFile(self: *MemoryStore, io: std.Io, path: []const u8, scratch: std.mem.Allocator) !void {
        const data = try std.Io.Dir.cwd().readFileAlloc(io, path, scratch, .unlimited);
        defer scratch.free(data);
        try self.deserialize(data);
    }
};

// --- tests ---
test "zmemory hybrid search" {
    const alloc = std.testing.allocator;
    var store = MemoryStore.init(alloc);
    defer store.deinit();

    try store.add("Compilar proyectos en Zig usando build.", &.{ 0.1, 0.2, 0.3 });
    try store.add("Optimizador Muon es rápido.", &.{ 0.5, 0.5, 0.5 });

    var results = std.ArrayList([]const u8).empty;
    defer results.deinit(alloc);

    // Buscar por coincidencia de términos
    try store.search("optimizador muon", &.{ 0.45, 0.45, 0.45 }, 1, &results);

    try std.testing.expectEqual(@as(usize, 1), results.items.len);
    try std.testing.expectEqualStrings("Optimizador Muon es rápido.", results.items[0]);
}

test "zmemory serialize then deserialize roundtrip" {
    const alloc = std.testing.allocator;
    var store = MemoryStore.init(alloc);
    defer store.deinit();

    try store.add("Zig es rápido.", &.{ 0.1, 0.2, 0.3 });
    try store.add("Zig es seguro.", &.{}); // sin embedding

    const buf = try store.serialize(alloc);
    defer alloc.free(buf);

    var loaded = MemoryStore.init(alloc);
    defer loaded.deinit();
    try loaded.deserialize(buf);

    try std.testing.expectEqual(@as(usize, 2), loaded.entries.items.len);
    try std.testing.expectEqualStrings("Zig es rápido.", loaded.entries.items[0].text);
    try std.testing.expectEqual(@as(usize, 3), loaded.entries.items[0].embedding.len);
    try std.testing.expectEqualStrings("Zig es seguro.", loaded.entries.items[1].text);
    try std.testing.expectEqual(@as(usize, 0), loaded.entries.items[1].embedding.len);
}

test "zmemory deserialize invalid magic" {
    const alloc = std.testing.allocator;
    var store = MemoryStore.init(alloc);
    defer store.deinit();
    try std.testing.expectError(error.InvalidFormat, store.deserialize("XXXX"));
}

test "zmemory deserialize unsupported version" {
    const alloc = std.testing.allocator;
    var store = MemoryStore.init(alloc);
    defer store.deinit();
    var bad = [_]u8{ 'Z', 'M', 'E', 'M' } ++ [_]u8{ 0xFF, 0xFF, 0xFF, 0xFF } ++ [_]u8{ 0, 0, 0, 0 };
    try std.testing.expectError(error.UnsupportedVersion, store.deserialize(&bad));
}

test "embedText determinista, normalizado y discriminante" {
    const alloc = std.testing.allocator;
    const e1 = try MemoryStore.embedText(alloc, "optimizador muon rápido", 64);
    defer alloc.free(e1);
    const e1b = try MemoryStore.embedText(alloc, "optimizador muon rápido", 64);
    defer alloc.free(e1b);
    const e2 = try MemoryStore.embedText(alloc, "compilar proyecto zig build", 64);
    defer alloc.free(e2);
    // determinista
    try std.testing.expectEqualSlices(f32, e1, e1b);
    // L2-normalizado (norma ~1)
    var n: f32 = 0;
    for (e1) |x| n += x * x;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), @sqrt(n), 1e-4);
    // un texto se parece más a sí mismo que a otro distinto
    try std.testing.expect(MemoryStore.cosineSimilarity(e1, e1b) > MemoryStore.cosineSimilarity(e1, e2));
}

test "searchText (embeddings + BM25) recupera la entrada relevante" {
    const alloc = std.testing.allocator;
    var store = MemoryStore.init(alloc);
    defer store.deinit();
    try store.addText("Compilar proyectos en Zig usando build.");
    try store.addText("El optimizador Muon es rápido y ortogonal.");
    try store.addText("La destilación MOHAWK transfiere de Transformer a SSM.");

    var results = std.ArrayList([]const u8).empty;
    defer results.deinit(alloc);
    try store.searchText("optimizador muon", 1, &results);
    try std.testing.expectEqual(@as(usize, 1), results.items.len);
    try std.testing.expectEqualStrings("El optimizador Muon es rápido y ortogonal.", results.items[0]);
}
