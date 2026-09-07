// libfluency - superficie C-ABI para embebido. Inferencia real con Transformer GGUF.
const std = @import("std");
const zload = @import("zload");
const ztok = @import("ztok");
const zarch = @import("zarch");

// ---------------------------------------------------------------------------
// C-ABI surface (estable)
// ---------------------------------------------------------------------------
const FluencyModel = extern struct {
    magic: u32,
    _opaque: usize,
};

const FluencyOpts = extern struct {
    max_tokens: u32 = 128,
    temperature: f32 = 0.0,
};

const TokenCb = *const fn ([*]const u8, usize, ?*anyopaque) callconv(.c) void;

// ---------------------------------------------------------------------------
// LibModel (interno)
// ---------------------------------------------------------------------------
const LibModel = struct {
    allocator: std.mem.Allocator,
    tf: zarch.Transformer,
    tok: ztok.Tokenizer,
    rng: std.Random.DefaultPrng,
    gguf_data: ?[]u8,
    arena: ?*std.heap.ArenaAllocator,
};

// ---------------------------------------------------------------------------
// Helpers de carga GGUF (replicados de main.zig para autonomia de lib.zig)
// ---------------------------------------------------------------------------
fn layerName(buf: []u8, layer: usize, suffix: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "blk.{d}.{s}", .{ layer, suffix });
}

fn loadDirect(tmp_alloc: std.mem.Allocator, gguf: zload.GgufFile, name: []const u8, dst: []f32) !void {
    const t = gguf.findTensor(name) orelse return error.MissingTensor;
    const src = try zload.dequantTensor(tmp_alloc, gguf, t);
    defer tmp_alloc.free(src);
    if (src.len != dst.len) return error.SizeMismatch;
    @memcpy(dst, src);
}

fn loadTransposed(tmp_alloc: std.mem.Allocator, gguf: zload.GgufFile, name: []const u8, dst: []f32, out_f: usize, in_f: usize) !void {
    const t = gguf.findTensor(name) orelse return error.MissingTensor;
    const src = try zload.dequantTensor(tmp_alloc, gguf, t);
    defer tmp_alloc.free(src);
    if (src.len != out_f * in_f or dst.len != out_f * in_f) return error.SizeMismatch;
    for (0..out_f) |o| {
        const row = o * in_f;
        for (0..in_f) |i| dst[i * out_f + o] = src[row + i];
    }
}

fn loadGgufTransformer(state_alloc: std.mem.Allocator, wa: std.mem.Allocator, tmp_alloc: std.mem.Allocator, gguf: zload.GgufFile, info: zload.ModelInfo, max_seq: usize) !zarch.Transformer {
    const dim: usize = info.embd;
    const hidden: usize = info.ffn;
    const L: usize = info.n_layers;
    const n_heads: usize = info.n_heads;
    const n_kv: usize = info.n_kv_heads;
    const kv_dim = (dim * n_kv) / n_heads;
    const vocab: usize = info.vocab;

    const cfg = zarch.Config{
        .dim = dim,
        .hidden_dim = hidden,
        .n_layers = L,
        .n_heads = n_heads,
        .n_kv_heads = n_kv,
        .vocab_size = vocab,
        .seq_len = max_seq,
        .theta = info.rope_freq,
        .eps = info.rms_eps,
        .rope_neox = !std.mem.startsWith(u8, info.arch, "llama"),
    };
    var tf = try zarch.Transformer.init(state_alloc, cfg);
    errdefer tf.deinit();

    const emb = try wa.alloc(f32, vocab * dim);
    const rms_att = try wa.alloc(f32, L * dim);
    const rms_ffn = try wa.alloc(f32, L * dim);
    const rms_final = try wa.alloc(f32, dim);
    const wq = try wa.alloc(f32, L * dim * dim);
    const wk = try wa.alloc(f32, L * kv_dim * dim);
    const wv = try wa.alloc(f32, L * kv_dim * dim);
    const wo = try wa.alloc(f32, L * dim * dim);
    const w1 = try wa.alloc(f32, L * dim * hidden);
    const w2 = try wa.alloc(f32, L * hidden * dim);
    const w3 = try wa.alloc(f32, L * dim * hidden);
    const bq = try wa.alloc(f32, L * dim);
    const bk = try wa.alloc(f32, L * kv_dim);
    const bv = try wa.alloc(f32, L * kv_dim);

    try loadDirect(tmp_alloc, gguf, "token_embd.weight", emb);
    try loadDirect(tmp_alloc, gguf, "output_norm.weight", rms_final);

    var nbuf: [64]u8 = undefined;
    for (0..L) |l| {
        try loadDirect(tmp_alloc, gguf, try layerName(&nbuf, l, "attn_norm.weight"), rms_att[l * dim ..][0..dim]);
        try loadDirect(tmp_alloc, gguf, try layerName(&nbuf, l, "ffn_norm.weight"), rms_ffn[l * dim ..][0..dim]);
        try loadTransposed(tmp_alloc, gguf, try layerName(&nbuf, l, "attn_q.weight"), wq[l * dim * dim ..][0 .. dim * dim], dim, dim);
        try loadTransposed(tmp_alloc, gguf, try layerName(&nbuf, l, "attn_k.weight"), wk[l * kv_dim * dim ..][0 .. kv_dim * dim], kv_dim, dim);
        try loadTransposed(tmp_alloc, gguf, try layerName(&nbuf, l, "attn_v.weight"), wv[l * kv_dim * dim ..][0 .. kv_dim * dim], kv_dim, dim);
        try loadTransposed(tmp_alloc, gguf, try layerName(&nbuf, l, "attn_output.weight"), wo[l * dim * dim ..][0 .. dim * dim], dim, dim);
        try loadTransposed(tmp_alloc, gguf, try layerName(&nbuf, l, "ffn_gate.weight"), w1[l * dim * hidden ..][0 .. dim * hidden], hidden, dim);
        try loadTransposed(tmp_alloc, gguf, try layerName(&nbuf, l, "ffn_up.weight"), w3[l * dim * hidden ..][0 .. dim * hidden], hidden, dim);
        try loadTransposed(tmp_alloc, gguf, try layerName(&nbuf, l, "ffn_down.weight"), w2[l * hidden * dim ..][0 .. hidden * dim], dim, hidden);
        try loadDirect(tmp_alloc, gguf, try layerName(&nbuf, l, "attn_q.bias"), bq[l * dim ..][0..dim]);
        try loadDirect(tmp_alloc, gguf, try layerName(&nbuf, l, "attn_k.bias"), bk[l * kv_dim ..][0..kv_dim]);
        try loadDirect(tmp_alloc, gguf, try layerName(&nbuf, l, "attn_v.bias"), bv[l * kv_dim ..][0..kv_dim]);
    }

    tf.token_embedding_table = emb;
    tf.rms_att_weight = rms_att;
    tf.rms_ffn_weight = rms_ffn;
    tf.rms_final_weight = rms_final;
    tf.wq = wq;
    tf.wk = wk;
    tf.wv = wv;
    tf.wo = wo;
    tf.w1 = w1;
    tf.w2 = w2;
    tf.w3 = w3;
    tf.bq = bq;
    tf.bk = bk;
    tf.bv = bv;
    tf.wcls = null;
    return tf;
}

fn buildTokenizer(alloc: std.mem.Allocator, gguf: zload.GgufFile) !ztok.Tokenizer {
    var t = try ztok.Tokenizer.init(alloc);
    errdefer t.deinit();

    const toks = gguf.getMeta("tokenizer.ggml.tokens") orelse return error.NoVocabInGguf;
    if (toks != .array or toks.array.type != .string) return error.NoVocabInGguf;
    var it = zload.arrayStrIter(toks.array);
    while (it.next()) |s| try t.addToken(s);

    if (gguf.getMeta("tokenizer.ggml.merges")) |m| {
        if (m == .array and m.array.type == .string) {
            var mit = zload.arrayStrIter(m.array);
            while (mit.next()) |s| try t.addMerge(s);
        }
    }
    t.bos_id = gguf.getU32("tokenizer.ggml.bos_token_id");
    t.eos_id = gguf.getU32("tokenizer.ggml.eos_token_id");
    return t;
}

// ---------------------------------------------------------------------------
// Sampling (greedy o con temperatura)
// ---------------------------------------------------------------------------
fn sampleToken(logits: []f32, temp: f32, rng: *std.Random.DefaultPrng) u32 {
    if (temp <= 0.0) {
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
    var maxv = logits[0];
    for (logits) |v| {
        if (v > maxv) maxv = v;
    }
    var sum: f32 = 0;
    for (logits) |*v| {
        v.* = @exp((v.* - maxv) / temp);
        sum += v.*;
    }
    const r = rng.random().float(f32) * sum;
    var acc: f32 = 0;
    for (logits, 0..) |v, i| {
        acc += v;
        if (acc >= r) return @intCast(i);
    }
    return @intCast(logits.len - 1);
}

// ---------------------------------------------------------------------------
// C-ABI exports
// ---------------------------------------------------------------------------
export fn fluency_version() [*:0]const u8 {
    return "0.7.0";
}

fn loadFromData(allocator: std.mem.Allocator, data: []u8) ?*FluencyModel {
    var gguf = zload.parseGguf(allocator, data) catch {
        allocator.free(data);
        return null;
    };

    const info = zload.modelInfo(gguf) catch {
        gguf.deinit();
        allocator.free(data);
        return null;
    };

    var tok = buildTokenizer(allocator, gguf) catch {
        gguf.deinit();
        allocator.free(data);
        return null;
    };

    var arena = std.heap.ArenaAllocator.init(allocator);
    var tf = loadGgufTransformer(allocator, arena.allocator(), allocator, gguf, info, 1024) catch {
        arena.deinit();
        tok.deinit();
        gguf.deinit();
        allocator.free(data);
        return null;
    };

    gguf.deinit();

    const arena_ptr = allocator.create(std.heap.ArenaAllocator) catch {
        tf.deinit();
        arena.deinit();
        tok.deinit();
        allocator.free(data);
        return null;
    };
    arena_ptr.* = arena;

    const model = allocator.create(LibModel) catch {
        allocator.destroy(arena_ptr);
        tf.deinit();
        arena.deinit();
        tok.deinit();
        allocator.free(data);
        return null;
    };

    model.* = .{
        .allocator = allocator,
        .tf = tf,
        .tok = tok,
        .rng = std.Random.DefaultPrng.init(0xF1),
        .gguf_data = data,
        .arena = arena_ptr,
    };

    const fm = allocator.create(FluencyModel) catch {
        allocator.destroy(model);
        allocator.destroy(arena_ptr);
        tf.deinit();
        arena.deinit();
        tok.deinit();
        allocator.free(data);
        return null;
    };

    fm.* = .{ .magic = 0x465a5341, ._opaque = @intFromPtr(model) };
    return fm;
}

export fn fluency_load(path: [*:0]const u8) ?*FluencyModel {
    if (comptime @import("builtin").target.cpu.arch == .wasm32) return null;
    const allocator = std.heap.page_allocator;
    const path_slice = std.mem.span(path);

    var threaded = std.Io.Threaded.init(allocator, .{});
    const io = threaded.io();
    defer threaded.deinit();

    const data = std.Io.Dir.cwd().readFileAlloc(io, path_slice, allocator, .unlimited) catch return null;
    return loadFromData(allocator, data);
}

export fn fluency_load_bytes(data: [*]const u8, len: usize) ?*FluencyModel {
    const allocator = std.heap.page_allocator;
    const copy = allocator.dupe(u8, data[0..len]) catch return null;
    return loadFromData(allocator, copy);
}

export fn fluency_generate(
    model: ?*FluencyModel,
    prompt: [*:0]const u8,
    opts: ?*const FluencyOpts,
    cb: ?TokenCb,
    user: ?*anyopaque,
) c_int {
    const fm = model orelse return -1;
    if (fm.magic != 0x465a5341) return -1;
    const m: *LibModel = @ptrFromInt(fm._opaque);
    const prompt_slice = std.mem.span(prompt);
    const callback = cb orelse return 0;
    const max_tokens = if (opts) |o| o.max_tokens else 128;
    const temp = if (opts) |o| o.temperature else 0.0;

    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(m.allocator);
    m.tok.encode(prompt_slice, &ids) catch return -1;

    var pos: usize = 0;
    var logits: []f32 = undefined;
    var any = false;
    for (ids.items) |id| {
        if (pos >= m.tf.config.seq_len) break;
        logits = m.tf.forward(id, pos);
        pos += 1;
        any = true;
    }
    if (!any) return 0;

    var buf: [256]u8 = undefined;
    var gen: usize = 0;
    while (gen < max_tokens and pos < m.tf.config.seq_len) : (gen += 1) {
        const next = sampleToken(logits, temp, &m.rng);
        if (m.tok.eos_id) |e| {
            if (next == e) break;
        }
        const piece = m.tok.decodeToken(next, &buf);
        callback(piece.ptr, piece.len, user);
        logits = m.tf.forward(next, pos);
        pos += 1;
    }
    return 0;
}

const GenBufCtx = struct {
    buf: [*]u8,
    cap: usize,
    pos: usize,
};

fn bufCallback(ptr: [*]const u8, len: usize, user: ?*anyopaque) callconv(.c) void {
    const ctx = @as(*GenBufCtx, @alignCast(@ptrCast(user.?)));
    const n = @min(len, ctx.cap - ctx.pos);
    if (n > 0) {
        @memcpy(ctx.buf + ctx.pos, ptr[0..n]);
        ctx.pos += n;
    }
}

export fn fluency_generate_buf(
    model: ?*FluencyModel,
    prompt: [*:0]const u8,
    max_tokens: u32,
    temp: f32,
    out_buf: [*]u8,
    out_cap: usize,
) c_int {
    const fm = model orelse return -1;
    if (fm.magic != 0x465a5341) return -1;
    const opts = FluencyOpts{ .max_tokens = max_tokens, .temperature = temp };
    var ctx = GenBufCtx{ .buf = out_buf, .cap = out_cap, .pos = 0 };
    const res = fluency_generate(fm, prompt, &opts, bufCallback, &ctx);
    if (res != 0) return res;
    if (ctx.pos < out_cap) out_buf[ctx.pos] = 0;
    return @intCast(ctx.pos);
}

export fn fluency_alloc(len: usize) ?[*]u8 {
    const alloc = std.heap.page_allocator;
    const ptr = alloc.alloc(u8, len) catch return null;
    return ptr.ptr;
}

export fn fluency_free_ptr(ptr: ?[*]u8, len: usize) void {
    if (ptr) |p| {
        const alloc = std.heap.page_allocator;
        alloc.free(p[0..len]);
    }
}

export fn fluency_embed(model: ?*FluencyModel, text: [*:0]const u8, out: [*]f32, out_len: c_int) c_int {
    const fm = model orelse return -1;
    if (fm.magic != 0x465a5341) return -1;
    const m: *LibModel = @ptrFromInt(fm._opaque);
    if (out_len <= 0) return -1;
    const text_slice = std.mem.span(text);

    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(m.allocator);
    m.tok.encode(text_slice, &ids) catch return -1;
    if (ids.items.len == 0) return 0;

    const dim = m.tf.config.dim;
    const n: usize = @min(@as(usize, @intCast(out_len)), dim);
    const emb_table = m.tf.token_embedding_table;
    const vocab = m.tf.config.vocab_size;

    var sum: []f32 = m.allocator.alloc(f32, dim) catch return -1;
    defer m.allocator.free(sum);
    @memset(sum, 0.0);

    for (ids.items) |id| {
        if (id >= vocab) continue;
        const e = emb_table[id * dim ..][0..dim];
        for (0..dim) |i| sum[i] += e[i];
    }

    const count: f32 = @floatFromInt(ids.items.len);
    for (0..n) |i| {
        out[i] = sum[i] / count;
    }
    return @intCast(n);
}

export fn fluency_free(model: ?*FluencyModel) void {
    const fm = model orelse return;
    if (fm.magic != 0x465a5341) return;
    const m: *LibModel = @ptrFromInt(fm._opaque);
    const alloc = m.allocator;
    m.tf.deinit();
    if (m.arena) |a| {
        a.deinit();
        alloc.destroy(a);
    }
    m.tok.deinit();
    if (m.gguf_data) |d| alloc.free(d);
    alloc.destroy(m);
    alloc.destroy(fm);
}

// Crea un modelo sintético mínimo en memoria (sin archivo GGUF).
// Util para demos embebidas donde no se dispone de un modelo real.
comptime { _ = &fluency_create_synthetic; }
export fn fluency_create_synthetic(
    dim: usize,
    hidden_dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    vocab_size: usize,
    seq_len: usize,
) ?*FluencyModel {
    const allocator = std.heap.page_allocator;

    const cfg = zarch.Config{
        .dim = dim,
        .hidden_dim = hidden_dim,
        .n_layers = n_layers,
        .n_heads = n_heads,
        .n_kv_heads = n_kv_heads,
        .vocab_size = vocab_size,
        .seq_len = seq_len,
        .theta = 10000.0,
        .eps = 1e-5,
        .rope_neox = true,
    };
    var tf = zarch.Transformer.init(allocator, cfg) catch return null;

    var arena = std.heap.ArenaAllocator.init(allocator);
    const aa = arena.allocator();

    const emb = aa.alloc(f32, vocab_size * dim) catch return null;
    @memset(emb, 0.1);
    // Hacer un token dominante (token 2) para que la generación sea determinista
    if (vocab_size > 2) {
        for (0..dim) |i| emb[2 * dim + i] = 10.0;
    }
    tf.token_embedding_table = emb;

    const rms_att = aa.alloc(f32, n_layers * dim) catch return null;
    @memset(rms_att, 1.0);
    tf.rms_att_weight = rms_att;
    const rms_ffn = aa.alloc(f32, n_layers * dim) catch return null;
    @memset(rms_ffn, 1.0);
    tf.rms_ffn_weight = rms_ffn;
    const rms_final = aa.alloc(f32, dim) catch return null;
    @memset(rms_final, 1.0);
    tf.rms_final_weight = rms_final;

    const wq = aa.alloc(f32, n_layers * dim * dim) catch return null;
    @memset(wq, 0.001);
    tf.wq = wq;
    const wk = aa.alloc(f32, n_layers * computeKvDim(dim, n_kv_heads, n_heads) * dim) catch return null;
    @memset(wk, 0.001);
    tf.wk = wk;
    const wv = aa.alloc(f32, n_layers * computeKvDim(dim, n_kv_heads, n_heads) * dim) catch return null;
    @memset(wv, 0.001);
    tf.wv = wv;
    const wo = aa.alloc(f32, n_layers * dim * dim) catch return null;
    @memset(wo, 0.001);
    tf.wo = wo;

    const w1 = aa.alloc(f32, n_layers * dim * hidden_dim) catch return null;
    @memset(w1, 0.001);
    tf.w1 = w1;
    const w2 = aa.alloc(f32, n_layers * hidden_dim * dim) catch return null;
    @memset(w2, 0.001);
    tf.w2 = w2;
    const w3 = aa.alloc(f32, n_layers * dim * hidden_dim) catch return null;
    @memset(w3, 0.001);
    tf.w3 = w3;

    const bq = aa.alloc(f32, n_layers * dim) catch return null;
    @memset(bq, 0.0);
    tf.bq = bq;
    const bk = aa.alloc(f32, n_layers * computeKvDim(dim, n_kv_heads, n_heads)) catch return null;
    @memset(bk, 0.0);
    tf.bk = bk;
    const bv = aa.alloc(f32, n_layers * computeKvDim(dim, n_kv_heads, n_heads)) catch return null;
    @memset(bv, 0.0);
    tf.bv = bv;
    tf.wcls = null;

    var tok = ztok.Tokenizer.init(allocator) catch return null;
    var tokens_ok = false;
    tok.addToken("<s>") catch return null;
    tok.addToken("a") catch return null;
    tok.addToken("b") catch return null;
    tok.addToken("</s>") catch return null;
    tokens_ok = true;
    if (!tokens_ok) {
        tok.deinit();
        return null;
    }
    tok.bos_id = 0;
    tok.eos_id = 3;

    const arena_ptr = allocator.create(std.heap.ArenaAllocator) catch return null;
    arena_ptr.* = arena;

    const model = allocator.create(LibModel) catch return null;
    model.* = .{
        .allocator = allocator,
        .tf = tf,
        .tok = tok,
        .rng = std.Random.DefaultPrng.init(42),
        .gguf_data = null,
        .arena = arena_ptr,
    };

    const fm = allocator.create(FluencyModel) catch return null;
    fm.* = .{ .magic = 0x465a5341, ._opaque = @intFromPtr(model) };
    return fm;
}

fn computeKvDim(dim_val: usize, n_kv: usize, n_h: usize) usize {
    return (dim_val * n_kv) / n_h;
}

/// Demo end-to-end: crea modelo sintético, genera texto y libera.
/// Usar en demos embebidas (WASM/Node) donde no se dispone de un modelo GGUF.
export fn fluency_demo_generate(
    prompt: [*:0]const u8,
    max_tokens: u32,
    out_buf: [*]u8,
    out_cap: usize,
) c_int {
    const model = fluency_create_synthetic(8, 16, 1, 2, 1, 4, 16) orelse return -1;
    const res = fluency_generate_buf(model, prompt, max_tokens, 0.0, out_buf, out_cap);
    fluency_free(model);
    return res;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
test "fluency_version" {
    try std.testing.expectEqualStrings("0.7.0", std.mem.span(fluency_version()));
}

test "fluency_load archivo inexistente devuelve null" {
    const ptr = fluency_load("ruta_que_no_existe_nunca.gguf");
    try std.testing.expect(ptr == null);
}

test "fluency_generate con modelo minimo produce salida real" {
    const alloc = std.testing.allocator;

    // Construir un Transformer minimo directamente (sin GGUF)
    const cfg = zarch.Config{
        .dim = 8,
        .hidden_dim = 16,
        .n_layers = 1,
        .n_heads = 2,
        .n_kv_heads = 1,
        .vocab_size = 4,
        .seq_len = 16,
        .theta = 10000.0,
        .eps = 1e-5,
        .rope_neox = true,
    };
    var tf = try zarch.Transformer.init(alloc, cfg);

    // Arena para los pesos (liberacion masiva al final)
    var arena = std.heap.ArenaAllocator.init(alloc);
    const aa = arena.allocator();

    // Pesos del embedding: token 2 ("b") tiene embedding dominante
    var emb = aa.alloc(f32, cfg.vocab_size * cfg.dim) catch unreachable;
    @memset(emb, 0.1);
    for (0..cfg.dim) |i| emb[2 * cfg.dim + i] = 10.0;
    tf.token_embedding_table = emb;

    // RMS weights a 1.0
    const rms_att = aa.alloc(f32, cfg.n_layers * cfg.dim) catch unreachable;
    @memset(rms_att, 1.0);
    tf.rms_att_weight = rms_att;
    const rms_ffn = aa.alloc(f32, cfg.n_layers * cfg.dim) catch unreachable;
    @memset(rms_ffn, 1.0);
    tf.rms_ffn_weight = rms_ffn;
    const rms_final = aa.alloc(f32, cfg.dim) catch unreachable;
    @memset(rms_final, 1.0);
    tf.rms_final_weight = rms_final;

    // Attention pesos pequenos
    const wq = aa.alloc(f32, cfg.n_layers * cfg.dim * cfg.dim) catch unreachable;
    @memset(wq, 0.001);
    tf.wq = wq;
    const wk = aa.alloc(f32, cfg.n_layers * cfg.dim * cfg.dim) catch unreachable;
    @memset(wk, 0.001);
    tf.wk = wk;
    const wv = aa.alloc(f32, cfg.n_layers * cfg.dim * cfg.dim) catch unreachable;
    @memset(wv, 0.001);
    tf.wv = wv;
    const wo = aa.alloc(f32, cfg.n_layers * cfg.dim * cfg.dim) catch unreachable;
    @memset(wo, 0.001);
    tf.wo = wo;

    // FFN pesos pequenos
    const w1 = aa.alloc(f32, cfg.n_layers * cfg.dim * cfg.hidden_dim) catch unreachable;
    @memset(w1, 0.001);
    tf.w1 = w1;
    const w2 = aa.alloc(f32, cfg.n_layers * cfg.hidden_dim * cfg.dim) catch unreachable;
    @memset(w2, 0.001);
    tf.w2 = w2;
    const w3 = aa.alloc(f32, cfg.n_layers * cfg.dim * cfg.hidden_dim) catch unreachable;
    @memset(w3, 0.001);
    tf.w3 = w3;

    // Bias cero
    const bq = aa.alloc(f32, cfg.n_layers * cfg.dim) catch unreachable;
    @memset(bq, 0.0);
    tf.bq = bq;
    const bk = aa.alloc(f32, cfg.n_layers * cfg.dim) catch unreachable;
    @memset(bk, 0.0);
    tf.bk = bk;
    const bv = aa.alloc(f32, cfg.n_layers * cfg.dim) catch unreachable;
    @memset(bv, 0.0);
    tf.bv = bv;

    tf.wcls = null;

    // Tokenizador minimo: <s>, a, b, </s>
    var tok = try ztok.Tokenizer.init(alloc);
    try tok.addToken("<s>");
    try tok.addToken("a");
    try tok.addToken("b");
    try tok.addToken("</s>");
    tok.bos_id = 0;
    tok.eos_id = 3;

    const model = try alloc.create(LibModel);
    model.* = .{
        .allocator = alloc,
        .tf = tf,
        .tok = tok,
        .rng = std.Random.DefaultPrng.init(42),
        .gguf_data = null,
        .arena = null,
    };

    var fm = FluencyModel{ .magic = 0x465a5341, ._opaque = @intFromPtr(model) };

    var out = std.ArrayList(u8).empty;
    defer out.deinit(alloc);

    const opts = FluencyOpts{ .max_tokens = 3, .temperature = 0.0 };

    const cb = struct {
        fn call(ptr: [*]const u8, len: usize, user: ?*anyopaque) callconv(.c) void {
            const arr = @as(*std.ArrayList(u8), @alignCast(@ptrCast(user.?)));
            arr.appendSlice(std.testing.allocator, ptr[0..len]) catch {};
        }
    }.call;

    const res = fluency_generate(&fm, "a", &opts, cb, &out);
    try std.testing.expectEqual(@as(c_int, 0), res);
    try std.testing.expect(out.items.len > 0);

    // La salida no debe ser el prompt ni el prefijo de echo anterior
    const is_echo = std.mem.startsWith(u8, out.items, "Fluency local response");
    try std.testing.expect(!is_echo);
    try std.testing.expect(!std.mem.eql(u8, out.items, "a"));

    // Liberacion manual (no fluency_free porque no hay arena ni gguf_data)
    model.tf.deinit();
    model.tok.deinit();
    arena.deinit();
    alloc.destroy(model);
}

test "fluency_embed con modelo minimo" {
    const alloc = std.testing.allocator;

    const cfg = zarch.Config{
        .dim = 8,
        .hidden_dim = 16,
        .n_layers = 1,
        .n_heads = 2,
        .n_kv_heads = 1,
        .vocab_size = 4,
        .seq_len = 16,
        .theta = 10000.0,
        .eps = 1e-5,
        .rope_neox = true,
    };
    var tf = try zarch.Transformer.init(alloc, cfg);

    var arena = std.heap.ArenaAllocator.init(alloc);
    const aa = arena.allocator();

    var emb = aa.alloc(f32, cfg.vocab_size * cfg.dim) catch unreachable;
    @memset(emb, 0.0);
    // Token 1 = "a" tiene embedding [0,1,0,0,0,0,0,0]
    for (0..cfg.dim) |i| emb[1 * cfg.dim + i] = @floatFromInt(i);
    tf.token_embedding_table = emb;

    const rms_att = aa.alloc(f32, cfg.n_layers * cfg.dim) catch unreachable;
    @memset(rms_att, 1.0);
    tf.rms_att_weight = rms_att;
    const rms_ffn = aa.alloc(f32, cfg.n_layers * cfg.dim) catch unreachable;
    @memset(rms_ffn, 1.0);
    tf.rms_ffn_weight = rms_ffn;
    const rms_final = aa.alloc(f32, cfg.dim) catch unreachable;
    @memset(rms_final, 1.0);
    tf.rms_final_weight = rms_final;
    const wq = aa.alloc(f32, cfg.n_layers * cfg.dim * cfg.dim) catch unreachable;
    @memset(wq, 0.0);
    tf.wq = wq;
    const wk = aa.alloc(f32, cfg.n_layers * cfg.dim * cfg.dim) catch unreachable;
    @memset(wk, 0.0);
    tf.wk = wk;
    const wv = aa.alloc(f32, cfg.n_layers * cfg.dim * cfg.dim) catch unreachable;
    @memset(wv, 0.0);
    tf.wv = wv;
    const wo = aa.alloc(f32, cfg.n_layers * cfg.dim * cfg.dim) catch unreachable;
    @memset(wo, 0.0);
    tf.wo = wo;
    const w1 = aa.alloc(f32, cfg.n_layers * cfg.dim * cfg.hidden_dim) catch unreachable;
    @memset(w1, 0.0);
    tf.w1 = w1;
    const w2 = aa.alloc(f32, cfg.n_layers * cfg.hidden_dim * cfg.dim) catch unreachable;
    @memset(w2, 0.0);
    tf.w2 = w2;
    const w3 = aa.alloc(f32, cfg.n_layers * cfg.dim * cfg.hidden_dim) catch unreachable;
    @memset(w3, 0.0);
    tf.w3 = w3;
    const bq = aa.alloc(f32, cfg.n_layers * cfg.dim) catch unreachable;
    @memset(bq, 0.0);
    tf.bq = bq;
    const bk = aa.alloc(f32, cfg.n_layers * cfg.dim) catch unreachable;
    @memset(bk, 0.0);
    tf.bk = bk;
    const bv = aa.alloc(f32, cfg.n_layers * cfg.dim) catch unreachable;
    @memset(bv, 0.0);
    tf.bv = bv;
    tf.wcls = null;

    var tok = try ztok.Tokenizer.init(alloc);
    try tok.addToken("<s>");
    try tok.addToken("a");
    try tok.addToken("b");
    try tok.addToken("</s>");
    tok.bos_id = 0;
    tok.eos_id = 3;

    const model = try alloc.create(LibModel);
    model.* = .{
        .allocator = alloc,
        .tf = tf,
        .tok = tok,
        .rng = std.Random.DefaultPrng.init(42),
        .gguf_data = null,
        .arena = null,
    };
    var fm = FluencyModel{ .magic = 0x465a5341, ._opaque = @intFromPtr(model) };

    var out: [8]f32 = undefined;
    const n = fluency_embed(&fm, "a", &out, 8);
    try std.testing.expectEqual(@as(c_int, 8), n);

    // El embedding debe reflejar los valores del token 1: [0,1,2,3,4,5,6,7]
    for (0..8) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(i)), out[i], 0.001);
    }

    model.tf.deinit();
    model.tok.deinit();
    arena.deinit();
    alloc.destroy(model);
}
