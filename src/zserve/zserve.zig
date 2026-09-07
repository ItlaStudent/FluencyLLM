// zserve - API OpenAI-compatible + SSE
const std = @import("std");
const http = std.http;
const net = std.Io.net;

pub const ChatMessage = struct {
    role: []const u8,
    content: []const u8,
};

pub const ModelInfo = struct {
    id: []const u8,
    owned_by: []const u8 = "fluency",
};

pub fn modelsResponse(allocator: std.mem.Allocator, models: []const ModelInfo) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"object\":\"list\",\"data\":[");
    for (models, 0..) |model, i| {
        if (i != 0) try out.append(allocator, ',');
        try out.appendSlice(allocator, "{\"id\":");
        try writeJsonString(&out, allocator, model.id);
        try out.appendSlice(allocator, ",\"object\":\"model\",\"created\":0,\"owned_by\":");
        try writeJsonString(&out, allocator, model.owned_by);
        try out.append(allocator, '}');
    }
    try out.appendSlice(allocator, "]}");
    return out.toOwnedSlice(allocator);
}

pub fn chatCompletionResponse(
    allocator: std.mem.Allocator,
    model: []const u8,
    content: []const u8,
    prompt_tokens: usize,
    completion_tokens: usize,
) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"id\":\"chatcmpl-fluency-local\",\"object\":\"chat.completion\",\"created\":0,\"model\":");
    try writeJsonString(&out, allocator, model);
    try out.appendSlice(allocator, ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
    try writeJsonString(&out, allocator, content);
    try out.appendSlice(allocator, "},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":");
    try out.print(allocator, "{},\"completion_tokens\":{},\"total_tokens\":{}", .{
        prompt_tokens,
        completion_tokens,
        prompt_tokens + completion_tokens,
    });
    try out.appendSlice(allocator, "}}");
    return out.toOwnedSlice(allocator);
}

pub fn sseChunk(allocator: std.mem.Allocator, content: []const u8) ![]u8 {
    var payload = std.ArrayList(u8).empty;
    defer payload.deinit(allocator);
    try payload.appendSlice(allocator, "{\"choices\":[{\"delta\":{\"content\":");
    try writeJsonString(&payload, allocator, content);
    try payload.appendSlice(allocator, "}}]}");

    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "data: ");
    try out.appendSlice(allocator, payload.items);
    try out.appendSlice(allocator, "\n\n");
    return out.toOwnedSlice(allocator);
}

pub fn sseDone(allocator: std.mem.Allocator) ![]u8 {
    return allocator.dupe(u8, "data: [DONE]\n\n");
}

pub fn ollamaTagsResponse(allocator: std.mem.Allocator, models: []const ModelInfo) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"models\":[");
    for (models, 0..) |model, i| {
        if (i != 0) try out.append(allocator, ',');
        try out.appendSlice(allocator, "{\"name\":");
        try writeJsonString(&out, allocator, model.id);
        try out.appendSlice(allocator, ",\"model\":");
        try writeJsonString(&out, allocator, model.id);
        try out.appendSlice(allocator, ",\"modified_at\":\"1970-01-01T00:00:00Z\",\"size\":0}");
    }
    try out.appendSlice(allocator, "]}");
    return out.toOwnedSlice(allocator);
}

pub fn ollamaChatResponse(allocator: std.mem.Allocator, model: []const u8, content: []const u8) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"model\":");
    try writeJsonString(&out, allocator, model);
    try out.appendSlice(allocator, ",\"created_at\":\"1970-01-01T00:00:00Z\",\"message\":{\"role\":\"assistant\",\"content\":");
    try writeJsonString(&out, allocator, content);
    try out.appendSlice(allocator, "},\"done\":true}");
    return out.toOwnedSlice(allocator);
}

pub fn ollamaGenerateResponse(allocator: std.mem.Allocator, model: []const u8, content: []const u8) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"model\":");
    try writeJsonString(&out, allocator, model);
    try out.appendSlice(allocator, ",\"created_at\":\"1970-01-01T00:00:00Z\",\"response\":");
    try writeJsonString(&out, allocator, content);
    try out.appendSlice(allocator, ",\"done\":true}");
    return out.toOwnedSlice(allocator);
}

fn writeJsonString(out: *std.ArrayList(u8), allocator: std.mem.Allocator, text: []const u8) !void {
    try out.append(allocator, '"');
    for (text) |c| {
        switch (c) {
            '"' => try out.appendSlice(allocator, "\\\""),
            '\\' => try out.appendSlice(allocator, "\\\\"),
            '\n' => try out.appendSlice(allocator, "\\n"),
            '\r' => try out.appendSlice(allocator, "\\r"),
            '\t' => try out.appendSlice(allocator, "\\t"),
            else => try out.append(allocator, c),
        }
    }
    try out.append(allocator, '"');
}

pub fn embeddingResponse(allocator: std.mem.Allocator, embedding: []const f32) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"object\":\"list\",\"data\":[{\"object\":\"embedding\",\"index\":0,\"embedding\":[");
    for (embedding, 0..) |value, i| {
        if (i != 0) try out.append(allocator, ',');
        try out.print(allocator, "{d}", .{value});
    }
    try out.appendSlice(allocator, "]}],\"usage\":{\"prompt_tokens\":0,\"total_tokens\":0}}");
    return out.toOwnedSlice(allocator);
}

pub fn ollamaEmbedResponse(allocator: std.mem.Allocator, embedding: []const f32) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"embeddings\":[[");
    for (embedding, 0..) |value, i| {
        if (i != 0) try out.append(allocator, ',');
        try out.print(allocator, "{d}", .{value});
    }
    try out.appendSlice(allocator, "]]}");
    return out.toOwnedSlice(allocator);
}

/// Respuesta de /api/show (metadatos del modelo). El formato sigue el shape
/// que esperan los clientes Ollama: details, modelfile, parameters, template.
pub fn ollamaShowResponse(allocator: std.mem.Allocator, options: ServeOptions) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"modelfile\":\"# Modelfile generated by Fluency\\nFROM ");
    try writeJsonString(&out, allocator, options.model);
    try out.appendSlice(allocator, "\\n\",\"parameters\":\"stop \"\"<|im_end|>\"\"\\nstop \"\"<|endoftext|>\"\"\",\"template\":\"{{ .System }}\\n{{ .Prompt }}\",\"details\":{\"parent_model\":\"\",\"format\":\"gguf\",\"family\":");
    try writeJsonString(&out, allocator, options.model_family);
    try out.appendSlice(allocator, ",\"families\":[");
    try writeJsonString(&out, allocator, options.model_family);
    try out.appendSlice(allocator, "],\"parameter_size\":");
    try writeJsonString(&out, allocator, options.model_params);
    try out.appendSlice(allocator, ",\"quantization_level\":");
    try writeJsonString(&out, allocator, options.model_quant);
    try out.appendSlice(allocator, "},\"model_info\":{\"general.architecture\":");
    try writeJsonString(&out, allocator, options.model_family);
    try out.appendSlice(allocator, ",\"general.context_length\":");
    try out.print(allocator, "{d}", .{options.model_ctx});
    try out.appendSlice(allocator, "}}");
    return out.toOwnedSlice(allocator);
}

/// Callback de inferencia: recibe el prompt y devuelve el texto generado (sólo la
/// continuación, sin el prompt). El que la implementa (main) posee el modelo.
pub const GenerateFn = *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, prompt: []const u8, max_tokens: usize) anyerror![]u8;

/// Emite un fragmento generado (un token) durante el streaming.
pub const EmitFn = *const fn (emit_ctx: *anyopaque, piece: []const u8) anyerror!void;
/// Callback de inferencia en streaming: llama a `emit` por cada token generado.
pub const GenerateStreamFn = *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, prompt: []const u8, max_tokens: usize, emit_ctx: *anyopaque, emit: EmitFn) anyerror!void;

/// Callback de embedding: recibe texto y devuelve un slice de f32 (embedding).
/// El caller es dueño del slice (debe liberarlo con allocator.free).
pub const EmbedFn = *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, text: []const u8) anyerror![]f32;

pub const ServeOptions = struct {
    host: []const u8 = "127.0.0.1",
    port: u16 = 11434,
    model: []const u8 = "Fluency-1.5b-v1",
    /// Si `generate` está presente, los endpoints de chat/generación producen
    /// salida REAL del modelo; si no, devuelven un texto por defecto (compat).
    engine_ctx: ?*anyopaque = null,
    generate: ?GenerateFn = null,
    /// Si está presente y la petición pide `stream:true`, se emite SSE/NDJSON
    /// incremental token a token.
    generate_stream: ?GenerateStreamFn = null,
    /// Si está presente, los endpoints de embeddings producen vectores reales.
    embed: ?EmbedFn = null,
    embed_ctx: ?*anyopaque = null,
    /// Metadatos del modelo para Ollama /api/show (opcionales; se rellenan en main
    /// cuando se carga un GGUF real).
    model_family: []const u8 = "unknown",
    model_params: []const u8 = "unknown",
    model_quant: []const u8 = "unknown",
    model_ctx: usize = 4096,
};

pub fn serveLocalhost(allocator: std.mem.Allocator, io: std.Io, options: ServeOptions) !void {
    var address = try net.IpAddress.parse(options.host, options.port);
    var tcp_server = try address.listen(io, .{ .reuse_address = true });
    defer tcp_server.deinit(io);

    while (true) {
        const stream = try tcp_server.accept(io);
        serveConnection(allocator, io, stream, options) catch {};
    }
}

fn serveConnection(allocator: std.mem.Allocator, io: std.Io, stream: net.Stream, options: ServeOptions) !void {
    defer {
        var copy = stream;
        copy.close(io);
    }

    var send_buffer: [8192]u8 = undefined;
    var recv_buffer: [8192]u8 = undefined;
    var connection_reader = stream.reader(io, &recv_buffer);
    var connection_writer = stream.writer(io, &send_buffer);
    var server: http.Server = .init(&connection_reader.interface, &connection_writer.interface);

    while (true) {
        var request = server.receiveHead() catch return;
        try serveRequest(allocator, &request, options);
        if (!request.head.keep_alive) return;
    }
}

fn serveRequest(allocator: std.mem.Allocator, request: *http.Server.Request, options: ServeOptions) !void {
    const target = stripQuery(request.head.target);

    // Leer el cuerpo de la petición (prompt / mensajes) si lo hay.
    var body_buf: [16384]u8 = undefined;
    var req_body: []const u8 = "";
    var req_body_owned: ?[]u8 = null;
    defer if (req_body_owned) |b| allocator.free(b);
    if (request.head.method.requestHasBody()) {
        const reader = if (request.head.expect != null)
            (request.readerExpectContinue(&body_buf) catch request.readerExpectNone(&body_buf))
        else
            request.readerExpectNone(&body_buf);
        if (reader.allocRemaining(allocator, std.Io.Limit.limited(1 << 20))) |b| {
            req_body_owned = b;
            req_body = b;
        } else |_| {}
    }

    // Ruta de streaming (SSE para OpenAI, NDJSON para Ollama) si se pide y hay motor.
    const is_gen = std.mem.eql(u8, target, "/v1/chat/completions") or std.mem.eql(u8, target, "/api/chat") or
        std.mem.eql(u8, target, "/api/generate") or std.mem.eql(u8, target, "/v1/completions");
    if (is_gen) {
        if (req_body.len > 0 and !isBodyJsonLike(req_body)) {
            const err_body = try errorJsonResponse(allocator, "invalid JSON body");
            defer allocator.free(err_body);
            try request.respond(err_body, .{
                .status = .bad_request,
                .extra_headers = &.{
                    .{ .name = "Content-Type", .value = "application/json" },
                    .{ .name = "Access-Control-Allow-Origin", .value = "*" },
                },
            });
            return;
        }
        if (!try checkModelField(allocator, req_body, options.model)) {
            const err_body = try errorJsonResponse(allocator, "model not found");
            defer allocator.free(err_body);
            try request.respond(err_body, .{
                .status = .bad_request,
                .extra_headers = &.{
                    .{ .name = "Content-Type", .value = "application/json" },
                    .{ .name = "Access-Control-Allow-Origin", .value = "*" },
                },
            });
            return;
        }
        if (options.generate_stream != null and options.engine_ctx != null and (extractBoolField(req_body, "stream") orelse false)) {
            try serveStreaming(allocator, request, target, req_body, options);
            return;
        }
    }

    const body = responseForTarget(allocator, target, req_body, options) catch |err| {
        const msg = if (err == error.OutOfMemory) "out of memory" else "internal error";
        const err_body = errorJsonResponse(allocator, msg) catch return;
        defer allocator.free(err_body);
        try request.respond(err_body, .{
            .status = .internal_server_error,
            .extra_headers = &.{
                .{ .name = "Content-Type", .value = "application/json" },
                .{ .name = "Access-Control-Allow-Origin", .value = "*" },
            },
        });
        return;
    };
    defer allocator.free(body);

    const status: http.Status = if (std.mem.eql(u8, body, not_found_body)) .not_found else .ok;
    try request.respond(body, .{
        .status = status,
        .extra_headers = &.{
            .{ .name = "Content-Type", .value = "application/json" },
            .{ .name = "Access-Control-Allow-Origin", .value = "*" },
            .{ .name = "Access-Control-Allow-Headers", .value = "authorization,content-type" },
            .{ .name = "Access-Control-Allow-Methods", .value = "GET,POST,OPTIONS" },
        },
    });
}

const not_found_body = "{\"error\":\"not found\"}";
const model_not_found_body = "{\"error\":\"model not found\"}";

fn errorJsonResponse(allocator: std.mem.Allocator, message: []const u8) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"error\":");
    try writeJsonString(&out, allocator, message);
    try out.append(allocator, '}');
    return out.toOwnedSlice(allocator);
}

/// Rechaza body que claramente no es JSON (no empieza con { / [ / ").
fn isBodyJsonLike(body: []const u8) bool {
    if (body.len == 0) return true;
    var i: usize = 0;
    while (i < body.len and (body[i] == ' ' or body[i] == '\t' or body[i] == '\n' or body[i] == '\r')) i += 1;
    if (i >= body.len) return true;
    const c = body[i];
    return c == '{' or c == '[' or c == '"';
}

fn checkModelField(allocator: std.mem.Allocator, req_body: []const u8, expected: []const u8) !bool {
    const key_pos = findKey(req_body, "model", false);
    if (key_pos == null) return true; // ausente → OK
    const requested = try extractStringAlloc(allocator, req_body, "model", false);
    defer if (requested) |r| allocator.free(r);
    if (requested) |r| {
        if (!std.mem.eql(u8, r, expected)) return false;
        return true;
    }
    // key presente pero valor no-string/malformado
    return false;
}

fn responseForTarget(allocator: std.mem.Allocator, target: []const u8, req_body: []const u8, options: ServeOptions) ![]u8 {
    const models = [_]ModelInfo{.{ .id = options.model }};
    const default_text = "Fluency motor de IA local listo.";

    if (std.mem.eql(u8, target, "/") or
        std.mem.eql(u8, target, "/health") or
        std.mem.eql(u8, target, "/api/health"))
    {
        var out = std.ArrayList(u8).empty;
        errdefer out.deinit(allocator);
        try out.appendSlice(allocator, "{\"ok\":true,\"name\":\"fluency\",\"version\":\"0.7.0\",\"model\":");
        try writeJsonString(&out, allocator, options.model);
        try out.append(allocator, '}');
        return out.toOwnedSlice(allocator);
    }
    if (std.mem.eql(u8, target, "/api/version")) {
        return allocator.dupe(u8, "{\"version\":\"0.7.0\"}");
    }
    if (std.mem.eql(u8, target, "/v1/models")) return modelsResponse(allocator, models[0..]);
    if (std.mem.eql(u8, target, "/api/tags")) return ollamaTagsResponse(allocator, models[0..]);
    if (std.mem.eql(u8, target, "/api/show")) {
        const requested = try extractStringAlloc(allocator, req_body, "name", false);
        defer if (requested) |r| allocator.free(r);
        if (requested) |r| {
            if (!std.mem.eql(u8, r, options.model)) {
                return allocator.dupe(u8, "{\"error\":\"model not found\"}");
            }
        }
        return ollamaShowResponse(allocator, options);
    }

    // Endpoints de generación: si hay motor cableado, producen salida real.
    if (std.mem.eql(u8, target, "/v1/chat/completions") or std.mem.eql(u8, target, "/api/chat")) {
        if (req_body.len > 0 and !isBodyJsonLike(req_body)) {
            return errorJsonResponse(allocator, "invalid JSON body");
        }
        if (!try checkModelField(allocator, req_body, options.model)) {
            return errorJsonResponse(allocator, "model not found");
        }
        const chat = std.mem.eql(u8, target, "/v1/chat/completions");
        const max_tokens = extractIntField(req_body, "max_tokens") orelse extractIntField(req_body, "num_predict") orelse 128;
        const prompt = try extractStringAlloc(allocator, req_body, "content", true);
        defer if (prompt) |p| allocator.free(p);
        if (prompt == null) {
            return errorJsonResponse(allocator, "missing prompt content");
        }
        const content = try runEngine(allocator, options, prompt.?, max_tokens, default_text);
        defer if (content.ptr != default_text.ptr) allocator.free(content);
        if (chat) {
            return chatCompletionResponse(allocator, options.model, content, prompt.?.len, content.len);
        }
        return ollamaChatResponse(allocator, options.model, content);
    }
    if (std.mem.eql(u8, target, "/api/generate") or std.mem.eql(u8, target, "/v1/completions")) {
        if (req_body.len > 0 and !isBodyJsonLike(req_body)) {
            return errorJsonResponse(allocator, "invalid JSON body");
        }
        if (!try checkModelField(allocator, req_body, options.model)) {
            return errorJsonResponse(allocator, "model not found");
        }
        const max_tokens = extractIntField(req_body, "max_tokens") orelse extractIntField(req_body, "num_predict") orelse 128;
        const prompt = try extractStringAlloc(allocator, req_body, "prompt", false);
        defer if (prompt) |p| allocator.free(p);
        if (prompt == null) {
            return errorJsonResponse(allocator, "missing prompt");
        }
        const content = try runEngine(allocator, options, prompt.?, max_tokens, default_text);
        defer if (content.ptr != default_text.ptr) allocator.free(content);
        return ollamaGenerateResponse(allocator, options.model, content);
    }
    if (std.mem.eql(u8, target, "/v1/embeddings")) {
        const input_text = try extractStringAlloc(allocator, req_body, "input", false);
        defer if (input_text) |t| allocator.free(t);
        const text = input_text orelse "";
        if (options.embed != null and options.embed_ctx != null and text.len > 0) {
            const emb = try options.embed.?(options.embed_ctx.?, allocator, text);
            defer allocator.free(emb);
            return embeddingResponse(allocator, emb);
        }
        const emb = [_]f32{ 0.125, 0.25, 0.5, 1.0 };
        return embeddingResponse(allocator, emb[0..]);
    }
    if (std.mem.eql(u8, target, "/api/embed") or std.mem.eql(u8, target, "/api/embeddings")) {
        // Ollama: el prompt a embedder puede venir en "prompt" (legacy) o en el
        // campo específico del body; usamos "input" si existe, sino "prompt".
        const input_text = try extractStringAlloc(allocator, req_body, "input", false);
        defer if (input_text) |t| allocator.free(t);
        const prompt_text = try extractStringAlloc(allocator, req_body, "prompt", false);
        defer if (prompt_text) |t| allocator.free(t);
        const text = if (input_text != null and input_text.?.len > 0) input_text.? else (prompt_text orelse "");
        if (options.embed != null and options.embed_ctx != null and text.len > 0) {
            const emb = try options.embed.?(options.embed_ctx.?, allocator, text);
            defer allocator.free(emb);
            return ollamaEmbedResponse(allocator, emb);
        }
        const emb = [_]f32{ 0.125, 0.25, 0.5, 1.0 };
        return ollamaEmbedResponse(allocator, emb[0..]);
    }
    return allocator.dupe(u8, not_found_body);
}

fn stripQuery(target: []const u8) []const u8 {
    if (std.mem.indexOfScalar(u8, target, '?')) |idx| return target[0..idx];
    return target;
}

// Llama al motor de inferencia si está cableado; si no, devuelve el texto por
// defecto (mismo puntero que `default_text`, para que el caller no lo libere).
fn runEngine(allocator: std.mem.Allocator, options: ServeOptions, prompt: []const u8, max_tokens: usize, default_text: []const u8) ![]u8 {
    if (options.generate) |gen| {
        if (options.engine_ctx) |ctx| {
            if (gen(ctx, allocator, prompt, max_tokens)) |txt| {
                return txt;
            } else |_| {
                return allocator.dupe(u8, "[error de generación]");
            }
        }
    }
    return @constCast(default_text);
}

// Busca la clave `"key"` en el cuerpo JSON (primera o última ocurrencia).
fn findKey(body: []const u8, key: []const u8, last: bool) ?usize {
    var needle_buf: [128]u8 = undefined;
    if (key.len + 2 > needle_buf.len) return null;
    needle_buf[0] = '"';
    @memcpy(needle_buf[1..][0..key.len], key);
    needle_buf[1 + key.len] = '"';
    const needle = needle_buf[0 .. key.len + 2];
    return if (last) std.mem.lastIndexOf(u8, body, needle) else std.mem.indexOf(u8, body, needle);
}

// Extrae y desescapa el valor string de `key` en el JSON. Devuelve memoria propia.
fn extractStringAlloc(allocator: std.mem.Allocator, body: []const u8, key: []const u8, last: bool) !?[]u8 {
    const start = findKey(body, key, last) orelse return null;
    var i = start + key.len + 2;
    while (i < body.len and body[i] != ':') i += 1;
    if (i >= body.len) return null;
    i += 1;
    while (i < body.len and (body[i] == ' ' or body[i] == '\t' or body[i] == '\n' or body[i] == '\r')) i += 1;
    if (i >= body.len or body[i] != '"') return null;
    i += 1;

    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    while (i < body.len) {
        const c = body[i];
        if (c == '"') break;
        if (c == '\\' and i + 1 < body.len) {
            i += 1;
            switch (body[i]) {
                'n' => try out.append(allocator, '\n'),
                't' => try out.append(allocator, '\t'),
                'r' => try out.append(allocator, '\r'),
                'b' => try out.append(allocator, 0x08),
                'f' => try out.append(allocator, 0x0c),
                '"' => try out.append(allocator, '"'),
                '\\' => try out.append(allocator, '\\'),
                '/' => try out.append(allocator, '/'),
                'u' => {
                    if (i + 4 < body.len) {
                        const cp = std.fmt.parseInt(u21, body[i + 1 ..][0..4], 16) catch 0xFFFD;
                        var ubuf: [4]u8 = undefined;
                        const n = std.unicode.utf8Encode(cp, &ubuf) catch 0;
                        try out.appendSlice(allocator, ubuf[0..n]);
                        i += 4;
                    }
                },
                else => try out.append(allocator, body[i]),
            }
            i += 1;
        } else {
            try out.append(allocator, c);
            i += 1;
        }
    }
    return try out.toOwnedSlice(allocator);
}

// Extrae un entero del valor de `key` (p. ej. "num_predict":128).
fn extractIntField(body: []const u8, key: []const u8) ?usize {
    const start = findKey(body, key, false) orelse return null;
    var i = start + key.len + 2;
    while (i < body.len and body[i] != ':') i += 1;
    if (i >= body.len) return null;
    i += 1;
    while (i < body.len and (body[i] == ' ' or body[i] == '\t')) i += 1;
    var j = i;
    while (j < body.len and body[j] >= '0' and body[j] <= '9') j += 1;
    if (j == i) return null;
    return std.fmt.parseInt(usize, body[i..j], 10) catch null;
}

// Extrae un booleano del valor de `key` (p. ej. "stream":true).
fn extractBoolField(body: []const u8, key: []const u8) ?bool {
    const start = findKey(body, key, false) orelse return null;
    var i = start + key.len + 2;
    while (i < body.len and body[i] != ':') i += 1;
    if (i >= body.len) return null;
    i += 1;
    while (i < body.len and (body[i] == ' ' or body[i] == '\t')) i += 1;
    if (std.mem.startsWith(u8, body[i..], "true")) return true;
    if (std.mem.startsWith(u8, body[i..], "false")) return false;
    return null;
}

fn writeJsonStringWriter(w: *std.Io.Writer, text: []const u8) !void {
    try w.writeByte('"');
    for (text) |c| {
        switch (c) {
            '"' => try w.writeAll("\\\""),
            '\\' => try w.writeAll("\\\\"),
            '\n' => try w.writeAll("\\n"),
            '\r' => try w.writeAll("\\r"),
            '\t' => try w.writeAll("\\t"),
            else => try w.writeByte(c),
        }
    }
    try w.writeByte('"');
}

// Contexto de emisión por token durante el streaming.
// Escribe un chunk de streaming con el shape correcto por protocolo/endpoint:
// - OpenAI (SSE):            data: {"choices":[{"delta":{"content":"<piece>"}}]}\n\n
// - Ollama /api/chat:        {"model":...,"message":{"role":"assistant","content":"<piece>"},"done":false}\n
// - Ollama /api/generate:    {"model":...,"response":"<piece>","done":false}\n
// El distingo chat↔generate en Ollama es obligatorio: los clientes de chat leen
// `message.content`, no `response`.
fn writeStreamChunk(w: *std.Io.Writer, ollama: bool, chat: bool, model: []const u8, piece: []const u8) !void {
    if (ollama) {
        try w.writeAll("{\"model\":");
        try writeJsonStringWriter(w, model);
        if (chat) {
            try w.writeAll(",\"message\":{\"role\":\"assistant\",\"content\":");
            try writeJsonStringWriter(w, piece);
            try w.writeAll("},\"done\":false}\n");
        } else {
            try w.writeAll(",\"response\":");
            try writeJsonStringWriter(w, piece);
            try w.writeAll(",\"done\":false}\n");
        }
    } else {
        try w.writeAll("data: {\"choices\":[{\"delta\":{\"content\":");
        try writeJsonStringWriter(w, piece);
        try w.writeAll("}}]}\n\n");
    }
}

const StreamCtx = struct {
    bw: *http.BodyWriter,
    ollama: bool,
    chat: bool,
    model: []const u8,

    fn emit(ctx: *anyopaque, piece: []const u8) anyerror!void {
        const self: *StreamCtx = @alignCast(@ptrCast(ctx));
        try writeStreamChunk(&self.bw.writer, self.ollama, self.chat, self.model, piece);
        try self.bw.flush();
    }
};

// Genera token a token y lo emite incrementalmente (SSE / NDJSON).
fn serveStreaming(allocator: std.mem.Allocator, request: *http.Server.Request, target: []const u8, req_body: []const u8, options: ServeOptions) !void {
    const ollama = std.mem.startsWith(u8, target, "/api/");
    const chat = std.mem.eql(u8, target, "/v1/chat/completions") or std.mem.eql(u8, target, "/api/chat");
    const max_tokens = extractIntField(req_body, "max_tokens") orelse extractIntField(req_body, "num_predict") orelse 128;
    const prompt = if (chat)
        try extractStringAlloc(allocator, req_body, "content", true)
    else
        try extractStringAlloc(allocator, req_body, "prompt", false);
    defer if (prompt) |p| allocator.free(p);

    var sbuf: [8192]u8 = undefined;
    const content_type = if (ollama) "application/x-ndjson" else "text/event-stream";
    var bw = request.respondStreaming(&sbuf, .{ .respond_options = .{ .extra_headers = &.{
        .{ .name = "Content-Type", .value = content_type },
        .{ .name = "Access-Control-Allow-Origin", .value = "*" },
        .{ .name = "Cache-Control", .value = "no-cache" },
    } } }) catch return;

    if (prompt == null) {
        const w = &bw.writer;
        if (ollama) {
            try w.writeAll("{\"error\":\"missing prompt\"}\n");
        } else {
            try w.writeAll("data: {\"error\":\"missing prompt\"}\n\n");
        }
        bw.end() catch {};
        return;
    }

    var sctx = StreamCtx{ .bw = &bw, .ollama = ollama, .chat = chat, .model = options.model };
    options.generate_stream.?(options.engine_ctx.?, allocator, prompt.?, max_tokens, &sctx, StreamCtx.emit) catch {};

    const w = &bw.writer;
    if (ollama) {
        // Mensaje final: mismo shape por endpoint que los chunks (message vs response).
        try w.writeAll("{\"model\":");
        try writeJsonStringWriter(w, options.model);
        if (chat) {
            try w.writeAll(",\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true}\n");
        } else {
            try w.writeAll(",\"response\":\"\",\"done\":true}\n");
        }
    } else {
        try w.writeAll("data: [DONE]\n\n");
    }
    bw.end() catch {};
}

test "zserve renders model and chat responses" {
    const allocator = std.testing.allocator;
    const models = [_]ModelInfo{.{ .id = "Fluency-1.5b-v1" }};
    const models_json = try modelsResponse(allocator, models[0..]);
    defer allocator.free(models_json);
    try std.testing.expect(std.mem.indexOf(u8, models_json, "Fluency-1.5b-v1") != null);

    const chat_json = try chatCompletionResponse(allocator, "Fluency-1.5b-v1", "ok", 1, 1);
    defer allocator.free(chat_json);
    try std.testing.expect(std.mem.indexOf(u8, chat_json, "\"finish_reason\":\"stop\"") != null);

    const tags_json = try responseForTarget(allocator, "/api/tags", "", .{ .model = "Fluency-1.5b-v1" });
    defer allocator.free(tags_json);
    try std.testing.expect(std.mem.indexOf(u8, tags_json, "\"models\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, tags_json, "Fluency-1.5b-v1") != null);

    // Sin motor cableado, /api/generate cae al texto por defecto (no genera).
    const gen_json = try responseForTarget(allocator, "/api/generate", "{\"prompt\":\"hola\"}", .{ .model = "Fluency-1.5b-v1" });
    defer allocator.free(gen_json);
    try std.testing.expect(std.mem.indexOf(u8, gen_json, "\"response\"") != null);

    // Extracción de campos JSON.
    const p = try extractStringAlloc(allocator, "{\"model\":\"x\",\"prompt\":\"a\\nb\"}", "prompt", false);
    defer if (p) |pp| allocator.free(pp);
    try std.testing.expectEqualStrings("a\nb", p.?);
    try std.testing.expectEqual(@as(?usize, 64), extractIntField("{\"num_predict\":64}", "num_predict"));
    try std.testing.expectEqual(@as(?bool, true), extractBoolField("{\"stream\":true}", "stream"));
    try std.testing.expectEqual(@as(?bool, false), extractBoolField("{\"stream\": false}", "stream"));
    try std.testing.expectEqual(@as(?bool, null), extractBoolField("{\"x\":1}", "stream"));
}

test "zserve discovery endpoints reflect configured model name" {
    const allocator = std.testing.allocator;
    const custom = "Fluency-3b-v2";
    const opts = ServeOptions{ .model = custom };

    const models_json = try responseForTarget(allocator, "/v1/models", "", opts);
    defer allocator.free(models_json);
    try std.testing.expect(std.mem.indexOf(u8, models_json, custom) != null);
    try std.testing.expect(std.mem.indexOf(u8, models_json, "\"object\":\"list\"") != null);

    const tags_json = try responseForTarget(allocator, "/api/tags", "", opts);
    defer allocator.free(tags_json);
    try std.testing.expect(std.mem.indexOf(u8, tags_json, custom) != null);
    try std.testing.expect(std.mem.indexOf(u8, tags_json, "\"models\"") != null);
}

test "zserve model field validation accepts matching or absent model" {
    const allocator = std.testing.allocator;
    const model = "Fluency-1.5b-v1";
    const opts = ServeOptions{ .model = model };

    // Petición sin campo model → aceptada
    const ok1 = try responseForTarget(allocator, "/api/chat", "{\"content\":\"hi\"}", opts);
    defer allocator.free(ok1);
    try std.testing.expect(std.mem.indexOf(u8, ok1, "model not found") == null);

    // Petición con model coincidente → aceptada
    const ok2 = try responseForTarget(allocator, "/api/chat", "{\"model\":\"Fluency-1.5b-v1\",\"content\":\"hi\"}", opts);
    defer allocator.free(ok2);
    try std.testing.expect(std.mem.indexOf(u8, ok2, "model not found") == null);

    // Petición con model distinto → rechazada con gracia
    const err = try responseForTarget(allocator, "/api/chat", "{\"model\":\"otro\",\"content\":\"hi\"}", opts);
    defer allocator.free(err);
    try std.testing.expect(std.mem.indexOf(u8, err, "model not found") != null);
}

test "zserve /api/version returns Ollama format" {
    const allocator = std.testing.allocator;
    const json = try responseForTarget(allocator, "/api/version", "", .{ .model = "Fluency-1.5b-v1" });
    defer allocator.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"version\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "0.7.0") != null);
    // No debe contener campos de health check genérico
    try std.testing.expect(std.mem.indexOf(u8, json, "\"ok\"") == null);
}

test "zserve /api/show returns metadata for matching model" {
    const allocator = std.testing.allocator;
    const opts = ServeOptions{
        .model = "Fluency-1.5b-v1",
        .model_family = "qwen2",
        .model_params = "1.5B",
        .model_quant = "Q4_K_M",
        .model_ctx = 32768,
    };

    const json = try responseForTarget(allocator, "/api/show", "{\"name\":\"Fluency-1.5b-v1\"}", opts);
    defer allocator.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"details\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "qwen2") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "1.5B") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "Q4_K_M") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "32768") != null);
}

test "zserve /api/show rejects wrong model name" {
    const allocator = std.testing.allocator;
    const opts = ServeOptions{ .model = "Fluency-1.5b-v1" };

    const json = try responseForTarget(allocator, "/api/show", "{\"name\":\"otro\"}", opts);
    defer allocator.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "model not found") != null);
}

test "zserve streaming chunk shape: Ollama chat usa message.content, generate usa response, OpenAI usa delta" {
    const allocator = std.testing.allocator;

    // Ollama /api/chat → debe llevar message.content y NO el campo response.
    {
        var aw = std.Io.Writer.Allocating.init(allocator);
        defer aw.deinit();
        try writeStreamChunk(&aw.writer, true, true, "Fluency-1.5b-v1", " hola");
        const s = aw.written();
        try std.testing.expect(std.mem.indexOf(u8, s, "\"message\":{\"role\":\"assistant\",\"content\":\" hola\"}") != null);
        try std.testing.expect(std.mem.indexOf(u8, s, "\"response\"") == null);
        try std.testing.expect(std.mem.indexOf(u8, s, "\"done\":false") != null);
    }
    // Ollama /api/generate → debe llevar response y NO message.
    {
        var aw = std.Io.Writer.Allocating.init(allocator);
        defer aw.deinit();
        try writeStreamChunk(&aw.writer, true, false, "Fluency-1.5b-v1", " hola");
        const s = aw.written();
        try std.testing.expect(std.mem.indexOf(u8, s, "\"response\":\" hola\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, s, "\"message\"") == null);
    }
    // OpenAI → SSE con delta.content.
    {
        var aw = std.Io.Writer.Allocating.init(allocator);
        defer aw.deinit();
        try writeStreamChunk(&aw.writer, false, true, "Fluency-1.5b-v1", " hola");
        const s = aw.written();
        try std.testing.expect(std.mem.indexOf(u8, s, "data: {\"choices\":[{\"delta\":{\"content\":\" hola\"}}]}") != null);
    }
}

test "zserve rejects non-JSON body with error" {
    const allocator = std.testing.allocator;
    const opts = ServeOptions{ .model = "Fluency-1.5b-v1" };

    const err = try responseForTarget(allocator, "/v1/chat/completions", "not json at all", opts);
    defer allocator.free(err);
    try std.testing.expect(std.mem.indexOf(u8, err, "invalid JSON body") != null);
}

test "zserve rejects malformed model field (non-string)" {
    const allocator = std.testing.allocator;
    const opts = ServeOptions{ .model = "Fluency-1.5b-v1" };

    const err = try responseForTarget(allocator, "/api/chat", "{\"model\":123,\"content\":\"hi\"}", opts);
    defer allocator.free(err);
    try std.testing.expect(std.mem.indexOf(u8, err, "model not found") != null);
}

test "zserve rejects missing content in chat completion" {
    const allocator = std.testing.allocator;
    const opts = ServeOptions{ .model = "Fluency-1.5b-v1" };

    const err = try responseForTarget(allocator, "/v1/chat/completions", "{\"model\":\"Fluency-1.5b-v1\"}", opts);
    defer allocator.free(err);
    try std.testing.expect(std.mem.indexOf(u8, err, "missing prompt content") != null);
}

test "zserve rejects empty body in chat completion" {
    const allocator = std.testing.allocator;
    const opts = ServeOptions{ .model = "Fluency-1.5b-v1" };

    const err = try responseForTarget(allocator, "/v1/chat/completions", "", opts);
    defer allocator.free(err);
    try std.testing.expect(std.mem.indexOf(u8, err, "missing prompt content") != null);
}

test "zserve rejects missing prompt in generate" {
    const allocator = std.testing.allocator;
    const opts = ServeOptions{ .model = "Fluency-1.5b-v1" };

    const err = try responseForTarget(allocator, "/api/generate", "{\"model\":\"Fluency-1.5b-v1\"}", opts);
    defer allocator.free(err);
    try std.testing.expect(std.mem.indexOf(u8, err, "missing prompt") != null);
}

test "zserve accepts valid JSON with extra fields" {
    const allocator = std.testing.allocator;
    const opts = ServeOptions{ .model = "Fluency-1.5b-v1" };

    const ok = try responseForTarget(allocator, "/api/generate", "{\"model\":\"Fluency-1.5b-v1\",\"prompt\":\"hi\",\"extra\":true}", opts);
    defer allocator.free(ok);
    try std.testing.expect(std.mem.indexOf(u8, ok, "missing prompt") == null);
    try std.testing.expect(std.mem.indexOf(u8, ok, "\"response\"") != null);
}

// Mock de embedding que devuelve un vector conocido para tests.
const MockEmbedCtx = struct {
    values: []const f32,
    fn embed(ctx: *anyopaque, alloc: std.mem.Allocator, text: []const u8) anyerror![]f32 {
        _ = text;
        const self: *MockEmbedCtx = @alignCast(@ptrCast(ctx));
        return alloc.dupe(f32, self.values);
    }
};

test "zserve /v1/embeddings usa embed_fn real cuando está cableado" {
    const allocator = std.testing.allocator;
    var mock = MockEmbedCtx{ .values = &.{ 0.1, 0.2, 0.3 } };
    const opts = ServeOptions{
        .model = "Fluency-1.5b-v1",
        .embed = MockEmbedCtx.embed,
        .embed_ctx = &mock,
    };

    const json = try responseForTarget(allocator, "/v1/embeddings", "{\"input\":\"hello\"}", opts);
    defer allocator.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "0.1") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "0.2") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "0.3") != null);
    // Debe tener shape OpenAI
    try std.testing.expect(std.mem.indexOf(u8, json, "\"object\":\"list\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"object\":\"embedding\"") != null);
}

test "zserve /api/embeddings usa embed_fn real cuando está cableado" {
    const allocator = std.testing.allocator;
    var mock = MockEmbedCtx{ .values = &.{ 0.4, 0.5, 0.6 } };
    const opts = ServeOptions{
        .model = "Fluency-1.5b-v1",
        .embed = MockEmbedCtx.embed,
        .embed_ctx = &mock,
    };

    const json = try responseForTarget(allocator, "/api/embeddings", "{\"input\":\"world\"}", opts);
    defer allocator.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "0.4") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "0.5") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "0.6") != null);
    // Shape Ollama
    try std.testing.expect(std.mem.indexOf(u8, json, "\"embeddings\"") != null);
}

test "zserve embeddings sin embed_fn cae a stub" {
    const allocator = std.testing.allocator;
    const opts = ServeOptions{ .model = "Fluency-1.5b-v1" };

    const json = try responseForTarget(allocator, "/v1/embeddings", "{\"input\":\"hello\"}", opts);
    defer allocator.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "0.125") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "0.25") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "0.5") != null);
    // Nota: {d} para f32 1.0 puede emitir "1" (sin .0) en Zig 0.16; buscamos "1".
    try std.testing.expect(std.mem.indexOf(u8, json, "1") != null);
}
