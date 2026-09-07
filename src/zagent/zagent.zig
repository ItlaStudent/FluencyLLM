// zagent — runtime agéntico + registry + bucle ReAct + secretos + auditoría
const std = @import("std");
const util = @import("util");
pub const tools = @import("tools/tools.zig");
pub const memory = @import("memory/zmemory.zig");
pub const trajectory = @import("trajectory.zig");

// ---------------------------------------------------------------------------
// SecretStore — secretos cifrados con ChaCha20-Poly1305 (Art. IV.3)
// ---------------------------------------------------------------------------
pub const SecretStore = struct {
    const Secret = struct {
        name: []const u8,
        ciphertext: []u8,
        nonce: [12]u8,
        tag: [16]u8,
        ad: []const u8,
    };

    secrets: std.StringHashMap(Secret),
    master_key: [32]u8,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, master_key: [32]u8) SecretStore {
        return .{
            .secrets = std.StringHashMap(Secret).init(allocator),
            .master_key = master_key,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *SecretStore) void {
        var it = self.secrets.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.value_ptr.name);
            self.allocator.free(entry.value_ptr.ciphertext);
            self.allocator.free(entry.value_ptr.ad);
        }
        self.secrets.deinit();
    }

    /// Guarda un secreto cifrado. Genera nonce criptográficamente seguro vía io.random.
    pub fn put(self: *SecretStore, io: std.Io, name: []const u8, plaintext: []const u8) !void {
        var nonce: [12]u8 = undefined;
        io.random(&nonce);
        try self.putWithNonce(name, plaintext, nonce);
    }

    /// Variante determinística (útil en tests o cuando io no está disponible).
    pub fn putWithNonce(self: *SecretStore, name: []const u8, plaintext: []const u8, nonce: [12]u8) !void {
        const ciphertext = try self.allocator.alloc(u8, plaintext.len);
        errdefer self.allocator.free(ciphertext);
        var tag: [16]u8 = undefined;
        const ad = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(ad);
        util.crypto.seal(self.master_key, nonce, plaintext, ad, ciphertext, &tag);

        const name_copy = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_copy);

        try self.secrets.put(name_copy, .{
            .name = name_copy,
            .ciphertext = ciphertext,
            .nonce = nonce,
            .tag = tag,
            .ad = ad,
        });
    }

    /// Recupera y descifra un secreto. El plaintext devuelto debe ser liberado por el caller.
    pub fn get(self: *SecretStore, name: []const u8, allocator: std.mem.Allocator) ![]u8 {
        const secret = self.secrets.get(name) orelse return error.SecretNotFound;
        const plaintext = try allocator.alloc(u8, secret.ciphertext.len);
        errdefer allocator.free(plaintext);
        try util.crypto.open(self.master_key, secret.nonce, secret.ciphertext, secret.tag, secret.ad, plaintext);
        return plaintext;
    }

    pub fn has(self: SecretStore, name: []const u8) bool {
        return self.secrets.contains(name);
    }
};

// ---------------------------------------------------------------------------
// AuditLog — registro append-only de eventos de seguridad
// ---------------------------------------------------------------------------
pub const AuditLog = struct {
    pub const Category = enum {
        tool_dispatch,
        tool_denied,
        secret_access,
        secret_denied,
        budget_exceeded,
    };

    pub const Event = struct {
        seq: u64,
        category: Category,
        description: []const u8,
        is_error: bool,
    };

    events: std.ArrayList(Event),
    next_seq: u64,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) AuditLog {
        return .{
            .events = std.ArrayList(Event).empty,
            .next_seq = 1,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *AuditLog) void {
        for (self.events.items) |ev| {
            self.allocator.free(ev.description);
        }
        self.events.deinit(self.allocator);
    }

    pub fn log(self: *AuditLog, category: Category, description: []const u8, is_error: bool) !void {
        const desc = try self.allocator.dupe(u8, description);
        errdefer self.allocator.free(desc);
        try self.events.append(self.allocator, .{
            .seq = self.next_seq,
            .category = category,
            .description = desc,
            .is_error = is_error,
        });
        self.next_seq += 1;
    }

    pub fn serializeTo(self: AuditLog, out: *std.ArrayList(u8), allocator: std.mem.Allocator) !void {
        for (self.events.items) |ev| {
            try out.appendSlice(allocator, "[");
            var buf: [32]u8 = undefined;
            const n = try std.fmt.bufPrint(&buf, "{d}", .{ev.seq});
            try out.appendSlice(allocator, n);
            try out.appendSlice(allocator, "] ");
            try out.appendSlice(allocator, @tagName(ev.category));
            try out.appendSlice(allocator, " | ");
            try out.appendSlice(allocator, ev.description);
            if (ev.is_error) {
                try out.appendSlice(allocator, " | ERROR");
            }
            try out.appendSlice(allocator, "\n");
        }
    }
};

// ---------------------------------------------------------------------------
// ToolCall — representación normalizada de una llamada a herramienta
// ---------------------------------------------------------------------------
pub const ToolCall = struct {
    id: []const u8,
    name: []const u8,
    arguments_json: []const u8,

    pub fn deinit(self: ToolCall, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        allocator.free(self.name);
        allocator.free(self.arguments_json);
    }
};

pub fn deinitToolCalls(allocator: std.mem.Allocator, calls: []ToolCall) void {
    for (calls) |call| {
        call.deinit(allocator);
    }
    allocator.free(calls);
}

// ---------------------------------------------------------------------------
// Parser de tool-calls (Hermes XML y OpenAI JSON)
// ---------------------------------------------------------------------------

fn extractName(value: util.json.Value) ![]const u8 {
    if (value.get("function")) |func| {
        if (func.get("name")) |n| {
            if (n == .string) return n.string;
        }
    }
    if (value.get("name")) |n| {
        if (n == .string) return n.string;
    }
    return error.MissingName;
}

fn extractId(value: util.json.Value) []const u8 {
    if (value.get("id")) |v| {
        if (v == .string) return v.string;
    }
    return "call_default";
}

fn extractArguments(alloc: std.mem.Allocator, value: util.json.Value) ![]const u8 {
    if (value.get("function")) |func| {
        if (func.get("arguments")) |args| {
            return try argumentsToJson(alloc, args);
        }
    }
    if (value.get("arguments")) |args| {
        return try argumentsToJson(alloc, args);
    }
    return error.MissingArguments;
}

fn argumentsToJson(alloc: std.mem.Allocator, args: util.json.Value) ![]const u8 {
    switch (args) {
        .string => |s| return alloc.dupe(u8, s),
        else => return try util.json.stringify(alloc, args),
    }
}

/// Parsea tool_calls desde texto libre.
/// Soporta:
///   - Hermes XML: `<tool_call>{"name":"...","arguments":{...}}</tool_call>`
///   - OpenAI JSON: `{"tool_calls":[...]}` con `function.name` y `function.arguments`
///   - Objeto directo: `{"name":"...","arguments":{...}}`
/// Devuelve slice vacío si no encuentra ninguna.
pub fn parseToolCalls(alloc: std.mem.Allocator, text: []const u8) ![]ToolCall {
    var results = std.ArrayList(ToolCall).empty;
    errdefer {
        for (results.items) |call| {
            call.deinit(alloc);
        }
        results.deinit(alloc);
    }

    // 1. Formato Hermes XML
    var search_start: usize = 0;
    while (true) {
        const start_tag = std.mem.indexOfPos(u8, text, search_start, "<tool_call>");
        if (start_tag == null) break;
        const start = start_tag.? + "<tool_call>".len;
        const end_tag = std.mem.indexOfPos(u8, text, start, "</tool_call>");
        if (end_tag == null) break;
        const json_text = std.mem.trim(u8, text[start..end_tag.?], " \t\n\r");

        var parsed = util.json.parse(alloc, json_text) catch break;
        defer parsed.deinit();

        const name = extractName(parsed.value) catch break;
        const args = extractArguments(alloc, parsed.value) catch break;
        const id = extractId(parsed.value);
        const id_copy = try alloc.dupe(u8, id);
        errdefer alloc.free(id_copy);
        const name_copy = try alloc.dupe(u8, name);
        errdefer alloc.free(name_copy);

        try results.append(alloc, .{
            .id = id_copy,
            .name = name_copy,
            .arguments_json = args,
        });
        search_start = end_tag.? + "</tool_call>".len;
    }

    if (results.items.len > 0) {
        return results.toOwnedSlice(alloc);
    }

    // 2. Formato OpenAI JSON puro
    var parsed = util.json.parse(alloc, text) catch return &[_]ToolCall{};
    defer parsed.deinit();

    const tool_calls = parsed.value.get("tool_calls");
    if (tool_calls == null) {
        // 3. Objeto directo con name + arguments
        if (parsed.value.get("name") != null and parsed.value.get("arguments") != null) {
            const name = extractName(parsed.value) catch return &[_]ToolCall{};
            const args = extractArguments(alloc, parsed.value) catch return &[_]ToolCall{};
            const id = extractId(parsed.value);
            const id_copy = try alloc.dupe(u8, id);
            errdefer alloc.free(id_copy);
            const name_copy = try alloc.dupe(u8, name);
            errdefer alloc.free(name_copy);
            try results.append(alloc, .{
                .id = id_copy,
                .name = name_copy,
                .arguments_json = args,
            });
            return results.toOwnedSlice(alloc);
        }
        return &[_]ToolCall{};
    }

    const arr = tool_calls.?;
    if (arr != .array) return &[_]ToolCall{};

    for (arr.array) |item| {
        const name = extractName(item) catch continue;
        const args = extractArguments(alloc, item) catch continue;
        const id = extractId(item);
        const id_copy = try alloc.dupe(u8, id);
        errdefer alloc.free(id_copy);
        const name_copy = try alloc.dupe(u8, name);
        errdefer alloc.free(name_copy);
        try results.append(alloc, .{
            .id = id_copy,
            .name = name_copy,
            .arguments_json = args,
        });
    }

    if (results.items.len == 0) return &[_]ToolCall{};
    return results.toOwnedSlice(alloc);
}

// ---------------------------------------------------------------------------
// ToolResult / Tool / Registry
// ---------------------------------------------------------------------------
pub const ToolResult = struct {
    content: []const u8,
    is_error: bool,
};

pub const Tool = struct {
    pub const VTable = struct {
        name: *const fn (ptr: *anyopaque) []const u8,
        description: *const fn (ptr: *anyopaque) []const u8,
        parameters_json: *const fn (ptr: *anyopaque) []const u8,
        execute: *const fn (ptr: *anyopaque, io: std.Io, args_json: []const u8, allocator: std.mem.Allocator) anyerror!ToolResult,
    };
    ptr: *anyopaque,
    vtable: *const VTable,
};

pub const Registry = struct {
    tools: std.StringHashMap(Tool),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Registry {
        return .{
            .tools = std.StringHashMap(Tool).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Registry) void {
        self.tools.deinit();
    }

    pub fn register(self: *Registry, tool: Tool) !void {
        try self.tools.put(tool.vtable.name(tool.ptr), tool);
    }

    pub fn dispatch(self: Registry, io: std.Io, name: []const u8, args_json: []const u8) !ToolResult {
        const tool = self.tools.get(name) orelse return ToolResult{
            .content = try self.allocator.dupe(u8, "Error: Herramienta no encontrada"),
            .is_error = true,
        };
        return tool.vtable.execute(tool.ptr, io, args_json, self.allocator);
    }
};

// ---------------------------------------------------------------------------
// Traza ReAct estructurada
// ---------------------------------------------------------------------------
pub const Trace = struct {
    steps: std.ArrayList(Step),

    pub const Step = union(enum) {
        thought: []const u8,
        action: ToolCall,
        observation: []const u8,
        answer: []const u8,
        budget_exceeded: void,
    };

    pub fn init(allocator: std.mem.Allocator) Trace {
        _ = allocator;
        return .{ .steps = std.ArrayList(Step).empty };
    }

    pub fn deinit(self: *Trace, allocator: std.mem.Allocator) void {
        for (self.steps.items) |step| {
            switch (step) {
                .thought => |t| allocator.free(t),
                .action => |a| a.deinit(allocator),
                .observation => |o| allocator.free(o),
                .answer => |a| allocator.free(a),
                .budget_exceeded => {},
            }
        }
        self.steps.deinit(allocator);
    }

    pub fn addThought(self: *Trace, allocator: std.mem.Allocator, text: []const u8) !void {
        const copy = try allocator.dupe(u8, text);
        errdefer allocator.free(copy);
        try self.steps.append(allocator, .{ .thought = copy });
    }

    pub fn addAction(self: *Trace, allocator: std.mem.Allocator, call: ToolCall) !void {
        const id_copy = try allocator.dupe(u8, call.id);
        errdefer allocator.free(id_copy);
        const name_copy = try allocator.dupe(u8, call.name);
        errdefer allocator.free(name_copy);
        const args_copy = try allocator.dupe(u8, call.arguments_json);
        errdefer allocator.free(args_copy);
        try self.steps.append(allocator, .{ .action = .{
            .id = id_copy,
            .name = name_copy,
            .arguments_json = args_copy,
        } });
    }

    pub fn addObservation(self: *Trace, allocator: std.mem.Allocator, result: ToolResult) !void {
        const copy = try allocator.dupe(u8, result.content);
        errdefer allocator.free(copy);
        try self.steps.append(allocator, .{ .observation = copy });
    }

    pub fn addAnswer(self: *Trace, allocator: std.mem.Allocator, text: []const u8) !void {
        const copy = try allocator.dupe(u8, text);
        errdefer allocator.free(copy);
        try self.steps.append(allocator, .{ .answer = copy });
    }

    pub fn addBudgetExceeded(self: *Trace, allocator: std.mem.Allocator) !void {
        try self.steps.append(allocator, .{ .budget_exceeded = {} });
    }

    pub fn serializeTo(self: Trace, out: *std.ArrayList(u8), allocator: std.mem.Allocator) !void {
        for (self.steps.items) |step| {
            switch (step) {
                .thought => |t| {
                    try out.appendSlice(allocator, "Pensamiento: ");
                    try out.appendSlice(allocator, t);
                    try out.appendSlice(allocator, "\n");
                },
                .action => |a| {
                    try out.appendSlice(allocator, "Acción: ");
                    try out.appendSlice(allocator, a.name);
                    try out.appendSlice(allocator, " con ");
                    try out.appendSlice(allocator, a.arguments_json);
                    try out.appendSlice(allocator, "\n");
                },
                .observation => |o| {
                    try out.appendSlice(allocator, "Observación: ");
                    try out.appendSlice(allocator, o);
                    try out.appendSlice(allocator, "\n");
                },
                .answer => |a| {
                    try out.appendSlice(allocator, "Respuesta: ");
                    try out.appendSlice(allocator, a);
                    try out.appendSlice(allocator, "\n");
                },
                .budget_exceeded => {
                    try out.appendSlice(allocator, "Error: Presupuesto de iteraciones agotado\n");
                },
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Agente con bucle ReAct + presupuesto + secretos + auditoría
// ---------------------------------------------------------------------------
pub const ApprovalFn = *const fn (ctx: ?*anyopaque, io: std.Io, tool_name: []const u8, args_json: []const u8) anyerror!bool;

pub const Agent = struct {
    model: []const u8,
    registry: *Registry,
    allocator: std.mem.Allocator,
    io: std.Io,
    memory: memory.MemoryStore,
    secret_store: ?*SecretStore,
    audit: ?*AuditLog,
    should_stop: ?*bool,
    trajectory: ?trajectory.Trajectory,
    generate_fn: ?*const fn (ctx: ?*anyopaque, prompt: []const u8, allocator: std.mem.Allocator) anyerror![]u8,
    generate_ctx: ?*anyopaque,
    /// Callback opcional de streaming: recibe cada token/pieza según se genera.
    stream_fn: ?*const fn (ctx: ?*anyopaque, piece: []const u8) anyerror!void = null,
    stream_ctx: ?*anyopaque = null,
    /// Generador en streaming: emite tokens a medida que decodifica.
    /// Si está presente, el agente lo prefiere sobre generate_fn para iteraciones > 0.
    /// El callback `emit` recibe cada pieza; el generador NO acumula buffer.
    generate_stream_fn: ?*const fn (
        ctx: ?*anyopaque,
        prompt: []const u8,
        allocator: std.mem.Allocator,
        max_tokens: usize,
        emit_ctx: ?*anyopaque,
        emit: *const fn (emit_ctx: ?*anyopaque, piece: []const u8) anyerror!void,
    ) anyerror!void = null,
    generate_stream_ctx: ?*anyopaque = null,
    /// Callback opcional de aprobación: se invoca antes de tool-calls destructivas.
    /// Si devuelve false, la tool-call se rechaza con error.
    approval_fn: ?ApprovalFn = null,
    approval_ctx: ?*anyopaque = null,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, model: []const u8, registry: *Registry) Agent {
        return .{
            .model = model,
            .registry = registry,
            .allocator = allocator,
            .io = io,
            .memory = memory.MemoryStore.init(allocator),
            .secret_store = null,
            .audit = null,
            .should_stop = null,
            .trajectory = null,
            .generate_fn = null,
            .generate_ctx = null,
            .stream_fn = null,
            .stream_ctx = null,
            .generate_stream_fn = null,
            .generate_stream_ctx = null,
        };
    }

    pub fn deinit(self: *Agent) void {
        self.memory.deinit();
        if (self.trajectory) |*t| t.deinit(self.allocator);
    }

    pub fn run(self: *Agent, prompt: []const u8, max_iters: usize) !std.ArrayList(u8) {
        var trace = Trace.init(self.allocator);
        defer trace.deinit(self.allocator);

        if (self.trajectory) |*t| t.deinit(self.allocator);
        self.trajectory = trajectory.Trajectory.init(prompt, self.model);

        try trace.addThought(self.allocator, prompt);
        try self.trajectory.?.addThought(self.allocator, prompt);

        // Contexto acumulado para iteraciones ReAct (empieza con el prompt).
        var context = std.ArrayList(u8).empty;
        defer context.deinit(self.allocator);
        try context.appendSlice(self.allocator, prompt);

        // Sólo se parsean tool-calls del texto NUEVO de cada iteración (el prompt
        // en iter 0, la salida del generador en iter>0). Sin esto, re-parsear todo
        // el contexto re-ejecutaría las mismas tool-calls en cada iteración.
        var parsed_from: usize = 0;
        var finished = false;

        var iter: usize = 0;
        while (iter < max_iters) : (iter += 1) {
            // Interrupción externa (Ctrl-C, botón stop, etc.)
            if (self.should_stop) |flag| {
                if (flag.*) {
                    try trace.addAnswer(self.allocator, "Interrumpido por el usuario.");
                    try self.trajectory.?.addInterrupted(self.allocator);
                    break;
                }
            }

            // En iteraciones > 0, si hay generador conectado, pedirle texto
            // al modelo a partir del contexto acumulado.
            var generated_text: []const u8 = "";
            var generated_owned: bool = false;
            if (iter > 0) {
                if (self.generate_stream_fn) |gen_stream| {
                    // Generador en streaming: emite tokens a medida que decodifica.
                    // Acumulamos el texto localmente mientras reenviamos al stream_fn.
                    var acc = std.ArrayList(u8).empty;
                    const Wrapper = struct {
                        sf: ?*const fn (ctx: ?*anyopaque, piece: []const u8) anyerror!void,
                        sctx: ?*anyopaque,
                        acc: *std.ArrayList(u8),
                        alloc: std.mem.Allocator,
                        pub fn emit(ctx: ?*anyopaque, piece: []const u8) !void {
                            const w = @as(*@This(), @ptrCast(@alignCast(ctx.?)));
                            try w.acc.appendSlice(w.alloc, piece);
                            if (w.sf) |sf| try sf(w.sctx, piece);
                        }
                    };
                    var wrapper = Wrapper{
                        .sf = self.stream_fn,
                        .sctx = self.stream_ctx,
                        .acc = &acc,
                        .alloc = self.allocator,
                    };
                    try gen_stream(self.generate_stream_ctx, context.items, self.allocator, 64, &wrapper, &Wrapper.emit);
                    generated_text = try acc.toOwnedSlice(self.allocator);
                    generated_owned = true;
                } else if (self.generate_fn) |gen| {
                    const generated = try gen(self.generate_ctx, context.items, self.allocator);
                    generated_text = generated;
                    generated_owned = true;
                    // Streaming token-a-token (re-emision por palabras del buffer completo).
                    if (self.stream_fn) |sf| {
                        var start: usize = 0;
                        for (generated, 0..) |b, i| {
                            if (b == ' ' or b == '\n') {
                                if (start <= i) {
                                    try sf(self.stream_ctx, generated[start .. i + 1]);
                                }
                                start = i + 1;
                            }
                        }
                        if (start < generated.len) {
                            try sf(self.stream_ctx, generated[start..]);
                        }
                    }
                }
                if (generated_text.len > 0) {
                    try context.appendSlice(self.allocator, "\n");
                    try context.appendSlice(self.allocator, generated_text);
                    try trace.addThought(self.allocator, generated_text);
                    try self.trajectory.?.addThought(self.allocator, generated_text);
                }
                if (generated_owned) {
                    self.allocator.free(generated_text);
                }
            }

            // Parsear sólo el texto añadido desde la última iteración.
            const fresh = context.items[parsed_from..];
            const calls = try parseToolCalls(self.allocator, fresh);
            defer deinitToolCalls(self.allocator, calls);

            if (calls.len == 0) {
                try trace.addAnswer(self.allocator, "No se requiere acción.");
                try self.trajectory.?.addAnswer(self.allocator, "No se requiere acción.");
                break;
            }

            for (calls) |call| {
                try trace.addAction(self.allocator, call);
                try self.trajectory.?.addAction(self.allocator, call.name, call.arguments_json);

                // Deny-by-default
                if (!self.registry.tools.contains(call.name)) {
                    if (self.audit) |audit| {
                        try audit.log(.tool_denied, call.name, true);
                    }
                    const err_res = ToolResult{
                        .content = try self.allocator.dupe(u8, "Error: Herramienta no registrada (deny-by-default)"),
                        .is_error = true,
                    };
                    try trace.addObservation(self.allocator, err_res);
                    try self.trajectory.?.addObservation(self.allocator, err_res.content, err_res.is_error);
                    self.allocator.free(err_res.content);

                    // Reflexión: incorporar al contexto para posible reintento
                    try context.appendSlice(self.allocator, "\n[Reflexión: herramienta ");
                    try context.appendSlice(self.allocator, call.name);
                    try context.appendSlice(self.allocator, " denegada por deny-by-default]");
                    continue;
                }

                // Aprobacion humana para tools destructivas (fs_write, bash)
                if (self.approval_fn) |approval| {
                    const needs_approval = std.mem.eql(u8, call.name, "fs_write") or
                        std.mem.eql(u8, call.name, "bash");
                    if (needs_approval) {
                        const approved = try approval(self.approval_ctx, self.io, call.name, call.arguments_json);
                        if (!approved) {
                            const err_res = ToolResult{
                                .content = try self.allocator.dupe(u8, "Error: tool-call rechazada por el usuario"),
                                .is_error = true,
                            };
                            try trace.addObservation(self.allocator, err_res);
                            try self.trajectory.?.addObservation(self.allocator, err_res.content, err_res.is_error);
                            try context.appendSlice(self.allocator, "\n[Reflexion: tool-call ");
                            try context.appendSlice(self.allocator, call.name);
                            try context.appendSlice(self.allocator, " rechazada por el usuario]");
                            self.allocator.free(err_res.content);
                            continue;
                        }
                    }
                }

                if (self.audit) |audit| {
                    try audit.log(.tool_dispatch, call.name, false);
                }
                const res = try self.registry.dispatch(self.io, call.name, call.arguments_json);
                try trace.addObservation(self.allocator, res);
                try self.trajectory.?.addObservation(self.allocator, res.content, res.is_error);

                // Alimentar observación al contexto para la siguiente iteración ReAct
                try context.appendSlice(self.allocator, "\n[Resultado de ");
                try context.appendSlice(self.allocator, call.name);
                try context.appendSlice(self.allocator, ": ");
                try context.appendSlice(self.allocator, res.content);
                try context.appendSlice(self.allocator, "]");
                if (res.is_error) {
                    try context.appendSlice(self.allocator, "\n[Reflexión: ocurrió un error, revisar estrategia]");
                }

                self.allocator.free(res.content);

                // La herramienta `finish` termina el bucle ReAct explícitamente.
                if (std.mem.eql(u8, call.name, "finish")) finished = true;
            }

            // Marcar todo lo consumido: la siguiente iteración sólo parsea texto nuevo.
            parsed_from = context.items.len;

            if (finished) {
                try trace.addAnswer(self.allocator, "Objetivo finalizado (finish).");
                try self.trajectory.?.addAnswer(self.allocator, "Objetivo finalizado (finish).");
                break;
            }
        } else {
            try trace.addBudgetExceeded(self.allocator);
            try self.trajectory.?.addBudgetExceeded(self.allocator);
            if (self.audit) |audit| {
                try audit.log(.budget_exceeded, "max_iters agotado", true);
            }
        }

        var history = std.ArrayList(u8).empty;
        errdefer history.deinit(self.allocator);
        try trace.serializeTo(&history, self.allocator);
        return history;
    }

    /// Acceso a secretos: si no existe o no hay store, loguea deny y devuelve error.
    pub fn getSecret(self: *Agent, name: []const u8, out_allocator: std.mem.Allocator) ![]u8 {
        const store = self.secret_store orelse {
            if (self.audit) |audit| {
                try audit.log(.secret_denied, name, true);
            }
            return error.SecretStoreNotConfigured;
        };
        if (!store.has(name)) {
            if (self.audit) |audit| {
                try audit.log(.secret_denied, name, true);
            }
            return error.SecretNotFound;
        }
        const plaintext = try store.get(name, out_allocator);
        if (self.audit) |audit| {
            try audit.log(.secret_access, name, false);
        }
        return plaintext;
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "Agent init and registry" {
    const alloc = std.testing.allocator;
    var reg = Registry.init(alloc);
    defer reg.deinit();

    var agent = Agent.init(alloc, undefined, "Fluency-100M", &reg);
    defer agent.deinit();
    try std.testing.expectEqualStrings("Fluency-100M", agent.model);
}

test "SecretStore round-trip ChaCha20-Poly1305" {
    const alloc = std.testing.allocator;
    const key = [_]u8{0xAB} ** 32;
    var store = SecretStore.init(alloc, key);
    defer store.deinit();

    const nonce = [_]u8{0xCD} ** 12;
    try store.putWithNonce("api_key", "shh-dont-tell", nonce);
    try std.testing.expect(store.has("api_key"));

    const pt = try store.get("api_key", alloc);
    defer alloc.free(pt);
    try std.testing.expectEqualStrings("shh-dont-tell", pt);
}

test "SecretStore get inexistente devuelve error" {
    const alloc = std.testing.allocator;
    const key = [_]u8{0x01} ** 32;
    var store = SecretStore.init(alloc, key);
    defer store.deinit();

    try std.testing.expectError(error.SecretNotFound, store.get("missing", alloc));
}

test "SecretStore tampered ciphertext falla" {
    const alloc = std.testing.allocator;
    const key = [_]u8{0x11} ** 32;
    var store = SecretStore.init(alloc, key);
    defer store.deinit();

    const nonce = [_]u8{0x22} ** 12;
    try store.putWithNonce("token", "secret", nonce);

    // Corromper ciphertext interno
    const secret = store.secrets.get("token").?;
    secret.ciphertext[0] ^= 0xFF;

    try std.testing.expectError(error.AuthenticationFailed, store.get("token", alloc));
}

test "AuditLog append y serialize" {
    const alloc = std.testing.allocator;
    var audit = AuditLog.init(alloc);
    defer audit.deinit();

    try audit.log(.tool_dispatch, "dummy", false);
    try audit.log(.tool_denied, "evil", true);

    try std.testing.expectEqual(@as(usize, 2), audit.events.items.len);
    try std.testing.expectEqual(@as(u64, 1), audit.events.items[0].seq);
    try std.testing.expectEqual(@as(u64, 2), audit.events.items[1].seq);

    var out = std.ArrayList(u8).empty;
    defer out.deinit(alloc);
    try audit.serializeTo(&out, alloc);
    try std.testing.expect(std.mem.indexOf(u8, out.items, "tool_dispatch") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.items, "tool_denied") != null);
}

test "Agent deny-by-default loguea tool_denied" {
    const alloc = std.testing.allocator;
    var reg = Registry.init(alloc);
    defer reg.deinit();

    var audit = AuditLog.init(alloc);
    defer audit.deinit();

    var agent = Agent.init(alloc, undefined, "test", &reg);
    defer agent.deinit();
    agent.audit = &audit;

    const prompt = "{\"tool_calls\":[{\"id\":\"c1\",\"function\":{\"name\":\"nonexistent\",\"arguments\":\"{}\"}}]}";
    var history = try agent.run(prompt, 5);
    defer history.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 1), audit.events.items.len);
    try std.testing.expect(audit.events.items[0].category == .tool_denied);
    try std.testing.expect(audit.events.items[0].is_error);
}

test "Agent getSecret sin store loguea deny" {
    const alloc = std.testing.allocator;
    var reg = Registry.init(alloc);
    defer reg.deinit();

    var audit = AuditLog.init(alloc);
    defer audit.deinit();

    var agent = Agent.init(alloc, undefined, "test", &reg);
    defer agent.deinit();
    agent.audit = &audit;

    const res = agent.getSecret("api_key", alloc);
    try std.testing.expectError(error.SecretStoreNotConfigured, res);
    try std.testing.expectEqual(@as(usize, 1), audit.events.items.len);
    try std.testing.expect(audit.events.items[0].category == .secret_denied);
}

test "parseToolCalls OpenAI format" {
    const alloc = std.testing.allocator;
    const text =
        \\{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"fs_read","arguments":"{\"path\":\"main.zig\"}"}}]}
    ;
    const calls = try parseToolCalls(alloc, text);
    defer deinitToolCalls(alloc, calls);

    try std.testing.expectEqual(@as(usize, 1), calls.len);
    try std.testing.expectEqualStrings("call_1", calls[0].id);
    try std.testing.expectEqualStrings("fs_read", calls[0].name);
    try std.testing.expectEqualStrings("{\"path\":\"main.zig\"}", calls[0].arguments_json);
}

test "parseToolCalls Hermes XML" {
    const alloc = std.testing.allocator;
    const text = "Razonando... <tool_call>{\"name\":\"bash\",\"arguments\":{\"command\":\"zig build test\"}}</tool_call>";
    const calls = try parseToolCalls(alloc, text);
    defer deinitToolCalls(alloc, calls);

    try std.testing.expectEqual(@as(usize, 1), calls.len);
    try std.testing.expectEqualStrings("bash", calls[0].name);
    try std.testing.expectEqualStrings("{\"command\":\"zig build test\"}", calls[0].arguments_json);
}

test "parseToolCalls direct object" {
    const alloc = std.testing.allocator;
    const text = "{\"name\":\"finish\",\"arguments\":{}}";
    const calls = try parseToolCalls(alloc, text);
    defer deinitToolCalls(alloc, calls);

    try std.testing.expectEqual(@as(usize, 1), calls.len);
    try std.testing.expectEqualStrings("finish", calls[0].name);
}

test "parseToolCalls empty text" {
    const alloc = std.testing.allocator;
    const calls = try parseToolCalls(alloc, "just plain text");
    defer deinitToolCalls(alloc, calls);
    try std.testing.expectEqual(@as(usize, 0), calls.len);
}

const DummyTool = struct {
    call_count: usize = 0,

    pub fn name(_: *anyopaque) []const u8 {
        return "dummy";
    }
    pub fn description(_: *anyopaque) []const u8 {
        return "dummy tool";
    }
    pub fn parameters_json(_: *anyopaque) []const u8 {
        return "{}";
    }
    pub fn execute(ptr: *anyopaque, _: std.Io, _: []const u8, allocator: std.mem.Allocator) anyerror!ToolResult {
        const self: *DummyTool = @alignCast(@ptrCast(ptr));
        self.call_count += 1;
        return ToolResult{ .content = try allocator.dupe(u8, "ok"), .is_error = false };
    }
    pub fn tool(self: *DummyTool) Tool {
        return .{
            .ptr = self,
            .vtable = &Tool.VTable{
                .name = name,
                .description = description,
                .parameters_json = parameters_json,
                .execute = execute,
            },
        };
    }
};

test "Agent despacha tool_calls JSON y traza ReAct" {
    const alloc = std.testing.allocator;
    var reg = Registry.init(alloc);
    defer reg.deinit();

    var dummy = DummyTool{};
    try reg.register(dummy.tool());

    var agent = Agent.init(alloc, undefined, "test", &reg);
    defer agent.deinit();
    const prompt = "{\"tool_calls\":[{\"id\":\"c1\",\"function\":{\"name\":\"dummy\",\"arguments\":\"{}\"}}]}";
    var history = try agent.run(prompt, 5);
    defer history.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 1), dummy.call_count);
    const hist_str = history.items;
    try std.testing.expect(std.mem.indexOf(u8, hist_str, "Pensamiento:") != null);
    try std.testing.expect(std.mem.indexOf(u8, hist_str, "Acción: dummy") != null);
    try std.testing.expect(std.mem.indexOf(u8, hist_str, "Observación: ok") != null);
}

test "Agent respeta presupuesto max_iters" {
    const alloc = std.testing.allocator;
    var reg = Registry.init(alloc);
    defer reg.deinit();

    var dummy = DummyTool{};
    try reg.register(dummy.tool());

    var agent = Agent.init(alloc, undefined, "test", &reg);
    defer agent.deinit();
    const prompt = "{\"tool_calls\":[{\"id\":\"c1\",\"function\":{\"name\":\"dummy\",\"arguments\":\"{}\"}}]}";
    var history = try agent.run(prompt, 0);
    defer history.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 0), dummy.call_count);
    const hist_str = history.items;
    try std.testing.expect(std.mem.indexOf(u8, hist_str, "Presupuesto de iteraciones agotado") != null);
}

test "Agent interrupción detiene ejecución" {
    const alloc = std.testing.allocator;
    var reg = Registry.init(alloc);
    defer reg.deinit();

    var stop = true;
    var agent = Agent.init(alloc, undefined, "test", &reg);
    defer agent.deinit();
    agent.should_stop = &stop;

    var history = try agent.run("Hola", 5);
    defer history.deinit(alloc);

    const hist_str = history.items;
    try std.testing.expect(std.mem.indexOf(u8, hist_str, "Interrumpido") != null);
}

test "Agent sin tool_calls responde directamente" {
    const alloc = std.testing.allocator;
    var reg = Registry.init(alloc);
    defer reg.deinit();

    var agent = Agent.init(alloc, undefined, "test", &reg);
    defer agent.deinit();
    var history = try agent.run("Hola", 5);
    defer history.deinit(alloc);

    const hist_str = history.items;
    try std.testing.expect(std.mem.indexOf(u8, hist_str, "No se requiere acción") != null);
}

test "Agent con generate_fn ejecuta tool en iteración 2" {
    const alloc = std.testing.allocator;
    var reg = Registry.init(alloc);
    defer reg.deinit();

    var dummy = DummyTool{};
    try reg.register(dummy.tool());

    const MockGen = struct {
        var call_count: usize = 0;
        fn generate(ctx: ?*anyopaque, prompt: []const u8, gpa: std.mem.Allocator) ![]u8 {
            _ = ctx;
            _ = prompt;
            call_count += 1;
            return try gpa.dupe(u8, "<tool_call>{\"name\":\"dummy\",\"arguments\":\"{}\"}</tool_call>");
        }
    };
    MockGen.call_count = 0;

    var agent = Agent.init(alloc, undefined, "test", &reg);
    defer agent.deinit();
    agent.generate_fn = &MockGen.generate;

    // Prompt con un tool-call para iteración 0; el generador provee otro para iteración 1
    const prompt = "{\"tool_calls\":[{\"id\":\"c1\",\"function\":{\"name\":\"dummy\",\"arguments\":\"{}\"}}]}";
    var history = try agent.run(prompt, 2);
    defer history.deinit(alloc);

    // El generador fue llamado una vez (iteración 1)
    try std.testing.expectEqual(@as(usize, 1), MockGen.call_count);
    // La tool dummy fue ejecutada dos veces (iteración 0 + iteración 1)
    try std.testing.expectEqual(@as(usize, 2), dummy.call_count);
}

const LogTool = struct {
    call_count: usize = 0,
    pub fn name(_: *anyopaque) []const u8 {
        return "log";
    }
    pub fn description(_: *anyopaque) []const u8 {
        return "log tool";
    }
    pub fn parameters_json(_: *anyopaque) []const u8 {
        return "{}";
    }
    pub fn execute(ptr: *anyopaque, _: std.Io, _: []const u8, allocator: std.mem.Allocator) anyerror!ToolResult {
        const self: *LogTool = @alignCast(@ptrCast(ptr));
        self.call_count += 1;
        return ToolResult{ .content = try allocator.dupe(u8, "logged"), .is_error = false };
    }
    pub fn tool(self: *LogTool) Tool {
        return .{ .ptr = self, .vtable = &Tool.VTable{ .name = name, .description = description, .parameters_json = parameters_json, .execute = execute } };
    }
};

const FinishDummy = struct {
    call_count: usize = 0,
    pub fn name(_: *anyopaque) []const u8 {
        return "finish";
    }
    pub fn description(_: *anyopaque) []const u8 {
        return "finish";
    }
    pub fn parameters_json(_: *anyopaque) []const u8 {
        return "{}";
    }
    pub fn execute(ptr: *anyopaque, _: std.Io, _: []const u8, allocator: std.mem.Allocator) anyerror!ToolResult {
        const self: *FinishDummy = @alignCast(@ptrCast(ptr));
        self.call_count += 1;
        return ToolResult{ .content = try allocator.dupe(u8, "fin"), .is_error = false };
    }
    pub fn tool(self: *FinishDummy) Tool {
        return .{ .ptr = self, .vtable = &Tool.VTable{ .name = name, .description = description, .parameters_json = parameters_json, .execute = execute } };
    }
};

test "Agent finish corta el bucle ReAct aunque haya generador y presupuesto" {
    const alloc = std.testing.allocator;
    var reg = Registry.init(alloc);
    defer reg.deinit();
    var fin = FinishDummy{};
    try reg.register(fin.tool());

    // Generador que SIEMPRE emitiría otra tool-call: si finish no cortara, se
    // re-llamaría en cada iteración hasta agotar el presupuesto.
    const MockGen = struct {
        var calls: usize = 0;
        fn generate(_: ?*anyopaque, _: []const u8, gpa: std.mem.Allocator) ![]u8 {
            calls += 1;
            return try gpa.dupe(u8, "<tool_call>{\"name\":\"finish\",\"arguments\":{}}</tool_call>");
        }
    };
    MockGen.calls = 0;

    var agent = Agent.init(alloc, undefined, "test", &reg);
    defer agent.deinit();
    agent.generate_fn = &MockGen.generate;

    const prompt = "<tool_call>{\"name\":\"finish\",\"arguments\":{}}</tool_call>";
    var history = try agent.run(prompt, 5);
    defer history.deinit(alloc);

    // finish se ejecuta UNA sola vez y el bucle corta en iter 0 (el generador
    // nunca llega a ejecutarse porque ya terminamos).
    try std.testing.expectEqual(@as(usize, 1), fin.call_count);
    try std.testing.expectEqual(@as(usize, 0), MockGen.calls);
    try std.testing.expect(std.mem.indexOf(u8, history.items, "finalizado") != null);
}

// --- Tests de tools (viven aquí porque tools.zig no puede ser root de test) ---

test "fs_read security check" {
    try std.testing.expectError(error.SecurityViolation, tools.validatePath("../fuera.txt"));
    try std.testing.expectError(error.SecurityViolation, tools.validatePath("C:\\Windows\\system32"));
    try tools.validatePath("src/main.zig");
}

test "bash allowlist rejects forbidden commands" {
    const alloc = std.testing.allocator;
    var bash = tools.BashTool{};
    const t = bash.tool();

    const res = try t.vtable.execute(t.ptr, undefined, "{\"command\":\"rm -rf /\"}", alloc);
    defer alloc.free(res.content);
    try std.testing.expect(res.is_error);
    try std.testing.expect(std.mem.indexOf(u8, res.content, "no permitido") != null);
}

test "bash allowlist rejects dangerous metacharacters" {
    const alloc = std.testing.allocator;
    var bash = tools.BashTool{};
    const t = bash.tool();

    const cases = &[_][]const u8{
        "{\"command\":\"zig build test | rm\"}",
        "{\"command\":\"zig build test > out.txt\"}",
        "{\"command\":\"zig build test && del\"}",
    };
    for (cases) |args| {
        const res = try t.vtable.execute(t.ptr, undefined, args, alloc);
        defer alloc.free(res.content);
        try std.testing.expect(res.is_error);
    }
}

test "http_fetch allowlist rejects forbidden host" {
    const alloc = std.testing.allocator;
    var fetch = tools.HttpFetchTool{};
    const t = fetch.tool();

    const res = try t.vtable.execute(t.ptr, undefined, "{\"url\":\"https://evil.com/data\"}", alloc);
    defer alloc.free(res.content);
    try std.testing.expect(res.is_error);
    try std.testing.expect(std.mem.indexOf(u8, res.content, "host no permitido") != null);
}

test "delegate rejects by default" {
    const alloc = std.testing.allocator;
    var del = tools.DelegateTool{};
    const t = del.tool();

    const res = try t.vtable.execute(t.ptr, undefined, "{\"goal\":\"hackear\"}", alloc);
    defer alloc.free(res.content);
    try std.testing.expect(res.is_error);
    try std.testing.expect(std.mem.indexOf(u8, res.content, "denegada") != null);
}

test "memory_write then memory_read roundtrip" {
    const alloc = std.testing.allocator;
    var store = memory.MemoryStore.init(alloc);
    defer store.deinit();

    var write_tool = tools.MemoryWriteTool{ .store = &store };
    var read_tool = tools.MemoryReadTool{ .store = &store };

    const w1 = try write_tool.tool().vtable.execute(write_tool.tool().ptr, undefined, "{\"text\":\"Zig es genial\"}", alloc);
    defer alloc.free(w1.content);
    try std.testing.expect(!w1.is_error);

    const r1 = try read_tool.tool().vtable.execute(read_tool.tool().ptr, undefined, "{\"query\":\"Zig\",\"top_k\":1}", alloc);
    defer alloc.free(r1.content);
    try std.testing.expect(std.mem.indexOf(u8, r1.content, "Zig es genial") != null);
}

test "memory_read without store returns error" {
    const alloc = std.testing.allocator;
    var read_tool = tools.MemoryReadTool{};
    const res = try read_tool.tool().vtable.execute(read_tool.tool().ptr, undefined, "{\"query\":\"x\"}", alloc);
    defer alloc.free(res.content);
    try std.testing.expect(res.is_error);
    try std.testing.expect(std.mem.indexOf(u8, res.content, "no inicializada") != null);
}

test "Agent streaming token-a-token emite pieces al callback" {
    const alloc = std.testing.allocator;
    var reg = Registry.init(alloc);
    defer reg.deinit();

    var dummy = DummyTool{};
    try reg.register(dummy.tool());

    const MockGen = struct {
        fn generate(_: ?*anyopaque, _: []const u8, gpa: std.mem.Allocator) ![]u8 {
            return try gpa.dupe(u8, "hello world");
        }
    };

    const StreamCapture = struct {
        var pieces: std.ArrayList(u8) = .empty;
        fn emit(_: ?*anyopaque, piece: []const u8) !void {
            try pieces.appendSlice(std.testing.allocator, piece);
        }
    };
    StreamCapture.pieces = .empty;
    defer StreamCapture.pieces.deinit(std.testing.allocator);

    var agent = Agent.init(alloc, undefined, "test", &reg);
    defer agent.deinit();
    agent.generate_fn = &MockGen.generate;
    agent.stream_fn = &StreamCapture.emit;

    // Prompt con tool-call no-finish para iter 0 -> generate en iter 1
    const prompt = "<tool_call>{\"name\":\"dummy\",\"arguments\":\"{}\"}</tool_call>";
    var history = try agent.run(prompt, 2);
    defer history.deinit(alloc);

    // El generador devolvió "hello world"; se emitió por palabras:
    // "hello " + "world"
    const captured = StreamCapture.pieces.items;
    try std.testing.expect(std.mem.indexOf(u8, captured, "hello") != null);
    try std.testing.expect(std.mem.indexOf(u8, captured, "world") != null);
    try std.testing.expectEqualStrings("hello world", captured);
}

// Generador mock en streaming: emite tokens/piezas a medida que "decodifica",
// en lugar de devolver un buffer completo.
fn mockGenerateStream(
    ctx: ?*anyopaque,
    prompt: []const u8,
    allocator: std.mem.Allocator,
    max_tokens: usize,
    emit_ctx: ?*anyopaque,
    emit: *const fn (emit_ctx: ?*anyopaque, piece: []const u8) anyerror!void,
) !void {
    _ = ctx;
    _ = prompt;
    _ = allocator;
    _ = max_tokens;
    try emit(emit_ctx, "hello");
    try emit(emit_ctx, " ");
    try emit(emit_ctx, "world");
}

// Gate-check E5 (F4): agente completa tarea multi-herramienta con
// tool-calls JSON válidos, presupuesto acotado y sin panic/leaks.
test "Gate-check E5: agente multi-herramienta con tool-calls validos y presupuesto" {
    const alloc = std.testing.allocator;
    var reg = Registry.init(alloc);
    defer reg.deinit();

    var dummy = DummyTool{};
    try reg.register(dummy.tool());
    var log_tool = LogTool{};
    try reg.register(log_tool.tool());
    var fin = FinishDummy{};
    try reg.register(fin.tool());

    const MockGen = struct {
        var call_count: usize = 0;
        fn generate(_: ?*anyopaque, _: []const u8, gpa: std.mem.Allocator) ![]u8 {
            call_count += 1;
            if (call_count == 1) {
                return try gpa.dupe(u8, "{\"name\":\"log\",\"arguments\":\"{}\"}");
            } else {
                return try gpa.dupe(u8, "{\"name\":\"finish\",\"arguments\":{}}");
            }
        }
    };
    MockGen.call_count = 0;

    var agent = Agent.init(alloc, undefined, "test", &reg);
    defer agent.deinit();
    agent.generate_fn = &MockGen.generate;

    // Iter 0: prompt con dummy; Iter 1: generate -> log; Iter 2: generate -> finish
    const prompt = "{\"name\":\"dummy\",\"arguments\":\"{}\"}";
    var history = try agent.run(prompt, 3);
    defer history.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 2), MockGen.call_count);
    try std.testing.expectEqual(@as(usize, 1), dummy.call_count);
    try std.testing.expectEqual(@as(usize, 1), log_tool.call_count);
    try std.testing.expectEqual(@as(usize, 1), fin.call_count);
    try std.testing.expect(std.mem.indexOf(u8, history.items, "finalizado") != null);
}

test "Agent generate_stream_fn emite tokens durante la generacion" {
    const alloc = std.testing.allocator;
    var reg = Registry.init(alloc);
    defer reg.deinit();

    var dummy = DummyTool{};
    try reg.register(dummy.tool());

    const StreamCapture = struct {
        var pieces: std.ArrayList(u8) = .empty;
        fn emit(_: ?*anyopaque, piece: []const u8) !void {
            try pieces.appendSlice(std.testing.allocator, piece);
        }
    };
    StreamCapture.pieces = .empty;
    defer StreamCapture.pieces.deinit(std.testing.allocator);

    var agent = Agent.init(alloc, undefined, "test", &reg);
    defer agent.deinit();
    agent.generate_stream_fn = &mockGenerateStream;
    agent.stream_fn = &StreamCapture.emit;

    const prompt = "<tool_call>{\"name\":\"dummy\",\"arguments\":\"{}\"}</tool_call>";
    var history = try agent.run(prompt, 2);
    defer history.deinit(alloc);

    const captured = StreamCapture.pieces.items;
    try std.testing.expectEqualStrings("hello world", captured);
    const hist_str = history.items;
    try std.testing.expect(std.mem.indexOf(u8, hist_str, "hello world") != null);
    try std.testing.expect(std.mem.indexOf(u8, hist_str, "dummy") != null);
}

test "Agent generate_stream_fn despacha tool-call desde texto acumulado" {
    const alloc = std.testing.allocator;
    var reg = Registry.init(alloc);
    defer reg.deinit();

    var dummy = DummyTool{};
    try reg.register(dummy.tool());
    var fin = FinishDummy{};
    try reg.register(fin.tool());

    const StreamGen = struct {
        fn generate(
            ctx: ?*anyopaque,
            prompt: []const u8,
            allocator: std.mem.Allocator,
            max_tokens: usize,
            emit_ctx: ?*anyopaque,
            emit: *const fn (emit_ctx: ?*anyopaque, piece: []const u8) anyerror!void,
        ) !void {
            _ = ctx;
            _ = prompt;
            _ = allocator;
            _ = max_tokens;
            try emit(emit_ctx, "<tool_call>{\"name\":\"finish\",\"arguments\":{}}</tool_call>");
        }
    };

    var agent = Agent.init(alloc, undefined, "test", &reg);
    defer agent.deinit();
    agent.generate_stream_fn = &StreamGen.generate;

    // Iter 0: dummy; iter 1: generate_stream_fn emite finish
    const prompt = "<tool_call>{\"name\":\"dummy\",\"arguments\":\"{}\"}</tool_call>";
    var history = try agent.run(prompt, 2);
    defer history.deinit(alloc);

    // finish debe haberse despachado exactamente una vez (en iter 1)
    try std.testing.expectEqual(@as(usize, 1), fin.call_count);
    // La historia debe mencionar finish
    try std.testing.expect(std.mem.indexOf(u8, history.items, "finish") != null);
}
