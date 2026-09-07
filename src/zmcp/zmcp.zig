// zmcp - MCP cliente + servidor
const std = @import("std");
const zagent = @import("zagent");

pub const protocol_version = "2025-11-25";

pub const ToolInfo = struct {
    name: []const u8,
    description: []const u8,
    parameters_json: []const u8,
};

pub const ToolResult = struct {
    content: []const u8,
    is_error: bool,
};

pub const ExecuteFn = *const fn (ctx: *anyopaque, name: []const u8, args_json: []const u8, allocator: std.mem.Allocator) anyerror!ToolResult;

pub const SampleFn = *const fn (ctx: ?*anyopaque, messages_json: []const u8, allocator: std.mem.Allocator) anyerror![]u8;

pub const Server = struct {
    allocator: std.mem.Allocator,
    tools: []const ToolInfo,
    ctx: *anyopaque,
    execute_fn: ExecuteFn,
    sample_fn: ?SampleFn,
    sample_ctx: ?*anyopaque,
    server_name: []const u8 = "fluency-zsa",

    pub fn stdioLoop(self: *Server, io: std.Io, reader: std.Io.File.Reader, writer: std.Io.File.Writer) !void {
        _ = io;
        var r = reader;
        var w = writer;
        while (true) {
            var line_buf = std.ArrayList(u8).empty;
            defer line_buf.deinit(self.allocator);
            while (true) {
                var byte_buf: [1]u8 = undefined;
                var vecs: [1][]u8 = .{&byte_buf};
                const n = r.interface.readVec(&vecs) catch |err| switch (err) {
                    error.EndOfStream => break,
                    else => return err,
                };
                if (n == 0) break;
                if (byte_buf[0] == '\n') break;
                try line_buf.append(self.allocator, byte_buf[0]);
            }
            if (line_buf.items.len == 0) break;
            const response = try self.handle(line_buf.items);
            defer self.allocator.free(response);
            try w.interface.writeAll(response);
            try w.interface.writeAll("\n");
            try w.interface.flush();
        }
    }

    pub fn init(
        allocator: std.mem.Allocator,
        tools: []const ToolInfo,
        ctx: *anyopaque,
        execute_fn: ExecuteFn,
    ) Server {
        return .{
            .allocator = allocator,
            .tools = tools,
            .ctx = ctx,
            .execute_fn = execute_fn,
            .sample_fn = null,
            .sample_ctx = null,
        };
    }

    pub fn handle(self: *Server, input: []const u8) ![]u8 {
        var req = parseRequest(self.allocator, input) catch {
            return makeError(self.allocator, .none, .parse_error, "invalid json-rpc request");
        };
        defer req.deinit();

        const method = req.method();
        if (std.mem.eql(u8, method, "initialize")) {
            const body = try self.initializeResult();
            defer self.allocator.free(body);
            return makeResult(self.allocator, req.id(), body);
        }
        if (std.mem.eql(u8, method, "tools/list")) {
            const body = try self.toolsListResult();
            defer self.allocator.free(body);
            return makeResult(self.allocator, req.id(), body);
        }
        if (std.mem.eql(u8, method, "tools/call")) {
            const result = try self.callTool(req.params());
            defer self.allocator.free(result);
            return makeResult(self.allocator, req.id(), result);
        }
        if (std.mem.eql(u8, method, "sampling/createMessage")) {
            const result = try self.createMessage(req.params());
            defer self.allocator.free(result);
            return makeResult(self.allocator, req.id(), result);
        }
        if (std.mem.eql(u8, method, "ping")) {
            return makeResult(self.allocator, req.id(), "{}");
        }

        return makeError(self.allocator, req.id(), .method_not_found, "method not found");
    }

    fn initializeResult(self: *Server) ![]u8 {
        var out = std.ArrayList(u8).empty;
        errdefer out.deinit(self.allocator);
        try out.appendSlice(self.allocator, "{\"protocolVersion\":\"");
        try out.appendSlice(self.allocator, protocol_version);
        try out.appendSlice(self.allocator, "\",\"capabilities\":{\"tools\":{\"listChanged\":true},\"sampling\":{}},\"serverInfo\":{\"name\":");
        try writeJsonString(&out, self.allocator, self.server_name);
        try out.appendSlice(self.allocator, ",\"version\":\"0.5.0\"}}");
        return out.toOwnedSlice(self.allocator);
    }

    fn toolsListResult(self: *Server) ![]u8 {
        var out = std.ArrayList(u8).empty;
        errdefer out.deinit(self.allocator);
        try out.appendSlice(self.allocator, "{\"tools\":[");
        for (self.tools, 0..) |tool, i| {
            if (i != 0) try out.append(allocatorFor(self), ',');
            try out.appendSlice(self.allocator, "{\"name\":");
            try writeJsonString(&out, self.allocator, tool.name);
            try out.appendSlice(self.allocator, ",\"description\":");
            try writeJsonString(&out, self.allocator, tool.description);
            try out.appendSlice(self.allocator, ",\"inputSchema\":");
            try out.appendSlice(self.allocator, tool.parameters_json);
            try out.append(allocatorFor(self), '}');
        }
        try out.appendSlice(self.allocator, "]}");
        return out.toOwnedSlice(self.allocator);
    }

    fn createMessage(self: *Server, params_value: ?std.json.Value) ![]u8 {
        const params = params_value orelse return error.InvalidParams;
        if (params != .object) return error.InvalidParams;

        if (self.sample_fn) |sample| {
            var msg_buf = std.ArrayList(u8).empty;
            defer msg_buf.deinit(self.allocator);
            try writeJsonValue(&msg_buf, self.allocator, params);
            const messages_json = try msg_buf.toOwnedSlice(self.allocator);
            defer self.allocator.free(messages_json);

            const generated = try sample(self.sample_ctx, messages_json, self.allocator);
            defer self.allocator.free(generated);

            var out = std.ArrayList(u8).empty;
            errdefer out.deinit(self.allocator);
            try out.appendSlice(self.allocator, "{\"role\":\"assistant\",\"content\":{\"type\":\"text\",\"text\":");
            try writeJsonString(&out, self.allocator, generated);
            try out.appendSlice(self.allocator, "}}");
            return out.toOwnedSlice(self.allocator);
        }

        // Si no hay sample_fn, devolver error
        var out = std.ArrayList(u8).empty;
        errdefer out.deinit(self.allocator);
        try out.appendSlice(self.allocator, "{\"role\":\"assistant\",\"content\":{\"type\":\"text\",\"text\":\"sampling not configured\"}}");
        return out.toOwnedSlice(self.allocator);
    }

    fn callTool(self: *Server, params_value: ?std.json.Value) ![]u8 {
        const params = params_value orelse return error.InvalidParams;
        if (params != .object) return error.InvalidParams;
        const obj = params.object;
        const name_v = obj.get("name") orelse return error.InvalidParams;
        if (name_v != .string) return error.InvalidParams;
        const name = name_v.string;
        const args_v = obj.get("arguments");
        const args_json = if (args_v) |v| try stringifyValue(self.allocator, v) else try self.allocator.dupe(u8, "{}");
        defer self.allocator.free(args_json);

        const tool_result = try self.execute_fn(self.ctx, name, args_json, self.allocator);
        defer self.allocator.free(tool_result.content);

        var out = std.ArrayList(u8).empty;
        errdefer out.deinit(self.allocator);
        try out.appendSlice(self.allocator, "{\"content\":[{\"type\":\"text\",\"text\":");
        try writeJsonString(&out, self.allocator, tool_result.content);
        try out.appendSlice(self.allocator, "}],\"isError\":");
        try out.appendSlice(self.allocator, if (tool_result.is_error) "true" else "false");
        try out.append(allocatorFor(self), '}');
        return out.toOwnedSlice(self.allocator);
    }
};

fn allocatorFor(self: *Server) std.mem.Allocator {
    return self.allocator;
}

fn stringifyValue(allocator: std.mem.Allocator, value: std.json.Value) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try writeJsonValue(&out, allocator, value);
    return out.toOwnedSlice(allocator);
}

fn writeJsonValue(out: *std.ArrayList(u8), allocator: std.mem.Allocator, value: std.json.Value) !void {
    switch (value) {
        .null => try out.appendSlice(allocator, "null"),
        .bool => |b| try out.appendSlice(allocator, if (b) "true" else "false"),
        .integer => |i| try out.print(allocator, "{}", .{i}),
        .float => |f| try out.print(allocator, "{d}", .{f}),
        .number_string => |s| try out.appendSlice(allocator, s),
        .string => |s| try writeJsonString(out, allocator, s),
        .array => |array| {
            try out.append(allocator, '[');
            for (array.items, 0..) |item, i| {
                if (i != 0) try out.append(allocator, ',');
                try writeJsonValue(out, allocator, item);
            }
            try out.append(allocator, ']');
        },
        .object => |object| {
            try out.append(allocator, '{');
            var it = object.iterator();
            var first = true;
            while (it.next()) |entry| {
                if (!first) try out.append(allocator, ',');
                first = false;
                try writeJsonString(out, allocator, entry.key_ptr.*);
                try out.append(allocator, ':');
                try writeJsonValue(out, allocator, entry.value_ptr.*);
            }
            try out.append(allocator, '}');
        },
    }
}

pub const ClientTool = struct {
    name: []const u8,
    description: []const u8,
    input_schema_json: []const u8,
};

// ---------------------------------------------------------------------------
// Cliente MCP (stdio)
// ---------------------------------------------------------------------------

pub const TransportFn = *const fn (ctx: ?*anyopaque, request: []const u8, allocator: std.mem.Allocator) anyerror![]u8;

pub const Client = struct {
    allocator: std.mem.Allocator,
    stdin: std.Io.File.Writer,
    stdout: std.Io.File.Reader,
    stdin_buf: [4096]u8,
    stdout_buf: [4096]u8,
    next_id: i64,
    /// Transporte alternativo para tests e2e (si es null, usa stdio).
    transport_fn: ?TransportFn = null,
    transport_ctx: ?*anyopaque = null,

    pub fn init(allocator: std.mem.Allocator, io: std.Io) Client {
        var client: Client = undefined;
        client.allocator = allocator;
        client.stdin_buf = undefined;
        client.stdout_buf = undefined;
        client.stdin = std.Io.File.stdout().writer(io, &client.stdin_buf);
        client.stdout = std.Io.File.stdin().reader(io, &client.stdout_buf);
        client.next_id = 1;
        client.transport_fn = null;
        client.transport_ctx = null;
        return client;
    }

    pub fn initPipes(allocator: std.mem.Allocator, stdin_writer: std.Io.File.Writer, stdout_reader: std.Io.File.Reader) Client {
        var client: Client = undefined;
        client.allocator = allocator;
        client.stdin_buf = undefined;
        client.stdout_buf = undefined;
        client.stdin = stdin_writer;
        client.stdout = stdout_reader;
        client.next_id = 1;
        client.transport_fn = null;
        client.transport_ctx = null;
        return client;
    }

    /// Envía un request JSON-RPC line-delimited y lee la respuesta.
    pub fn request(self: *Client, method: []const u8, params: []const u8) ![]u8 {
        const id = self.next_id;
        self.next_id += 1;

        var req_buf = std.ArrayList(u8).empty;
        defer req_buf.deinit(self.allocator);
        try req_buf.appendSlice(self.allocator, "{\"jsonrpc\":\"2.0\",\"id\":");
        try req_buf.print(self.allocator, "{}", .{id});
        try req_buf.appendSlice(self.allocator, ",\"method\":\"");
        try req_buf.appendSlice(self.allocator, method);
        try req_buf.appendSlice(self.allocator, "\",\"params\":");
        try req_buf.appendSlice(self.allocator, params);
        try req_buf.appendSlice(self.allocator, "}\n");

        if (self.transport_fn) |transport| {
            return transport(self.transport_ctx, req_buf.items, self.allocator);
        }

        try self.stdin.interface.writeAll(req_buf.items);
        try self.stdin.interface.flush();

        // Leer línea de respuesta byte a byte (evita problemas con takeDelimiterExclusive en pipes)
        var line_buf = std.ArrayList(u8).empty;
        defer line_buf.deinit(self.allocator);
        while (true) {
            var byte_buf: [1]u8 = undefined;
            var vecs: [1][]u8 = .{&byte_buf};
            const n = try self.stdout.interface.readVec(&vecs);
            if (n == 0) return error.EndOfStream;
            if (byte_buf[0] == '\n') break;
            try line_buf.append(self.allocator, byte_buf[0]);
        }
        return try self.allocator.dupe(u8, line_buf.items);
    }

    pub fn initialize(self: *Client, client_name: []const u8) !void {
        const params = try std.fmt.allocPrint(self.allocator, "{{\"protocolVersion\":\"{s}\",\"capabilities\":{{}},\"clientInfo\":{{\"name\":\"{s}\",\"version\":\"0.5.0\"}}}}", .{ protocol_version, client_name });
        defer self.allocator.free(params);
        const res = try self.request("initialize", params);
        defer self.allocator.free(res);
    }

    pub fn toolsList(self: *Client) ![]ClientTool {
        const res = try self.request("tools/list", "{}");
        defer self.allocator.free(res);

        var parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, res, .{});
        defer parsed.deinit();

        const result = parsed.value.object.get("result") orelse return &[_]ClientTool{};
        const tools_arr = result.object.get("tools") orelse return &[_]ClientTool{};
        const items = tools_arr.array.items;

        var out = try self.allocator.alloc(ClientTool, items.len);
        errdefer self.allocator.free(out);

        for (items, 0..) |item, i| {
            const name = item.object.get("name").?.string;
            const desc = if (item.object.get("description")) |d| d.string else "";
            const schema = if (item.object.get("inputSchema")) |s| blk: {
                var schema_buf = std.ArrayList(u8).empty;
                defer schema_buf.deinit(self.allocator);
                try writeJsonValue(&schema_buf, self.allocator, s);
                break :blk try schema_buf.toOwnedSlice(self.allocator);
            } else try self.allocator.dupe(u8, "{}");

            out[i] = .{
                .name = try self.allocator.dupe(u8, name),
                .description = try self.allocator.dupe(u8, desc),
                .input_schema_json = schema,
            };
        }
        return out;
    }

    pub fn toolsCall(self: *Client, name: []const u8, args_json: []const u8) !ToolResult {
        const params = try std.fmt.allocPrint(self.allocator, "{{\"name\":\"{s}\",\"arguments\":{s}}}", .{ name, args_json });
        defer self.allocator.free(params);
        const res = try self.request("tools/call", params);
        defer self.allocator.free(res);

        var parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, res, .{});
        defer parsed.deinit();

        const result = parsed.value.object.get("result") orelse return ToolResult{ .content = try self.allocator.dupe(u8, "error: no result"), .is_error = true };
        const content_arr = result.object.get("content") orelse return ToolResult{ .content = try self.allocator.dupe(u8, "error: no content"), .is_error = true };

        var text_buf = std.ArrayList(u8).empty;
        defer text_buf.deinit(self.allocator);
        for (content_arr.array.items) |c| {
            if (c.object.get("text")) |t| {
                try text_buf.appendSlice(self.allocator, t.string);
            }
        }
        const is_error = if (result.object.get("isError")) |e| e == .bool and e.bool else false;
        return ToolResult{
            .content = try text_buf.toOwnedSlice(self.allocator),
            .is_error = is_error,
        };
    }

    pub fn deinitToolsList(self: *Client, tools: []ClientTool) void {
        for (tools) |t| {
            self.allocator.free(t.name);
            self.allocator.free(t.description);
            self.allocator.free(t.input_schema_json);
        }
        self.allocator.free(tools);
    }
};

// ---------------------------------------------------------------------------
// Utilidades
// ---------------------------------------------------------------------------

pub fn makeInitializeRequest(allocator: std.mem.Allocator, id: i64, client_name: []const u8) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.print(allocator, "{{\"jsonrpc\":\"2.0\",\"id\":{},\"method\":\"initialize\",\"params\":{{\"protocolVersion\":\"{s}\",\"capabilities\":{{}},\"clientInfo\":{{\"name\":", .{ id, protocol_version });
    try writeJsonString(&out, allocator, client_name);
    try out.appendSlice(allocator, ",\"version\":\"0.5.0\"}}}}}");
    return out.toOwnedSlice(allocator);
}

const ErrorCode = enum(i32) {
    parse_error = -32700,
    method_not_found = -32601,
};

const Id = union(enum) {
    none,
    integer: i64,
    string: []const u8,

    fn fromValue(value: ?std.json.Value) Id {
        const v = value orelse return .none;
        return switch (v) {
            .integer => |i| .{ .integer = i },
            .string => |s| .{ .string = s },
            else => .none,
        };
    }
};

const RawRequest = struct {
    jsonrpc: []const u8 = "2.0",
    id: ?std.json.Value = null,
    method: []const u8,
    params: ?std.json.Value = null,
};

const ParsedRequest = struct {
    parsed: std.json.Parsed(RawRequest),

    fn deinit(self: *ParsedRequest) void {
        self.parsed.deinit();
    }

    fn id(self: ParsedRequest) Id {
        return Id.fromValue(self.parsed.value.id);
    }

    fn method(self: ParsedRequest) []const u8 {
        return self.parsed.value.method;
    }

    fn params(self: ParsedRequest) ?std.json.Value {
        return self.parsed.value.params;
    }
};

fn parseRequest(allocator: std.mem.Allocator, input: []const u8) !ParsedRequest {
    const parsed = try std.json.parseFromSlice(RawRequest, allocator, input, .{ .ignore_unknown_fields = true });
    return .{ .parsed = parsed };
}

fn makeResult(allocator: std.mem.Allocator, id: Id, result_json: []const u8) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"jsonrpc\":\"2.0\",\"id\":");
    try writeId(&out, allocator, id);
    try out.appendSlice(allocator, ",\"result\":");
    try out.appendSlice(allocator, result_json);
    try out.append(allocator, '}');
    return out.toOwnedSlice(allocator);
}

fn makeError(allocator: std.mem.Allocator, id: Id, code: ErrorCode, message: []const u8) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"jsonrpc\":\"2.0\",\"id\":");
    try writeId(&out, allocator, id);
    try out.print(allocator, ",\"error\":{{\"code\":{},\"message\":", .{@intFromEnum(code)});
    try writeJsonString(&out, allocator, message);
    try out.appendSlice(allocator, "}}}");
    return out.toOwnedSlice(allocator);
}

fn writeId(out: *std.ArrayList(u8), allocator: std.mem.Allocator, id: Id) !void {
    switch (id) {
        .none => try out.appendSlice(allocator, "null"),
        .integer => |i| try out.print(allocator, "{}", .{i}),
        .string => |s| try writeJsonString(out, allocator, s),
    }
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

test "zmcp lists tools over json-rpc" {
    const allocator = std.testing.allocator;
    const tools = [_]ToolInfo{.{
        .name = "finish",
        .description = "finish task",
        .parameters_json = "{\"type\":\"object\",\"properties\":{}}",
    }};

    const Ctx = struct {
        fn execute(_: *anyopaque, _: []const u8, _: []const u8, alloc: std.mem.Allocator) !ToolResult {
            return .{ .content = try alloc.dupe(u8, "ok"), .is_error = false };
        }
    };

    var dummy: u8 = 0;
    var server = Server.init(allocator, tools[0..], &dummy, Ctx.execute);
    const response = try server.handle("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}");
    defer allocator.free(response);
    try std.testing.expect(std.mem.indexOf(u8, response, "\"name\":\"finish\"") != null);
}

test "zmcp sampling/createMessage con sample_fn" {
    const allocator = std.testing.allocator;
    const tools = [_]ToolInfo{};

    const Ctx = struct {
        fn execute(_: *anyopaque, _: []const u8, _: []const u8, alloc: std.mem.Allocator) !ToolResult {
            return .{ .content = try alloc.dupe(u8, "ok"), .is_error = false };
        }
        fn sample(_: ?*anyopaque, _: []const u8, alloc: std.mem.Allocator) ![]u8 {
            return try alloc.dupe(u8, "generated text");
        }
    };

    var dummy: u8 = 0;
    var server = Server.init(allocator, tools[0..], &dummy, Ctx.execute);
    server.sample_fn = Ctx.sample;

    const response = try server.handle("{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"sampling/createMessage\",\"params\":{\"messages\":[]}}");
    defer allocator.free(response);
    try std.testing.expect(std.mem.indexOf(u8, response, "generated text") != null);
    try std.testing.expect(std.mem.indexOf(u8, response, "\"role\":\"assistant\"") != null);
}

// Transporte en memoria que conecta Client a Server directamente (sin stdio).
pub const MemTransportCtx = struct {
    server: *Server,
};
pub fn memTransport(ctx: ?*anyopaque, request: []const u8, _: std.mem.Allocator) anyerror![]u8 {
    const t: *MemTransportCtx = @alignCast(@ptrCast(ctx.?));
    // Server.handle espera la línea sin el \n final
    var end = request.len;
    while (end > 0 and request[end - 1] == '\n') end -= 1;
    const trimmed = request[0..end];
    return t.server.handle(trimmed);
}

test "zmcp e2e client->server initialize + tools/list + tools/call" {
    const allocator = std.testing.allocator;

    const tools = [_]ToolInfo{
        .{ .name = "echo", .description = "echo", .parameters_json = "{\"type\":\"object\",\"properties\":{\"msg\":{\"type\":\"string\"}}}" },
        .{ .name = "finish", .description = "finish", .parameters_json = "{}" },
    };

    const Ctx = struct {
        fn execute(_: *anyopaque, name: []const u8, args_json: []const u8, alloc: std.mem.Allocator) !ToolResult {
            _ = name;
            return .{ .content = try alloc.dupe(u8, args_json), .is_error = false };
        }
    };

    var dummy: u8 = 0;
    var server = Server.init(allocator, tools[0..], &dummy, Ctx.execute);
    var tctx = MemTransportCtx{ .server = &server };

    var client = Client.init(allocator, undefined);
    client.transport_fn = &memTransport;
    client.transport_ctx = &tctx;

    // Initialize
    try client.initialize("test-client");

    // tools/list
    const list = try client.toolsList();
    defer client.deinitToolsList(list);
    try std.testing.expectEqual(@as(usize, 2), list.len);
    try std.testing.expectEqualStrings("echo", list[0].name);
    try std.testing.expectEqualStrings("finish", list[1].name);

    // tools/call
    const result = try client.toolsCall("echo", "{\"msg\":\"hola\"}");
    defer allocator.free(result.content);
    try std.testing.expect(!result.is_error);
    try std.testing.expect(std.mem.indexOf(u8, result.content, "hola") != null);
}

test "zmcp e2e client toolsCall error propagation" {
    const allocator = std.testing.allocator;

    const tools = [_]ToolInfo{.{
        .name = "fail",
        .description = "always fails",
        .parameters_json = "{}",
    }};

    const Ctx = struct {
        fn execute(_: *anyopaque, _: []const u8, _: []const u8, alloc: std.mem.Allocator) !ToolResult {
            return .{ .content = try alloc.dupe(u8, "boom"), .is_error = true };
        }
    };

    var dummy: u8 = 0;
    var server = Server.init(allocator, tools[0..], &dummy, Ctx.execute);
    var tctx = MemTransportCtx{ .server = &server };

    var client = Client.init(allocator, undefined);
    client.transport_fn = &memTransport;
    client.transport_ctx = &tctx;

    const result = try client.toolsCall("fail", "{}");
    defer allocator.free(result.content);
    try std.testing.expect(result.is_error);
    try std.testing.expectEqualStrings("boom", result.content);
}

// ---------------------------------------------------------------------------
// Bridge: conecta zmcp.Client con zagent.Registry
// ---------------------------------------------------------------------------

/// Envuelve un Client para que cada tool MCP externa aparezca
/// como una tool nativa en el registry del agente.
pub const McpToolWrapper = struct {
    client: *Client,
    tool_name: []const u8,

    pub fn name(ptr: *anyopaque) []const u8 {
        const self = @as(*McpToolWrapper, @ptrCast(@alignCast(ptr)));
        return self.tool_name;
    }

    pub fn description(_: *anyopaque) []const u8 {
        return "MCP tool";
    }

    pub fn parameters_json(_: *anyopaque) []const u8 {
        return "{\"type\":\"object\",\"properties\":{}}";
    }

    pub fn execute(ctx: *anyopaque, io: std.Io, args_json: []const u8, allocator: std.mem.Allocator) !zagent.ToolResult {
        _ = io;
        const self = @as(*McpToolWrapper, @ptrCast(@alignCast(ctx)));
        const result = try self.client.toolsCall(self.tool_name, args_json);
        defer allocator.free(result.content);
        return zagent.ToolResult{
            .content = try allocator.dupe(u8, result.content),
            .is_error = result.is_error,
        };
    }

    pub fn tool(self: *McpToolWrapper) zagent.Tool {
        return .{
            .ptr = self,
            .vtable = &.{
                .name = name,
                .description = description,
                .parameters_json = parameters_json,
                .execute = execute,
            },
        };
    }
};

/// Registra todas las tools de un servidor MCP en el registry del agente.
/// El caller debe mantener vivo el `client` y los wrappers durante la vida del agente.
pub fn registerTools(client: *Client, registry: *zagent.Registry, allocator: std.mem.Allocator) !void {
    const tools = try client.toolsList();
    defer client.deinitToolsList(tools);

    for (tools) |t| {
        const wrapper = try allocator.create(McpToolWrapper);
        wrapper.* = .{
            .client = client,
            .tool_name = try allocator.dupe(u8, t.name),
        };
        try registry.register(wrapper.tool());
    }
}
