// zrpc - nucleo JSON-RPC 2.0
const std = @import("std");

pub const ErrorCode = enum(i32) {
    parse_error = -32700,
    invalid_request = -32600,
    method_not_found = -32601,
    invalid_params = -32602,
    internal_error = -32603,
};

pub const Id = union(enum) {
    none,
    integer: i64,
    string: []const u8,

    pub fn fromValue(value: ?std.json.Value) Id {
        const v = value orelse return .none;
        return switch (v) {
            .integer => |i| .{ .integer = i },
            .string => |s| .{ .string = s },
            else => .none,
        };
    }
};

pub const RawRequest = struct {
    jsonrpc: []const u8 = "2.0",
    id: ?std.json.Value = null,
    method: []const u8,
    params: ?std.json.Value = null,
};

pub const ParsedRequest = struct {
    parsed: std.json.Parsed(RawRequest),

    pub fn deinit(self: *ParsedRequest) void {
        self.parsed.deinit();
    }

    pub fn id(self: ParsedRequest) Id {
        return Id.fromValue(self.parsed.value.id);
    }

    pub fn method(self: ParsedRequest) []const u8 {
        return self.parsed.value.method;
    }

    pub fn params(self: ParsedRequest) ?std.json.Value {
        return self.parsed.value.params;
    }
};

pub fn parseRequest(allocator: std.mem.Allocator, input: []const u8) !ParsedRequest {
    const parsed = try std.json.parseFromSlice(RawRequest, allocator, input, .{ .ignore_unknown_fields = true });
    if (!std.mem.eql(u8, parsed.value.jsonrpc, "2.0")) {
        parsed.deinit();
        return error.InvalidJsonRpcVersion;
    }
    return .{ .parsed = parsed };
}

pub fn makeResult(allocator: std.mem.Allocator, id: Id, result_json: []const u8) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"jsonrpc\":\"2.0\",\"id\":");
    try writeId(&out, allocator, id);
    try out.appendSlice(allocator, ",\"result\":");
    try out.appendSlice(allocator, result_json);
    try out.append(allocator, '}');
    return out.toOwnedSlice(allocator);
}

pub fn makeError(allocator: std.mem.Allocator, id: Id, code: ErrorCode, message: []const u8) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"jsonrpc\":\"2.0\",\"id\":");
    try writeId(&out, allocator, id);
    try out.print(allocator, ",\"error\":{{\"code\":{},\"message\":", .{@intFromEnum(code)});
    try writeJsonString(&out, allocator, message);
    try out.appendSlice(allocator, "}}");
    try out.append(allocator, '}');
    return out.toOwnedSlice(allocator);
}

pub fn frameNewline(allocator: std.mem.Allocator, payload: []const u8) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, payload);
    try out.append(allocator, '\n');
    return out.toOwnedSlice(allocator);
}

pub fn frameContentLength(allocator: std.mem.Allocator, payload: []const u8) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.print(allocator, "Content-Length: {}\r\n\r\n", .{payload.len});
    try out.appendSlice(allocator, payload);
    return out.toOwnedSlice(allocator);
}

fn writeId(out: *std.ArrayList(u8), allocator: std.mem.Allocator, id: Id) !void {
    switch (id) {
        .none => try out.appendSlice(allocator, "null"),
        .integer => |i| try out.print(allocator, "{}", .{i}),
        .string => |s| try writeJsonString(out, allocator, s),
    }
}

pub fn writeJsonString(out: *std.ArrayList(u8), allocator: std.mem.Allocator, text: []const u8) !void {
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

test "zrpc parses and frames request" {
    const allocator = std.testing.allocator;
    var req = try parseRequest(allocator, "{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"ping\"}");
    defer req.deinit();

    try std.testing.expectEqualStrings("ping", req.method());
    try std.testing.expectEqual(@as(i64, 7), req.id().integer);

    const response = try makeResult(allocator, req.id(), "{\"ok\":true}");
    defer allocator.free(response);
    try std.testing.expect(std.mem.indexOf(u8, response, "\"result\":{\"ok\":true}") != null);

    const framed = try frameNewline(allocator, response);
    defer allocator.free(framed);
    try std.testing.expectEqual(@as(u8, '\n'), framed[framed.len - 1]);
}
