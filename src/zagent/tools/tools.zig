// tools.zig — herramientas nativas para zagent
const std = @import("std");
const zagent = @import("../zagent.zig");

// ---------------------------------------------------------------------------
// Helper: validar que la ruta no escape del workspace
// ---------------------------------------------------------------------------
pub fn validatePath(path: []const u8) !void {
    if (std.mem.indexOf(u8, path, "..") != null or
        std.mem.startsWith(u8, path, "/") or
        std.mem.indexOf(u8, path, ":") != null)
    {
        return error.SecurityViolation;
    }
}

// ---------------------------------------------------------------------------
// Tool: fs_read
// ---------------------------------------------------------------------------
pub const FsReadTool = struct {
    name_str: []const u8 = "fs_read",
    desc_str: []const u8 = "Lee el contenido de un archivo dentro del workspace. Argumentos: {\"path\": \"nombre_archivo\"}",
    params_str: []const u8 = "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}},\"required\":[\"path\"]}",

    pub fn name(ptr: *anyopaque) []const u8 {
        const self: *FsReadTool = @alignCast(@ptrCast(ptr));
        return self.name_str;
    }
    pub fn description(ptr: *anyopaque) []const u8 {
        const self: *FsReadTool = @alignCast(@ptrCast(ptr));
        return self.desc_str;
    }
    pub fn parameters_json(ptr: *anyopaque) []const u8 {
        const self: *FsReadTool = @alignCast(@ptrCast(ptr));
        return self.params_str;
    }
    pub fn execute(_: *anyopaque, io: std.Io, args_json: []const u8, allocator: std.mem.Allocator) anyerror!zagent.ToolResult {
        const Args = struct { path: []const u8 };
        const parsed = std.json.parseFromSlice(Args, allocator, args_json, .{ .ignore_unknown_fields = true }) catch |err| {
            var msg = std.ArrayList(u8).empty;
            defer msg.deinit(allocator);
            try msg.print(allocator, "Error al parsear argumentos: {s}", .{@errorName(err)});
            return zagent.ToolResult{ .content = try msg.toOwnedSlice(allocator), .is_error = true };
        };
        defer parsed.deinit();

        const path = parsed.value.path;
        validatePath(path) catch {
            return zagent.ToolResult{
                .content = try allocator.dupe(u8, "Error de seguridad: Acceso fuera del espacio de trabajo denegado"),
                .is_error = true,
            };
        };

        const dir = std.Io.Dir.cwd();
        const content = dir.readFileAlloc(io, path, allocator, .unlimited) catch |err| {
            var msg = std.ArrayList(u8).empty;
            defer msg.deinit(allocator);
            try msg.print(allocator, "Error al leer archivo: {s}", .{@errorName(err)});
            return zagent.ToolResult{ .content = try msg.toOwnedSlice(allocator), .is_error = true };
        };
        return zagent.ToolResult{ .content = content, .is_error = false };
    }
    pub fn tool(self: *FsReadTool) zagent.Tool {
        return .{
            .ptr = self,
            .vtable = &zagent.Tool.VTable{
                .name = name,
                .description = description,
                .parameters_json = parameters_json,
                .execute = execute,
            },
        };
    }
};

// ---------------------------------------------------------------------------
// Tool: fs_write
// ---------------------------------------------------------------------------
pub const FsWriteTool = struct {
    name_str: []const u8 = "fs_write",
    desc_str: []const u8 = "Escribe o sobreescribe contenido en un archivo dentro del workspace. Argumentos: {\"path\": \"archivo\", \"content\": \"datos\"}",
    params_str: []const u8 = "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}},\"required\":[\"path\",\"content\"]}",

    pub fn name(ptr: *anyopaque) []const u8 {
        const self: *FsWriteTool = @alignCast(@ptrCast(ptr));
        return self.name_str;
    }
    pub fn description(ptr: *anyopaque) []const u8 {
        const self: *FsWriteTool = @alignCast(@ptrCast(ptr));
        return self.desc_str;
    }
    pub fn parameters_json(ptr: *anyopaque) []const u8 {
        const self: *FsWriteTool = @alignCast(@ptrCast(ptr));
        return self.params_str;
    }
    pub fn execute(_: *anyopaque, io: std.Io, args_json: []const u8, allocator: std.mem.Allocator) anyerror!zagent.ToolResult {
        const Args = struct { path: []const u8, content: []const u8 };
        const parsed = std.json.parseFromSlice(Args, allocator, args_json, .{ .ignore_unknown_fields = true }) catch |err| {
            var msg = std.ArrayList(u8).empty;
            defer msg.deinit(allocator);
            try msg.print(allocator, "Error al parsear argumentos: {s}", .{@errorName(err)});
            return zagent.ToolResult{ .content = try msg.toOwnedSlice(allocator), .is_error = true };
        };
        defer parsed.deinit();

        const path = parsed.value.path;
        validatePath(path) catch {
            return zagent.ToolResult{
                .content = try allocator.dupe(u8, "Error de seguridad: Acceso fuera del espacio de trabajo denegado"),
                .is_error = true,
            };
        };

        const dir = std.Io.Dir.cwd();
        dir.writeFile(io, .{ .sub_path = path, .data = parsed.value.content }) catch |err| {
            var msg = std.ArrayList(u8).empty;
            defer msg.deinit(allocator);
            try msg.print(allocator, "Error al escribir archivo: {s}", .{@errorName(err)});
            return zagent.ToolResult{ .content = try msg.toOwnedSlice(allocator), .is_error = true };
        };
        return zagent.ToolResult{
            .content = try allocator.dupe(u8, "Archivo escrito exitosamente"),
            .is_error = false,
        };
    }
    pub fn tool(self: *FsWriteTool) zagent.Tool {
        return .{
            .ptr = self,
            .vtable = &zagent.Tool.VTable{
                .name = name,
                .description = description,
                .parameters_json = parameters_json,
                .execute = execute,
            },
        };
    }
};

// ---------------------------------------------------------------------------
// Tool: bash (sandbox + allowlist + metacaracteres)
// ---------------------------------------------------------------------------
const bash_allowlist = [_][]const u8{
    "zig build test",
    "zig build",
    "zig version",
    "dir ",
    "echo ",
    "type ",
};

fn hasDangerousChars(cmd: []const u8) bool {
    const dangerous = &[_]u8{ '|', '>', '<', '&', ';', '$', '`', '\n', '\r' };
    for (dangerous) |d| {
        if (std.mem.indexOfScalar(u8, cmd, d) != null) return true;
    }
    return false;
}

fn isCommandAllowed(cmd: []const u8) bool {
    if (hasDangerousChars(cmd)) return false;
    for (bash_allowlist) |allowed| {
        if (std.mem.startsWith(u8, cmd, allowed)) return true;
    }
    return false;
}

pub const BashTool = struct {
    name_str: []const u8 = "bash",
    desc_str: []const u8 = "Ejecuta un comando permitido en la shell y retorna stdout/stderr. Argumentos: {\"command\": \"zig build test\"}",
    params_str: []const u8 = "{\"type\":\"object\",\"properties\":{\"command\":{\"type\":\"string\"}},\"required\":[\"command\"]}",

    pub fn name(ptr: *anyopaque) []const u8 {
        const self: *BashTool = @alignCast(@ptrCast(ptr));
        return self.name_str;
    }
    pub fn description(ptr: *anyopaque) []const u8 {
        const self: *BashTool = @alignCast(@ptrCast(ptr));
        return self.desc_str;
    }
    pub fn parameters_json(ptr: *anyopaque) []const u8 {
        const self: *BashTool = @alignCast(@ptrCast(ptr));
        return self.params_str;
    }
    pub fn execute(_: *anyopaque, io: std.Io, args_json: []const u8, allocator: std.mem.Allocator) anyerror!zagent.ToolResult {
        const Args = struct { command: []const u8 };
        const parsed = std.json.parseFromSlice(Args, allocator, args_json, .{ .ignore_unknown_fields = true }) catch |err| {
            var msg = std.ArrayList(u8).empty;
            defer msg.deinit(allocator);
            try msg.print(allocator, "Error al parsear argumentos: {s}", .{@errorName(err)});
            return zagent.ToolResult{ .content = try msg.toOwnedSlice(allocator), .is_error = true };
        };
        defer parsed.deinit();

        const command = parsed.value.command;
        if (!isCommandAllowed(command)) {
            return zagent.ToolResult{
                .content = try allocator.dupe(u8, "Error de seguridad: comando no permitido por allowlist"),
                .is_error = true,
            };
        }

        var child = try std.process.spawn(io, .{
            .argv = &.{ "cmd.exe", "/C", command },
            .stdout = .pipe,
            .stderr = .pipe,
        });

        var aw_out = std.Io.Writer.Allocating.init(allocator);
        defer aw_out.deinit();

        var buf_out: [4096]u8 = undefined;
        var buf_err: [4096]u8 = undefined;
        var stdout_reader = child.stdout.?.reader(io, &buf_out);
        var stderr_reader = child.stderr.?.reader(io, &buf_err);
        _ = stdout_reader.interface.streamRemaining(&aw_out.writer) catch {};
        _ = stderr_reader.interface.streamRemaining(&aw_out.writer) catch {};

        _ = try child.wait(io);

        const out = try aw_out.toOwnedSlice();
        return zagent.ToolResult{ .content = out, .is_error = false };
    }
    pub fn tool(self: *BashTool) zagent.Tool {
        return .{
            .ptr = self,
            .vtable = &zagent.Tool.VTable{
                .name = name,
                .description = description,
                .parameters_json = parameters_json,
                .execute = execute,
            },
        };
    }
};

// ---------------------------------------------------------------------------
// Tool: http_fetch (allowlist de hosts)
// ---------------------------------------------------------------------------
const http_allowlist = [_][]const u8{
    "127.0.0.1",
    "localhost",
    "api.github.com",
    "raw.githubusercontent.com",
};

fn extractHost(url: []const u8) []const u8 {
    var start: usize = 0;
    if (std.mem.indexOf(u8, url, "://")) |pos| {
        start = pos + "://".len;
    }
    var end = url.len;
    if (std.mem.indexOfPos(u8, url, start, "/")) |slash| {
        end = slash;
    }
    if (std.mem.indexOfPos(u8, url, start, ":")) |colon| {
        if (colon < end) end = colon;
    }
    return url[start..end];
}

fn isHostAllowed(host: []const u8) bool {
    for (http_allowlist) |allowed| {
        if (std.mem.eql(u8, host, allowed)) return true;
    }
    return false;
}

pub const HttpFetchTool = struct {
    name_str: []const u8 = "http_fetch",
    desc_str: []const u8 = "Realiza una petición HTTP GET a un host permitido. Argumentos: {\"url\": \"http://127.0.0.1:11434/api/tags\"}",
    params_str: []const u8 = "{\"type\":\"object\",\"properties\":{\"url\":{\"type\":\"string\"}},\"required\":[\"url\"]}",

    pub fn name(ptr: *anyopaque) []const u8 {
        const self: *HttpFetchTool = @alignCast(@ptrCast(ptr));
        return self.name_str;
    }
    pub fn description(ptr: *anyopaque) []const u8 {
        const self: *HttpFetchTool = @alignCast(@ptrCast(ptr));
        return self.desc_str;
    }
    pub fn parameters_json(ptr: *anyopaque) []const u8 {
        const self: *HttpFetchTool = @alignCast(@ptrCast(ptr));
        return self.params_str;
    }
    pub fn execute(_: *anyopaque, io: std.Io, args_json: []const u8, allocator: std.mem.Allocator) anyerror!zagent.ToolResult {
        const Args = struct { url: []const u8 };
        const parsed = std.json.parseFromSlice(Args, allocator, args_json, .{ .ignore_unknown_fields = true }) catch |err| {
            var msg = std.ArrayList(u8).empty;
            defer msg.deinit(allocator);
            try msg.print(allocator, "Error al parsear argumentos: {s}", .{@errorName(err)});
            return zagent.ToolResult{ .content = try msg.toOwnedSlice(allocator), .is_error = true };
        };
        defer parsed.deinit();

        const host = extractHost(parsed.value.url);
        if (!isHostAllowed(host)) {
            return zagent.ToolResult{
                .content = try allocator.dupe(u8, "Error de seguridad: host no permitido por allowlist"),
                .is_error = true,
            };
        }

        var client = std.http.Client{ .allocator = allocator, .io = io };
        defer client.deinit();

        var res_writer = std.Io.Writer.Allocating.init(allocator);
        defer res_writer.deinit();

        const res = client.fetch(.{
            .location = .{ .url = parsed.value.url },
            .method = .GET,
            .response_writer = &res_writer.writer,
        }) catch |err| {
            var msg = std.ArrayList(u8).empty;
            defer msg.deinit(allocator);
            try msg.print(allocator, "Error HTTP: {s}", .{@errorName(err)});
            return zagent.ToolResult{ .content = try msg.toOwnedSlice(allocator), .is_error = true };
        };

        if (res.status != .ok) {
            var msg = std.ArrayList(u8).empty;
            defer msg.deinit(allocator);
            try msg.print(allocator, "HTTP status: {s}", .{@tagName(res.status)});
            return zagent.ToolResult{ .content = try msg.toOwnedSlice(allocator), .is_error = true };
        }

        const body = res_writer.written();
        return zagent.ToolResult{ .content = try allocator.dupe(u8, body), .is_error = false };
    }
    pub fn tool(self: *HttpFetchTool) zagent.Tool {
        return .{
            .ptr = self,
            .vtable = &zagent.Tool.VTable{
                .name = name,
                .description = description,
                .parameters_json = parameters_json,
                .execute = execute,
            },
        };
    }
};

// ---------------------------------------------------------------------------
// Tool: search (búsqueda en archivos del workspace)
// ---------------------------------------------------------------------------
fn isTextExtension(name: []const u8) bool {
    const exts = &[_][]const u8{ ".zig", ".md", ".txt", ".json", ".zon", ".rs", ".py", ".js", ".ts", ".c", ".cpp", ".h", ".hpp" };
    for (exts) |ext| {
        if (std.mem.endsWith(u8, name, ext)) return true;
    }
    return false;
}

fn searchInDir(alloc: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, query: []const u8, depth: usize, max_depth: usize, out: *std.ArrayList(u8)) !void {
    if (depth > max_depth) return;
    var it = dir.iterate();
    while (it.next(io) catch return) |entry| {
        if (std.mem.eql(u8, entry.name, ".git") or std.mem.eql(u8, entry.name, ".zig-cache")) continue;
        if (entry.kind == .file and isTextExtension(entry.name)) {
            const content = dir.readFileAlloc(io, entry.name, alloc, .unlimited) catch continue;
            defer alloc.free(content);
            if (std.mem.indexOf(u8, content, query) != null) {
                try out.appendSlice(alloc, entry.name);
                try out.appendSlice(alloc, "\n");
            }
        } else if (entry.kind == .directory and depth < max_depth) {
            const sub = dir.openDir(io, entry.name, .{ .iterate = true }) catch continue;
            defer sub.close(io);
            try searchInDir(alloc, io, sub, query, depth + 1, max_depth, out);
        }
    }
}

pub const SearchTool = struct {
    name_str: []const u8 = "search",
    desc_str: []const u8 = "Busca texto en archivos del workspace. Argumentos: {\"query\": \"texto_a_buscar\", \"path\": \".\"}",
    params_str: []const u8 = "{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"},\"path\":{\"type\":\"string\"}},\"required\":[\"query\"]}",

    pub fn name(ptr: *anyopaque) []const u8 {
        const self: *SearchTool = @alignCast(@ptrCast(ptr));
        return self.name_str;
    }
    pub fn description(ptr: *anyopaque) []const u8 {
        const self: *SearchTool = @alignCast(@ptrCast(ptr));
        return self.desc_str;
    }
    pub fn parameters_json(ptr: *anyopaque) []const u8 {
        const self: *SearchTool = @alignCast(@ptrCast(ptr));
        return self.params_str;
    }
    pub fn execute(_: *anyopaque, io: std.Io, args_json: []const u8, allocator: std.mem.Allocator) anyerror!zagent.ToolResult {
        const Args = struct { query: []const u8, path: []const u8 = "." };
        const parsed = std.json.parseFromSlice(Args, allocator, args_json, .{ .ignore_unknown_fields = true }) catch |err| {
            var msg = std.ArrayList(u8).empty;
            defer msg.deinit(allocator);
            try msg.print(allocator, "Error al parsear argumentos: {s}", .{@errorName(err)});
            return zagent.ToolResult{ .content = try msg.toOwnedSlice(allocator), .is_error = true };
        };
        defer parsed.deinit();

        const query = parsed.value.query;
        const path = parsed.value.path;
        validatePath(path) catch {
            return zagent.ToolResult{
                .content = try allocator.dupe(u8, "Error de seguridad: path no permitido"),
                .is_error = true,
            };
        }
        ;

        var results = std.ArrayList(u8).empty;
        defer results.deinit(allocator);

        const dir = std.Io.Dir.cwd();
        try searchInDir(allocator, io, dir, query, 0, 2, &results);

        if (results.items.len == 0) {
            return zagent.ToolResult{ .content = try allocator.dupe(u8, "(sin resultados)"), .is_error = false };
        }
        return zagent.ToolResult{ .content = try allocator.dupe(u8, results.items), .is_error = false };
    }
    pub fn tool(self: *SearchTool) zagent.Tool {
        return .{
            .ptr = self,
            .vtable = &zagent.Tool.VTable{
                .name = name,
                .description = description,
                .parameters_json = parameters_json,
                .execute = execute,
            },
        };
    }
};

// ---------------------------------------------------------------------------
// Tool: memory_read
// ---------------------------------------------------------------------------
pub const MemoryReadTool = struct {
    name_str: []const u8 = "memory_read",
    desc_str: []const u8 = "Recupera entradas relevantes de la memoria del agente. Argumentos: {\"query\": \"texto\", \"top_k\": 3}",
    params_str: []const u8 = "{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"},\"top_k\":{\"type\":\"integer\"}},\"required\":[\"query\"]}",
    store: ?*zagent.memory.MemoryStore = null,

    pub fn name(ptr: *anyopaque) []const u8 {
        const self: *MemoryReadTool = @alignCast(@ptrCast(ptr));
        return self.name_str;
    }
    pub fn description(ptr: *anyopaque) []const u8 {
        const self: *MemoryReadTool = @alignCast(@ptrCast(ptr));
        return self.desc_str;
    }
    pub fn parameters_json(ptr: *anyopaque) []const u8 {
        const self: *MemoryReadTool = @alignCast(@ptrCast(ptr));
        return self.params_str;
    }
    pub fn execute(ptr: *anyopaque, _: std.Io, args_json: []const u8, allocator: std.mem.Allocator) anyerror!zagent.ToolResult {
        const self: *MemoryReadTool = @alignCast(@ptrCast(ptr));
        const Args = struct { query: []const u8, top_k: i64 = 3 };
        const parsed = std.json.parseFromSlice(Args, allocator, args_json, .{ .ignore_unknown_fields = true }) catch |err| {
            var msg = std.ArrayList(u8).empty;
            defer msg.deinit(allocator);
            try msg.print(allocator, "Error al parsear argumentos: {s}", .{@errorName(err)});
            return zagent.ToolResult{ .content = try msg.toOwnedSlice(allocator), .is_error = true };
        };
        defer parsed.deinit();

        const store = self.store orelse return zagent.ToolResult{
            .content = try allocator.dupe(u8, "Error: memoria no inicializada"),
            .is_error = true,
        };

        var results = std.ArrayList([]const u8).empty;
        defer results.deinit(allocator);
        const top_k: usize = if (parsed.value.top_k > 0) @intCast(parsed.value.top_k) else 3;
        // searchText genera el embedding de la query (híbrida real: coseno + BM25)
        try store.searchText(parsed.value.query, top_k, &results);

        var out = std.ArrayList(u8).empty;
        defer out.deinit(allocator);
        for (results.items) |r| {
            try out.appendSlice(allocator, r);
            try out.appendSlice(allocator, "\n");
        }
        return zagent.ToolResult{ .content = try allocator.dupe(u8, out.items), .is_error = false };
    }
    pub fn tool(self: *MemoryReadTool) zagent.Tool {
        return .{
            .ptr = self,
            .vtable = &zagent.Tool.VTable{
                .name = name,
                .description = description,
                .parameters_json = parameters_json,
                .execute = execute,
            },
        };
    }
};

// ---------------------------------------------------------------------------
// Tool: memory_write
// ---------------------------------------------------------------------------
pub const MemoryWriteTool = struct {
    name_str: []const u8 = "memory_write",
    desc_str: []const u8 = "Guarda texto en la memoria del agente. Argumentos: {\"text\": \"contenido\"}",
    params_str: []const u8 = "{\"type\":\"object\",\"properties\":{\"text\":{\"type\":\"string\"}},\"required\":[\"text\"]}",
    store: ?*zagent.memory.MemoryStore = null,

    pub fn name(ptr: *anyopaque) []const u8 {
        const self: *MemoryWriteTool = @alignCast(@ptrCast(ptr));
        return self.name_str;
    }
    pub fn description(ptr: *anyopaque) []const u8 {
        const self: *MemoryWriteTool = @alignCast(@ptrCast(ptr));
        return self.desc_str;
    }
    pub fn parameters_json(ptr: *anyopaque) []const u8 {
        const self: *MemoryWriteTool = @alignCast(@ptrCast(ptr));
        return self.params_str;
    }
    pub fn execute(ptr: *anyopaque, _: std.Io, args_json: []const u8, allocator: std.mem.Allocator) anyerror!zagent.ToolResult {
        const self: *MemoryWriteTool = @alignCast(@ptrCast(ptr));
        const Args = struct { text: []const u8 };
        const parsed = std.json.parseFromSlice(Args, allocator, args_json, .{ .ignore_unknown_fields = true }) catch |err| {
            var msg = std.ArrayList(u8).empty;
            defer msg.deinit(allocator);
            try msg.print(allocator, "Error al parsear argumentos: {s}", .{@errorName(err)});
            return zagent.ToolResult{ .content = try msg.toOwnedSlice(allocator), .is_error = true };
        };
        defer parsed.deinit();

        const store = self.store orelse return zagent.ToolResult{
            .content = try allocator.dupe(u8, "Error: memoria no inicializada"),
            .is_error = true,
        };

        try store.addText(parsed.value.text);
        return zagent.ToolResult{
            .content = try allocator.dupe(u8, "Entrada guardada en memoria"),
            .is_error = false,
        };
    }
    pub fn tool(self: *MemoryWriteTool) zagent.Tool {
        return .{
            .ptr = self,
            .vtable = &zagent.Tool.VTable{
                .name = name,
                .description = description,
                .parameters_json = parameters_json,
                .execute = execute,
            },
        };
    }
};

// ---------------------------------------------------------------------------
// Tool: delegate
// ---------------------------------------------------------------------------
pub const DelegateTool = struct {
    name_str: []const u8 = "delegate",
    desc_str: []const u8 = "Delega una subtarea a un sub-agente (requiere permiso explícito). Argumentos: {\"goal\": \"objetivo\"}",
    params_str: []const u8 = "{\"type\":\"object\",\"properties\":{\"goal\":{\"type\":\"string\"}},\"required\":[\"goal\"]}",

    pub fn name(ptr: *anyopaque) []const u8 {
        const self: *DelegateTool = @alignCast(@ptrCast(ptr));
        return self.name_str;
    }
    pub fn description(ptr: *anyopaque) []const u8 {
        const self: *DelegateTool = @alignCast(@ptrCast(ptr));
        return self.desc_str;
    }
    pub fn parameters_json(ptr: *anyopaque) []const u8 {
        const self: *DelegateTool = @alignCast(@ptrCast(ptr));
        return self.params_str;
    }
    pub fn execute(_: *anyopaque, _: std.Io, _: []const u8, allocator: std.mem.Allocator) anyerror!zagent.ToolResult {
        return zagent.ToolResult{
            .content = try allocator.dupe(u8, "Error: delegación denegada por deny-by-default (Art. IV.5). Usa el agente principal."),
            .is_error = true,
        };
    }
    pub fn tool(self: *DelegateTool) zagent.Tool {
        return .{
            .ptr = self,
            .vtable = &zagent.Tool.VTable{
                .name = name,
                .description = description,
                .parameters_json = parameters_json,
                .execute = execute,
            },
        };
    }
};

// ---------------------------------------------------------------------------
// Tool: finish
// ---------------------------------------------------------------------------
pub const FinishTool = struct {
    name_str: []const u8 = "finish",
    desc_str: []const u8 = "Finaliza el bucle agéntico indicando que se ha logrado el objetivo. Argumentos: {}",
    params_str: []const u8 = "{\"type\":\"object\",\"properties\":{}}",

    pub fn name(ptr: *anyopaque) []const u8 {
        const self: *FinishTool = @alignCast(@ptrCast(ptr));
        return self.name_str;
    }
    pub fn description(ptr: *anyopaque) []const u8 {
        const self: *FinishTool = @alignCast(@ptrCast(ptr));
        return self.desc_str;
    }
    pub fn parameters_json(ptr: *anyopaque) []const u8 {
        const self: *FinishTool = @alignCast(@ptrCast(ptr));
        return self.params_str;
    }
    pub fn execute(_: *anyopaque, _: std.Io, _: []const u8, allocator: std.mem.Allocator) anyerror!zagent.ToolResult {
        return zagent.ToolResult{
            .content = try allocator.dupe(u8, "Objetivo finalizado correctamente"),
            .is_error = false,
        };
    }
    pub fn tool(self: *FinishTool) zagent.Tool {
        return .{
            .ptr = self,
            .vtable = &zagent.Tool.VTable{
                .name = name,
                .description = description,
                .parameters_json = parameters_json,
                .execute = execute,
            },
        };
    }
};

// ---------------------------------------------------------------------------
// Tests (viven en zagent.zig como root de test; aquí no hay tests para
// evitar import relativos fuera del module path cuando tools.zig es root).
// ---------------------------------------------------------------------------
