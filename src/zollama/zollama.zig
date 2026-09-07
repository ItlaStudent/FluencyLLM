// zollama — cliente HTTP Ollama
const std = @import("std");

pub const OllamaClient = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    host_url: []const u8,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, host_url: []const u8) OllamaClient {
        return .{
            .allocator = allocator,
            .io = io,
            .host_url = host_url,
        };
    }

    /// Descarga un modelo en Ollama
    pub fn pullModel(self: *OllamaClient, model_name: []const u8) !void {
        var client = std.http.Client{ .allocator = self.allocator, .io = self.io };
        defer client.deinit();

        var aw = std.Io.Writer.Allocating.init(self.allocator);
        defer aw.deinit();
        try aw.writer.print("{{\"name\":\"{s}\",\"stream\":false}}", .{model_name});
        const payload = aw.written();

        var res_writer = std.Io.Writer.Allocating.init(self.allocator);
        defer res_writer.deinit();

        var url_buf: [512]u8 = undefined;
        const url = try std.fmt.bufPrint(&url_buf, "{s}/api/pull", .{self.host_url});

        const res = try client.fetch(.{
            .location = .{ .url = url },
            .method = .POST,
            .payload = payload,
            .response_writer = &res_writer.writer,
        });

        if (res.status != .ok) {
            return error.OllamaRequestFailed;
        }
    }

    /// Genera una respuesta completa a partir de un prompt
    pub fn generate(self: *OllamaClient, model: []const u8, prompt: []const u8, out_text: *std.ArrayList(u8)) !void {
        var client = std.http.Client{ .allocator = self.allocator, .io = self.io };
        defer client.deinit();

        var aw = std.Io.Writer.Allocating.init(self.allocator);
        defer aw.deinit();
        // Escapar caracteres del prompt para JSON de forma segura y sencilla
        try aw.writer.print("{{\"model\":\"{s}\",\"prompt\":\"{s}\",\"stream\":false}}", .{ model, prompt });
        const payload = aw.written();

        var res_writer = std.Io.Writer.Allocating.init(self.allocator);
        defer res_writer.deinit();

        var url_buf: [512]u8 = undefined;
        const url = try std.fmt.bufPrint(&url_buf, "{s}/api/generate", .{self.host_url});

        const res = try client.fetch(.{
            .location = .{ .url = url },
            .method = .POST,
            .payload = payload,
            .response_writer = &res_writer.writer,
        });

        if (res.status != .ok) {
            return error.OllamaRequestFailed;
        }

        const body = res_writer.written();
        
        // Parsear respuesta JSON {"response": "..."}
        const parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, body, .{});
        defer parsed.deinit();

        if (parsed.value.object.get("response")) |v| {
            if (v == .string) {
                try out_text.appendSlice(self.allocator, v.string);
            }
        }
    }

    /// Obtiene embeddings vectoriales para un texto
    pub fn embeddings(self: *OllamaClient, model: []const u8, prompt: []const u8, out_embeddings: *std.ArrayList(f32)) !void {
        var client = std.http.Client{ .allocator = self.allocator, .io = self.io };
        defer client.deinit();

        var aw = std.Io.Writer.Allocating.init(self.allocator);
        defer aw.deinit();
        try aw.writer.print("{{\"model\":\"{s}\",\"prompt\":\"{s}\"}}", .{ model, prompt });
        const payload = aw.written();

        var res_writer = std.Io.Writer.Allocating.init(self.allocator);
        defer res_writer.deinit();

        var url_buf: [512]u8 = undefined;
        const url = try std.fmt.bufPrint(&url_buf, "{s}/api/embeddings", .{self.host_url});

        const res = try client.fetch(.{
            .location = .{ .url = url },
            .method = .POST,
            .payload = payload,
            .response_writer = &res_writer.writer,
        });

        if (res.status != .ok) {
            return error.OllamaRequestFailed;
        }

        const body = res_writer.written();

        const parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, body, .{});
        defer parsed.deinit();

        if (parsed.value.object.get("embedding")) |v| {
            if (v == .array) {
                for (v.array.items) |elem| {
                    if (elem == .float) {
                        try out_embeddings.append(self.allocator, @floatCast(elem.float));
                    } else if (elem == .integer) {
                        try out_embeddings.append(self.allocator, @floatFromInt(elem.integer));
                    }
                }
            }
        }
    }
};

// --- tests ---
test "OllamaClient init and URL formatting" {
    const alloc = std.testing.allocator;
    const io = std.testing.io; // I/O de testing
    const client = OllamaClient.init(alloc, io, "http://127.0.0.1:11434");
    try std.testing.expectEqualStrings("http://127.0.0.1:11434", client.host_url);
}
