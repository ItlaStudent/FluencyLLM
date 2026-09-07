// trajectory.zig — captura de trayectorias ReAct para entrenamiento de tool-use
const std = @import("std");

/// Paso individual dentro de una trayectoria ReAct.
pub const Step = union(enum) {
    thought: []const u8,
    action: struct { name: []const u8, arguments_json: []const u8 },
    observation: struct { content: []const u8, is_error: bool },
    answer: []const u8,
    budget_exceeded: void,
    interrupted: void,
};

/// Trayectoria completa de una ejecución del agente.
pub const Trajectory = struct {
    steps: std.ArrayList(Step),
    prompt: []const u8,
    model: []const u8,

    pub fn init(prompt: []const u8, model: []const u8) Trajectory {
        return .{
            .steps = std.ArrayList(Step).empty,
            .prompt = prompt,
            .model = model,
        };
    }

    pub fn deinit(self: *Trajectory, alloc: std.mem.Allocator) void {
        for (self.steps.items) |s| {
            switch (s) {
                .thought => |t| alloc.free(t),
                .action => |a| {
                    alloc.free(a.name);
                    alloc.free(a.arguments_json);
                },
                .observation => |o| alloc.free(o.content),
                .answer => |a| alloc.free(a),
                .budget_exceeded, .interrupted => {},
            }
        }
        self.steps.deinit(alloc);
    }

    pub fn addThought(self: *Trajectory, alloc: std.mem.Allocator, text: []const u8) !void {
        const copy = try alloc.dupe(u8, text);
        errdefer alloc.free(copy);
        try self.steps.append(alloc, .{ .thought = copy });
    }

    pub fn addAction(self: *Trajectory, alloc: std.mem.Allocator, name: []const u8, args: []const u8) !void {
        const n = try alloc.dupe(u8, name);
        errdefer alloc.free(n);
        const a = try alloc.dupe(u8, args);
        errdefer alloc.free(a);
        try self.steps.append(alloc, .{ .action = .{ .name = n, .arguments_json = a } });
    }

    pub fn addObservation(self: *Trajectory, alloc: std.mem.Allocator, content: []const u8, is_error: bool) !void {
        const c = try alloc.dupe(u8, content);
        errdefer alloc.free(c);
        try self.steps.append(alloc, .{ .observation = .{ .content = c, .is_error = is_error } });
    }

    pub fn addAnswer(self: *Trajectory, alloc: std.mem.Allocator, text: []const u8) !void {
        const copy = try alloc.dupe(u8, text);
        errdefer alloc.free(copy);
        try self.steps.append(alloc, .{ .answer = copy });
    }

    pub fn addBudgetExceeded(self: *Trajectory, alloc: std.mem.Allocator) !void {
        try self.steps.append(alloc, .{ .budget_exceeded = {} });
    }

    pub fn addInterrupted(self: *Trajectory, alloc: std.mem.Allocator) !void {
        try self.steps.append(alloc, .{ .interrupted = {} });
    }

    /// Serializa la trayectoria a JSON simple (formato propio ZTRAJ).
    pub fn serializeTo(self: Trajectory, out: *std.ArrayList(u8), alloc: std.mem.Allocator) !void {
        try out.appendSlice(alloc, "{\"prompt\":");
        try writeJsonString(out, alloc, self.prompt);
        try out.appendSlice(alloc, ",\"model\":");
        try writeJsonString(out, alloc, self.model);
        try out.appendSlice(alloc, ",\"steps\":[");

        for (self.steps.items, 0..) |step, i| {
            if (i != 0) try out.appendSlice(alloc, ",");
            try out.appendSlice(alloc, "{");
            switch (step) {
                .thought => |t| {
                    try out.appendSlice(alloc, "\"type\":\"thought\",\"text\":");
                    try writeJsonString(out, alloc, t);
                },
                .action => |a| {
                    try out.appendSlice(alloc, "\"type\":\"action\",\"name\":");
                    try writeJsonString(out, alloc, a.name);
                    try out.appendSlice(alloc, ",\"arguments\":");
                    try writeJsonString(out, alloc, a.arguments_json);
                },
                .observation => |o| {
                    try out.appendSlice(alloc, "\"type\":\"observation\",\"content\":");
                    try writeJsonString(out, alloc, o.content);
                    try out.appendSlice(alloc, ",\"is_error\":");
                    try out.appendSlice(alloc, if (o.is_error) "true" else "false");
                },
                .answer => |a| {
                    try out.appendSlice(alloc, "\"type\":\"answer\",\"text\":");
                    try writeJsonString(out, alloc, a);
                },
                .budget_exceeded => try out.appendSlice(alloc, "\"type\":\"budget_exceeded\""),
                .interrupted => try out.appendSlice(alloc, "\"type\":\"interrupted\""),
            }
            try out.appendSlice(alloc, "}");
        }
        try out.appendSlice(alloc, "]}");
    }
};

fn writeJsonString(out: *std.ArrayList(u8), alloc: std.mem.Allocator, s: []const u8) !void {
    try out.append(alloc, '"');
    for (s) |c| {
        switch (c) {
            '"' => try out.appendSlice(alloc, "\\\""),
            '\\' => try out.appendSlice(alloc, "\\\\"),
            '\n' => try out.appendSlice(alloc, "\\n"),
            '\r' => try out.appendSlice(alloc, "\\r"),
            '\t' => try out.appendSlice(alloc, "\\t"),
            else => try out.append(alloc, c),
        }
    }
    try out.append(alloc, '"');
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "Trajectory round-trip serialize" {
    const alloc = std.testing.allocator;
    var traj = Trajectory.init("Hola", "test-model");
    defer traj.deinit(alloc);

    try traj.addThought(alloc, "Pienso");
    try traj.addAction(alloc, "fs_read", "{\"path\":\"a\"}");
    try traj.addObservation(alloc, "contenido", false);
    try traj.addAnswer(alloc, "Listo");

    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(alloc);
    try traj.serializeTo(&buf, alloc);

    const s = buf.items;
    try std.testing.expect(std.mem.indexOf(u8, s, "\"type\":\"thought\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, s, "\"type\":\"action\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, s, "\"type\":\"observation\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, s, "\"type\":\"answer\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, s, "\"prompt\":\"Hola\"") != null);
}

test "Trajectory budget_exceeded and interrupted" {
    const alloc = std.testing.allocator;
    var traj = Trajectory.init("x", "m");
    defer traj.deinit(alloc);

    try traj.addBudgetExceeded(alloc);
    try traj.addInterrupted(alloc);

    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(alloc);
    try traj.serializeTo(&buf, alloc);

    const s = buf.items;
    try std.testing.expect(std.mem.indexOf(u8, s, "\"type\":\"budget_exceeded\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, s, "\"type\":\"interrupted\"") != null);
}
