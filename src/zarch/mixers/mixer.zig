// mixer.zig — trait comptime para mixers subcuadráticos y atención
const std = @import("std");

/// Cada mixer implementa init, forward, y tiene state para recurrencia.
/// forward(seq_len, dim) -> output[seq_len * dim]
pub fn Mixer(comptime State: type, comptime forwardFn: fn (state: *State, inp: []const f32, out: []f32, seq_len: usize, dim: usize, pos: usize) void) type {
    return struct {
        state: State,
        dim: usize,

        const Self = @This();

        pub fn init(alloc: std.mem.Allocator, dim_: usize) !Self {
            return .{ .state = try State.init(alloc, dim_), .dim = dim_ };
        }

        pub fn deinit(self: *Self, alloc: std.mem.Allocator) void {
            self.state.deinit(alloc);
        }

        pub fn forward(self: *Self, inp: []const f32, out: []f32, seq_len: usize, pos: usize) void {
            forwardFn(&self.state, inp, out, seq_len, self.dim, pos);
        }
    };
}
