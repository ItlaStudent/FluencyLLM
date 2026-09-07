const std = @import("std");
const platform = @import("platform");

test "saxpy.spv magic bytes" {
    const alloc = std.testing.allocator;
    var threaded = std.Io.Threaded.init(alloc, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const data = std.Io.Dir.cwd().readFileAlloc(io, "zig-out/saxpy.spv", alloc, .unlimited) catch |err| switch (err) {
        error.FileNotFound => {
            std.log.warn("saxpy.spv no encontrado; corre `zig build spv` primero", .{});
            return error.SkipZigTest;
        },
        else => |e| return e,
    };
    defer alloc.free(data);

    if (data.len < 4) {
        std.log.warn("saxpy.spv vacio o incompleto; saltando", .{});
        return error.SkipZigTest;
    }
    const spv_magic = [_]u8{ 0x03, 0x02, 0x23, 0x07 };
    try std.testing.expectEqualSlices(u8, &spv_magic, data[0..4]);
}

test "detect gpu" {
    const alloc = std.testing.allocator;
    var ctx = platform.vulkan.VulkanContext.init(alloc) catch |err| switch (err) {
        error.NoVulkan => {
            std.log.warn("No hay Vulkan disponible; saltando test GPU", .{});
            return error.SkipZigTest;
        },
        else => |e| return e,
    };
    defer ctx.deinit();

    const info = ctx.gpuInfo();
    const name_len = std.mem.indexOfScalar(u8, &info.device_name, 0) orelse info.device_name.len;
    const name = info.device_name[0..name_len];
    std.log.info("GPU detectada: {s} (queue family {d})", .{ name, info.queue_family_index });
    try std.testing.expect(name.len > 0);
}
