// platform — shims de SO/driver + terminal cruda
const std = @import("std");

pub const tty = @import("tty.zig");
pub const vulkan = @import("vulkan/device.zig");
