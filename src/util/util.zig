// util — utilidades base del proyecto
const std = @import("std");

pub const json = @import("json.zig");
pub const bytes = @import("bytes.zig");
pub const hash = @import("hash.zig");
pub const rng = @import("rng.zig");
pub const crypto = @import("crypto.zig");
pub const parallel = @import("parallel.zig");

test "util submodules" {
    _ = json;
    _ = bytes;
    _ = hash;
    _ = rng;
    _ = crypto;
    _ = parallel;
}
