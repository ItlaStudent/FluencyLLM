// bf16 emulado: u16 + conversión f32 <-> bf16
const std = @import("std");

pub fn toF32(x: u16) f32 {
    const u: u32 = @as(u32, x) << 16;
    return @bitCast(u);
}

pub fn fromF32(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u + 0x8000) >> 16);
}
