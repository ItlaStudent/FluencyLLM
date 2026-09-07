// saxpy_vulkan.zig — kernel SAXPY para Vulkan compute (SPIR-V)
// Compila con: zig build-obj -target spirv64-vulkan -fno-llvm -O ReleaseFast saxpy_vulkan.zig

const MAX_N = 4096;

// Buffers de storage (descriptor set)
extern var x: *addrspace(.storage_buffer) const [MAX_N]f32;
extern var y: *addrspace(.storage_buffer) const [MAX_N]f32;
extern var r: *addrspace(.storage_buffer) [MAX_N]f32;

// Push constants
const PushConstants = extern struct {
    n: u32,
    a: f32,
};
extern var push: *addrspace(.push_constant) PushConstants;

export fn saxpy() callconv(.spirv_kernel) void {
    var i: u32 = 0;
    while (i < push.n) : (i += 1) {
        r[i] = push.a * x[i] + y[i];
    }
}
