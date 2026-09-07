// saxpy.zig — kernel SAXPY para Vulkan compute (SPIR-V)
// Compila con: zig build-obj -target spirv64-vulkan -fno-llvm -O ReleaseFast saxpy.zig

const MAX_N = 4096;

export fn saxpy(
    n: u32,
    a: f32,
    x: *addrspace(.global) const [MAX_N]f32,
    y: *addrspace(.global) const [MAX_N]f32,
    r: *addrspace(.global) [MAX_N]f32,
) callconv(.spirv_kernel) void {
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        r[i] = a * x[i] + y[i];
    }
}
