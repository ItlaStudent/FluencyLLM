const MAX_N = 4096;

export fn saxpy(
    n: u32,
    a: f32,
    x: *addrspace(.storage_buffer) const [MAX_N]f32,
    y: *addrspace(.storage_buffer) const [MAX_N]f32,
    r: *addrspace(.storage_buffer) [MAX_N]f32,
) callconv(.spirv_kernel) void {
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        r[i] = a * x[i] + y[i];
    }
}
