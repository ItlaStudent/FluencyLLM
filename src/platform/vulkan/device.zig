// device.zig — shim Vulkan mínimo para detectar GPU y crear device+cola compute.
// Windows-only MVP: usa LoadLibraryW + GetProcAddress directamente (std.DynLib
// en Zig 0.16 Windows tiene un bug con lookup).
const std = @import("std");
const vk = @import("vk.zig");
const win = std.os.windows;
extern "kernel32" fn LoadLibraryW(lpLibFileName: [*:0]const u16) ?win.HMODULE;
extern "kernel32" fn GetProcAddress(hModule: win.HMODULE, lpProcName: [*:0]const u8) ?*const anyopaque;
extern "kernel32" fn FreeLibrary(hLibModule: win.HMODULE) win.BOOL;

pub const GpuInfo = struct {
    device_name: [256]u8,
    queue_family_index: u32,
};

pub const VulkanContext = struct {
    hmodule: win.HMODULE,
    instance: vk.VkInstance,
    physical_device: vk.VkPhysicalDevice,
    device: vk.VkDevice,
    queue: vk.VkQueue,
    queue_family_index: u32,

    vki: InstanceDispatch,
    vkd: DeviceDispatch,

    pub const InstanceDispatch = struct {
        createInstance: vk.PFN_vkCreateInstance,
        destroyInstance: vk.PFN_vkDestroyInstance,
        enumeratePhysicalDevices: vk.PFN_vkEnumeratePhysicalDevices,
        getPhysicalDeviceProperties: vk.PFN_vkGetPhysicalDeviceProperties,
        getPhysicalDeviceQueueFamilyProperties: vk.PFN_vkGetPhysicalDeviceQueueFamilyProperties,
        getPhysicalDeviceMemoryProperties: vk.PFN_vkGetPhysicalDeviceMemoryProperties,
        createDevice: vk.PFN_vkCreateDevice,
        getPhysicalDeviceFeatures: vk.PFN_vkGetPhysicalDeviceFeatures,
    };

    pub const DeviceDispatch = struct {
        destroyDevice: vk.PFN_vkDestroyDevice,
        getDeviceQueue: vk.PFN_vkGetDeviceQueue,
        createBuffer: vk.PFN_vkCreateBuffer,
        destroyBuffer: vk.PFN_vkDestroyBuffer,
        getBufferMemoryRequirements: vk.PFN_vkGetBufferMemoryRequirements,
        allocateMemory: vk.PFN_vkAllocateMemory,
        freeMemory: vk.PFN_vkFreeMemory,
        bindBufferMemory: vk.PFN_vkBindBufferMemory,
        mapMemory: vk.PFN_vkMapMemory,
        unmapMemory: vk.PFN_vkUnmapMemory,
        createDescriptorSetLayout: vk.PFN_vkCreateDescriptorSetLayout,
        destroyDescriptorSetLayout: vk.PFN_vkDestroyDescriptorSetLayout,
        createDescriptorPool: vk.PFN_vkCreateDescriptorPool,
        destroyDescriptorPool: vk.PFN_vkDestroyDescriptorPool,
        allocateDescriptorSets: vk.PFN_vkAllocateDescriptorSets,
        updateDescriptorSets: vk.PFN_vkUpdateDescriptorSets,
        createShaderModule: vk.PFN_vkCreateShaderModule,
        destroyShaderModule: vk.PFN_vkDestroyShaderModule,
        createPipelineLayout: vk.PFN_vkCreatePipelineLayout,
        destroyPipelineLayout: vk.PFN_vkDestroyPipelineLayout,
        createComputePipelines: vk.PFN_vkCreateComputePipelines,
        destroyPipeline: vk.PFN_vkDestroyPipeline,
        createCommandPool: vk.PFN_vkCreateCommandPool,
        destroyCommandPool: vk.PFN_vkDestroyCommandPool,
        allocateCommandBuffers: vk.PFN_vkAllocateCommandBuffers,
        freeCommandBuffers: vk.PFN_vkFreeCommandBuffers,
        beginCommandBuffer: vk.PFN_vkBeginCommandBuffer,
        endCommandBuffer: vk.PFN_vkEndCommandBuffer,
        cmdBindPipeline: vk.PFN_vkCmdBindPipeline,
        cmdBindDescriptorSets: vk.PFN_vkCmdBindDescriptorSets,
        cmdDispatch: vk.PFN_vkCmdDispatch,
        createFence: vk.PFN_vkCreateFence,
        destroyFence: vk.PFN_vkDestroyFence,
        queueSubmit: vk.PFN_vkQueueSubmit,
        queueWaitIdle: vk.PFN_vkQueueWaitIdle,
        deviceWaitIdle: vk.PFN_vkDeviceWaitIdle,
        waitForFences: vk.PFN_vkWaitForFences,
        resetFences: vk.PFN_vkResetFences,
        getDeviceProcAddr: vk.PFN_vkGetDeviceProcAddr,
    };

    pub const Error = error{
        NoVulkan,
        OutOfMemory,
        IncompatibleDriver,
        DeviceLost,
        Unknown,
        InvalidUtf8,
    };

    pub fn init(alloc: std.mem.Allocator) Error!VulkanContext {
        _ = alloc;

        const wide_name = try std.unicode.utf8ToUtf16LeAllocZ(std.heap.page_allocator, "vulkan-1.dll");
        defer std.heap.page_allocator.free(wide_name);

        const hmodule = LoadLibraryW(wide_name) orelse {
            std.log.warn("No se pudo cargar vulkan-1.dll", .{});
            return error.NoVulkan;
        };
        errdefer _ = FreeLibrary(hmodule);

        const getInstanceProcAddr = @as(vk.PFN_vkGetInstanceProcAddr, @ptrCast(GetProcAddress(hmodule, "vkGetInstanceProcAddr") orelse return error.NoVulkan));

        var vki: InstanceDispatch = undefined;
        vki.createInstance = @ptrCast(getInstanceProcAddr(0, "vkCreateInstance") orelse return error.NoVulkan);
        vki.destroyInstance = @ptrCast(getInstanceProcAddr(0, "vkDestroyInstance") orelse return error.NoVulkan);
        vki.enumeratePhysicalDevices = @ptrCast(getInstanceProcAddr(0, "vkEnumeratePhysicalDevices") orelse return error.NoVulkan);
        vki.getPhysicalDeviceProperties = @ptrCast(getInstanceProcAddr(0, "vkGetPhysicalDeviceProperties") orelse return error.NoVulkan);
        vki.getPhysicalDeviceQueueFamilyProperties = @ptrCast(getInstanceProcAddr(0, "vkGetPhysicalDeviceQueueFamilyProperties") orelse return error.NoVulkan);
        vki.getPhysicalDeviceMemoryProperties = @ptrCast(getInstanceProcAddr(0, "vkGetPhysicalDeviceMemoryProperties") orelse return error.NoVulkan);
        vki.createDevice = @ptrCast(getInstanceProcAddr(0, "vkCreateDevice") orelse return error.NoVulkan);
        vki.getPhysicalDeviceFeatures = @ptrCast(getInstanceProcAddr(0, "vkGetPhysicalDeviceFeatures") orelse return error.NoVulkan);

        const app_info = vk.VkApplicationInfo{};
        const create_info = vk.VkInstanceCreateInfo{
            .pApplicationInfo = &app_info,
        };
        var instance: vk.VkInstance = undefined;
        try checkResult(vki.createInstance(&create_info, null, &instance));
        errdefer vki.destroyInstance(instance, null);

        var device_count: u32 = 0;
        try checkResult(vki.enumeratePhysicalDevices(instance, &device_count, null));
        if (device_count == 0) return error.NoVulkan;

        const devices = std.heap.page_allocator.alloc(vk.VkPhysicalDevice, device_count) catch return error.OutOfMemory;
        defer std.heap.page_allocator.free(devices);
        try checkResult(vki.enumeratePhysicalDevices(instance, &device_count, @ptrCast(devices.ptr)));

        var selected_pd: vk.VkPhysicalDevice = 0;
        var selected_family: u32 = 0;

        for (devices) |pd| {
            var props: vk.VkPhysicalDeviceProperties = undefined;
            vki.getPhysicalDeviceProperties(pd, &props);

            var family_count: u32 = 0;
            vki.getPhysicalDeviceQueueFamilyProperties(pd, &family_count, null);
            const families = std.heap.page_allocator.alloc(vk.VkPhysicalDeviceQueueFamilyProperties, family_count) catch return error.OutOfMemory;
            defer std.heap.page_allocator.free(families);
            vki.getPhysicalDeviceQueueFamilyProperties(pd, &family_count, families.ptr);

            var compute_family: u32 = 0;
            var has_compute = false;
            for (families, 0..) |f, i| {
                if (f.queueFlags & vk.VK_QUEUE_COMPUTE_BIT != 0) {
                    compute_family = @intCast(i);
                    has_compute = true;
                    break;
                }
            }
            if (!has_compute) continue;

            if (props.deviceType == .discrete_gpu) {
                selected_pd = pd;
                selected_family = compute_family;
                break;
            }
            if (selected_pd == 0) {
                selected_pd = pd;
                selected_family = compute_family;
            }
        }

        if (selected_pd == 0) return error.NoVulkan;

        const queue_priority: f32 = 1.0;
        const queue_create_info = vk.VkDeviceQueueCreateInfo{
            .queueFamilyIndex = selected_family,
            .pQueuePriorities = &queue_priority,
        };
        const device_create_info = vk.VkDeviceCreateInfo{
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queue_create_info,
        };
        const destroyDevice: vk.PFN_vkDestroyDevice = @ptrCast(getInstanceProcAddr(instance, "vkDestroyDevice") orelse return error.NoVulkan);

        var device: vk.VkDevice = undefined;
        try checkResult(vki.createDevice(selected_pd, &device_create_info, null, &device));
        errdefer destroyDevice(device, null);

        var queue: vk.VkQueue = undefined;
        const getDeviceQueue: vk.PFN_vkGetDeviceQueue = @ptrCast(getInstanceProcAddr(instance, "vkGetDeviceQueue") orelse return error.NoVulkan);
        getDeviceQueue(device, selected_family, 0, &queue);

        const gdpa: vk.PFN_vkGetDeviceProcAddr = @ptrCast(getInstanceProcAddr(instance, "vkGetDeviceProcAddr") orelse return error.NoVulkan);

        var vkd: DeviceDispatch = undefined;
        vkd.destroyDevice = @ptrCast(getInstanceProcAddr(instance, "vkDestroyDevice") orelse return error.NoVulkan);
        vkd.getDeviceQueue = getDeviceQueue;
        vkd.createBuffer = @ptrCast(gdpa(device, "vkCreateBuffer") orelse return error.NoVulkan);
        vkd.destroyBuffer = @ptrCast(gdpa(device, "vkDestroyBuffer") orelse return error.NoVulkan);
        vkd.getBufferMemoryRequirements = @ptrCast(gdpa(device, "vkGetBufferMemoryRequirements") orelse return error.NoVulkan);
        vkd.allocateMemory = @ptrCast(gdpa(device, "vkAllocateMemory") orelse return error.NoVulkan);
        vkd.freeMemory = @ptrCast(gdpa(device, "vkFreeMemory") orelse return error.NoVulkan);
        vkd.bindBufferMemory = @ptrCast(gdpa(device, "vkBindBufferMemory") orelse return error.NoVulkan);
        vkd.mapMemory = @ptrCast(gdpa(device, "vkMapMemory") orelse return error.NoVulkan);
        vkd.unmapMemory = @ptrCast(gdpa(device, "vkUnmapMemory") orelse return error.NoVulkan);
        vkd.createDescriptorSetLayout = @ptrCast(gdpa(device, "vkCreateDescriptorSetLayout") orelse return error.NoVulkan);
        vkd.destroyDescriptorSetLayout = @ptrCast(gdpa(device, "vkDestroyDescriptorSetLayout") orelse return error.NoVulkan);
        vkd.createDescriptorPool = @ptrCast(gdpa(device, "vkCreateDescriptorPool") orelse return error.NoVulkan);
        vkd.destroyDescriptorPool = @ptrCast(gdpa(device, "vkDestroyDescriptorPool") orelse return error.NoVulkan);
        vkd.allocateDescriptorSets = @ptrCast(gdpa(device, "vkAllocateDescriptorSets") orelse return error.NoVulkan);
        vkd.updateDescriptorSets = @ptrCast(gdpa(device, "vkUpdateDescriptorSets") orelse return error.NoVulkan);
        vkd.createShaderModule = @ptrCast(gdpa(device, "vkCreateShaderModule") orelse return error.NoVulkan);
        vkd.destroyShaderModule = @ptrCast(gdpa(device, "vkDestroyShaderModule") orelse return error.NoVulkan);
        vkd.createPipelineLayout = @ptrCast(gdpa(device, "vkCreatePipelineLayout") orelse return error.NoVulkan);
        vkd.destroyPipelineLayout = @ptrCast(gdpa(device, "vkDestroyPipelineLayout") orelse return error.NoVulkan);
        vkd.createComputePipelines = @ptrCast(gdpa(device, "vkCreateComputePipelines") orelse return error.NoVulkan);
        vkd.destroyPipeline = @ptrCast(gdpa(device, "vkDestroyPipeline") orelse return error.NoVulkan);
        vkd.createCommandPool = @ptrCast(gdpa(device, "vkCreateCommandPool") orelse return error.NoVulkan);
        vkd.destroyCommandPool = @ptrCast(gdpa(device, "vkDestroyCommandPool") orelse return error.NoVulkan);
        vkd.allocateCommandBuffers = @ptrCast(gdpa(device, "vkAllocateCommandBuffers") orelse return error.NoVulkan);
        vkd.freeCommandBuffers = @ptrCast(gdpa(device, "vkFreeCommandBuffers") orelse return error.NoVulkan);
        vkd.beginCommandBuffer = @ptrCast(gdpa(device, "vkBeginCommandBuffer") orelse return error.NoVulkan);
        vkd.endCommandBuffer = @ptrCast(gdpa(device, "vkEndCommandBuffer") orelse return error.NoVulkan);
        vkd.cmdBindPipeline = @ptrCast(gdpa(device, "vkCmdBindPipeline") orelse return error.NoVulkan);
        vkd.cmdBindDescriptorSets = @ptrCast(gdpa(device, "vkCmdBindDescriptorSets") orelse return error.NoVulkan);
        vkd.cmdDispatch = @ptrCast(gdpa(device, "vkCmdDispatch") orelse return error.NoVulkan);
        vkd.createFence = @ptrCast(gdpa(device, "vkCreateFence") orelse return error.NoVulkan);
        vkd.destroyFence = @ptrCast(gdpa(device, "vkDestroyFence") orelse return error.NoVulkan);
        vkd.queueSubmit = @ptrCast(gdpa(device, "vkQueueSubmit") orelse return error.NoVulkan);
        vkd.queueWaitIdle = @ptrCast(gdpa(device, "vkQueueWaitIdle") orelse return error.NoVulkan);
        vkd.deviceWaitIdle = @ptrCast(gdpa(device, "vkDeviceWaitIdle") orelse return error.NoVulkan);
        vkd.waitForFences = @ptrCast(gdpa(device, "vkWaitForFences") orelse return error.NoVulkan);
        vkd.resetFences = @ptrCast(gdpa(device, "vkResetFences") orelse return error.NoVulkan);
        vkd.getDeviceProcAddr = gdpa;

        return .{
            .hmodule = hmodule,
            .instance = instance,
            .physical_device = selected_pd,
            .device = device,
            .queue = queue,
            .queue_family_index = selected_family,
            .vki = vki,
            .vkd = vkd,
        };
    }

    pub fn deinit(self: *VulkanContext) void {
        self.vkd.destroyDevice(self.device, null);
        self.vki.destroyInstance(self.instance, null);
        _ = FreeLibrary(self.hmodule);
        self.* = undefined;
    }

    pub fn gpuInfo(self: *const VulkanContext) GpuInfo {
        var props: vk.VkPhysicalDeviceProperties = undefined;
        self.vki.getPhysicalDeviceProperties(self.physical_device, &props);
        return .{
            .device_name = props.deviceName,
            .queue_family_index = self.queue_family_index,
        };
    }

    fn checkResult(result: vk.VkResult) Error!void {
        switch (result) {
            .success => {},
            .error_out_of_host_memory, .error_out_of_device_memory, .error_out_of_pool_memory => return error.OutOfMemory,
            .error_initialization_failed, .error_incompatible_driver => return error.IncompatibleDriver,
            .error_device_lost => return error.DeviceLost,
            else => return error.Unknown,
        }
    }
};
