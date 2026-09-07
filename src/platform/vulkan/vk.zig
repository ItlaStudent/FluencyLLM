// vk.zig — declaraciones Vulkan mínimas, 100% Zig (sin @cImport).
// Subconjunto para: createInstance → enumeratePhysicalDevices → createDevice → dispatch compute.
const std = @import("std");

// ── handles ──
pub const VkInstance = u64;
pub const VkPhysicalDevice = u64;
pub const VkDevice = u64;
pub const VkQueue = u64;
pub const VkFence = u64;
pub const VkCommandPool = u64;
pub const VkCommandBuffer = u64;
pub const VkBuffer = u64;
pub const VkDeviceMemory = u64;
pub const VkDescriptorSetLayout = u64;
pub const VkDescriptorPool = u64;
pub const VkDescriptorSet = u64;
pub const VkPipelineLayout = u64;
pub const VkPipeline = u64;
pub const VkShaderModule = u64;
pub const VkDeviceSize = u64;
pub const VkSampleCountFlags = u32;

// ── result ──
pub const VkResult = enum(i32) {
    success = 0,
    not_ready = 1,
    timeout = 2,
    event_set = 3,
    event_reset = 4,
    incomplete = 5,
    error_out_of_host_memory = -1,
    error_out_of_device_memory = -2,
    error_initialization_failed = -3,
    error_device_lost = -4,
    error_memory_map_failed = -5,
    error_layer_not_present = -6,
    error_extension_not_present = -7,
    error_feature_not_present = -8,
    error_incompatible_driver = -9,
    error_too_many_objects = -10,
    error_format_not_supported = -11,
    error_fragmented_pool = -12,
    error_unknown = -13,
    error_out_of_pool_memory = -1000069000,
    error_invalid_external_handle = -1000072003,
    error_fragmentation = -1000161000,
    error_invalid_opaque_capture_address = -1000257000,
    error_surface_lost_khr = -1000000000,
    error_native_window_in_use_khr = -1000000001,
    suboptimal_khr = 1000001003,
    error_out_of_date_khr = -1000001004,
    error_incompatible_display_khr = -1000003001,
    error_validation_failed_ext = -1000011001,
    error_invalid_shader_nv = -1000012000,
    error_image_usage_not_supported_khr = -1000023000,
    error_video_picture_layout_not_supported_ext = -1000023001,
    error_video_profile_operation_not_supported_ext = -1000023002,
    error_video_profile_format_not_supported_ext = -1000023003,
    error_video_profile_codec_not_supported_ext = -1000023004,
    error_video_std_version_not_supported_ext = -1000023005,
    error_not_permitted_khr = -1000174001,
    error_full_screen_exclusive_mode_lost_ext = -1000255000,
    _,
};

// ── flags ──
pub const VkQueueFlags = u32;
pub const VK_QUEUE_GRAPHICS_BIT: VkQueueFlags = 0x00000001;
pub const VK_QUEUE_COMPUTE_BIT: VkQueueFlags = 0x00000002;
pub const VK_QUEUE_TRANSFER_BIT: VkQueueFlags = 0x00000004;

pub const VkMemoryPropertyFlags = u32;
pub const VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: VkMemoryPropertyFlags = 0x00000001;
pub const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: VkMemoryPropertyFlags = 0x00000002;
pub const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: VkMemoryPropertyFlags = 0x00000004;
pub const VK_MEMORY_PROPERTY_HOST_CACHED_BIT: VkMemoryPropertyFlags = 0x00000008;

pub const VkMemoryHeapFlags = u32;
pub const VK_MEMORY_HEAP_DEVICE_LOCAL_BIT: VkMemoryHeapFlags = 0x00000001;

pub const VkBufferUsageFlags = u32;
pub const VK_BUFFER_USAGE_TRANSFER_SRC_BIT: VkBufferUsageFlags = 0x00000001;
pub const VK_BUFFER_USAGE_TRANSFER_DST_BIT: VkBufferUsageFlags = 0x00000002;
pub const VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT: VkBufferUsageFlags = 0x00000010;
pub const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: VkBufferUsageFlags = 0x00000020;

pub const VkShaderStageFlags = u32;
pub const VK_SHADER_STAGE_COMPUTE_BIT: VkShaderStageFlags = 0x00000020;

pub const VkDescriptorType = enum(u32) {
    sampler = 0,
    combined_image_sampler = 1,
    sampled_image = 2,
    storage_image = 3,
    uniform_texel_buffer = 4,
    storage_texel_buffer = 5,
    uniform_buffer = 6,
    storage_buffer = 7,
    uniform_buffer_dynamic = 8,
    storage_buffer_dynamic = 9,
    input_attachment = 10,
};

pub const VkPipelineBindPoint = enum(u32) {
    graphics = 0,
    compute = 1,
};

pub const VkCommandPoolCreateFlags = u32;
pub const VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: VkCommandPoolCreateFlags = 0x00000002;

pub const VkCommandBufferUsageFlags = u32;
pub const VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: VkCommandBufferUsageFlags = 0x00000001;

pub const VkFenceCreateFlags = u32;
pub const VK_FENCE_CREATE_SIGNALED_BIT: VkFenceCreateFlags = 0x00000001;

pub const VkDescriptorPoolCreateFlags = u32;
pub const VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT: VkDescriptorPoolCreateFlags = 0x00000001;

// ── structs ──
pub const VkApplicationInfo = extern struct {
    sType: u32 = 0,
    pNext: ?*const anyopaque = null,
    pApplicationName: [*:0]const u8 = "fluency",
    applicationVersion: u32 = 1,
    pEngineName: [*:0]const u8 = "fluency",
    engineVersion: u32 = 1,
    apiVersion: u32 = makeApiVersion(1, 0, 0, 0),
};

pub const VkInstanceCreateInfo = extern struct {
    sType: u32 = 1,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    pApplicationInfo: ?*const VkApplicationInfo = null,
    enabledLayerCount: u32 = 0,
    ppEnabledLayerNames: ?*const [*:0]const u8 = null,
    enabledExtensionCount: u32 = 0,
    ppEnabledExtensionNames: ?*const [*:0]const u8 = null,
};

pub const VkPhysicalDeviceProperties = extern struct {
    apiVersion: u32,
    driverVersion: u32,
    vendorID: u32,
    deviceID: u32,
    deviceType: VkPhysicalDeviceType,
    deviceName: [256]u8,
    pipelineCacheUUID: [16]u8,
    limits: VkPhysicalDeviceLimits,
    sparseProperties: VkPhysicalDeviceSparseProperties,
};

pub const VkPhysicalDeviceType = enum(u32) {
    other = 0,
    integrated_gpu = 1,
    discrete_gpu = 2,
    virtual_gpu = 3,
    cpu = 4,
};

pub const VkPhysicalDeviceLimits = extern struct {
    maxImageDimension1D: u32,
    maxImageDimension2D: u32,
    maxImageDimension3D: u32,
    maxImageDimensionCube: u32,
    maxImageArrayLayers: u32,
    maxTexelBufferElements: u32,
    maxUniformBufferRange: u32,
    maxStorageBufferRange: u32,
    maxPushConstantsSize: u32,
    maxMemoryAllocationCount: u32,
    maxSamplerAllocationCount: u32,
    bufferImageGranularity: VkDeviceSize,
    sparseAddressSpaceSize: VkDeviceSize,
    maxBoundDescriptorSets: u32,
    maxPerStageDescriptorSamplers: u32,
    maxPerStageDescriptorUniformBuffers: u32,
    maxPerStageDescriptorStorageBuffers: u32,
    maxPerStageDescriptorSampledImages: u32,
    maxPerStageDescriptorStorageImages: u32,
    maxPerStageDescriptorInputAttachments: u32,
    maxPerStageResources: u32,
    maxDescriptorSetSamplers: u32,
    maxDescriptorSetUniformBuffers: u32,
    maxDescriptorSetUniformBuffersDynamic: u32,
    maxDescriptorSetStorageBuffers: u32,
    maxDescriptorSetStorageBuffersDynamic: u32,
    maxDescriptorSetSampledImages: u32,
    maxDescriptorSetStorageImages: u32,
    maxDescriptorSetInputAttachments: u32,
    maxVertexInputAttributes: u32,
    maxVertexInputBindings: u32,
    maxVertexInputAttributeOffset: u32,
    maxVertexInputBindingStride: u32,
    maxVertexOutputComponents: u32,
    maxTessellationGenerationLevel: u32,
    maxTessellationPatchSize: u32,
    maxTessellationControlPerVertexInputComponents: u32,
    maxTessellationControlPerVertexOutputComponents: u32,
    maxTessellationControlPerPatchOutputComponents: u32,
    maxTessellationControlTotalOutputComponents: u32,
    maxTessellationEvaluationInputComponents: u32,
    maxTessellationEvaluationOutputComponents: u32,
    maxGeometryShaderInvocations: u32,
    maxGeometryInputComponents: u32,
    maxGeometryOutputComponents: u32,
    maxGeometryOutputVertices: u32,
    maxGeometryTotalOutputComponents: u32,
    maxFragmentInputComponents: u32,
    maxFragmentOutputAttachments: u32,
    maxFragmentDualSrcAttachments: u32,
    maxFragmentCombinedOutputResources: u32,
    maxComputeSharedMemorySize: u32,
    maxComputeWorkGroupCount: [3]u32,
    maxComputeWorkGroupInvocations: u32,
    maxComputeWorkGroupSize: [3]u32,
    subPixelPrecisionBits: u32,
    subTexelPrecisionBits: u32,
    mipmapPrecisionBits: u32,
    maxDrawIndexedIndexValue: u32,
    maxDrawIndirectCount: u32,
    maxSamplerLodBias: f32,
    maxSamplerAnisotropy: f32,
    maxViewports: u32,
    maxViewportDimensions: [2]u32,
    viewportBoundsRange: [2]f32,
    viewportSubPixelBits: u32,
    minMemoryMapAlignment: usize,
    minTexelBufferOffsetAlignment: VkDeviceSize,
    minUniformBufferOffsetAlignment: VkDeviceSize,
    minStorageBufferOffsetAlignment: VkDeviceSize,
    minTexelOffset: i32,
    maxTexelOffset: u32,
    minTexelGatherOffset: i32,
    maxTexelGatherOffset: u32,
    minInterpolationOffset: f32,
    maxInterpolationOffset: f32,
    subPixelInterpolationOffsetBits: u32,
    maxFramebufferWidth: u32,
    maxFramebufferHeight: u32,
    maxFramebufferLayers: u32,
    framebufferColorSampleCounts: VkSampleCountFlags,
    framebufferDepthSampleCounts: VkSampleCountFlags,
    framebufferStencilSampleCounts: VkSampleCountFlags,
    framebufferNoAttachmentsSampleCounts: VkSampleCountFlags,
    maxColorAttachments: u32,
    sampledImageColorSampleCounts: VkSampleCountFlags,
    sampledImageIntegerSampleCounts: VkSampleCountFlags,
    sampledImageDepthSampleCounts: VkSampleCountFlags,
    sampledImageStencilSampleCounts: VkSampleCountFlags,
    storageImageSampleCounts: VkSampleCountFlags,
    maxSampleMaskWords: u32,
    timestampComputeAndGraphics: u32,
    timestampPeriod: f32,
    maxClipDistances: u32,
    maxCullDistances: u32,
    maxCombinedClipAndCullDistances: u32,
    discreteQueuePriorities: u32,
    pointSizeRange: [2]f32,
    lineWidthRange: [2]f32,
    pointSizeGranularity: f32,
    lineWidthGranularity: f32,
    strictLines: u32,
    standardSampleLocations: u32,
    optimalBufferCopyOffsetAlignment: VkDeviceSize,
    optimalBufferCopyRowPitchAlignment: VkDeviceSize,
    nonCoherentAtomSize: VkDeviceSize,
};

pub const VkPhysicalDeviceSparseProperties = extern struct {
    residencyStandard2DBlockShape: u32,
    residencyStandard2DMultisampleBlockShape: u32,
    residencyStandard3DBlockShape: u32,
    residencyAlignedMipSize: u32,
    residencyNonResidentStrict: u32,
};

pub const VkPhysicalDeviceQueueFamilyProperties = extern struct {
    queueFlags: VkQueueFlags,
    queueCount: u32,
    timestampValidBits: u32,
    minImageTransferGranularity: VkExtent3D,
};

pub const VkExtent3D = extern struct {
    width: u32,
    height: u32,
    depth: u32,
};

pub const VkDeviceQueueCreateInfo = extern struct {
    sType: u32 = 2,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    queueFamilyIndex: u32,
    queueCount: u32 = 1,
    pQueuePriorities: ?*const f32 = null,
};

pub const VkDeviceCreateInfo = extern struct {
    sType: u32 = 3,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    queueCreateInfoCount: u32,
    pQueueCreateInfos: ?*const VkDeviceQueueCreateInfo,
    enabledLayerCount: u32 = 0,
    ppEnabledLayerNames: ?*const [*:0]const u8 = null,
    enabledExtensionCount: u32 = 0,
    ppEnabledExtensionNames: ?*const [*:0]const u8 = null,
    pEnabledFeatures: ?*const VkPhysicalDeviceFeatures = null,
};

pub const VkPhysicalDeviceFeatures = extern struct {
    robustBufferAccess: u32,
    fullDrawIndexUint32: u32,
    imageCubeArray: u32,
    independentBlend: u32,
    geometryShader: u32,
    tessellationShader: u32,
    sampleRateShading: u32,
    dualSrcBlend: u32,
    logicOp: u32,
    multiDrawIndirect: u32,
    drawIndirectFirstInstance: u32,
    depthClamp: u32,
    depthBiasClamp: u32,
    fillModeNonSolid: u32,
    depthBounds: u32,
    wideLines: u32,
    largePoints: u32,
    alphaToOne: u32,
    multiViewport: u32,
    samplerAnisotropy: u32,
    textureCompressionETC2: u32,
    textureCompressionASTC_LDR: u32,
    textureCompressionBC: u32,
    occlusionQueryPrecise: u32,
    pipelineStatisticsQuery: u32,
    vertexPipelineStoresAndAtomics: u32,
    fragmentStoresAndAtomics: u32,
    shaderTessellationAndGeometryPointSize: u32,
    shaderImageGatherExtended: u32,
    shaderStorageImageExtendedFormats: u32,
    shaderStorageImageMultisample: u32,
    shaderStorageImageReadWithoutFormat: u32,
    shaderStorageImageWriteWithoutFormat: u32,
    shaderUniformBufferArrayDynamicIndexing: u32,
    shaderSampledImageArrayDynamicIndexing: u32,
    shaderStorageBufferArrayDynamicIndexing: u32,
    shaderStorageImageArrayDynamicIndexing: u32,
    shaderClipDistance: u32,
    shaderCullDistance: u32,
    shaderFloat64: u32,
    shaderInt64: u32,
    shaderInt16: u32,
    shaderResourceResidency: u32,
    shaderResourceMinLod: u32,
    sparseBinding: u32,
    sparseResidencyBuffer: u32,
    sparseResidencyImage2D: u32,
    sparseResidencyImage3D: u32,
    sparseResidency2Samples: u32,
    sparseResidency4Samples: u32,
    sparseResidency8Samples: u32,
    sparseResidency16Samples: u32,
    sparseResidencyAliased: u32,
    variableMultisampleRate: u32,
    inheritedQueries: u32,
};

pub const VkBufferCreateInfo = extern struct {
    sType: u32 = 12,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    size: VkDeviceSize,
    usage: VkBufferUsageFlags,
    sharingMode: VkSharingMode = .exclusive,
    queueFamilyIndexCount: u32 = 0,
    pQueueFamilyIndices: ?*const u32 = null,
};

pub const VkSharingMode = enum(u32) {
    exclusive = 0,
    concurrent = 1,
};

pub const VkMemoryRequirements = extern struct {
    size: VkDeviceSize,
    alignment: VkDeviceSize,
    memoryTypeBits: u32,
};

pub const VkPhysicalDeviceMemoryProperties = extern struct {
    memoryTypeCount: u32,
    memoryTypes: [32]VkMemoryType,
    memoryHeapCount: u32,
    memoryHeaps: [16]VkMemoryHeap,
};

pub const VkMemoryType = extern struct {
    propertyFlags: VkMemoryPropertyFlags,
    heapIndex: u32,
};

pub const VkMemoryHeap = extern struct {
    size: VkDeviceSize,
    flags: VkMemoryHeapFlags,
};

pub const VkMemoryAllocateInfo = extern struct {
    sType: u32 = 5,
    pNext: ?*const anyopaque = null,
    allocationSize: VkDeviceSize,
    memoryTypeIndex: u32,
};

pub const VkDescriptorSetLayoutBinding = extern struct {
    binding: u32,
    descriptorType: VkDescriptorType,
    descriptorCount: u32,
    stageFlags: VkShaderStageFlags,
    pImmutableSamplers: ?*const anyopaque = null,
};

pub const VkDescriptorSetLayoutCreateInfo = extern struct {
    sType: u32 = 30,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    bindingCount: u32,
    pBindings: ?*const VkDescriptorSetLayoutBinding,
};

pub const VkDescriptorPoolSize = extern struct {
    type: VkDescriptorType,
    descriptorCount: u32,
};

pub const VkDescriptorPoolCreateInfo = extern struct {
    sType: u32 = 33,
    pNext: ?*const anyopaque = null,
    flags: VkDescriptorPoolCreateFlags = 0,
    maxSets: u32,
    poolSizeCount: u32,
    pPoolSizes: ?*const VkDescriptorPoolSize,
};

pub const VkDescriptorSetAllocateInfo = extern struct {
    sType: u32 = 34,
    pNext: ?*const anyopaque = null,
    descriptorPool: VkDescriptorPool,
    descriptorSetCount: u32,
    pSetLayouts: ?*const VkDescriptorSetLayout,
};

pub const VkWriteDescriptorSet = extern struct {
    sType: u32 = 35,
    pNext: ?*const anyopaque = null,
    dstSet: VkDescriptorSet,
    dstBinding: u32,
    dstArrayElement: u32,
    descriptorCount: u32,
    descriptorType: VkDescriptorType,
    pImageInfo: ?*const anyopaque = null,
    pBufferInfo: ?*const VkDescriptorBufferInfo = null,
    pTexelBufferView: ?*const anyopaque = null,
};

pub const VkDescriptorBufferInfo = extern struct {
    buffer: VkBuffer,
    offset: VkDeviceSize,
    range: VkDeviceSize,
};

pub const VkShaderModuleCreateInfo = extern struct {
    sType: u32 = 16,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    codeSize: usize,
    pCode: ?*const u32,
};

pub const VkPipelineLayoutCreateInfo = extern struct {
    sType: u32 = 28,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    setLayoutCount: u32,
    pSetLayouts: ?*const VkDescriptorSetLayout,
    pushConstantRangeCount: u32 = 0,
    pPushConstantRanges: ?*const anyopaque = null,
};

pub const VkComputePipelineCreateInfo = extern struct {
    sType: u32 = 29,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    stage: VkPipelineShaderStageCreateInfo,
    layout: VkPipelineLayout,
    basePipelineHandle: VkPipeline = 0,
    basePipelineIndex: i32 = -1,
};

pub const VkPipelineShaderStageCreateInfo = extern struct {
    sType: u32 = 18,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    stage: VkShaderStageFlags,
    module: VkShaderModule,
    pName: [*:0]const u8 = "main",
    pSpecializationInfo: ?*const anyopaque = null,
};

pub const VkCommandPoolCreateInfo = extern struct {
    sType: u32 = 39,
    pNext: ?*const anyopaque = null,
    flags: VkCommandPoolCreateFlags = 0,
    queueFamilyIndex: u32,
};

pub const VkCommandBufferAllocateInfo = extern struct {
    sType: u32 = 40,
    pNext: ?*const anyopaque = null,
    commandPool: VkCommandPool,
    level: VkCommandBufferLevel = .primary,
    commandBufferCount: u32,
};

pub const VkCommandBufferLevel = enum(u32) {
    primary = 0,
    secondary = 1,
};

pub const VkCommandBufferBeginInfo = extern struct {
    sType: u32 = 42,
    pNext: ?*const anyopaque = null,
    flags: VkCommandBufferUsageFlags = 0,
    pInheritanceInfo: ?*const anyopaque = null,
};

pub const VkSubmitInfo = extern struct {
    sType: u32 = 4,
    pNext: ?*const anyopaque = null,
    waitSemaphoreCount: u32 = 0,
    pWaitSemaphores: ?*const VkSemaphore = null,
    pWaitDstStageMask: ?*const VkPipelineStageFlags = null,
    commandBufferCount: u32,
    pCommandBuffers: ?*const VkCommandBuffer,
    signalSemaphoreCount: u32 = 0,
    pSignalSemaphores: ?*const VkSemaphore = null,
};

pub const VkSemaphore = u64;
pub const VkPipelineStageFlags = u32;

pub const VkFenceCreateInfo = extern struct {
    sType: u32 = 12,
    pNext: ?*const anyopaque = null,
    flags: VkFenceCreateFlags = 0,
};

// ── helpers ──
fn makeApiVersion(variant: u32, major: u32, minor: u32, patch: u32) u32 {
    return (variant << 29) | (major << 22) | (minor << 12) | patch;
}

// ── PFNs (function pointer types) ──
pub const PFN_vkGetInstanceProcAddr = *const fn (instance: VkInstance, pName: [*:0]const u8) callconv(.c) ?*const anyopaque;
pub const PFN_vkCreateInstance = *const fn (pCreateInfo: *const VkInstanceCreateInfo, pAllocator: ?*const anyopaque, pInstance: *VkInstance) callconv(.c) VkResult;
pub const PFN_vkDestroyInstance = *const fn (instance: VkInstance, pAllocator: ?*const anyopaque) callconv(.c) void;
pub const PFN_vkEnumeratePhysicalDevices = *const fn (instance: VkInstance, pPhysicalDeviceCount: *u32, pPhysicalDevices: ?[*]VkPhysicalDevice) callconv(.c) VkResult;
pub const PFN_vkGetPhysicalDeviceProperties = *const fn (physicalDevice: VkPhysicalDevice, pProperties: *VkPhysicalDeviceProperties) callconv(.c) void;
pub const PFN_vkGetPhysicalDeviceQueueFamilyProperties = *const fn (physicalDevice: VkPhysicalDevice, pQueueFamilyPropertyCount: *u32, pQueueFamilyProperties: ?[*]VkPhysicalDeviceQueueFamilyProperties) callconv(.c) void;
pub const PFN_vkGetPhysicalDeviceMemoryProperties = *const fn (physicalDevice: VkPhysicalDevice, pMemoryProperties: *VkPhysicalDeviceMemoryProperties) callconv(.c) void;
pub const PFN_vkCreateDevice = *const fn (physicalDevice: VkPhysicalDevice, pCreateInfo: *const VkDeviceCreateInfo, pAllocator: ?*const anyopaque, pDevice: *VkDevice) callconv(.c) VkResult;
pub const PFN_vkDestroyDevice = *const fn (device: VkDevice, pAllocator: ?*const anyopaque) callconv(.c) void;
pub const PFN_vkGetDeviceQueue = *const fn (device: VkDevice, queueFamilyIndex: u32, queueIndex: u32, pQueue: *VkQueue) callconv(.c) void;

// Device-level functions (resolved via vkGetDeviceProcAddr or vkGetInstanceProcAddr)
pub const PFN_vkCreateBuffer = *const fn (device: VkDevice, pCreateInfo: *const VkBufferCreateInfo, pAllocator: ?*const anyopaque, pBuffer: *VkBuffer) callconv(.c) VkResult;
pub const PFN_vkDestroyBuffer = *const fn (device: VkDevice, buffer: VkBuffer, pAllocator: ?*const anyopaque) callconv(.c) void;
pub const PFN_vkGetBufferMemoryRequirements = *const fn (device: VkDevice, buffer: VkBuffer, pMemoryRequirements: *VkMemoryRequirements) callconv(.c) void;
pub const PFN_vkAllocateMemory = *const fn (device: VkDevice, pAllocateInfo: *const VkMemoryAllocateInfo, pAllocator: ?*const anyopaque, pMemory: *VkDeviceMemory) callconv(.c) VkResult;
pub const PFN_vkFreeMemory = *const fn (device: VkDevice, memory: VkDeviceMemory, pAllocator: ?*const anyopaque) callconv(.c) void;
pub const PFN_vkBindBufferMemory = *const fn (device: VkDevice, buffer: VkBuffer, memory: VkDeviceMemory, memoryOffset: VkDeviceSize) callconv(.c) VkResult;
pub const PFN_vkMapMemory = *const fn (device: VkDevice, memory: VkDeviceMemory, offset: VkDeviceSize, size: VkDeviceSize, flags: u32, ppData: **anyopaque) callconv(.c) VkResult;
pub const PFN_vkUnmapMemory = *const fn (device: VkDevice, memory: VkDeviceMemory) callconv(.c) void;
pub const PFN_vkCreateDescriptorSetLayout = *const fn (device: VkDevice, pCreateInfo: *const VkDescriptorSetLayoutCreateInfo, pAllocator: ?*const anyopaque, pSetLayout: *VkDescriptorSetLayout) callconv(.c) VkResult;
pub const PFN_vkDestroyDescriptorSetLayout = *const fn (device: VkDevice, descriptorSetLayout: VkDescriptorSetLayout, pAllocator: ?*const anyopaque) callconv(.c) void;
pub const PFN_vkCreateDescriptorPool = *const fn (device: VkDevice, pCreateInfo: *const VkDescriptorPoolCreateInfo, pAllocator: ?*const anyopaque, pDescriptorPool: *VkDescriptorPool) callconv(.c) VkResult;
pub const PFN_vkDestroyDescriptorPool = *const fn (device: VkDevice, descriptorPool: VkDescriptorPool, pAllocator: ?*const anyopaque) callconv(.c) void;
pub const PFN_vkAllocateDescriptorSets = *const fn (device: VkDevice, pAllocateInfo: *const VkDescriptorSetAllocateInfo, pDescriptorSets: *VkDescriptorSet) callconv(.c) VkResult;
pub const PFN_vkUpdateDescriptorSets = *const fn (device: VkDevice, descriptorWriteCount: u32, pDescriptorWrites: ?*const VkWriteDescriptorSet, descriptorCopyCount: u32, pDescriptorCopies: ?*const anyopaque) callconv(.c) void;
pub const PFN_vkCreateShaderModule = *const fn (device: VkDevice, pCreateInfo: *const VkShaderModuleCreateInfo, pAllocator: ?*const anyopaque, pShaderModule: *VkShaderModule) callconv(.c) VkResult;
pub const PFN_vkDestroyShaderModule = *const fn (device: VkDevice, shaderModule: VkShaderModule, pAllocator: ?*const anyopaque) callconv(.c) void;
pub const PFN_vkCreatePipelineLayout = *const fn (device: VkDevice, pCreateInfo: *const VkPipelineLayoutCreateInfo, pAllocator: ?*const anyopaque, pPipelineLayout: *VkPipelineLayout) callconv(.c) VkResult;
pub const PFN_vkDestroyPipelineLayout = *const fn (device: VkDevice, pipelineLayout: VkPipelineLayout, pAllocator: ?*const anyopaque) callconv(.c) void;
pub const PFN_vkCreateComputePipelines = *const fn (device: VkDevice, pipelineCache: VkPipelineCache, createInfoCount: u32, pCreateInfos: *const VkComputePipelineCreateInfo, pAllocator: ?*const anyopaque, pPipelines: *VkPipeline) callconv(.c) VkResult;
pub const PFN_vkDestroyPipeline = *const fn (device: VkDevice, pipeline: VkPipeline, pAllocator: ?*const anyopaque) callconv(.c) void;
pub const PFN_vkCreateCommandPool = *const fn (device: VkDevice, pCreateInfo: *const VkCommandPoolCreateInfo, pAllocator: ?*const anyopaque, pCommandPool: *VkCommandPool) callconv(.c) VkResult;
pub const PFN_vkDestroyCommandPool = *const fn (device: VkDevice, commandPool: VkCommandPool, pAllocator: ?*const anyopaque) callconv(.c) void;
pub const PFN_vkAllocateCommandBuffers = *const fn (device: VkDevice, pAllocateInfo: *const VkCommandBufferAllocateInfo, pCommandBuffers: *VkCommandBuffer) callconv(.c) VkResult;
pub const PFN_vkFreeCommandBuffers = *const fn (device: VkDevice, commandPool: VkCommandPool, commandBufferCount: u32, pCommandBuffers: *const VkCommandBuffer) callconv(.c) void;
pub const PFN_vkBeginCommandBuffer = *const fn (commandBuffer: VkCommandBuffer, pBeginInfo: *const VkCommandBufferBeginInfo) callconv(.c) VkResult;
pub const PFN_vkEndCommandBuffer = *const fn (commandBuffer: VkCommandBuffer) callconv(.c) VkResult;
pub const PFN_vkCmdBindPipeline = *const fn (commandBuffer: VkCommandBuffer, pipelineBindPoint: VkPipelineBindPoint, pipeline: VkPipeline) callconv(.c) void;
pub const PFN_vkCmdBindDescriptorSets = *const fn (commandBuffer: VkCommandBuffer, pipelineBindPoint: VkPipelineBindPoint, layout: VkPipelineLayout, firstSet: u32, descriptorSetCount: u32, pDescriptorSets: *const VkDescriptorSet, dynamicOffsetCount: u32, pDynamicOffsets: ?*const u32) callconv(.c) void;
pub const PFN_vkCmdDispatch = *const fn (commandBuffer: VkCommandBuffer, groupCountX: u32, groupCountY: u32, groupCountZ: u32) callconv(.c) void;
pub const PFN_vkCreateFence = *const fn (device: VkDevice, pCreateInfo: *const VkFenceCreateInfo, pAllocator: ?*const anyopaque, pFence: *VkFence) callconv(.c) VkResult;
pub const PFN_vkDestroyFence = *const fn (device: VkDevice, fence: VkFence, pAllocator: ?*const anyopaque) callconv(.c) void;
pub const PFN_vkQueueSubmit = *const fn (queue: VkQueue, submitCount: u32, pSubmits: *const VkSubmitInfo, fence: VkFence) callconv(.c) VkResult;
pub const PFN_vkQueueWaitIdle = *const fn (queue: VkQueue) callconv(.c) VkResult;
pub const PFN_vkDeviceWaitIdle = *const fn (device: VkDevice) callconv(.c) VkResult;
pub const PFN_vkWaitForFences = *const fn (device: VkDevice, fenceCount: u32, pFences: *const VkFence, waitAll: u32, timeout: u64) callconv(.c) VkResult;
pub const PFN_vkResetFences = *const fn (device: VkDevice, fenceCount: u32, pFences: *const VkFence) callconv(.c) VkResult;
pub const PFN_vkGetDeviceProcAddr = *const fn (device: VkDevice, pName: [*:0]const u8) callconv(.c) ?*const anyopaque;
pub const PFN_vkGetPhysicalDeviceFeatures = *const fn (physicalDevice: VkPhysicalDevice, pFeatures: *VkPhysicalDeviceFeatures) callconv(.c) void;

pub const VkPipelineCache = u64;
