// zload — lectores GGUF + safetensors (+ dequantización a f32)
const std = @import("std");

// --- helpers little-endian ---
fn readU16LE(data: []const u8, offset: usize) u16 {
    return std.mem.readInt(u16, data[offset..][0..2], .little);
}
fn readU32LE(data: []const u8, offset: usize) u32 {
    return std.mem.readInt(u32, data[offset..][0..4], .little);
}
fn readU64LE(data: []const u8, offset: usize) u64 {
    return std.mem.readInt(u64, data[offset..][0..8], .little);
}

/// f16 (IEEE half) -> f32. Zig tiene f16 nativo, así que basta reinterpretar y ensanchar.
pub fn f16ToF32(bits: u16) f32 {
    const h: f16 = @bitCast(bits);
    return h;
}

// --- GGUF types ---
pub const GgufValueType = enum(u32) {
    uint8, int8, uint16, int16, uint32, int32, float32, bool, string, array, uint64, int64, float64,
};

// IDs de tipo de tensor ggml. Enum NO exhaustivo (`_`) para tolerar tipos que no
// dequantizamos todavía sin caer en comportamiento indefinido al usar @enumFromInt.
pub const GgufTensorType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    bf16 = 30,
    _,
};

pub const GgufTensorInfo = struct {
    name: []const u8,
    n_dims: u32,
    dims: [4]u64,
    type: GgufTensorType,
    offset: u64,

    pub fn numElements(self: GgufTensorInfo) usize {
        var n: usize = 1;
        for (0..self.n_dims) |i| n *= u64ToUsize(self.dims[i]) catch return 0;
        return n;
    }
};

pub const GgufMetadata = struct {
    key: []const u8,
    value_type: GgufValueType,
    value: GgufValue,
};

pub const GgufValue = union(GgufValueType) {
    uint8: u8,
    int8: i8,
    uint16: u16,
    int16: i16,
    uint32: u32,
    int32: i32,
    float32: f32,
    bool: bool,
    string: []const u8,
    array: GgufArray,
    uint64: u64,
    int64: i64,
    float64: f64,
};

pub const GgufArray = struct {
    type: GgufValueType,
    len: u64,
    /// Bytes crudos del array tal cual en disco (para strings incluye los prefijos de longitud).
    data: []const u8,
};

pub const GgufFile = struct {
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
    alignment: usize,
    tensor_data_offset: usize,
    tensors: []GgufTensorInfo,
    metadata: []GgufMetadata,
    data: []const u8,
    allocator: std.mem.Allocator,

    pub fn findTensor(self: GgufFile, name: []const u8) ?*const GgufTensorInfo {
        for (self.tensors) |*t| {
            if (std.mem.eql(u8, t.name, name)) return t;
        }
        return null;
    }

    pub fn getTensorData(self: GgufFile, info: *const GgufTensorInfo) []const u8 {
        const off = u64ToUsize(info.offset) catch unreachable;
        const start = self.tensor_data_offset + off;
        const size = tensorByteSize(info);
        return self.data[start..][0..size];
    }

    pub fn getMeta(self: GgufFile, key: []const u8) ?GgufValue {
        for (self.metadata) |m| if (std.mem.eql(u8, m.key, key)) return m.value;
        return null;
    }

    pub fn getStr(self: GgufFile, key: []const u8) ?[]const u8 {
        return switch (self.getMeta(key) orelse return null) {
            .string => |s| s,
            else => null,
        };
    }

    pub fn getU32(self: GgufFile, key: []const u8) ?u32 {
        return switch (self.getMeta(key) orelse return null) {
            .uint8 => |v| v,
            .uint16 => |v| v,
            .uint32 => |v| v,
            .int32 => |v| std.math.cast(u32, v) orelse return null,
            .uint64 => |v| std.math.cast(u32, v) orelse return null,
            .int64 => |v| std.math.cast(u32, v) orelse return null,
            else => null,
        };
    }

    pub fn getF32(self: GgufFile, key: []const u8) ?f32 {
        return switch (self.getMeta(key) orelse return null) {
            .float32 => |v| v,
            .float64 => |v| @floatCast(v),
            .uint32 => |v| @floatFromInt(v),
            else => null,
        };
    }

    /// Cuenta de tipos de cuantización presentes (para inspección).
    pub fn quantHistogram(self: GgufFile, alloc: std.mem.Allocator) !std.AutoHashMap(GgufTensorType, usize) {
        var map = std.AutoHashMap(GgufTensorType, usize).init(alloc);
        for (self.tensors) |t| {
            const gop = try map.getOrPut(t.type);
            if (!gop.found_existing) gop.value_ptr.* = 0;
            gop.value_ptr.* += 1;
        }
        return map;
    }

    pub fn deinit(self: GgufFile) void {
        for (self.tensors) |t| self.allocator.free(t.name);
        self.allocator.free(self.tensors);
        for (self.metadata) |m| {
            self.allocator.free(m.key);
            switch (m.value) {
                .string => |s| self.allocator.free(s),
                .array => |a| self.allocator.free(a.data),
                else => {},
            }
        }
        self.allocator.free(self.metadata);
    }
};

// --- tamaños de bloque cuantizado (elems por bloque, bytes por bloque) ---
const QuantBlock = struct { elems: usize, bytes: usize };

fn quantBlock(t: GgufTensorType) ?QuantBlock {
    return switch (t) {
        .q4_0 => .{ .elems = 32, .bytes = 18 },
        .q4_1 => .{ .elems = 32, .bytes = 20 },
        .q5_0 => .{ .elems = 32, .bytes = 22 },
        .q5_1 => .{ .elems = 32, .bytes = 24 },
        .q8_0 => .{ .elems = 32, .bytes = 34 },
        .q8_1 => .{ .elems = 32, .bytes = 36 },
        .q2_k => .{ .elems = 256, .bytes = 84 },
        .q3_k => .{ .elems = 256, .bytes = 110 },
        .q4_k => .{ .elems = 256, .bytes = 144 },
        .q5_k => .{ .elems = 256, .bytes = 176 },
        .q6_k => .{ .elems = 256, .bytes = 210 },
        .q8_k => .{ .elems = 256, .bytes = 292 },
        else => null,
    };
}

fn scalarTypeSize(t: GgufTensorType) usize {
    return switch (t) {
        .f32 => 4,
        .f16, .bf16 => 2,
        else => 0,
    };
}

pub fn tensorByteSize(info: *const GgufTensorInfo) usize {
    const nelem = info.numElements();
    if (quantBlock(info.type)) |qb| {
        return (nelem + qb.elems - 1) / qb.elems * qb.bytes;
    }
    return nelem * scalarTypeSize(info.type);
}

fn alignUp(n: usize, alignment: usize) usize {
    return ((n + alignment - 1) / alignment) * alignment;
}

fn u64ToUsize(v: u64) !usize {
    return std.math.cast(usize, v) orelse error.Overflow;
}

// --- parse GGUF ---
pub fn parseGguf(alloc: std.mem.Allocator, data: []const u8) !GgufFile {
    if (data.len < 24) return error.TooSmall;
    if (!std.mem.eql(u8, data[0..4], "GGUF")) return error.InvalidMagic;
    const version = readU32LE(data, 4);
    if (version != 3) return error.UnsupportedVersion;
    const tensor_count = readU64LE(data, 8);
    const metadata_kv_count = readU64LE(data, 16);

    var off: usize = 24;
    const metadata_n = try u64ToUsize(metadata_kv_count);
    const metadata = try alloc.alloc(GgufMetadata, metadata_n);
    errdefer alloc.free(metadata);
    for (0..metadata_n) |i| {
        const key_len = try u64ToUsize(readU64LE(data, off));
        off += 8;
        const key = try alloc.dupe(u8, data[off..][0..key_len]);
        off += key_len;
        const vt: GgufValueType = @enumFromInt(readU32LE(data, off));
        off += 4;
        const val = try parseValue(alloc, data, &off, vt);
        metadata[i] = .{ .key = key, .value_type = vt, .value = val };
    }

    const tensor_n = try u64ToUsize(tensor_count);
    const tensors = try alloc.alloc(GgufTensorInfo, tensor_n);
    errdefer alloc.free(tensors);
    for (0..tensor_n) |i| {
        const name_len = try u64ToUsize(readU64LE(data, off));
        off += 8;
        const name = try alloc.dupe(u8, data[off..][0..name_len]);
        off += name_len;
        const n_dims = readU32LE(data, off);
        off += 4;
        var dims: [4]u64 = .{ 1, 1, 1, 1 };
        for (0..n_dims) |d| {
            dims[d] = readU64LE(data, off);
            off += 8;
        }
        const ttype: GgufTensorType = @enumFromInt(readU32LE(data, off));
        off += 4;
        const toffset = readU64LE(data, off);
        off += 8;
        tensors[i] = .{ .name = name, .n_dims = n_dims, .dims = dims, .type = ttype, .offset = toffset };
    }

    // El offset real de los datos de tensores es el fin del bloque de tensor-info,
    // alineado a general.alignment (default 32). Lo capturamos en vez de recalcularlo.
    var alignment: usize = 32;
    for (metadata) |m| {
        if (std.mem.eql(u8, m.key, "general.alignment")) {
            switch (m.value) {
                .uint32 => |v| alignment = v,
                .uint64 => |v| alignment = try u64ToUsize(v),
                else => {},
            }
        }
    }
    const tensor_data_offset = alignUp(off, alignment);

    return GgufFile{
        .version = version,
        .tensor_count = tensor_count,
        .metadata_kv_count = metadata_kv_count,
        .alignment = alignment,
        .tensor_data_offset = tensor_data_offset,
        .tensors = tensors,
        .metadata = metadata,
        .data = data,
        .allocator = alloc,
    };
}

fn fixedElemSize(t: GgufValueType) usize {
    return switch (t) {
        .uint8, .int8, .bool => 1,
        .uint16, .int16 => 2,
        .uint32, .int32, .float32 => 4,
        .uint64, .int64, .float64 => 8,
        .string, .array => 0, // variable / no soportado aquí
    };
}

fn parseValue(alloc: std.mem.Allocator, data: []const u8, off: *usize, vt: GgufValueType) !GgufValue {
    switch (vt) {
        .uint8 => { const v = data[off.*]; off.* += 1; return .{ .uint8 = v }; },
        .int8 => { const v = @as(i8, @bitCast(data[off.*])); off.* += 1; return .{ .int8 = v }; },
        .uint16 => { const v = readU16LE(data, off.*); off.* += 2; return .{ .uint16 = v }; },
        .int16 => { const v = @as(i16, @bitCast(readU16LE(data, off.*))); off.* += 2; return .{ .int16 = v }; },
        .uint32 => { const v = readU32LE(data, off.*); off.* += 4; return .{ .uint32 = v }; },
        .int32 => { const v = @as(i32, @bitCast(readU32LE(data, off.*))); off.* += 4; return .{ .int32 = v }; },
        .float32 => { const v = @as(f32, @bitCast(readU32LE(data, off.*))); off.* += 4; return .{ .float32 = v }; },
        .bool => { const v = data[off.*] != 0; off.* += 1; return .{ .bool = v }; },
        .uint64 => { const v = readU64LE(data, off.*); off.* += 8; return .{ .uint64 = v }; },
        .int64 => { const v = @as(i64, @bitCast(readU64LE(data, off.*))); off.* += 8; return .{ .int64 = v }; },
        .float64 => { const v = @as(f64, @bitCast(readU64LE(data, off.*))); off.* += 8; return .{ .float64 = v }; },
        .string => {
            const len = try u64ToUsize(readU64LE(data, off.*));
            off.* += 8;
            const v = try alloc.dupe(u8, data[off.*..][0..len]);
            off.* += len;
            return .{ .string = v };
        },
        .array => {
            const atype: GgufValueType = @enumFromInt(readU32LE(data, off.*));
            off.* += 4;
            const alen = readU64LE(data, off.*);
            off.* += 8;
            const start = off.*;
            switch (atype) {
                .array => return error.NestedArray,
                .string => {
                    const alen_usize = try u64ToUsize(alen);
                    var i: usize = 0;
                    while (i < alen_usize) : (i += 1) {
                        const slen = try u64ToUsize(readU64LE(data, off.*));
                        off.* += 8 + slen;
                    }
                },
                else => {
                    off.* += (try u64ToUsize(alen)) * fixedElemSize(atype);
                },
            }
            const adata = try alloc.dupe(u8, data[start..off.*]);
            return .{ .array = .{ .type = atype, .len = alen, .data = adata } };
        },
    }
}

/// Devuelve el string en la posición `idx` de un array de strings (p. ej. el vocab).
pub fn arrayStringAt(arr: GgufArray, idx: usize) ?[]const u8 {
    if (arr.type != .string or idx >= u64ToUsize(arr.len) catch return null) return null;
    var off: usize = 0;
    var i: usize = 0;
    const len = u64ToUsize(arr.len) catch return null;
    while (i < len) : (i += 1) {
        const slen = u64ToUsize(readU64LE(arr.data, off)) catch return null;
        off += 8;
        if (i == idx) return arr.data[off..][0..slen];
        off += slen;
    }
    return null;
}

/// Iterador lineal O(n) sobre un array de strings del GGUF (p. ej.
/// tokenizer.ggml.tokens / .merges). `arrayStringAt` es O(n) por llamada, así que
/// para recorrer vocabularios grandes (~150k) hay que usar este iterador.
pub const ArrayStrIter = struct {
    arr: GgufArray,
    off: usize = 0,
    i: u64 = 0,

    pub fn next(self: *ArrayStrIter) ?[]const u8 {
        if (self.arr.type != .string or self.i >= self.arr.len) return null;
        const slen = u64ToUsize(readU64LE(self.arr.data, self.off)) catch return null;
        self.off += 8;
        const s = self.arr.data[self.off..][0..slen];
        self.off += slen;
        self.i += 1;
        return s;
    }
};

pub fn arrayStrIter(arr: GgufArray) ArrayStrIter {
    return .{ .arr = arr };
}

/// Itera un array de strings (p. ej. tokenizer.ggml.tokens) llamando `cb` por cada uno.
pub fn forEachArrayString(arr: GgufArray, ctx: anytype, cb: fn (@TypeOf(ctx), usize, []const u8) void) void {
    if (arr.type != .string) return;
    var off: usize = 0;
    var i: usize = 0;
    const len = u64ToUsize(arr.len) catch return;
    while (i < len) : (i += 1) {
        const slen = u64ToUsize(readU64LE(arr.data, off)) catch return;
        off += 8;
        cb(ctx, i, arr.data[off..][0..slen]);
        off += slen;
    }
}

// --- dequantización a f32 ---

fn dequantQ8_0(src: []const u8, dst: []f32) void {
    const nb = dst.len / 32;
    var si: usize = 0;
    var di: usize = 0;
    var b: usize = 0;
    while (b < nb) : (b += 1) {
        const d = f16ToF32(readU16LE(src, si));
        si += 2;
        for (0..32) |i| {
            const q: i8 = @bitCast(src[si + i]);
            dst[di + i] = d * @as(f32, @floatFromInt(q));
        }
        si += 32;
        di += 32;
    }
}

fn dequantQ4_0(src: []const u8, dst: []f32) void {
    const nb = dst.len / 32;
    var si: usize = 0;
    var di: usize = 0;
    var b: usize = 0;
    while (b < nb) : (b += 1) {
        const d = f16ToF32(readU16LE(src, si));
        si += 2;
        for (0..16) |l| {
            const byte = src[si + l];
            const x0: i32 = @as(i32, byte & 0x0F) - 8;
            const x1: i32 = @as(i32, byte >> 4) - 8;
            dst[di + l] = d * @as(f32, @floatFromInt(x0));
            dst[di + l + 16] = d * @as(f32, @floatFromInt(x1));
        }
        si += 16;
        di += 32;
    }
}

// get_scale_min_k4 de ggml: extrae escala (d) y mínimo (m) de 6 bits del sub-bloque j.
fn getScaleMinK4(j: usize, q: []const u8) struct { d: u8, m: u8 } {
    if (j < 4) {
        return .{ .d = q[j] & 63, .m = q[j + 4] & 63 };
    } else {
        const d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        const m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        return .{ .d = d, .m = m };
    }
}

fn dequantQ4_K(src: []const u8, dst: []f32) void {
    const QK_K = 256;
    const nb = dst.len / QK_K;
    var si: usize = 0;
    var y: usize = 0;
    var b: usize = 0;
    while (b < nb) : (b += 1) {
        const d = f16ToF32(readU16LE(src, si));
        const dmin = f16ToF32(readU16LE(src, si + 2));
        const scales = src[si + 4 ..][0..12];
        const q = src[si + 16 ..][0..128];
        si += 144;

        var is: usize = 0;
        var qoff: usize = 0;
        var j: usize = 0;
        while (j < QK_K) : (j += 64) {
            const sm0 = getScaleMinK4(is, scales);
            const d1 = d * @as(f32, @floatFromInt(sm0.d));
            const m1 = dmin * @as(f32, @floatFromInt(sm0.m));
            const sm1 = getScaleMinK4(is + 1, scales);
            const d2 = d * @as(f32, @floatFromInt(sm1.d));
            const m2 = dmin * @as(f32, @floatFromInt(sm1.m));
            for (0..32) |l| {
                dst[y] = d1 * @as(f32, @floatFromInt(q[qoff + l] & 0x0F)) - m1;
                y += 1;
            }
            for (0..32) |l| {
                dst[y] = d2 * @as(f32, @floatFromInt(q[qoff + l] >> 4)) - m2;
                y += 1;
            }
            qoff += 32;
            is += 2;
        }
    }
}

fn dequantQ6_K(src: []const u8, dst: []f32) void {
    const QK_K = 256;
    const nb = dst.len / QK_K;
    var si: usize = 0;
    var ybase: usize = 0;
    var b: usize = 0;
    while (b < nb) : (b += 1) {
        const ql = src[si..][0..128];
        const qh = src[si + 128 ..][0..64];
        const sc = src[si + 192 ..][0..16];
        const d = f16ToF32(readU16LE(src, si + 208));
        si += 210;

        var n: usize = 0;
        var qlb: usize = 0;
        var qhb: usize = 0;
        var scb: usize = 0;
        while (n < QK_K) : (n += 128) {
            for (0..32) |l| {
                const is = l / 16;
                const lo0: i32 = @as(i32, ql[qlb + l] & 0x0F) | (@as(i32, (qh[qhb + l] >> 0) & 3) << 4);
                const lo1: i32 = @as(i32, ql[qlb + l + 32] & 0x0F) | (@as(i32, (qh[qhb + l] >> 2) & 3) << 4);
                const hi0: i32 = @as(i32, ql[qlb + l] >> 4) | (@as(i32, (qh[qhb + l] >> 4) & 3) << 4);
                const hi1: i32 = @as(i32, ql[qlb + l + 32] >> 4) | (@as(i32, (qh[qhb + l] >> 6) & 3) << 4);
                const q1 = lo0 - 32;
                const q2 = lo1 - 32;
                const q3 = hi0 - 32;
                const q4 = hi1 - 32;
                const s0: i8 = @bitCast(sc[scb + is + 0]);
                const s2: i8 = @bitCast(sc[scb + is + 2]);
                const s4: i8 = @bitCast(sc[scb + is + 4]);
                const s6: i8 = @bitCast(sc[scb + is + 6]);
                dst[ybase + n + l] = d * @as(f32, @floatFromInt(s0)) * @as(f32, @floatFromInt(q1));
                dst[ybase + n + l + 32] = d * @as(f32, @floatFromInt(s2)) * @as(f32, @floatFromInt(q2));
                dst[ybase + n + l + 64] = d * @as(f32, @floatFromInt(s4)) * @as(f32, @floatFromInt(q3));
                dst[ybase + n + l + 96] = d * @as(f32, @floatFromInt(s6)) * @as(f32, @floatFromInt(q4));
            }
            qlb += 64;
            qhb += 32;
            scb += 8;
        }
        ybase += QK_K;
    }
}

/// Dequantiza un tensor GGUF a un buffer f32 nuevo (propiedad del caller).
pub fn dequantTensor(alloc: std.mem.Allocator, gguf: GgufFile, info: *const GgufTensorInfo) ![]f32 {
    const nelem = info.numElements();
    const raw = gguf.getTensorData(info);
    const out = try alloc.alloc(f32, nelem);
    errdefer alloc.free(out);
    switch (info.type) {
        .f32 => for (0..nelem) |i| {
            out[i] = @bitCast(readU32LE(raw, i * 4));
        },
        .f16 => for (0..nelem) |i| {
            out[i] = f16ToF32(readU16LE(raw, i * 2));
        },
        .q8_0 => dequantQ8_0(raw, out),
        .q4_0 => dequantQ4_0(raw, out),
        .q4_k => dequantQ4_K(raw, out),
        .q6_k => dequantQ6_K(raw, out),
        else => {
            alloc.free(out);
            return error.UnsupportedQuant;
        },
    }
    return out;
}

// --- extracción de configuración del modelo ---
pub const ModelInfo = struct {
    arch: []const u8,
    n_layers: u32,
    n_heads: u32,
    n_kv_heads: u32,
    embd: u32,
    ffn: u32,
    vocab: u32,
    ctx: u32,
    rope_freq: f32,
    rms_eps: f32,
};

fn archKey(buf: []u8, arch: []const u8, suffix: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "{s}.{s}", .{ arch, suffix }) catch suffix;
}

pub fn modelInfo(gguf: GgufFile) !ModelInfo {
    const arch = gguf.getStr("general.architecture") orelse return error.NoArchitecture;
    var kb: [128]u8 = undefined;

    const n_layers = gguf.getU32(archKey(&kb, arch, "block_count")) orelse return error.MissingMeta;
    const n_heads = gguf.getU32(archKey(&kb, arch, "attention.head_count")) orelse return error.MissingMeta;
    const n_kv_heads = gguf.getU32(archKey(&kb, arch, "attention.head_count_kv")) orelse n_heads;
    const embd = gguf.getU32(archKey(&kb, arch, "embedding_length")) orelse return error.MissingMeta;
    const ffn = gguf.getU32(archKey(&kb, arch, "feed_forward_length")) orelse 0;
    const ctx = gguf.getU32(archKey(&kb, arch, "context_length")) orelse 0;
    const rope_freq = gguf.getF32(archKey(&kb, arch, "rope.freq_base")) orelse 10000.0;
    const rms_eps = gguf.getF32(archKey(&kb, arch, "attention.layer_norm_rms_epsilon")) orelse 1e-5;

    // vocab: preferir tokenizer.ggml.tokens; fallback a {arch}.vocab_size.
    var vocab: u32 = 0;
    if (gguf.getMeta("tokenizer.ggml.tokens")) |m| {
        if (m == .array) vocab = std.math.cast(u32, m.array.len) orelse 0;
    }
    if (vocab == 0) vocab = gguf.getU32(archKey(&kb, arch, "vocab_size")) orelse 0;

    return ModelInfo{
        .arch = arch,
        .n_layers = n_layers,
        .n_heads = n_heads,
        .n_kv_heads = n_kv_heads,
        .embd = embd,
        .ffn = ffn,
        .vocab = vocab,
        .ctx = ctx,
        .rope_freq = rope_freq,
        .rms_eps = rms_eps,
    };
}

// --- lector safetensors ---
//
// Formato: [u64 LE: longitud cabecera N][N bytes: JSON][bytes de datos].
// El JSON mapea nombre -> {dtype, shape:[...], data_offsets:[ini,fin]} (offsets
// relativos al inicio de la zona de datos = 8 + N). Clave "__metadata__" opcional.

pub const StDType = enum {
    f32,
    f16,
    bf16,
    i8,
    u8,
    i16,
    i32,
    i64,
    bool,
    unknown,

    fn parse(s: []const u8) StDType {
        const map = .{
            .{ "F32", StDType.f32 }, .{ "F16", StDType.f16 },   .{ "BF16", StDType.bf16 },
            .{ "I8", StDType.i8 },   .{ "U8", StDType.u8 },     .{ "I16", StDType.i16 },
            .{ "I32", StDType.i32 }, .{ "I64", StDType.i64 },   .{ "BOOL", StDType.bool },
        };
        inline for (map) |m| if (std.mem.eql(u8, s, m[0])) return m[1];
        return .unknown;
    }
};

pub const StTensor = struct {
    name: []const u8, // propiedad de SafeTensors (duplicado)
    dtype: StDType,
    shape: [8]u64 = .{0} ** 8,
    ndim: u8 = 0,
    begin: u64 = 0, // relativo al inicio de datos
    end: u64 = 0,

    pub fn elemCount(self: StTensor) u64 {
        var c: u64 = 1;
        for (0..self.ndim) |i| c *= self.shape[i];
        return c;
    }
};

pub const SafeTensors = struct {
    data: []const u8,
    data_start: usize,
    tensors: []StTensor,
    allocator: std.mem.Allocator,

    pub fn deinit(self: SafeTensors) void {
        for (self.tensors) |t| self.allocator.free(t.name);
        self.allocator.free(self.tensors);
    }

    pub fn findTensor(self: SafeTensors, name: []const u8) ?*const StTensor {
        for (self.tensors) |*t| if (std.mem.eql(u8, t.name, name)) return t;
        return null;
    }

    pub fn tensorBytes(self: SafeTensors, t: *const StTensor) []const u8 {
        return self.data[self.data_start + t.begin ..][0 .. t.end - t.begin];
    }

    /// Dequantiza a f32 (soporta F32/F16/BF16). Caller libera.
    pub fn dequant(self: SafeTensors, alloc: std.mem.Allocator, t: *const StTensor) ![]f32 {
        const raw = self.tensorBytes(t);
        const n = try u64ToUsize(t.elemCount());
        const out = try alloc.alloc(f32, n);
        errdefer alloc.free(out);
        switch (t.dtype) {
            .f32 => {
                if (raw.len < n * 4) return error.TruncatedTensor;
                for (0..n) |i| out[i] = @bitCast(readU32LE(raw, i * 4));
            },
            .f16 => {
                if (raw.len < n * 2) return error.TruncatedTensor;
                for (0..n) |i| out[i] = f16ToF32(@intCast(readU16LE(raw, i * 2)));
            },
            .bf16 => {
                if (raw.len < n * 2) return error.TruncatedTensor;
                for (0..n) |i| {
                    const bits: u32 = @as(u32, readU16LE(raw, i * 2)) << 16;
                    out[i] = @bitCast(bits);
                }
            },
            else => return error.UnsupportedDtype,
        }
        return out;
    }
};

// Mini-parser JSON específico para la cabecera safetensors.
const StHeaderParser = struct {
    s: []const u8,
    i: usize = 0,

    fn ws(self: *StHeaderParser) void {
        while (self.i < self.s.len and (self.s[self.i] == ' ' or self.s[self.i] == '\t' or self.s[self.i] == '\n' or self.s[self.i] == '\r')) self.i += 1;
    }
    fn expect(self: *StHeaderParser, c: u8) !void {
        self.ws();
        if (self.i >= self.s.len or self.s[self.i] != c) return error.BadSafetensorsHeader;
        self.i += 1;
    }
    fn peek(self: *StHeaderParser) ?u8 {
        self.ws();
        return if (self.i < self.s.len) self.s[self.i] else null;
    }
    // String JSON sin escapes complejos (los nombres de safetensors no los usan).
    fn str(self: *StHeaderParser) ![]const u8 {
        try self.expect('"');
        const start = self.i;
        while (self.i < self.s.len and self.s[self.i] != '"') {
            if (self.s[self.i] == '\\') self.i += 1;
            self.i += 1;
        }
        if (self.i >= self.s.len) return error.BadSafetensorsHeader;
        const out = self.s[start..self.i];
        self.i += 1;
        return out;
    }
    fn u64v(self: *StHeaderParser) !u64 {
        self.ws();
        const start = self.i;
        while (self.i < self.s.len and self.s[self.i] >= '0' and self.s[self.i] <= '9') self.i += 1;
        if (self.i == start) return error.BadSafetensorsHeader;
        return std.fmt.parseInt(u64, self.s[start..self.i], 10);
    }
    // Salta un valor JSON arbitrario (para "__metadata__").
    fn skipValue(self: *StHeaderParser) !void {
        const c = self.peek() orelse return error.BadSafetensorsHeader;
        switch (c) {
            '{' => {
                try self.expect('{');
                if (self.peek() != @as(?u8, '}')) {
                    while (true) {
                        _ = try self.str();
                        try self.expect(':');
                        try self.skipValue();
                        if (self.peek() == @as(?u8, ',')) {
                            self.i += 1;
                        } else break;
                    }
                }
                try self.expect('}');
            },
            '[' => {
                try self.expect('[');
                if (self.peek() != @as(?u8, ']')) {
                    while (true) {
                        try self.skipValue();
                        if (self.peek() == @as(?u8, ',')) {
                            self.i += 1;
                        } else break;
                    }
                }
                try self.expect(']');
            },
            '"' => _ = try self.str(),
            else => {
                // número / true / false / null
                while (self.i < self.s.len and self.s[self.i] != ',' and self.s[self.i] != '}' and self.s[self.i] != ']') self.i += 1;
            },
        }
    }
};

pub fn parseSafetensors(alloc: std.mem.Allocator, data: []const u8) !SafeTensors {
    if (data.len < 8) return error.TruncatedSafetensors;
    const header_len = u64ToUsize(readU64LE(data, 0)) catch return error.TruncatedSafetensors;
    if (8 + header_len > data.len) return error.TruncatedSafetensors;
    const header = data[8 .. 8 + header_len];
    const data_start = 8 + header_len;

    var list: std.ArrayList(StTensor) = .empty;
    errdefer {
        for (list.items) |t| alloc.free(t.name);
        list.deinit(alloc);
    }

    var p = StHeaderParser{ .s = header };
    try p.expect('{');
    if (p.peek() != @as(?u8, '}')) {
        while (true) {
            const name = try p.str();
            try p.expect(':');
            if (std.mem.eql(u8, name, "__metadata__")) {
                try p.skipValue();
            } else {
                // {"dtype":"F32","shape":[...],"data_offsets":[a,b]}
                var t = StTensor{ .name = "", .dtype = .unknown };
                try p.expect('{');
                while (true) {
                    const field = try p.str();
                    try p.expect(':');
                    if (std.mem.eql(u8, field, "dtype")) {
                        t.dtype = StDType.parse(try p.str());
                    } else if (std.mem.eql(u8, field, "shape")) {
                        try p.expect('[');
                        if (p.peek() != @as(?u8, ']')) {
                            while (true) {
                                if (t.ndim < t.shape.len) {
                                    t.shape[t.ndim] = try p.u64v();
                                    t.ndim += 1;
                                } else _ = try p.u64v();
                                if (p.peek() == @as(?u8, ',')) {
                                    p.i += 1;
                                } else break;
                            }
                        }
                        try p.expect(']');
                    } else if (std.mem.eql(u8, field, "data_offsets")) {
                        try p.expect('[');
                        t.begin = try p.u64v();
                        try p.expect(',');
                        t.end = try p.u64v();
                        try p.expect(']');
                    } else {
                        try p.skipValue();
                    }
                    if (p.peek() == @as(?u8, ',')) {
                        p.i += 1;
                    } else break;
                }
                try p.expect('}');
                t.name = try alloc.dupe(u8, name);
                try list.append(alloc, t);
            }
            if (p.peek() == @as(?u8, ',')) {
                p.i += 1;
            } else break;
        }
    }
    try p.expect('}');

    return SafeTensors{
        .data = data,
        .data_start = data_start,
        .tensors = try list.toOwnedSlice(alloc),
        .allocator = alloc,
    };
}

// --- tests ---
test "parse minimal GGUF" {
    const alloc = std.testing.allocator;
    var buf: [1024]u8 = undefined;
    @memcpy(buf[0..4], "GGUF");
    std.mem.writeInt(u32, buf[4..8], 3, .little);
    std.mem.writeInt(u64, buf[8..16], 0, .little); // tensor_count
    std.mem.writeInt(u64, buf[16..24], 0, .little); // metadata_kv_count
    const gf = try parseGguf(alloc, buf[0..24]);
    defer gf.deinit();
    try std.testing.expectEqual(@as(u32, 3), gf.version);
    try std.testing.expectEqual(@as(u64, 0), gf.tensor_count);
}

test "f16 -> f32" {
    try std.testing.expectEqual(@as(f32, 1.0), f16ToF32(@bitCast(@as(f16, 1.0))));
    try std.testing.expectEqual(@as(f32, 0.5), f16ToF32(@bitCast(@as(f16, 0.5))));
    try std.testing.expectEqual(@as(f32, -2.0), f16ToF32(@bitCast(@as(f16, -2.0))));
}

test "dequant Q8_0 contra referencia" {
    // Un bloque de 32: d=0.5, qs = 0,1,2,...,31  =>  out[i] = 0.5*i
    var src: [34]u8 = undefined;
    const d: f16 = 0.5;
    std.mem.writeInt(u16, src[0..2], @bitCast(d), .little);
    for (0..32) |i| src[2 + i] = @bitCast(@as(i8, @intCast(i)));
    var out: [32]f32 = undefined;
    dequantQ8_0(src[0..], out[0..]);
    for (0..32) |i| {
        try std.testing.expectApproxEqAbs(0.5 * @as(f32, @floatFromInt(i)), out[i], 1e-4);
    }
}

test "dequant Q4_0 contra referencia" {
    // d=2.0; cada byte tiene dos nibbles. nibble n => (n-8)*d.
    var src: [18]u8 = undefined;
    const d: f16 = 2.0;
    std.mem.writeInt(u16, src[0..2], @bitCast(d), .little);
    for (0..16) |l| src[2 + l] = @intCast((l & 0x0F)); // low nibble = l, high nibble = 0
    var out: [32]f32 = undefined;
    dequantQ4_0(src[0..], out[0..]);
    // out[l] = d*((l)-8); out[l+16] = d*((0)-8) = -16
    for (0..16) |l| {
        try std.testing.expectApproxEqAbs(2.0 * (@as(f32, @floatFromInt(l)) - 8.0), out[l], 1e-4);
        try std.testing.expectApproxEqAbs(-16.0, out[l + 16], 1e-4);
    }
}

test "array de strings de longitud variable se parsea bien" {
    const alloc = std.testing.allocator;
    // GGUF con 1 metadata: key="t" array<string>=["ab","c"]
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(alloc);
    try buf.appendSlice(alloc, "GGUF");
    try appendInt(&buf, alloc, u32, 3);
    try appendInt(&buf, alloc, u64, 0); // tensor_count
    try appendInt(&buf, alloc, u64, 1); // metadata_kv_count
    // key
    try appendInt(&buf, alloc, u64, 1);
    try buf.append(alloc, 't');
    try appendInt(&buf, alloc, u32, @intFromEnum(GgufValueType.array)); // value_type=array
    try appendInt(&buf, alloc, u32, @intFromEnum(GgufValueType.string)); // array elem type
    try appendInt(&buf, alloc, u64, 2); // array len
    try appendInt(&buf, alloc, u64, 2);
    try buf.appendSlice(alloc, "ab");
    try appendInt(&buf, alloc, u64, 1);
    try buf.append(alloc, 'c');

    const gf = try parseGguf(alloc, buf.items);
    defer gf.deinit();
    const m = gf.getMeta("t").?;
    try std.testing.expectEqual(@as(u64, 2), m.array.len);
    try std.testing.expectEqual(GgufValueType.string, m.array.type);
}

fn appendInt(buf: *std.ArrayList(u8), alloc: std.mem.Allocator, comptime T: type, v: T) !void {
    var tmp: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &tmp, v, .little);
    try buf.appendSlice(alloc, &tmp);
}

test "parse safetensors F32 + dequant" {
    const alloc = std.testing.allocator;
    const header = "{\"__metadata__\":{\"format\":\"pt\"},\"w\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}";

    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(alloc);
    try appendInt(&buf, alloc, u64, header.len);
    try buf.appendSlice(alloc, header);
    const vals = [_]f32{ 1.5, -2.0, 3.25, 4.0 };
    for (vals) |v| try appendInt(&buf, alloc, u32, @bitCast(v));

    var st = try parseSafetensors(alloc, buf.items);
    defer st.deinit();
    try std.testing.expectEqual(@as(usize, 1), st.tensors.len);
    const t = st.findTensor("w").?;
    try std.testing.expectEqual(StDType.f32, t.dtype);
    try std.testing.expectEqual(@as(u8, 2), t.ndim);
    try std.testing.expectEqual(@as(u64, 4), t.elemCount());

    const deq = try st.dequant(alloc, t);
    defer alloc.free(deq);
    for (vals, deq) |want, got| try std.testing.expectEqual(want, got);
}
