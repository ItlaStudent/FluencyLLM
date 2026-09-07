// ztensor — tensores N-dimensionales, dtypes, bf16 emulado, mmap
const std = @import("std");

pub const bf16 = @import("bf16.zig");

pub const DType = enum(u8) {
    f32,
    f16,
    bf16,
    i32,
    i64,
    u8,

    pub fn size(self: DType) usize {
        return switch (self) {
            .f32 => 4,
            .f16 => 2, .bf16 => 2,
            .i32 => 4,
            .i64 => 8,
            .u8 => 1,
        };
    }
};

pub const Tensor = struct {
    dtype: DType,
    ndim: u8,
    shape: []const u32,
    strides: []const u32,
    data: [*]u8,
    nbytes: usize,
    allocator: ?std.mem.Allocator,

    pub fn shapeAt(self: Tensor, i: usize) u32 {
        return self.shape[i];
    }

    pub fn elemCount(self: Tensor) usize {
        var c: usize = 1;
        for (self.shape) |s| c *= s;
        return c;
    }

    pub fn byteOffset(self: Tensor, indices: []const u32) usize {
        var off: usize = 0;
        for (self.strides, 0..) |st, i| {
            off += @as(usize, st) * @as(usize, indices[i]);
        }
        return off * self.dtype.size();
    }

    pub fn ptr(self: Tensor, comptime T: type, indices: []const u32) *T {
        const offset = self.byteOffset(indices);
        return @ptrCast(@alignCast(self.data + offset));
    }

    pub fn getF32(self: Tensor, indices: []const u32) f32 {
        return switch (self.dtype) {
            .f32 => self.ptr(f32, indices).*,
            .bf16 => bf16.toF32(self.ptr(u16, indices).*),
            else => unreachable,
        };
    }

    pub fn setF32(self: Tensor, indices: []const u32, v: f32) void {
        switch (self.dtype) {
            .f32 => self.ptr(f32, indices).* = v,
            .bf16 => self.ptr(u16, indices).* = bf16.fromF32(v),
            else => unreachable,
        }
    }

    pub fn deinit(self: Tensor) void {
        if (self.allocator) |a| {
            a.free(self.shape);
            a.free(self.strides);
            a.free(self.data[0..self.nbytes]);
        }
    }
};

pub fn empty(alloc: std.mem.Allocator, dtype: DType, shape: []const u32) !Tensor {
    var nelem: usize = 1;
    for (shape) |s| nelem *= s;
    const nbytes = nelem * dtype.size();
    const data = try alloc.alloc(u8, nbytes);
    @memset(data, 0);

    const shape_copy = try alloc.dupe(u32, shape);
    const strides = try alloc.alloc(u32, shape.len);
    computeStrides(shape, strides);

    return Tensor{
        .dtype = dtype,
        .ndim = @intCast(shape.len),
        .shape = shape_copy,
        .strides = strides,
        .data = data.ptr,
        .nbytes = nbytes,
        .allocator = alloc,
    };
}

pub fn fromSliceF32(alloc: std.mem.Allocator, shape: []const u32, slice: []const f32) !Tensor {
    std.debug.assert(slice.len == elemCountFromShape(shape));
    var t = try empty(alloc, .f32, shape);
    const dst = t.data[0 .. t.nbytes];
    @memcpy(dst, std.mem.sliceAsBytes(slice));
    return t;
}

fn elemCountFromShape(shape: []const u32) usize {
    var c: usize = 1;
    for (shape) |s| c *= s;
    return c;
}

fn computeStrides(shape: []const u32, strides: []u32) void {
    var stride: u32 = 1;
    var i: usize = shape.len;
    while (i > 0) {
        i -= 1;
        strides[i] = stride;
        stride *= shape[i];
    }
}

// --- memory-mapping ---
//
// `MappedFile` da acceso de solo-lectura al contenido completo de un archivo
// como un slice de bytes. La intención es mapear pesos GGUF sin copiarlos.
//
// NOTA SOBRE EL BACKEND (Windows): el verdadero mmap de Windows
// (CreateFileMappingW/MapViewOfFile) requiere declarar funciones foráneas
// de kernel32, lo cual viola la Constitución del proyecto (100% Zig, sin C
// interop / sin enlace foráneo de funciones). La std de Zig 0.16 tampoco expone
// esos bindings de forma pública. Por lo tanto, en TODAS las plataformas
// usamos el FALLBACK documentado: abrir el archivo, reservar un buffer con el
// allocator y leer todo su contenido; `deinit` libera ese buffer con el mismo
// allocator. La API de `MappedFile` es idéntica independientemente del backend,
// de modo que el código cliente no necesita saber cuál se usó. Si en el futuro
// la std expone mmap de Windows en Zig puro, se puede cambiar el backend sin
// tocar la API.
pub const MappedFile = struct {
    /// Contenido completo del archivo (solo-lectura para el cliente).
    bytes: []const u8,
    /// Allocator que reservó `bytes` (presente en el backend de fallback).
    allocator: ?std.mem.Allocator,

    pub fn len(self: MappedFile) usize {
        return self.bytes.len;
    }

    pub fn deinit(self: *MappedFile) void {
        if (self.allocator) |a| {
            a.free(self.bytes);
        }
        self.* = undefined;
    }
};

/// Mapea (en la práctica: lee completo) el archivo en `path` relativo al cwd.
/// El llamador debe invocar `MappedFile.deinit` para liberar.
///
/// Se requiere un `std.Io` porque toda E/S de archivos en Zig 0.16 pasa por la
/// interfaz `std.Io` (igual que el resto del proyecto, p. ej. `main.zig`).
pub fn mapFile(alloc: std.mem.Allocator, io: std.Io, path: []const u8) !MappedFile {
    const dir = std.Io.Dir.cwd();
    const data = try dir.readFileAlloc(io, path, alloc, .unlimited);
    return MappedFile{ .bytes = data, .allocator = alloc };
}

/// Variante que mapea desde un `std.Io.Dir` explícito (útil en tests con
/// directorios temporales).
pub fn mapFileIn(alloc: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, path: []const u8) !MappedFile {
    const data = try dir.readFileAlloc(io, path, alloc, .unlimited);
    return MappedFile{ .bytes = data, .allocator = alloc };
}

// --- tests ---
test "tensor shape and strides" {
    const alloc = std.testing.allocator;
    var t = try empty(alloc, .f32, &.{ 2, 3, 4 });
    defer t.deinit();
    try std.testing.expectEqual(@as(u8, 3), t.ndim);
    try std.testing.expectEqual(@as(u32, 2), t.shapeAt(0));
    try std.testing.expectEqual(@as(u32, 3), t.shapeAt(1));
    try std.testing.expectEqual(@as(u32, 4), t.shapeAt(2));
    try std.testing.expectEqual(@as(u32, 12), t.strides[0]);
    try std.testing.expectEqual(@as(u32, 4), t.strides[1]);
    try std.testing.expectEqual(@as(u32, 1), t.strides[2]);
}

test "tensor get/set f32" {
    const alloc = std.testing.allocator;
    var t = try empty(alloc, .f32, &.{ 2, 3 });
    defer t.deinit();
    t.setF32(&.{ 1, 2 }, 42.0);
    try std.testing.expectEqual(@as(f32, 42.0), t.getF32(&.{ 1, 2 }));
}

test "tensor bf16 roundtrip" {
    const alloc = std.testing.allocator;
    var t = try empty(alloc, .bf16, &.{ 1, 1 });
    defer t.deinit();
    t.setF32(&.{ 0, 0 }, 3.14159);
    const v = t.getF32(&.{ 0, 0 });
    try std.testing.expect(@abs(v - 3.14159) < 0.01);
}

test "mapFile maps file contents" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;

    var td = std.testing.tmpDir(.{});
    defer td.cleanup();

    const contents = "GGUF\x00datos binarios \xff\x01\x02 fin";
    try td.dir.writeFile(io, .{ .sub_path = "weights.bin", .data = contents });

    var mapped = try mapFileIn(alloc, io, td.dir, "weights.bin");
    defer mapped.deinit();

    try std.testing.expectEqual(contents.len, mapped.len());
    try std.testing.expectEqualSlices(u8, contents, mapped.bytes);
}
