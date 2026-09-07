const std = @import("std");

// Art. I — Pureza de lenguaje. Este gate recorre el árbol del proyecto y FALLA
// si encuentra (1) archivos fuente de otro lenguaje, (2) enlaces a terceros en
// build.zig*, o (3) FFI/@cImport en el código de producto (src/, fuera de platform).

const banned_exts: []const []const u8 = &.{
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx",
    ".rs", ".py", ".js", ".ts", ".go", ".java", ".rb",
    ".cu", ".cuh", ".metal", ".cl", ".S", ".asm",
};

// Directorios que NO se versionan ni forman parte del producto (toolchain, caches).
// Incluye el compilador Zig vendorizado (zig-0.16.0), que trae miles de .h/.c de libc.
const excluded_dirs: []const []const u8 = &.{
    ".git", ".jj", ".hg", ".svn", "node_modules",
    ".zig-cache", "zig-cache", "zig-out", "zig-0.16.0",
};

const banned_in_build: []const []const u8 = &.{ "linkSystemLibrary", "linkLibC", "@cImport", "addTranslateC" };

// Patrones de FFI hacia otro lenguaje (Art. I.3). OJO: `extern struct/union/enum`
// son layout C-ABI en Zig puro y SON válidos; sólo se prohíbe importar símbolos
// foráneos (`extern fn`, `extern "c"`) y `@cImport`/`@cInclude`.
const ffi_patterns: []const []const u8 = &.{ "@cImport", "@cInclude", "extern fn", "extern \"" };

fn isExcludedDir(name: []const u8) bool {
    for (excluded_dirs) |d| if (std.mem.eql(u8, name, d)) return true;
    return false;
}

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;
    var violations: u32 = 0;
    var scanned: u32 = 0;

    const src_prefix = "src" ++ std.fs.path.sep_str;
    const platform_prefix = "src" ++ std.fs.path.sep_str ++ "platform";

    // cwd() NO es iterable; hay que abrir el directorio con .iterate = true.
    var root = try std.Io.Dir.cwd().openDir(io, ".", .{ .iterate = true });
    defer root.close(io);

    var walker = try root.walkSelectively(alloc);
    defer walker.deinit();

    while (try walker.next(io)) |e| {
        if (e.kind == .directory) {
            // Sólo descendemos en directorios no excluidos (evita los 18k .h de zig-0.16.0).
            if (!isExcludedDir(e.basename)) walker.enter(io, e) catch {};
            continue;
        }
        if (e.kind != .file) continue;
        scanned += 1;
        const path = e.path;

        // 1. Extensión prohibida (cualquier parte del árbol versionado).
        for (banned_exts) |ext| {
            if (std.mem.endsWith(u8, path, ext)) {
                std.log.err("VIOLACION PUREZA: archivo de otro lenguaje: {s}", .{path});
                violations += 1;
            }
        }

        // 2. FFI/@cImport SÓLO en código de producto (src/), excepto src/platform.
        //    (Se excluye tools/ porque los propios gates contienen estos literales como datos.)
        if (std.mem.endsWith(u8, path, ".zig") and
            std.mem.startsWith(u8, path, src_prefix) and
            !std.mem.startsWith(u8, path, platform_prefix))
        {
            const content = e.dir.readFileAlloc(io, e.basename, alloc, .limited(8 << 20)) catch null;
            if (content) |c| {
                defer alloc.free(c);
                for (ffi_patterns) |pat| {
                    if (std.mem.indexOf(u8, c, pat) != null) {
                        std.log.err("VIOLACION PUREZA: FFI a otro lenguaje ('{s}') en {s}", .{ pat, path });
                        violations += 1;
                    }
                }
            }
        }
    }

    // 3. build.zig y build.zig.zon: sin enlaces a terceros.
    const build_files: []const []const u8 = &.{ "build.zig", "build.zig.zon" };
    for (build_files) |bf| {
        const content = root.readFileAlloc(io, bf, alloc, .limited(1 << 20)) catch |err| {
            std.log.err("gate_purity: no se pudo leer {s}: {s}", .{ bf, @errorName(err) });
            violations += 1;
            continue;
        };
        defer alloc.free(content);
        for (banned_in_build) |kw| {
            if (std.mem.indexOf(u8, content, kw) != null) {
                std.log.err("VIOLACION PUREZA: '{s}' encontrado en {s}", .{ kw, bf });
                violations += 1;
            }
        }
    }

    if (violations > 0) {
        std.log.err("gate_purity FALLO: {d} violaciones (escaneados {d} archivos).", .{ violations, scanned });
        std.process.exit(1);
    }
    std.log.info("gate_purity: OK — 0 archivos no-Zig, 0 FFI, 0 enlaces a terceros ({d} archivos escaneados).", .{scanned});
}
