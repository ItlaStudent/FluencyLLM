const std = @import("std");

// Art. IX.2 — Integridad constitucional: verifica que existan los archivos de
// gobernanza y que CONSTITUCION.md no haya cambiado (hash SHA-256 fijado en
// tools/constitution.sha256). Cambiarla requiere enmienda humana (Art. X) que
// re-genere ese hash.

const required_files: []const []const u8 = &.{
    "AGENTS.md",
};

fn toHexLower(digest: [32]u8) [64]u8 {
    const hexchars = "0123456789abcdef";
    var out: [64]u8 = undefined;
    for (digest, 0..) |b, i| {
        out[i * 2] = hexchars[b >> 4];
        out[i * 2 + 1] = hexchars[b & 0x0f];
    }
    return out;
}

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;
    const dir = std.Io.Dir.cwd();
    var violations: u32 = 0;

    // 1. Archivos de gobernanza por-IDE.
    for (required_files) |rf| {
        dir.access(io, rf, .{}) catch {
            std.log.err("VIOLACION: falta archivo de gobernanza: {s}", .{rf});
            violations += 1;
        };
    }

    // 2. tools/gate_* existen.
    const gate_files: []const []const u8 = &.{ "gate_purity.zig", "gate_constitution.zig", "gate_quality.zig", "gate_budgets.zig" };
    for (gate_files) |gf| {
        const path = try std.fs.path.join(alloc, &.{ "tools", gf });
        defer alloc.free(path);
        dir.access(io, path, .{}) catch {
            std.log.err("VIOLACION: falta gate: {s}", .{path});
            violations += 1;
        };
    }

    // 3. Integridad de CONSTITUCION.md (hash SHA-256).
    const constitution = dir.readFileAlloc(io, "CONSTITUCION.md", alloc, .limited(2 << 20)) catch |err| {
        std.log.err("VIOLACION: no se pudo leer CONSTITUCION.md: {s}", .{@errorName(err)});
        std.process.exit(1);
    };
    defer alloc.free(constitution);

    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(constitution, &digest, .{});
    const actual = toHexLower(digest);

    const expected_raw = dir.readFileAlloc(io, "tools/constitution.sha256", alloc, .limited(1024)) catch {
        std.log.err("VIOLACION: falta tools/constitution.sha256. Hash actual de CONSTITUCION.md = {s}", .{actual});
        std.log.err("Si el cambio es una enmienda humana legítima (Art. X), fija ese hash en el archivo.", .{});
        std.process.exit(1);
    };
    defer alloc.free(expected_raw);
    const expected = std.mem.trim(u8, expected_raw, " \t\r\n");

    if (!std.ascii.eqlIgnoreCase(expected, &actual)) {
        std.log.err("VIOLACION: CONSTITUCION.md fue MODIFICADA sin enmienda.", .{});
        std.log.err("  esperado: {s}", .{expected});
        std.log.err("  actual:   {s}", .{actual});
        std.log.err("  Sólo un humano autorizado (Art. X) puede cambiarla y re-fijar el hash.", .{});
        violations += 1;
    }

    if (violations > 0) {
        std.log.err("gate_constitution FALLO: {d} violaciones.", .{violations});
        std.process.exit(1);
    }
    std.log.info("gate_constitution: OK — gobernanza intacta, CONSTITUCION.md íntegra.", .{});
}
