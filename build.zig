const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // --- módulos ---
    const util = b.addModule("util", .{
        .root_source_file = b.path("src/util/util.zig"),
    });
    const ztensor = b.addModule("ztensor", .{
        .root_source_file = b.path("src/ztensor/ztensor.zig"),
    });
    const zkernel = b.addModule("zkernel", .{
        .root_source_file = b.path("src/zkernel/zkernel.zig"),
    });
    const zgrad = b.addModule("zgrad", .{
        .root_source_file = b.path("src/zgrad/zgrad.zig"),
    });
    const zload = b.addModule("zload", .{
        .root_source_file = b.path("src/zload/zload.zig"),
    });
    const ztok = b.addModule("ztok", .{
        .root_source_file = b.path("src/ztok/ztok.zig"),
    });
    const zollama = b.addModule("zollama", .{
        .root_source_file = b.path("src/zollama/zollama.zig"),
    });
    const zarch = b.addModule("zarch", .{
        .root_source_file = b.path("src/zarch/zarch.zig"),
    });
    const ztrain = b.addModule("ztrain", .{
        .root_source_file = b.path("src/ztrain/ztrain.zig"),
    });
    const zdistill = b.addModule("zdistill", .{
        .root_source_file = b.path("src/zdistill/zdistill.zig"),
    });
    const zagent = b.addModule("zagent", .{
        .root_source_file = b.path("src/zagent/zagent.zig"),
    });
    zagent.addImport("util", util);
    const zrpc = b.addModule("zrpc", .{
        .root_source_file = b.path("src/zrpc/zrpc.zig"),
    });
    const zmcp = b.addModule("zmcp", .{
        .root_source_file = b.path("src/zmcp/zmcp.zig"),
    });
    zmcp.addImport("zagent", zagent);
    const zserve = b.addModule("zserve", .{
        .root_source_file = b.path("src/zserve/zserve.zig"),
    });
    const zlsp = b.addModule("zlsp", .{
        .root_source_file = b.path("src/zlsp/zlsp.zig"),
    });
    const platform = b.addModule("platform", .{
        .root_source_file = b.path("src/platform/platform.zig"),
    });
    const ztui_engine = b.addModule("ztui_engine", .{
        .root_source_file = b.path("src/ztui/engine.zig"),
    });
    ztui_engine.addImport("zagent", zagent);
    ztui_engine.addImport("platform", platform);

    const ztui = b.addModule("ztui", .{
        .root_source_file = b.path("src/ztui/ztui.zig"),
    });
    ztui.addImport("zagent", zagent);
    ztui.addImport("platform", platform);
    ztui.addImport("ztui_engine", ztui_engine);

    const zspeculate = b.addModule("zspeculate", .{
        .root_source_file = b.path("src/zspeculate.zig"),
    });
    zspeculate.addImport("zarch", zarch);
    zspeculate.addImport("ztok", ztok);

    // === GPU kernels SPIR-V (M1) — REQUIERE REVISIÓN HUMANA (Art. IX.5) ===
    const spv_step = b.step("spv", "Compila kernels GPU a SPIR-V");
    const zig_exe = b.graph.zig_exe;
    const run_saxpy_spv = b.addSystemCommand(&.{
        zig_exe,
        "build-obj",
        "-target", "spirv64-vulkan",
        "-fno-llvm",
        "-OReleaseFast",
    });
    // Se emite en la caché y luego se instala en <prefix>/saxpy.spv (zig-out/ por defecto),
    // así funciona en un clon limpio donde zig-out/ aún no existe.
    const saxpy_spv = run_saxpy_spv.addPrefixedOutputFileArg("-femit-bin=", "saxpy.spv");
    run_saxpy_spv.addFileArg(b.path("src/zgpu/kernels/saxpy.zig"));
    run_saxpy_spv.setName("compilar saxpy.spv");
    spv_step.dependOn(&b.addInstallFile(saxpy_spv, "saxpy.spv").step);

    // Cableado del grafo de módulos (imports entre módulos).
    zarch.addImport("zkernel", zkernel);
    ztrain.addImport("ztensor", ztensor);
    // zdistill (motor de aprendizaje F3) usa el autodiff y las utilidades de train.
    zdistill.addImport("zgrad", zgrad);
    zdistill.addImport("ztrain", ztrain);

    // --- ejecutable CLI ---
    const fluency_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    fluency_mod.addImport("util", util);
    fluency_mod.addImport("ztensor", ztensor);
    fluency_mod.addImport("zkernel", zkernel);
    fluency_mod.addImport("zgrad", zgrad);
    fluency_mod.addImport("zload", zload);
    fluency_mod.addImport("ztok", ztok);
    fluency_mod.addImport("zollama", zollama);
    fluency_mod.addImport("zarch", zarch);
    fluency_mod.addImport("ztrain", ztrain);
    fluency_mod.addImport("zdistill", zdistill);
    fluency_mod.addImport("zagent", zagent);
    fluency_mod.addImport("zrpc", zrpc);
    fluency_mod.addImport("zmcp", zmcp);
    fluency_mod.addImport("zserve", zserve);
    fluency_mod.addImport("zlsp", zlsp);
    fluency_mod.addImport("ztui", ztui);
    fluency_mod.addImport("platform", platform);
    fluency_mod.addImport("zspeculate", zspeculate);

    const fluency = b.addExecutable(.{
        .name = "fluency",
        .root_module = fluency_mod,
    });
    b.installArtifact(fluency);

    // --- librería C-ABI ---
    const libfluency_mod = b.createModule(.{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });
    libfluency_mod.addImport("zload", zload);
    libfluency_mod.addImport("ztok", ztok);
    libfluency_mod.addImport("zarch", zarch);
    libfluency_mod.addImport("zkernel", zkernel);
    libfluency_mod.addImport("ztensor", ztensor);
    const libfluency = b.addLibrary(.{
        .name = "fluency",
        .linkage = .dynamic,
        .root_module = libfluency_mod,
    });
    b.installArtifact(libfluency);

    // --- WASM target ---
    const wasm_target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .freestanding,
    });
    const wasm_mod = b.createModule(.{
        .root_source_file = b.path("src/lib.zig"),
        .target = wasm_target,
        .optimize = optimize,
    });
    wasm_mod.export_symbol_names = &.{
        "fluency_version",
        "fluency_demo_generate",
        "fluency_alloc",
        "fluency_free_ptr",
    };
    wasm_mod.addImport("zload", zload);
    wasm_mod.addImport("ztok", ztok);
    wasm_mod.addImport("zarch", zarch);
    wasm_mod.addImport("zkernel", zkernel);
    wasm_mod.addImport("ztensor", ztensor);
    const wasm_exe = b.addExecutable(.{
        .name = "fluency",
        .root_module = wasm_mod,
    });
    wasm_exe.entry = .disabled;
    wasm_exe.link_gc_sections = false;
    b.installArtifact(wasm_exe);

    // --- tests ---
    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_mod.addImport("util", util);
    test_mod.addImport("ztensor", ztensor);
    test_mod.addImport("zkernel", zkernel);
    test_mod.addImport("zgrad", zgrad);
    test_mod.addImport("zload", zload);
    test_mod.addImport("ztok", ztok);
    test_mod.addImport("zollama", zollama);
    test_mod.addImport("zarch", zarch);
    test_mod.addImport("ztrain", ztrain);
    test_mod.addImport("zdistill", zdistill);
    test_mod.addImport("zagent", zagent);
    test_mod.addImport("zrpc", zrpc);
    test_mod.addImport("zmcp", zmcp);
    test_mod.addImport("zserve", zserve);
    test_mod.addImport("zlsp", zlsp);
    test_mod.addImport("ztui", ztui);
    test_mod.addImport("platform", platform);
    test_mod.addImport("zspeculate", zspeculate);

    const test_step = b.step("test", "Run tests (con detección de leaks)");
    test_step.dependOn(spv_step); // compila kernels SPIR-V antes de testear

    // Root (main.zig) con todos los módulos.
    const main_tests = b.addTest(.{ .root_module = test_mod });
    test_step.dependOn(&b.addRunArtifact(main_tests).step);

    // Tests por módulo auto-contenido (solo std + imports relativos).
    // NOTA: la raíz importa módulos por NOMBRE, y `zig test` no ejecuta los tests
    // de módulos importados así; hay que compilar cada módulo como su propia raíz.
    const standalone_test_files = [_][]const u8{
        "src/util/util.zig",
        "src/ztensor/ztensor.zig",
        "src/zkernel/zkernel.zig",
        "src/zgrad/zgrad.zig",
        "src/zload/zload.zig",
        "src/ztok/ztok.zig",
        "src/zollama/zollama.zig",
        "src/zrpc/zrpc.zig",
        "src/zmcp/zmcp.zig",
        "src/zserve/zserve.zig",
        "src/platform/tty.zig",
    };
    for (standalone_test_files) |tf| {
        const mt = b.addTest(.{ .root_module = b.createModule(.{
            .root_source_file = b.path(tf),
            .target = target,
            .optimize = optimize,
        }) });
        test_step.dependOn(&b.addRunArtifact(mt).step);
    }

    // zgpu depende del módulo platform (shim Vulkan).
    const zgpu_test_mod = b.createModule(.{
        .root_source_file = b.path("src/zgpu/zgpu.zig"),
        .target = target,
        .optimize = optimize,
    });
    zgpu_test_mod.addImport("platform", platform);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = zgpu_test_mod })).step);

    // lib.zig (C-ABI) depende de zload, ztok, zarch, zkernel, ztensor.
    const lib_test_mod = b.createModule(.{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib_test_mod.addImport("zload", zload);
    lib_test_mod.addImport("ztok", ztok);
    lib_test_mod.addImport("zarch", zarch);
    lib_test_mod.addImport("zkernel", zkernel);
    lib_test_mod.addImport("ztensor", ztensor);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = lib_test_mod })).step);

    // ztrain depende del módulo ztensor.
    const ztrain_test_mod = b.createModule(.{
        .root_source_file = b.path("src/ztrain/ztrain.zig"),
        .target = target,
        .optimize = optimize,
    });
    ztrain_test_mod.addImport("ztensor", ztensor);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = ztrain_test_mod })).step);

    // zarch (Transformer) depende del módulo zkernel.
    const zarch_test_mod = b.createModule(.{
        .root_source_file = b.path("src/zarch/zarch.zig"),
        .target = target,
        .optimize = optimize,
    });
    zarch_test_mod.addImport("zkernel", zkernel);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = zarch_test_mod })).step);

    // zagent (runtime agéntico) depende de util (json parser).
    const zagent_test_mod = b.createModule(.{
        .root_source_file = b.path("src/zagent/zagent.zig"),
        .target = target,
        .optimize = optimize,
    });
    zagent_test_mod.addImport("util", util);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = zagent_test_mod })).step);

    // zagent/memory (store híbrido) solo usa std.
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("src/zagent/memory/zmemory.zig"),
        .target = target,
        .optimize = optimize,
    }) })).step);

    // ztui (TUI interactiva) depende de zagent.
    const ztui_test_mod = b.createModule(.{
        .root_source_file = b.path("src/ztui/ztui.zig"),
        .target = target,
        .optimize = optimize,
    });
    ztui_test_mod.addImport("zagent", zagent);
    ztui_test_mod.addImport("platform", platform);
    ztui_test_mod.addImport("ztui_engine", ztui_engine);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = ztui_test_mod })).step);

    // ztui/engine.zig: motor cell-based. Se compila como su propio root de test
    // (zig test no corre los tests de un módulo importado por NOMBRE). Sus imports
    // relativos screen.zig y transcript.zig se testean al ser raíz.
    const ztui_engine_test_mod = b.createModule(.{
        .root_source_file = b.path("src/ztui/engine.zig"),
        .target = target,
        .optimize = optimize,
    });
    ztui_engine_test_mod.addImport("zagent", zagent);
    ztui_engine_test_mod.addImport("platform", platform);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = ztui_engine_test_mod })).step);

    // zspeculate depende de zarch y ztok.
    const zspeculate_test_mod = b.createModule(.{
        .root_source_file = b.path("src/zspeculate.zig"),
        .target = target,
        .optimize = optimize,
    });
    zspeculate_test_mod.addImport("zarch", zarch);
    zspeculate_test_mod.addImport("ztok", ztok);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = zspeculate_test_mod })).step);

    // zdistill (FluencyNano + Trainer + eval) depende de zgrad y ztrain.
    const zdistill_test_mod = b.createModule(.{
        .root_source_file = b.path("src/zdistill/zdistill.zig"),
        .target = target,
        .optimize = optimize,
    });
    zdistill_test_mod.addImport("zgrad", zgrad);
    zdistill_test_mod.addImport("ztrain", ztrain);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = zdistill_test_mod })).step);

    // F2: mixers subcuadráticos + híbrido + gramática. Cada archivo se compila
    // como su propio root de test (zig test no corre los tests de los @import).
    const mixer_test_files = [_][]const u8{
        "src/zarch/mixers/mamba2.zig",
        "src/zarch/mixers/swa.zig",
        "src/zarch/mixers/gated_deltanet.zig",
        "src/zarch/mixers/rwkv7.zig",
        "src/zarch/mixers/hybrid.zig",
    };
    for (mixer_test_files) |tf| {
        const mt = b.createModule(.{
            .root_source_file = b.path(tf),
            .target = target,
            .optimize = optimize,
        });
        mt.addImport("zkernel", zkernel); // los mixers (y hybrid vía import) usan zkernel
        test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mt })).step);
    }
    // grammar.zig sólo usa std.
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("src/zarch/grammar.zig"),
        .target = target,
        .optimize = optimize,
    }) })).step);

    // --- GATES ---
    const gate = b.step("gate", "Enforcement constitucional (Art. IX)");
    inline for (&.{ "gate_purity", "gate_constitution", "gate_quality", "gate_budgets" }) |g| {
        const gate_mod = b.createModule(.{
            .root_source_file = b.path("tools/" ++ g ++ ".zig"),
            .target = target,
            .optimize = optimize,
        });
        const exe = b.addExecutable(.{
            .name = g,
            .root_module = gate_mod,
        });
        gate.dependOn(&b.addRunArtifact(exe).step);
    }

    // --- preview de la TUI (dev): usa el motor real para volcar frames ---
    const preview_mod = b.createModule(.{
        .root_source_file = b.path("src/ztui/preview.zig"),
        .target = target,
        .optimize = optimize,
    });
    preview_mod.addImport("ztui_engine", ztui_engine);
    const preview_exe = b.addExecutable(.{ .name = "tui_preview", .root_module = preview_mod });
    const preview_step = b.step("preview", "Vista previa no-interactiva de la TUI (frames de texto)");
    preview_step.dependOn(&b.addRunArtifact(preview_exe).step);

    // --- eval / bench ---
    _ = b.step("eval", "Evaluación del modelo");
    _ = b.step("bench", "Benchmarks de kernels y latencia");
}
