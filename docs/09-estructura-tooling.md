# 09 — Estructura del proyecto y tooling

[← Plan Maestro](../PLAN-MAESTRO.md) · [← 08 Roadmap](08-roadmap-fases.md) · [→ 10 Riesgos y decisiones](10-riesgos-decisiones.md)

Layout del repositorio, sistema de build (`build.zig`), testing y los gates de CI escritos en Zig que materializan el Art. IX.

---

## 1. Layout del repositorio

```
Proyecto - Zig Super Agent/
├── PLAN-MAESTRO.md            # índice del plan
├── CONSTITUCION.md            # reglas inviolables (protegido por CODEOWNERS)
├── AGENTS.md                  # gobernanza
├── build.zig                  # grafo de build (protegido)
├── build.zig.zon              # manifiesto + Zig fijado (protegido)
├── budgets.zon                # presupuestos: tamaño binario, RSS, latencia/token
├── CODEOWNERS                 # revisión humana de zonas críticas
│
├── docs/                      # este plan (01..11) + adr/ + PROFESORES.md
│   ├── 01..11-*.md
│   ├── adr/                   # Architecture Decision Records (0001, 0002, ...)
│   └── PROFESORES.md          # registro de modelos profesores + licencias
│
├── src/
│   ├── main.zig               # CLI `fluency` (dispatch de subcomandos + arranca TUI)
│   ├── lib.zig                # superficie de librería (para libfluency/wasm)
│   ├── util/                  # json, bytes, hash, rng, crypto
│   ├── ztensor/               # tensores, dtypes, bf16, mmap
│   ├── zkernel/               # kernels SIMD + *_ref + backend vtable
│   │   ├── cpu/               # backend CPU SIMD (default)
│   │   ├── wasm/              # backend WASM SIMD128
│   │   └── gpu/               # backend GPU-Zig (SPIR-V/PTX) — F7
│   ├── zgrad/                 # autodiff
│   ├── zload/                 # GGUF + safetensors
│   ├── ztok/                  # tokenizadores
│   ├── zollama/               # cliente Ollama
│   ├── zarch/                 # bloque híbrido + mixers/
│   │   └── mixers/            # gla.zig mamba2.zig gdn.zig rwkv7.zig swa.zig sparse.zig
│   ├── ztrain/                # bucle + optimizadores
│   ├── zdistill/              # mohawk, kd, onpolicy, dpo
│   ├── zagent/                # runtime agéntico + tools/ + memory/
│   ├── zrpc/                  # JSON-RPC 2.0 + framings
│   ├── zmcp/                  # cliente + servidor MCP
│   ├── zserve/                # API OpenAI + Ollama-style + SSE
│   ├── zlsp/                  # LSP (opcional)
│   ├── ztui/                  # CLI/TUI (render celdas, input, markdown, diff)
│   └── platform/              # shims de SO/driver + terminal cruda (única zona con extern, por ADR)
│
├── tools/                     # GATES (protegidos) — escritos en Zig
│   ├── gate_purity.zig
│   ├── gate_constitution.zig
│   ├── gate_quality.zig
│   └── gate_budgets.zig
│
├── configs/                   # *.zon de modelos (subz-100m, subz-1b, patrón híbrido)
├── data/                      # recetas de datos (no datasets pesados)
├── eval/                      # suites de evaluación
├── plugin/                    # empaquetado como plugin MCP (plugin.json, .mcp.json)
└── .github/ (o CI equivalente)/workflows/  # corre `zig build gate` y `zig build test`
```

> **Nota:** sólo hay archivos `.zig`, `.md`, `.zon` y assets de datos. **Cero** `.c/.py/.rs/...` (lo garantiza `gate_purity`).

---

## 2. `build.zig` (esquema)

```zig
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // --- módulos ---
    const util   = b.addModule("util",   .{ .root_source_file = b.path("src/util/util.zig") });
    const ztensor= b.addModule("ztensor",.{ .root_source_file = b.path("src/ztensor/ztensor.zig") });
    // ... zkernel, zgrad, zload, ztok, zarch, ztrain, zdistill, zagent, zrpc, zmcp, zserve, ztui ...

    // --- ejecutable CLI ---
    const fluency = b.addExecutable(.{ .name = "fluency", .root_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"), .target = target, .optimize = optimize,
    })});
    // imports de módulos en fluency.root_module ...
    b.installArtifact(fluency);

    // --- librería C-ABI ---
    const libfluency = b.addLibrary(.{ .name = "fluency", .linkage = .dynamic, .root_module = /* lib.zig */ });
    b.installArtifact(libfluency);

    // --- WASM ---  (target wasm32-freestanding, ReleaseSmall, -rdynamic)

    // --- tests ---
    const tests = b.addTest(.{ .root_module = /* raíz que importa todos los módulos */ });
    const run_tests = b.addRunArtifact(tests);
    b.step("test", "Run tests (con detección de leaks)").dependOn(&run_tests.step);

    // --- GATES ---
    const gate = b.step("gate", "Enforcement constitucional (Art. IX)");
    inline for (.{ "gate_purity", "gate_constitution", "gate_quality", "gate_budgets" }) |g| {
        const exe = b.addExecutable(.{ .name = g, .root_module = b.createModule(.{
            .root_source_file = b.path("tools/" ++ g ++ ".zig"), .target = target, .optimize = optimize }) });
        gate.dependOn(&b.addRunArtifact(exe).step);
    }

    // --- eval / bench ---
    b.step("eval", "Evaluación del modelo");
    b.step("bench", "Benchmarks de kernels y latencia");
}
```

Recordatorios de la API 0.16 (verificados): módulo raíz explícito (`root_module`, ya no implícito), `addLibrary` con `linkage`. **Nunca** `linkLibC`/`linkSystemLibrary`/`addTranslateC` (los prohíbe la Constitución y los detecta `gate_purity`).

---

## 3. `build.zig.zon` (esquema)

```zig
.{
    .name = .zig_super_agent,
    .version = "0.0.0",
    .minimum_zig_version = "0.16.0",
    .fingerprint = 0x...,           // generado por zig
    .dependencies = .{},            // VACÍO por defecto. Toda dependencia debe ser Zig puro y aprobada por ADR.
    .paths = .{ "build.zig", "build.zig.zon", "src", "tools", "configs", "eval" },
}
```
> `dependencies` vacío es la postura por defecto (Art. I). Si alguna vez se añade una, `gate_purity` verifica que apunte a un paquete **Zig puro** y debe haber un ADR.

---

## 4. `budgets.zon` (presupuestos verificados por `gate_budgets`)

```zig
.{
    .infer_binary_bytes_max = 5_000_000,   // binario de inferencia release
    .agent_binary_bytes_max = 8_000_000,
    .rss_peak_mb_max        = 64,          // inferencia Fluency-350M en CPU (objetivo)
    .latency_ms_per_token_max = 50,        // por target/modelo (tabla en el .zon)
    .wasm_bytes_max         = 3_000_000,
}
```
Regresión por encima del umbral ⇒ el gate falla, salvo ADR que actualice el presupuesto (cambio visible y justificado, Art. V.5).

---

## 5. Los gates (qué verifica cada uno)

| Gate | Verifica | Falla si… |
|---|---|---|
| `gate_purity` | extensiones de archivo; ausencia de `linkLibC`/`linkSystemLibrary`/`@cImport`/`addTranslateC`; `extern` sólo en `src/platform/` con ADR | hay un archivo no-Zig o un enlace/FFI a terceros |
| `gate_constitution` | hash de `CONSTITUCION.md`; presencia y coherencia de AGENTS.md; integridad de `tools/gate_*` | la Constitución cambió sin enmienda firmada, o falta/diverge un archivo de gobernanza |
| `gate_quality` | `zig build test` sin leaks; cada módulo público con tests; cada kernel con test `*_ref`; ratio subcuadrático ≥ umbral en configs por defecto | hay leak, módulo sin test, kernel sin referencia, o config viola II.3 |
| `gate_budgets` | tamaños y latencias contra `budgets.zon` | regresión sobre umbral sin ADR |

Todos corren en **pre-commit** (versión rápida, `gate-fast`) y en **CI** (versión completa). Art. IX.5/IX.6: se autoprotegen y no se pueden desactivar sin revisión humana.

---

## 6. Testing

- **Unidad:** `test "..."` por módulo; allocator de testing (detecta leaks → Art. V.1).
- **Kernels:** cada kernel SIMD vs su `*_ref` escalar, igualdad dentro de tolerancia; tests de estabilidad numérica en secuencias largas (Art. V.2, §03 §6).
- **Determinismo:** misma semilla ⇒ misma salida (Art. V.3).
- **Integración:** F1 valida inferencia vs Ollama; F3 valida destilación vs baseline; F4 valida tareas agénticas; F5 valida MCP/embebido.
- **Eval de modelo (`fluency eval`):** suites de calidad + regresión, comparación contra profesor/baseline, reportes versionados.

---

## 7. CI (esquema)

```yaml
# pseudo-CI (adaptar al proveedor)
steps:
  - setup-zig: { version: "0.16.0" }   # versión fijada
  - run: zig build gate                # pureza + constitución + calidad + presupuestos
  - run: zig build test                # tests + leaks
  - run: zig build eval                # si cambió el modelo
  - protected-paths-review:            # CODEOWNERS: CONSTITUCION.md, build.zig*, tools/gate_*, src/platform/, gobernanza
```

## 8. Convenciones de código

- Estilo Zig idiomático 0.16; `zig fmt` obligatorio (gate opcional).
- Allocator explícito en toda API que asigna; documentar ownership (evita el footgun de vtable colgante de [05](05-runtime-agentico.md) §10).
- Errores como `error{...}!T`; **nunca `panic`** en rutas de producción (sí en `unreachable` de invariantes internas comprobadas).
- Comentarios en español o inglés, consistentes con el archivo; densidad como el código circundante.
- Un ADR (`docs/adr/NNNN-*.md`) por decisión arquitectónica (Art. VIII).

[→ Siguiente: 10 — Riesgos y decisiones](10-riesgos-decisiones.md)
