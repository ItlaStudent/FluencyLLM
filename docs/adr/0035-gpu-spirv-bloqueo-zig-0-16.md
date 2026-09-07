# ADR-0035: Backend GPU vía Zig a SPIR-V — ~~BLOQUEADO~~ DESBLOQUEADO (corregido 2026-06-06)

**Estado:** ⚠️ **CORREGIDO** — el veredicto original ("bloqueado") era **erróneo**: usaba triples de target inválidos. La compilación Zig→SPIR-V **SÍ funciona** en zig 0.16.0. Ver ADR-0037.
**Fecha:** 2026-06-06
**Decisores:** ZSA (Zig Super Agent)
**Contexto:** TAREA 4 — FASE 7 (GPU exploratorio)

> ## 🔴 CORRECCIÓN (auditoría 2026-06-06)
>
> Este ADR concluyó "backend SPIR-V no funcional". **Es falso.** El intento usó
> triples **incorrectos**: `spirv64` (sin componente de OS → `MissingOperatingSystem`,
> error esperado) y `spirv64-freestanding` (OS inválido para SPIR-V). Los targets GPU
> reales que expone `zig targets` son `spirv64-vulkan`, `spirv64-opencl` y
> `amdgcn-amdhsa`.
>
> **Verificación empírica (este equipo, zig 0.16.0):**
> - `zig build-obj -target spirv64-vulkan kernel.zig` → **`kernel.spv` válido** (magic
>   `03 02 23 07`). Un kernel SAXPY realista (punteros `addrspace(.global)` a arrays,
>   bucle, multiply-add, 5 parámetros, `callconv(.kernel)`) compila en **Debug**
>   (14.308 b) y en **`-fno-llvm -O ReleaseFast`** (12.808 b, optimizado).
> - Causa de los crashes en `-O Release*` SIN `-fno-llvm`: enrutan por el backend
>   `stage2_llvm`, que **no soporta** `callconv(.spirv_kernel)`
>   (`error: calling convention 'spirv_kernel' not supported by ... 'stage2_llvm'`).
>   **El backend correcto es el self-hosted**, que es el default en Debug y se fuerza
>   en Release con **`-fno-llvm`**.
> - Runtime presente: `vulkan-1.dll` v1.4.309 + ICD AMD `amdxc64.dll` (RX 6600 XT).
>
> **Conclusión corregida:** la *compilación* Zig→SPIR-V está **desbloqueada y probada**.
> Lo que falta (no bloqueado, es ingeniería) es la *ejecución end-to-end*: un shim
> Vulkan en `src/platform/` que cargue el `.spv`, despache en la GPU y lea resultados,
> validado bit-a-bit contra el matmul `_ref` de CPU. Plan en **ADR-0037**.

## Contexto

F7.D del roadmap pide un kernel matmul a SPIR-V compilado desde Zig puro, validado bit-a-bit contra el matmul `_ref` de CPU. La restricción constitucional prohíbe CUDA/Triton/Metal/ROCm como dependencia. El único camino viable es el backend SPIR-V nativo del compilador Zig.

## Intento de verificación

Se intentó compilar el programa mínimo siguiente con `zig-0.16.0`:

```zig
export fn add(a: f32, b: f32) f32 {
    return a + b;
}
```

Comandos probados y resultados:

1. `zig build-obj gpu_test.zig -target spirv64`
   - **Resultado:** `error: unable to parse target query 'spirv64': MissingOperatingSystem`
2. `zig build-obj gpu_test.zig -target spirv64-freestanding -mcpu baseline`
   - **Resultado:** Sin output, sin archivo objeto generado, sin error reportado (exit code indeterminado).
3. `zig build-obj gpu_test.zig -target spirv64-freestanding -femit-bin=gpu_test.spv`
   - **Resultado:** `error: unrecognized file extension of parameter '.spv'`
4. `zig build-obj gpu_test.zig -target spirv64-freestanding -femit-bin=gpu_test.o`
   - **Resultado:** Sin output, sin archivo `.o` generado.

## Decisión

**NO forzar el backend GPU.** El backend SPIR-V de Zig 0.16.0 no es funcional para este caso de uso:
- No acepta `spirv64` sin OS (requiere hack `freestanding` que no documenta).
- No emite archivos objeto/binario `.o`, `.spv` ni sin extensión explícita.
- No reporta errores útiles cuando falla.

Se registra este ADR como bloqueo honesto. El código del proyecto **no se modifica** para incorporar stubs GPU ni FFI a terceros. Si una versión futura de Zig (≥0.17) soporta SPIR-V/Vulkan limpiamente, se reevaluará.

## Consecuencias

- GPU permanece como **capacidad futura**, no entregable de F7.
- El proyecto mantiene 100 % CPU-Zig, sin dependencias GPU externas.
- No se añade vtable `Backend` ni código muerto de GPU.

## Alternativas consideradas

- **Usar `@cImport` + Vulkan C SDK:** Rechazado por Constitución (no FFI, no C).
- **Emitir SPIR-V a mano (ensamblador):** Rechazado. Fuera de alcance, no valida el objetivo de compilar *desde Zig*.
- **Usar WASM + WebGPU:** Rechazado. WebGPU requiere bindings JS/Host; no es Zig puro ni comparable bit-a-bit con CPU.
- **Forzar un stub `Backend` con SPIR-V hardcodeado:** Rechazado. Violaba el espíritu de la tarea (kernel compilado desde Zig) y añadiría código muerto.
