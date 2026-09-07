# ADR-0037 — Backend GPU vía Vulkan compute + SPIR-V compilado desde Zig

**Estado:** Bloqueado en M3 — re-evaluar en Zig ≥0.17
**Fecha:** 2026-06-06
**Contexto:** F7.D (GPU) — ADR-0035 declaró el backend SPIR-V "no funcional" por triples inválidos; ADR-0037 corrigió el triple y validó compilación (M1/M2), pero el backend self-hosted SPIR-V de Zig 0.16 no genera código ejecutable en Vulkan compute.

## Contexto

### Validación M1 — Compilación a SPIR-V
La auditoría 2026-06-06 demostró que el triple correcto (`spirv64-vulkan`) sí produce un
archivo `.spv` con magic válido (`03 02 23 07`):

- `zig build-obj -target spirv64-vulkan -fno-llvm kernel.zig` genera `saxpy.spv` (~2 KB).
- Test `saxpy.spv magic bytes` pasa (verifica magic big-endian). *(Archivo conservado como
evidence de bitstream — Art. I.5.)*
- Runtime Vulkan presente: `vulkan-1.dll` v1.4.309 + ICD AMD `amdxc64.dll` (RX 6600 XT, RDNA2).

### Validación M2 — Detección de GPU viva
El shim en `src/platform/vulkan/` enumera el dispositivo sin `linkSystemLibrary`:

- `enumeratePhysicalDevices` devuelve "AMD Radeon RX 6600 XT".
- Degradación limpia `error.NoVulkan` cuando no hay driver (sin panic, Art. IV.2).
- Test `detect gpu` **skip** en CI / máquinas sin GPU; **PASS** en vivo con la RX 6600 XT.

### Bloqueo M3 — SPIR-V emitido no es válido para Vulkan compute
El backend self-hosted SPIR-V de Zig 0.16 emite un bitstream con **estructura OpenCL**
en vez de **Vulkan compute**. Los 3 defectos constatados impiden crear un pipeline
Vulkan válido:

| Defecto | Evidencia del `.spv` | Impacto en Vulkan |
|---------|----------------------|-------------------|
| **1. Address space ignorado** | `addrspace(.global)` y `addrspace(.storage_buffer)` producen **SPIR-V bit-idéntico** (desensamblado `saxpy_disasm.txt`). No aparece `OpMemoryModel Physical64` ni `OpVariable … StorageBuffer`. | `vkCreateShaderModule` requiere `OpMemoryModel Logical` + variables `StorageBuffer` con `Block`/`Offset`. |
| **2. OpVariable sin StorageBuffer** | `OpVariable` usa clase de almacenamiento `Function` (stack) en vez de `StorageBuffer` para los punteros de entrada/salida. | Los buffers device-local no se pueden enlazar a descriptor sets. |
| **3. Sin OpExecutionMode LocalSize** | No hay decoración de work-group size (`LocalSize 1 1 1`). | Vulkan compute exige `OpExecutionMode %main LocalSize X Y Z` explícita. |

**Conclusión técnica:** Zig 0.16.0 compila a SPIR-V, pero el backend self-hosted **ignora
semántica de Vulkan compute** (address spaces, descriptor bindings, execution mode). Es un
**bug del compilador**, no del código Fluency. Intentar forzar un workaround (p.ej. ensamblar
SPIR-V a mano o inyectar opcodes) violaría Art. I.4 (código Zig compilado, no artefactos
manuales) y Art. V.2 (referencia verificable).

## Decisión

1. **Mantenido M1/M2** como validación del pipeline build + detección de GPU.
2. **M3 declarado BLOQUEADO** por el compilador Zig 0.16. No se añade código nuevo de GPU
   ni se modifica `build.zig` sin revisión humana (Art. IX.5).
3. **Plan de reactivación:** re-evaluar el backend SPIR-V en Zig 0.17 o posterior. Si el
   backend self-hosted corrige `OpVariable StorageBuffer`, `OpExecutionMode LocalSize` y
   decoraciones `Binding`/`DescriptorSet`, se reanuda M3 sin cambios arquitectónicos.
4. **Fallback CPU intacto:** el entrenamiento e inferencia siguen sobre CPU 100 % funcional.

### Por qué es constitucional

- **Art. I.4:** la única ruta GPU aceptable es código Zig → SPIR-V nativo. No se recurre
  a CUDA/HIP/Metal/OpenCL-C/Triton ni a artefactos ensamblados a mano.
- **Art. I.5:** el `.spv` existente (`saxpy.spv`, `saxpy_v2.spv`) se conserva como
  evidencia de bitstream M1; NO se usa en runtime (no es funcional para Vulkan).
- **gate_purity:** no se introduce FFI ni librerías no-Zig.

## Plan de implementación (hitos verificables) — actualizado

| Hito | Estado | Evidencia |
|------|--------|-----------|
| **M1 — Kernel + artefacto build** | ✅ **HECHO** | `zig build spv` produce `.spv` reproducible; test `saxpy.spv magic bytes` pasa. |
| **M2 — Shim Vulkan mínimo** | ✅ **HECHO** | Test `detect gpu` enumera RX 6600 XT en vivo; skip limpio sin GPU. |
| **M3 — Dispatch SAXPY end-to-end** | ❌ **BLOQUEADO** | Backend Zig 0.16 ignora address spaces / sin `LocalSize` / `OpVariable` en `Function`. |
| **M4 — `matmul` GPU** | ⏸️ **PENDIENTE** | Bloqueado por M3. |
| **M5 — Integración entrenamiento** | ⏸️ **PENDIENTE** | Bloqueado por M3. |

## Consecuencias

- La aceleración GPU queda **fuera de alcance hasta Zig ≥0.17** (o hasta que el backend
  self-hosted SPIR-V soporte Vulkan compute semánticamente).
- **Sin regresión:** gates, tests, y fallback CPU permanecen verdes; `saxpy.spv` sigue
  sirviendo como test de bitstream M1.
- Opción B (`amdgcn-amdhsa`) sigue teóricamente viable pero prácticamente descartada en
  Windows sin runtime ROCm/HSA.

## Alternativas consideradas

- **`amdgcn-amdhsa`:** requiere runtime ROCm/HSA para cargar el ELF; no hay soporte nativo
  en Windows para gfx1032 (RX 6600 XT). Descartada prácticamente.
- **Ensamblar SPIR-V a mano / inyectar opcodes:** violaría Art. I.4 (artefacto no-Zig
  ensamblado) y Art. V.2 (sin referencia CPU verificable). Rechazada.
- **WebGPU/WASM:** requiere host JS; no es Zig puro. Rechazada.

## Artículo afectado

Art. I.3 (shim de plataforma con ADR), Art. I.4 (Zig→GPU), Art. I.5 (bitstream conservado),
Art. VII.1 (vtable `Backend`), Art. V.2 (kernel validado vs referencia), Art. IX.5
(revisión humana de `build.zig`/`platform`).
