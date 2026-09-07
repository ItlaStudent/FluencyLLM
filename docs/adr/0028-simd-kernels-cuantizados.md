# ADR-0028: SIMD en kernels cuantizados (matmulVecQ*)

**Estado:** Aceptado  
**Fecha:** 2026-06-02  
**Decisores:** ZSA (Zig Super Agent)  
**Contexto:** TAREA 1 — FASE 7 (rendimiento / evolución)

## Contexto

Los kernels `matmulVecQ4_0`/`Q8_0`/`Q4_K`/`Q6_K` dequantizan bloque a bloque en un buffer stack y luego hacen el dot-product con el vector de entrada de forma **escalar** (bucle `for` elemento a elemento). En un modelo Qwen2.5-coder-1.5B Q4_K_M esto produce ~0.37 tokens/s en CPU (ReleaseFast), lo cual es lento para una TUI o serve interactivo.

El objetivo es acelerar el dot-product por bloque usando SIMD de Zig (`@Vector` + `@reduce`), manteniendo 100 % Zig puro, sin función foránea, sin intrínsecos de C, y sin tocar la lógica de dequantización (que es correcta y validada).

## Decisión

1. **Mantener dequantización escalar, vectorizar sólo el dot-product.** La dequantización de bloque (extracción de nibbles, lookups de escala, etc.) es compleja de vectorizar con `@Vector` sin operaciones de empaquetado/desempaquetado de laneas de distinto tamaño. El dot-product, en cambio, es un acumulador SIMD directo: `acc_vec += dequantized * x`.

2. **Usar `@Vector(VecWidth, f32)` con `VecWidth = 8`** (256 bits). Esto mapea a AVX2 en x86_64 moderno; el compilador Zig 0.16 lo baja a instrucciones SIMD nativas. Cargas alineadas o desalineadas se hacen con `*align(1) const Vec` (patrón ya usado en `matmul` SIMD de `zkernel.zig`).

3. **Helper `inline fn dotSimd(deq, x)`** que:
   - Acumula en un vector `@Vector(8, f32)` inicializado a cero.
   - Procesa el bloque en chunks de 8 elementos.
   - Reduce el vector vía `@reduce(.Add, acc_vec)` al final del bloque (tree reduction, más preciso que acumulación escalar secuencial).
   - Procesa cola escalar si el bloque no es múltiplo de 8 (defensa futura, aunque los bloques actuales sí lo son).

4. **Conservar `_ref` escalares.** Se añaden `matmulVecQ4_0Ref`/`Q8_0Ref`/`Q4_KRef`/`Q6_KRef` como copias exactas del código escalar original. Sirven para:
   - Validación numérica en tests (`SIMD vs _ref`).
   - Fallback futuro si se detecta regresión en alguna arquitectura.
   - Cumplir Art. V.2 de la Constitución: "kernels numéricos DEBEN validarse contra implementación de referencia lenta y legible".

5. **Sin cambios en `zarch.zig` ni en la carga.** Los nombres de función públicos no cambian; `zarch.zig` sigue llamando a `matmulVecQ4_0` etc. La optimización es transparente para el resto del sistema.

## Consecuencias

**Positivas:**
- **Speedup medible en vivo:** ~1.6× más rápido en el modelo real Q4_K_M (0.37 → 0.59 tokens/s, ReleaseFast).
- Sin dependencias ni código no-Zig. Cumple Art. I de la Constitución.
- Los tests existentes y los nuevos "SIMD vs ref" garantizan igualdad numérica dentro de tolerancia de cuantización.

**Negativas / trade-offs / limitaciones honestas:**
- El speedup (~1.6×) es **modesto** porque la dequantización escalar todavía consume una fracción significativa del tiempo. Vectorizar también la dequantización (especialmente Q8_0, donde es trivial) podría dar un extra ~10-20 %.
- En debug el speedup es menor (o incluso imperceptible) porque el compilador no vectoriza agresivamente; ReleaseFast es donde se materializa.
- La tolerancia numérica entre SIMD y escalar es mayor para Q6_K (hasta ~1e-2) debido a la magnitud de los valores y al tree reduction de `@reduce(.Add, ...)`, que es **más preciso** que el acumulador escalar secuencial pero produce resultados ligeramente distintos. Esto es aceptable para inferencia cuantizada.

## Alternativas consideradas

- **Vectorizar también la dequantización Q8_0** (cargar 8 i8, `@floatFromInt` a `@Vector(8, f32)`, multiplicar por scale vector): Rechazado para esta tarea porque el ganador claro es el dot-product y queríamos un cambio mínimo y seguro. Se puede retomar como mejora incremental.
- **Procesar múltiples filas N en paralelo con SIMD**: Rechazado. Cambiaría la estructura de memoria (se necesitaría transponer o intercalar dequantización de varias filas), con riesgo de complejidad y regresión. El dot-product por bloque es el punto de fricción identificado.
- **Usar inline assembly o intrínsecos de C**: Prohibido por Constitución (Art. I.2/I.3). `@Vector` es la vía Zig pura.

## Evidencia

- `zig build`: ✅ compila.
- `zig build test`: **180/180 tests pasan**, sin fugas (subió de 176).
- `zig build gate`: 4 gates OK.
- Tests nuevos: `matmulVecQ4_0 SIMD vs ref`, `matmulVecQ8_0 SIMD vs ref`, `matmulVecQ4_K SIMD vs ref`, `matmulVecQ6_K SIMD vs ref` (K multi-bloque para ejercitar múltiples iteraciones del vector).
- **Medición en vivo** (Qwen2.5-coder-1.5B-Q4_K_M, ReleaseFast, prompt "def fib(n):", 12 tokens, Windows 11, CPU desconocida):
  - Escalar (_ref): **32.5 s** → **0.37 tokens/s**
  - SIMD: **20.2 s** → **0.59 tokens/s**
  - **Speedup: ~1.6×**
  - **Misma salida greedy:** `def fib(n):\n    if n == 0:\n        return 0`
