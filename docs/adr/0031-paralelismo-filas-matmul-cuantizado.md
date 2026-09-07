# ADR-0031 — Paralelismo de filas en matmul cuantizado + dequantización SIMD Q8_0

## Contexto

Los kernels cuantizados on-the-fly (`matmulVecQ4_0`, `Q8_0`, `Q4_K`, `Q6_K`) en `src/zkernel/zkernel.zig` ya usaban `dotSimd` (acumulación `@Vector(8, f32)` + `@reduce(.Add, ...)`), lo que dio un speedup de ~1.6× sobre la versión escalar pura (de 0.37 a 0.59 tok/s en Qwen2.5-coder-1.5B-Q4_K_M, CPU). Sin embargo, la **dequantización por bloque** seguía siendo escalar: cada nibble/byte se convertía individualmente a `f32` antes de hacer el `dotSimd`. Además, las filas del matmul-vec se procesaban en **un único hilo**, desperdiciando los núcleos de CPU disponibles.

## Decisión

1. **Vectorizar la dequantización Q8_0 por bloque** (`dequantQ8_0_simd`): leer 8 bytes de `i8` como `@Vector(8, i8)`, castear a `@Vector(8, f32)` vía `@floatFromInt`, y multiplicar por escala splat (`@Vector(8, f32)`). Para los bytes restantes (< 8), fallback escalar.

2. **Paralelizar las filas del matmul-vec** con un helper `parallelForRows` inline en `zkernel.zig` (sin tocar `build.zig` ni añadir dependencias externas):
   - Cada fila `n` es independiente: `out[n] = dot(dequantized_row_n, x)`.
   - Usar `std.Thread.spawn`+`join` con chunking por rango contiguo.
   - Fallback a serial si: (a) target `single_threaded` (ej. WASM), (b) `n < 256`, o (c) `cpu_count <= 1`.
   - Umbral de trabajo adicional: `N * K >= 100_000` (evita overhead en operaciones pequeñas).

3. **Conservar las funciones `*_Ref`** como oráculo de corrección numérica.

## Consecuencias

- **Speedup medido en vivo** (Qwen2.5-coder-1.5B-Q4_K_M, ReleaseFast, prompt `"def fib(n):"`, 12 tokens):
  - Baseline SIMD (TAREA 1): **20.2 s** → **0.59 tok/s**
  - Con esta optimización: **13.1 s** → **0.91 tok/s**
  - **Speedup total: ~1.55×** sobre la línea SIMD actual; **~2.46×** sobre la línea escalar pura.
  - Salida greedy **idéntica** bit a bit: `def fib(n):\n    if n == 0:\n        return 0`.

- **Tests**: 5 nuevos (`dequant Q8_0 SIMD vs ref`, `matmulVecQ4_0/Q8_0/Q4_K/Q6_K parallel vs ref`). Tolerancias ajustadas: Q4_0/Q8_0 `1e-4`, Q4_K `5e-3`, Q6_K `5e-2` (el tree reduction SIMD acumula diferencias numéricas frente a la suma escalar en datasets grandes).

- **WAM seguro**: `parallelForRows` cae a serial en `builtin.single_threaded` (WASM `wasm32-freestanding`).

- **Limitación honesta**: el speedup está por debajo del objetivo aspiracional ≥2× sobre SIMD. El cuello de botalla residual es el overhead de `spawn`+`join` por cada llamada a `matmulVec` (~200 veces por token, 28 capas × 7 pesos). Un pool de hilos persistente (requiere `std.Thread.Pool`, eliminado en Zig 0.16, o implementación propia) podría reducir este overhead y acercarse al 2×. La dequantización de Q4_K/Q6_K es estructuralmente compleja (scales por sub-bloque) y no se vectorizó en este paso.

## Alternativas consideradas

- **Pool de hilos persistente**: mejoraría el speedup pero requiere un estado global en `zkernel.zig` o un cambio en la interfaz de `matmulVec`. Posible en iteración futura.
- **OpenMP / rayon**: vetado por la Constitución (Art. I: 100 % Zig, sin librerías de terceros).
- **Vectorizar Q4_0/Q4_K/Q6_K dequant**: Q4_0 requiere unpack de nibbles (shuffle/blast en SIMD), que en Zig puro sin intrínsecas dedicadas es complejo y el beneficio marginal frente al paralelismo. Q4_K/Q6_K tienen estructuras irregulares (scales por sub-bloque) que no se prestan fácilmente a vectorización uniforme.

## Artículo afectado

Art. V (calidad: kernels validados contra referencia), Art. I (pureza: 100 % Zig).
