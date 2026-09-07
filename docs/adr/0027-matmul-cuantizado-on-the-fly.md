# ADR-0027: Matmul cuantizado on-the-fly para inferencia GGUF

**Estado:** Aceptado  
**Fecha:** 2026-06-02  
**Decisores:** ZSA (Zig Super Agent)  
**Contexto:** TAREA 5 — FASE 5 (embebido / escalabilidad)

## Contexto

`fluency agent --gguf`, `fluency serve --gguf` y `fluency distill` cargan modelos GGUF dequantizando **todos** los pesos a f32. Un modelo Qwen2.5-1.5B cuantizado a Q4_K_M pesa ~1.1 GB en disco, pero dequantizado a f32 consume ~6 GB de RAM. Esto causa OOM en máquinas con <16 GB y limita severamente la escala de inferencia local.

El objetivo es hacer inferencia **directamente desde pesos cuantizados**, dequantizando bloque a bloque on-the-fly durante el matmul, sin mantener una copia f32 completa de los pesos lineales.

## Decisión

1. **No transponer los pesos cuantizados**. Los formatos GGUF cuantizan por bloques a lo largo de la dimensión interna (K). Si los transponemos a `[in, out]` (como hace la ruta f32), rompemos la alineación de bloques. En su lugar, mantenemos el layout GGUF original `[out, in]` y usamos un matmul "vector × matriz" donde el vector de entrada es f32 y la matriz de pesos está cuantizada.

2. **Kernels vector×matriz cuantizados** (`matmulVecQ*`) en `zkernel.zig`: para cada fila de salida N, dequantizan sus bloques K uno a uno en un buffer stack, hacen dot product con el vector de entrada, y escriben el resultado. Esto minimiza memoria temporal (un bloque, no todo el peso).

3. **Formatos soportados en MVP**: Q4_0, Q8_0, Q4_K, Q6_K. Son los más comunes en modelos Qwen/Llama cuantizados con llama.cpp. Cada formato tiene su propia función de dequantización de bloque y su kernel `matmulVec`.

4. **Dualidad en `Transformer`**: los campos `wq`/`wk`/... siguen siendo `[]const f32` (compatibilidad hacia atrás), y se añaden campos `wq_quant`/`wk_quant`/... como `?[]const u8` **más un `*_fmt: QuantFormat` POR-TENSOR** (`wq_fmt`…`w3_fmt`). `forward` usa `matmulVec`/`matmulVecIn` que reciben el `fmt` de ESE tensor y enrutan según el formato del peso y la presencia de `*_quant`. (Inicialmente se usó un único `quant_format` global; ver Actualización 2026-06-02.)

5. **Carga cuantizada en `main.zig`**: `loadGgufTransformerQuantized` copia los bytes crudos de los tensores lineales del GGUF (sin dequantizar ni transponer), y dequantiza a f32 solo los pesos no-lineales (embeddings, norms, bias) que no se usan en matmuls pesados.

## Consecuencias

**Positivas:**
- Memoria de pesos lineales ~4× menor (Q4_0) o ~8× menor (Q8_0) vs f32.
- Inferencia de modelos 7B+ cuantizados viable en RAM de consumo (8-16 GB).
- Sin dependencias externas: todo en Zig puro, sin llama.cpp, sin BLAS.

**Negativas / trade-offs:**
- Más lento que un kernel SIMD optimizado para Q4_K (nuestro MVP usa dequant bloque-a-bloque scalar). La optimización con SIMD/arm_neon/avx512 es futuro trabajo (F7).
- Tolerancia numérica: Q4_0 introduce error de cuantización ~scale/2; Q8_0 ~scale/128. Los logits difieren de la ruta f32, pero la salida greedy suele coincidir para la mayoría de tokens.
- Los pesos MoE (si existen) no tienen ruta cuantizada todavía; se dequantizan a f32.

## Alternativas consideradas

- **Dequantizar todo a f32 y luego re-cuantizar en formato propio**: Rechazado. Añade complejidad innecesaria; los formatos GGUF ya están optimizados.
- **Transponer bloques cuantizados**: Rechazado. Extremadamente complejo; requeriría re-empaquetar escalas y offsets por bloque.
- **Usar llama.cpp como backend**: Rechazado por Constitución (100% Zig, sin FFI).

## Actualización 2026-06-02 — formato POR-CAPA-POR-TENSOR + carga multi-capa (3 bugs corregidos)

La versión original tenía **tres defectos** que la hacían fallar sobre el modelo objetivo real (Qwen2.5-coder-1.5B **Q4_K_M**), descubiertos al ejecutarla en vivo (no se había hecho antes):

1. **Formato único global.** `Transformer.quant_format` + `error.MixedQuantFormats` exigían que todos los pesos compartieran formato, afirmando que "cubre el 99% de los GGUF reales". **Falso**: `fluency inspect` muestra `Q4_K x168, Q6_K x29, F32 x141` — los pesos mezclan Q4_K (attn/gate/up) y Q6_K (ffn_down). La mezcla es la norma en los K-quants `_M`/`_S`/`_L`, no la excepción.
2. **Carga de una sola capa.** `loadGgufTransformerQuantized` buscaba `"attn_q.weight"` (sin prefijo `blk.{l}.` ni bucle de capas) → `error.MissingTensor`; y `forward` no desplazaba el peso cuantizado por capa. El test sintético de 1 capa lo ocultaba.
3. **Formato variable ENTRE capas.** Incluso por tipo de peso, el formato varía por capa (ffn_down/attn_v alternan Q4_K y Q6_K según la capa), con tamaños de bloque distintos (Q4_K=144 B, Q6_K=210 B) → no concatenables con stride uniforme.

**Corrección**: tipo `QuantWeight { layers: [][]const u8, fmts: []QuantFormat }` con bytes y formato **por-capa**; los 7 pesos lineales del `Transformer` son `?QuantWeight`. `matmulVec`/`matmulVecIn` reciben `(qw, layer)` y seleccionan `qw.layers[layer]` + `qw.fmts[layer]`. `loadQuantWeight` carga `blk.{0..L-1}.{suffix}` por capa con su formato. Eliminados `error.MixedQuantFormats`/global format.

**Verificado EN VIVO** sobre el modelo real (28 capas, Q4_K+Q6_K mezclados entre tipos y entre capas): `fluency run <Q4_K_M> "def fib(n):" 12 --quantized` genera código Python coherente (`def fib(n):\n    if n == 0:\n        return 0`), **pico de RAM 3082 MB** (vs ~6 GB del dequant f32 → sin OOM en máquina de 16 GB). El bug de capa única se blinda con el test de regresión multi-capa.

## Evidencia

- `zig build test`: 170/170 tests pasan, sin fugas.
- `zig build gate`: pureza ✅, constitución ✅, calidad ✅, presupuestos ✅.
- Tests unitarios de kernels (`matmulVecQ4_0`, `matmulVecQ8_0`, `matmulVecQ4_K`, `matmulVecQ6_K`) vs dequantización manual de referencia.
- Test de integración `Transformer forward cuantizado Q8_0 coincide con f32 (tolerancia Q8)`.
- **Test de formatos mezclados** `...formatos cuantizados MEZCLADOS (w1=Q4_0, resto=Q8_0)...` — enrutado per-tensor con dos formatos en un mismo forward.
- **Test de regresión multi-capa** `...MULTI-CAPA (2 capas, pesos+formato distintos por capa)...` — bloquea el bug de carga/uso de una sola capa (w1 alterna q8_0/q4_0 entre capas).
- **Corrida EN VIVO** (modelo real, ReleaseFast): `fluency run <Qwen2.5-coder-1.5B-Q4_K_M> "def fib(n):" 12 --quantized` → genera Python coherente, **pico RAM 3082 MB** (sin OOM; el dequant f32 requería ~6 GB). CLI `--quantized` también en `fluency agent` y `fluency serve`.
