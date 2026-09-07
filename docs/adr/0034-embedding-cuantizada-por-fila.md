# ADR-0034 — Embedding/output table cuantizada: dequant por-fila bajo demanda

## Contexto

En `loadGgufTransformerQuantized` (`src/main.zig`) la tabla de embeddings `token_embd.weight` (vocab 151936 × dim 1536) se dequantizaba completa a `f32` → ~933 MB sólo en embeddings. Sumando KV-cache, buffers de estado y pesos lineales cuantizados, el proceso alcanzaba ~3 GB de RAM (3.3× el archivo GGUF de 941 MB). llama.cpp mantiene los embeddings en su formato GGUF original y los dequantiza bajo demanda.

## Decisión

1. **Mantener `token_embd.weight` en formato cuantizado GGUF crudo** dentro del `Transformer` via un nuevo campo `token_embedding_table_q: ?QuantEmbedding` (`bytes: []const u8, fmt: QuantFormat`).

2. **Dequantización por-fila bajo demanda** en dos puntos:
   - **Embedding lookup** (`forward`, `forwardWithHidden`, `forwardBatch`): cuando llega un `token`, dequantizar únicamente esa fila a un buffer temporal (`self.x` o `x_batch`) usando `zkernel.dequantRowQ*`.
   - **Proyección final (tied embeddings)** (`wcls == null`): en lugar de iterar fila a fila sobre una tabla `f32`, reutilizar directamente `zkernel.matmulVecQ*` sobre el blob cuantizado completo. Esto calcula `logits[v] = dot(x, emb_row_v)` para todo el vocabulario sin materializar la matriz `f32`.

3. **Reutilizar kernels existentes**: `dequantRowQ4_0`, `dequantRowQ8_0`, `dequantRowQ4_K`, `dequantRowQ6_K` se añadieron a `zkernel.zig` como wrappers de una sola fila sobre los dequantizadores de bloque ya validados (`dequantQ*_simd` / `dequantQ*_block`). No se duplicó lógica de dequantización.

4. **Conservar fallback f32**: si `token_embedding_table_q` es `null`, se usa el camino f32 anterior sin cambios (sin regresión para modelos que carguen embeddings en f32).

## Consecuencias

- **Reducción de RAM medida en vivo** (Qwen2.5-coder-1.5B-Q4_K_M, `token_embd.weight` = Q6_K):
  - Pico de RAM antes (baseline documentada F7): **~3 GB**
  - Pico de RAM después (medición externa vía `Get-Process fluency | PeakWorkingSet64`): **~2376 MB**
  - **Reducción observada: ~600 MB** (~751 MB teóricos: 933 MB f32 → ~182 MB Q6_K).

> **Corrección (auditoría 2026-06-06).** La versión inicial de este ADR atribuía
> el footprint residual (~2.4 GB) al KV-cache con `seq_len=32768` (~1.8 GB). **Eso
> es incorrecto para el comando `run`**: `runGenerate` fija
> `max_seq = min(prompt+n_tokens+4, ctx)` (p.ej. 21 para 16 tokens), así que el
> KV-cache real es ~1 MB, no 1.8 GB. La causa real del residual era la **doble
> residencia del GGUF**: el buffer crudo del archivo (`data`, ~940 MB vía
> `readFileAlloc`) permanecía vivo Y los pesos se **copiaban** (`@memcpy`) al arena
> (~900 MB). Resuelto en **ADR-0036 (carga zero-copy)**: tras referenciar los pesos
> in situ en `data`, el pico bajó a **~1441 MB**, salida byte-idéntica. Ver ADR-0036.

- **Corrección numérica**: la salida greedy con el modelo real es **idéntica en tokens** al camino anterior (mismo prompt `"def fib(n):"`, 12 tokens). Los logits difieren en los últimos ULP debido a distinto orden de acumulación (matmulVecQ* hace tree-reduction SIMD vs suma escalar del loop f32), pero el argmax greedy no cambia.

- **Tests de regresión**:
  - 4 tests nuevos en `zkernel.zig`: `dequantRowQ4_0/Q8_0/Q4_K/Q6_K vs manual dequant` (oráculo bloque-a-bloque, tolerancia 1e-6).
  - 1 test nuevo en `zarch.zig`: `embedding cuantizado Q8_0: forward idéntico a f32` (mini-Transformer con embedding cuantizado manual vs oráculo f32, tolerancia 1e-3 en logits).
  - Conteo total de tests subió de 231 → 236.

- **Sin fugas**: `zig build gate` verifica 0 fugas; `loadQuantEmbedding` asigna en el arena `wa` que vive con el Transformer y se libera en `deinit` (aunque `token_embedding_table_q.bytes` es un slice del arena, no requiere free individual).

## Alternativas consideradas

- **Dequantizar la tabla completa al inicio pero en memoria mapeada / comprimida**: vetado porque requiere dependencias externas o código C para mmap/compression, violando Art. I (100 % Zig, sin FFI).
- **Mantener una caché f32 de filas recientes**: aumentaría complejidad y footprint para un beneficio marginal (el embedding lookup es 1 fila por token, negligible frente al costo de atención/FFN).
- **Usar `matmulVecQ*` directamente para lookup**: no es aplicable porque el lookup necesita la fila completa como vector de entrada para la atención, no un escalar dot-product.

## Artículo afectado

Art. I (pureza 100 % Zig), Art. V (calidad: kernels validados contra referencia), Art. VI (allocator explícito, cero fugas).
