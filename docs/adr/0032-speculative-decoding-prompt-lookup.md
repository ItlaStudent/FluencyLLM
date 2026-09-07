# ADR-0032 — Speculative decoding 100% Zig (prompt-lookup / n-gram)

## Contexto

La generación token-a-token en `ServeEngine.generate` y `generateStream` (`src/main.zig`) es secuencial: un forward por token. El throughput en CPU (~0.59 tok/s en Qwen2.5-coder-1.5B-Q4_K_M, línea SIMD previa a F7.A) está limitado por la latencia de cada pasada del Transformer. Speculative decoding permite **amortizar el costo de los forwards** proponiendo múltiples tokens candidatos y verificándolos en batch (posición por posición) con un único modelo.

La aproximación "prompt-lookup" (también llamada n-gram draft) NO requiere un modelo borrador separado: busca repeticiones del propio contexto ya generado y propone continuar la secuencia. Es puramente algorítmica, 100% Zig, y no añade memoria ni pesos adicionales.

## Decisión

1. **Módulo `zspeculate`** (`src/zspeculate.zig`), auto-contenido, importado por `main.zig` y registrado en `build.zig`.
   - `SpeculativeGenerator` con dos métodos: `generate` (buffer) y `generateStream` (emit callback).
   - **Draft**: `promptLookupDraft` busca un n-grama de longitud `ngram_len` (default 4) en el contexto previo (excluyendo el final para evitar overlap) que coincida con los últimos `ngram_len` tokens. Devuelve hasta `max_draft` (default 15) tokens siguientes al match.
   - **Verify**: para cada token propuesto, ejecuta `Transformer.forward`, toma `argmax`, y compara con el draft. Si coincide, se acepta y continúa. Si no, se acepta el token greedy del modelo (que es EXACTAMENTE el mismo que habría salido sin speculative) y se aborta el draft.
   - **Fallback**: si no hay draft o se agota, greedy simple un token a la vez.

2. **Integración en `ServeEngine`**: campo `speculate: bool`. Cuando está activo, `generate` y `generateStream` delegan a `SpeculativeGenerator`. El CLI expone `--speculate` en `run`, `serve`, y `agent`.

3. **Invariante de salida**: el speculative decoding NO modifica la distribución. La única fuente de tokens es `argmax(logits)`, igual que greedy baseline. Por tanto, la salida es bit-a-bit idéntica a greedy (salvo que speculative puede aceptar varios tokens en un solo ciclo de draft+verify).

## Consecuencias

- **Tests**: 3 nuevos tests unitarios en `src/zspeculate.zig` (promptLookupDraft: repetición, overlap, sin match). Conteo total de tests sube de 220 a 221.
- **Speedup esperado**: en texto repetitivo (código, listas, repeticiones de frases), el draft acepta múltiples tokens por forward, reduciendo el número de pasadas del Transformer. En texto altamente entrópico (sin repeticiones), el overhead del lookup es mínimo (búsqueda lineal en contexto local).
- **Memoria**: cero asignación adicional en el camino caliente; `ids` y `out` reutilizan los buffers del caller.
- **Limitación honesta**: prompt-lookup funciona mejor en contextos con repeticiones locales. Para texto abierto/altamente entropico, el speedup puede ser cercano a 1× (sin regresión). Un modelo borrador dedicado (futuro) podría mejorar la tasa de aceptación en cualquier dominio, pero requiere pesos adicionales y memoria.
- **Compatibilidad**: no altera el comportamiento de `temp > 0` o gramática; en esos casos se ignora speculative y se cae a greedy normal.

## Actualización 2026-06-05 (auditoría + correcciones reales)

Tras auditoría en vivo con modelo real (Qwen2.5-coder-1.5B-Q4_K_M), se encontraron y corrigieron bugs que la entrega anterior había marcado como "✅" sin evidencia en vivo:

1. **Bug de divergencia cuando draft se rompe:** `zspeculate.zig` copiaba logits del batch del último token *aceptado* en lugar de hacer `forward` del token *rechazado* que acababa de emitir. Esto rompía la identidad greedy bit-a-bit cuando el draft no era perfecto. Fix: si `accepted < accept_limit`, se hace `self.tf.forward(ids.items[pos - 1], pos - 1)`.
2. **Texto decorativo rompía identidad:** `main.zig` inyectaba `[speculative: prompt-lookup]` y `(N tokens generados, speculative)`. Eliminado; la salida speculative debe ser indistinguible de greedy.
3. **Buffers cacheados:** `forwardBatch` alocaba 5 buffers (`x_batch`, `xb_batch`, `q_batch`, `k_batch`, `v_batch`) en cada llamada. Ahora se cachean como campos `batch_*` en `Transformer` (lazy-realloc). `SpeculativeGenerator` también cachea `logits_buf` y `draft_buf` con `deinit()` para liberación.
4. **Speedup real con cuantización: ~0.98×** (NO acelera). Causa raíz: `matmulBatch` sobre pesos cuantizados (Q4_K/Q6_K) itera fila por fila llamando `matmulVecQ*`; cada llamada dequantiza el peso completo de forma independiente. Con K tokens, se dequantiza K veces el mismo peso → sin ahorro real. El speedup > 1× del test sintético (f32) no se materializa en cuantizado. La atención sigue siendo token-por-token.
5. **Salida greedy = speculative confirmada en vivo** (prompt `"def fib(n):"`, 20 tokens, modelo real Q4_K_M).

## Actualización 2026-06-06 (matmulBatch cuantizado dequant-once)

Se implementaron `matmulBatchQ4_0`, `matmulBatchQ8_0`, `matmulBatchQ4_K`, `matmulBatchQ6_K` en `src/zkernel/zkernel.zig`. Cada kernel dequantiza un bloque del peso **UNA vez** y lo aplica a los M tokens del batch, eliminando la re-dequantización redundante que causaba ~0.98×.

- **Cambio en `zarch.zig`**: `matmulBatch` ahora llama `zkernel.matmulBatchQ*` en lugar de iterar `matmulVecQ*` token por token.
- **Tests**: 4 nuevos (`matmulBatchQ4_0/Q8_0/Q4_K/Q6_K vs M×matmulVecQ*`) con tolerancias 1e-4/1e-4/5e-3/5e-2.
- **Speedup medido en vivo** (Qwen2.5-coder-1.5B-Q4_K_M, ReleaseFast, prompt `"def fib(n):"`, 12 tokens):
  - Greedy: **10.74 s** → **1.12 tok/s**
  - Speculative: **10.32 s** → **1.16 tok/s**
  - **Speedup: ~1.04×** (antes ~0.98×)
  - Salida **bit-idéntica** a greedy confirmada.
- **Limitación honesta**: el speedup sigue siendo modesto porque la atención en `forwardBatch` sigue siendo token-por-token (causal). El beneficio principal se materializa en QKV+FFN batch por capa. Para aceleraciones mayores se requeriría batching de la atención causal (difícil sin quebrar causalidad) o un modelo borrador que proponga drafts más largos y de mayor calidad.

## Alternativas consideradas

- **Modelo borrador separado** (p. ej. FluencyNano): mejor cobertura de draft pero requiere cargar dos modelos, más memoria, más código. Posible iteración futura.
- **Múltiples n-gramas concurrentes**: complejidad innecesaria para esta iteración; un único match reciente es suficiente.
- **Librerías externas (llama.cpp, vLLM speculative)**: vetado por la Constitución (Art. I: 100 % Zig, sin binarios ajenos).

## Artículo afectado

Art. I (pureza: 100 % Zig), Art. V (calidad: no regresión, salida idéntica a greedy).
