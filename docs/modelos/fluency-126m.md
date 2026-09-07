# Fluency-126M / Fluency-139M

> Modelo de lenguaje entrenable 100 % Zig, definido en `src/zdistill.zig`.
> Estado: **configuración validada, pipeline CPU funcional** (entrenamiento de pre-entrenamiento fuera de alcance por presupuesto de cómpute).

> **Nota de naming (2026-06-08).** La misma arquitectura recibe **dos nombres
> oficiales** según el vocabulario del profesor con el que se destila:
>
> - **Fluency-126M** = arquitectura con `vocab = 32 000` (teacher TinyLlama).
>   Total exacto: **126 052 608** parámetros entrenables.
> - **Fluency-139M** = misma arquitectura con `vocab = 49 152` (teacher
>   SmolLM2-360M-Instruct). Total exacto: **139 225 344** parámetros entrenables
>   — el delta de +13.17 M proviene íntegramente de la tabla de embeddings
>   atados (`vocab × d_model`).
>
> En código sigue habiendo un solo símbolo (`Fluency126MConfig` en
> `src/zdistill.zig`); el `vocab` se reescribe en tiempo de carga según el
> profesor GGUF y el runtime imprime el `ParamCount` real. El número "139M"
> que aparece en el README público es el que corresponde al checkpoint
> publicado, entrenado con SmolLM2 como profesor.

## Arquitectura

| Hiperparámetro | Valor |
|----------------|-------|
| `vocab`        | 32 000 |
| `d_model`      | 768 |
| `n_layers`     | 12 |
| `n_heads`      | 12 (SWA causal donde aplica) |
| `ff_dim`       | 3 072 (ratio 4×) |
| `eps`          | 1e-5 |
| `attn_interval`| 6 |
| `attn_window`  | 128 |

### Patrón híbrido (subcuadrático predominante)

- Capas múltiplo de 6 (`0`, `6`) = **atención causal SWA** (2 capas, ~16.7 %).
- Resto = **scan SSM diagonal** (10 capas, ~83.3 %).

La atención está acotada por una **ventana fija W = 128**: cada posición `i`
solo atiende a claves en `[max(0, i−W+1), i]`. Complejidad forward/backward
**O(T·W)** por cabeza (no O(T²)), y estado recurrente en decode **O(W·D)**
(últimos W pares KV). Esto prioriza eficiencia subcuadrática sin eliminar la
capacidad de capturar dependencias locales de corto/medio alcance.

## Conteo de parámetros

Desglose exacto (vía `printParamCount` en `FluencyNano`):

| Componente | Cálculo | Parámetros |
|------------|---------|------------|
| Embeddings (input/output atados) | 32 000 × 768 | 24 576 000 |
| Mixer scan (10 capas) | 10 × (2×768² + 768) | 11 804 160 |
| Mixer attn (2 capas) | 2 × (4×768²) | 4 718 592 |
| FFN dense (12 capas) | 12 × (3×768×3 072) | 84 934 656 |
| Normas (RMSNorm 2× por capa + final) | 12 × 2×768 + 768 | 19 200 |
| **Total** | | **~126 052 608** |

**126 052 608 parámetros entrenables: CUMPLE.**

## Pipeline demostrado

Los siguientes puntos del pipeline están **cubiertos por tests automáticos** en
`src/zdistill.zig`:

1. **Instanciación sin fugas** — `Fluency-126M` se construye con `std.testing.allocator`
   (GPA) y `defer model.deinit()`; el gate de calidad verifica cero fugas.
2. **Conteo exacto ≥126 M** — test `"Fluency-126M: ≥126M parámetros y desglose
   imprimible"` (`paramCount() >= 126_000_000`).
3. **Gradient-check** — test `"Fluency-tiny: gradient-check analítico vs numérico en
   sub-módulo"` compara gradientes del embedding obtenidos por backprop vs
   diferencias finitas (±1e-3, tolerancia 5e-2).
4. **Convergencia en pequeño** — test `"Fluency-tiny: converge en corpus pequeño"`
   entrena 200 pasos en una secuencia de 6 tokens; la CE baja >50 % y la accuracy
   sube ≥80 %.
5. **Checkpoint round-trip** — test `"Fluency-tiny: checkpoint round-trip reproduce
   forward bit-a-bit"` entrena 20 pasos, guarda `.ckpt`, recarga y verifica que el
   forward produce **valores idénticos bit-a-bit**.
6. **Seq-KD end-to-end CI-safe** — test `"seqkd end-to-end: teacher sintetico,
   student aprende"` genera 20 secuencias greedy con un teacher sintético,
   entrena al student por seq-KD (held-out split), verifica que la CE de entrenamiento
   baja >50 % y que el checkpoint round-trip del modelo entrenado reproduce logits
   bit-a-bit.
7. **Seq-KD dataset round-trip** — test `"seqkd dataset round-trip"` guarda/recarga
   un dataset `.seqkd` y valida que las secuencias se reproducen idénticas.

## Pipeline de destilación (seq-KD)

El comando `fluency distill` ahora implementa seq-KD completo tanto para profesor
GGUF real como para profesor sintético (`--synthetic`):

- **Alineación de vocabulario**: el student hereda el vocabulario exacto del teacher
  GGUF (vía `zload` + `ztok`). Si el vocab del teacher no es 32 000, `Fluency126MConfig`
  se ajusta y `printParamCount` reporta el conteo real (p. ej. ~305 M para vocab
  Qwen2.5 de 151 936).
- **Generación de datos**: prompts semilla hardcodeados se expanden greedy con el
  teacher GGUF (`generateSeqWithTeacher`) o con un `FluencyNano` sintético
  (`generateGreedyNano`).
- **Entrenamiento seq-KD**: `trainSeqKD` usa cross-entropy sobre el siguiente token
  con split train/held-out y scheduler cosine con warmup.
- **Evaluación antes/después**: `evaluateSeqDataset` reporta CE, PPL y accuracy
  top-1 sobre el dataset generado.
- **Muestra greedy**: tras entrenar, el student genera 12 tokens greedy desde el
  mismo prompt semilla para inspección cualitativa.
- **Checkpoint**: `saveCheckpoint` / `loadCheckpoint` permiten persistir el modelo.

## Limitaciones honestas

- **Entrenamiento de pre-entrenamiento a escala**: un modelo de ~126 M params
  requiere ~250–500 GB de tokens y decenas de horas de cómpute en CPU. Esto está
  **fuera del presupuesto de este proyecto** (ver `budgets.zon`). El pipeline está
  listo; el pre-entrenamiento real sería el siguiente paso con recursos de
  producción.
- **Profesor GGUF**: el pipeline seq-KD real requiere un archivo `.gguf` descargado
  (p. ej. TinyLlama-1.1B, Apache-2.0). Sin él, solo el modo `--synthetic` demuestra
  el pipeline; no produce un modelo competitivo.
- **Prompts semilla**: la corrida piloto usa prompts hardcodeados (8 prompts fijos).
  Un entrenamiento real necesitaría miles de prompts diversos o un corpus curado.
- **Generalización**: el test CI-safe demuestra memorización de secuencias cortas
  (12 tokens de continuación); no es evidencia de generalización a largo plazo.
- **GPU acelerada**: bloqueada por el backend SPIR-V de Zig 0.16 (ver
  `docs/adr/0037-backend-gpu-vulkan-spirv-zig.md`). El entrenamiento corre sobre
  CPU (fallback siempre disponible).
- **Cuantización**: la inferencia del modelo entrenado en FP32 sería lenta; la
  ruta cuantizada (`Q4_K_M`, etc.) existe para modelos GGUF cargados, pero no
  para checkpoints nativos de `FluencyNano` (trabajo futuro).

## Referencias

- Config: `src/zdistill.zig` → `Fluency126MConfig`
- Tests: `src/zdistill.zig` → sección "Fluency-126M / Fluency-Tiny configs y tests"
- Pipeline de entrenamiento: `src/ztrain.zig`, `src/zgrad.zig`
- ADR GPU: `docs/adr/0037-backend-gpu-vulkan-spirv-zig.md`
