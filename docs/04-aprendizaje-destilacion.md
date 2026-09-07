# 04 — Aprendizaje y destilación

[← Plan Maestro](../PLAN-MAESTRO.md) · [← 03 Núcleo](03-nucleo-subcuadratico.md) · [→ 05 Runtime agéntico](05-runtime-agentico.md)

Este documento responde el requisito central: **"aprender de otros LLM descargables, pero todo lo aprendido 100 % en Zig"** y **"auto-entrenarse desde modelos vía Ollama"**. Todo el código aquí descrito es Zig (`zgrad`/`ztrain`/`zdistill`/`zload`/`zollama`).

> **Estado de implementación (F3, 2026-05-30):** ✅ realizado y verificado (gate-check E4). Decisiones en [ADR-0011](adr/0011-autodiff-tensorial-y-fluencynano.md); evidencia y próximos pasos en [ESTADO-FASES.md](ESTADO-FASES.md). Resumen: autodiff tensorial con VJP de scan/delta/attn (gradient-checked), alumno `FluencyNano` híbrido, destilación del Qwen real (KL 10.21→0.034, acuerdo top-1 87.5%, reproducible, sin fugas) por las dos rutas (zload + zollama). Pendiente de escala: Fluency-100M/350M sobre 5B tokens y MOHAWK B/C en vivo.

---

## 1. El principio: aprender = destilar, no copiar

ZSA no reempaqueta pesos ajenos (Art. III). Aprende **transfiriendo conocimiento** de un modelo **profesor** (Transformer abierto) a un modelo **alumno** Fluency (subcuadrático), usando cuatro mecanismos combinables:

1. **Destilación cruzada de arquitectura (MOHAWK):** Transformer → SSM. *El mecanismo estrella.*
2. **KD por logits/secuencias:** igualar distribuciones de salida.
3. **On-policy distillation:** el alumno genera, el profesor corrige (KL inversa).
4. **DPO/ORPO:** afinado por preferencias sin reward model.

El profesor se consulta de dos formas, ambas 100 % Zig en *nuestro* lado:
- **Ruta pura:** corremos el profesor con **nuestro** motor (`zload` GGUF + `zarch` todo-atención). Garantiza "todo en Zig" extremo a extremo.
- **Ruta pragmática:** consultamos un profesor servido por **Ollama** (`zollama` HTTP) como oráculo black-box. Más simple y rápido para arrancar; nuestro código sigue siendo Zig (Ollama es un proceso externo, no una dependencia de build — no viola Art. I).

> **Licencias (Art. III.3) — Fluency es producto PROPIETARIO/CERRADO:** esto restringe qué profesores son legales. Cada profesor se registra en `docs/PROFESORES.md` con su licencia y un campo explícito "permite destilar a producto cerrado/comercial: sí/no/condiciones".
> - ✅ **Preferir Apache-2.0 / MIT** (permiten derivados propietarios y uso de salidas): p. ej. **Qwen2.5** (la mayoría de tamaños son Apache-2.0), **Mistral/Mixtral base** (Apache-2.0).
> - ⚠️ **Revisar con lupa / probablemente evitar:** **Llama** (licencia comunitaria de Meta: exige incluir "Llama" en el nombre del derivado + uso aceptable; incómodo para un producto cerrado), **Gemma** (licencia con restricciones de uso), y cualquiera con cláusula que prohíba usar salidas para entrenar otros modelos.
> - ❌ **Prohibido:** modelos/código **no comercial (NC)** o de "solo investigación" (p. ej. el código de referencia de Gated-DeltaNet-2 es NC → solo sirve como lectura, no para destilar a Fluency).

---

## 2. Formatos de entrada (qué sabe leer `zload`)

### 2.1 GGUF (el formato de Ollama/llama.cpp)
Archivo auto-descriptivo y mmap-friendly. Orden: `header → tensor_info[] → padding(alineación) → tensor_data`.

- **Header:** magic `"GGUF"` (`0x47475546`), `version`=3, `tensor_count:u64`, `metadata_kv_count:u64`, luego los KV.
- **Strings:** `u64 len` + bytes UTF-8 (sin null).
- **Metadata KV:** `key(string)` + `value_type(u32)` + `value`. Tipos: 0-12 (uint8..float64, bool, string, array). **Array:** `type(u32)` + `len(u64)` + elementos.
- **Tensor info:** `name(string≤64)` + `n_dims(u32)` + `dims[u64]` + `type(u32 ggml_type)` + `offset(u64)`.
- **Alineación:** `general.alignment` (default **32** si ausente); cada `offset` múltiplo de la alineación.
- **Metadata clave:** `general.architecture` (`llama`/`qwen2`/`gpt2`…), `{arch}.block_count`, `{arch}.embedding_length`, `{arch}.attention.head_count[_kv]`, `{arch}.rope.dimension_count`, etc.
- **Tokenizer embebido:** `tokenizer.ggml.model`, `.tokens[]`, `.scores[]`, `.token_type[]`, `.merges[]`, `.bos/eos/...token_id`. → un GGUF basta para reconstruir el tokenizer.

### 2.2 Dequantización de bloques (implementable desde cero)
Constantes: bloques de 32 (legacy) o superbloques de 256 (K-quants). `ggml_half` = fp16.

Bloques legacy (32 elems):
```
block_q4_0 {f16 d; u8 qs[16]}            → w = d * (nibble - 8)      # simétrico
block_q4_1 {f16 d; f16 m; u8 qs[16]}     → w = d * nibble + m        # asimétrico
block_q8_0 {f16 d; i8 qs[32]}            → w = d * q
block_q5_0/1 {..., u8 qh[4], ...}        → 5º bit desde qh (máscara u32 LE)
```
K-quants (superbloque 256, dos niveles de escala): `Q4_K` (144 B), `Q5_K` (176 B), `Q6_K` (210 B), `Q2_K` (84 B), `Q3_K` (110 B). Modelo general: `w = d·sc_j·q − dmin·m_j` (asimétricos Q4_K/Q5_K/Q2_K) o `w = d·scale_j·q` (simétricos Q6_K/Q3_K), con `sc_j`/`m_j` sub-escalas de 6 bits empaquetadas en `scales[12]`.

> **Plan de implementación:** portar las funciones de referencia `dequantize_row_q4_K`/`q6_K` **reescritas en Zig** (no enlazar ggml). Para cada tipo, test contra vectores conocidos. Empezar por F32/F16/Q8_0/Q4_0/Q4_K (cubren la mayoría de modelos). IQ-quants y MXFP4 (codebook) se dejan para más tarde.

### 2.3 safetensors (HuggingFace)
`[u64 len header LE][header JSON][datos]`. Header = dict `nombre → {dtype, shape, data_offsets:[ini,fin]}` + `__metadata__`. Offsets relativos al buffer de datos; little-endian, C-order, sin huecos. dtypes incl. `BF16`, `F8_E4M3/E5M2`. La arquitectura y el tokenizer vienen en `config.json`/`tokenizer.json` aparte (a diferencia de GGUF).

---

## 3. Tokenización (`ztok`)
Implementar las cuatro familias (compatibilidad con modelos populares):
- **BPE byte-level** (GPT-2/tiktoken): base de 256 bytes (sin `<UNK>`), regex de pre-tokenización + remapeo byte↔unicode + lista ordenada de `merges`. → **Llama-3, Qwen2/2.5/3**.
- **SentencePiece unigram** (Viterbi sobre lattice, `scores` log-prob, `▁` para espacios). → **Llama-1/2, Mistral**.
- **WordPiece** (greedy longest-match, prefijo `##`). → BERT.
- Manejo de **tokens especiales/añadidos** antes del pase BPE/unigram.

> El detalle crítico: replicar **exactamente** el regex de split y el mapeo de bytes, o los IDs no coinciden con el profesor (output basura). Test: re-tokenizar un corpus y comparar IDs contra la referencia del modelo.

---

## 4. MOHAWK — destilar Transformer → subcuadrático (el método estrella)

Ref: **MOHAWK** `arXiv:2408.10189` (código de referencia `goombalab/phi-mamba`, **se reimplementa en Zig**), **Mamba-in-the-Llama** `arXiv:2408.15237`, **Llamba** `arXiv:2502.14458`.

**Insight:** tanto la auto-atención como un SSM aplican una **matriz de mezcla de secuencia**. La de la atención es `softmax(QKᵀ)`; la del SSM Mamba-2 es la **matriz semiseparable materializada**. Son comparables → se alinean por etapas, de gruesa a fina:

**Etapa A — Orientación de matrices.** Minimizar la distancia Frobenius entre la matriz de mezcla del profesor y la del alumno SSM:
```
min_φ  ‖ MixProfesor(u) − MixAlumno_φ(u) ‖_F
```

**Etapa B — Alineación de estados ocultos.** Con los mismos inputs, igualar las salidas por bloque (L2):
```
min_φ  ‖ BloqueAtn(u) − BloqueMixerAlumno_φ(u) ‖₂
```

**Etapa C — Transferencia de pesos + KD end-to-end.** Copiar **directamente** del profesor: embeddings de token, norm final, LM head, **bloques MLP** y norms de entrada por bloque. **Sólo se aprende el mixer** (atención→SSM); las proyecciones Q/K/V/O inicializan las proyecciones de entrada del SSM. Luego destilar logits:
```
min_φ  L_CE/KL( Profesor(x), Alumno_φ(x) )
```

**Presupuesto:** Phi-Mamba se entrenó con **~3B tokens**; el híbrido con **~5B** — <1 % de entrenar desde cero. **Esto es lo que hace viable el proyecto en CPU/Zig.** Mantener una fracción de capas de atención (híbrido) recupera el recall que el SSM puro pierde.

**Flujo en Zig:**
```
fluency distill mohawk --teacher <ollama|gguf:path> --student model.zon \
    --stage A --tokens 1B --out ckpt_A
fluency distill mohawk ... --stage B --init ckpt_A --tokens 1B --out ckpt_B
fluency distill mohawk ... --stage C --init ckpt_B --tokens 3B --out fluency-1b
```

---

## 5. Otras formas de destilación

### 5.1 KD por logits (clásica)
Softmax con temperatura `T`, KL hacia adelante + cross-entropy dura:
```
L = α·T²·KL( softmax(z_prof/T) ‖ softmax(z_alum/T) ) + (1−α)·CE(y, z_alum)
```
`T∈[1,10]`, `α∈[0.5,0.9]`. Requiere logits del profesor (ruta pura) o top-k logprobs (algunas APIs).

### 5.2 Sequence-level KD (Seq-KD)
SFT del alumno sobre **secuencias generadas** por el profesor (greedy/beam). Simple y efectiva; base de la "destilación por generación". Funciona incluso con profesor black-box (sólo texto).

### 5.3 On-policy distillation (KL inversa)
El alumno **genera sus propios rollouts**; el profesor puntúa cada token con KL inversa `KL(alumno‖profesor)` (mode-seeking, evita que el alumno reparta masa donde el profesor no la pone). Recipe: (1) muestrear del alumno, (2) pedir `logprobs` del profesor sobre esa trayectoria (sin gradiente), (3) ventaja por token = −KL, (4) loss tipo RL con importance sampling. **Sin reward model.** Mucho más sample-efficient que RL; entrena sobre los **propios errores** del alumno. Ref: blog On-Policy Distillation (Thinking Machines).

### 5.4 DPO / ORPO (preferencias)
- **DPO:** dados pares (elegido `y_w`, rechazado `y_l`), loss Bradley-Terry que sube `logπ(y_w)` y baja `logπ(y_l)` contra una política de referencia congelada. Sin reward model, estable.
- **ORPO:** funde SFT + preferencias en **una etapa, sin modelo de referencia** (penalización odds-ratio sobre la loss SFT). El de menor overhead.
- **RLAIF:** las etiquetas de preferencia las pone un **juez IA** (otro modelo fuerte) → datos de preferencia baratos.

---

## 6. Entrenamiento (`zgrad` + `ztrain`)

### 6.1 Autodiff (`zgrad`)
Reverse-mode con cinta. Op set mínimo con su VJP: matmul, add/mul, softmax, RMSNorm/LayerNorm, SiLU/GELU, embedding, cross-entropy, RoPE, y los **mixers** (scan SSD, delta-WY, atención). Recompute en backward para los scans (ahorra memoria, patrón FlashAttention).

### 6.2 Optimizadores
- **AdamW** — default robusto (2 tensores de estado).
- **Muon** — "MomentUm Orthogonalized by Newton-Schulz": ortogonaliza el momentum de matrices 2-D; **un solo buffer** (~50 % menos memoria de optimizador), ~2× más rápido reportado, usado en GLM-4.5/Kimi. Aplicar a matrices ocultas; AdamW para embeddings/norms/LM-head. Variantes: AdaMuon, NorMuon.
- (Lion como alternativa ligera.)

### 6.3 Precisión mixta y memoria
- Cómputo en **bf16** (emulado, §02) o f32; **copia maestra de pesos y estado de optimizador en f32**. El estado recurrente de los SSM **siempre f32** (§03 §6).
- **Acumulación de gradiente** (k micro-batches → batch efectivo k×), **clipping** de norma global (~1.0), schedule **warmup + coseno/WSD**.

### 6.4 Pipeline de datos
- Pre-tokenizar el corpus una vez a un `.bin` de `u16/u32` mmap (estilo nanoGPT). **Packing:** concatenar documentos separados por EOS y cortar en ventanas `seq_len` (sin padding); máscara de frontera de documento opcional. Shuffle a nivel de shard. Determinismo con semilla (Art. V.3).

### 6.5 Bucle (pseudocódigo)
```
for batch in data:
    logits = model.forward(batch.x)          # zarch
    loss   = (CE(logits, batch.y) + distill_terms) / k
    loss.backward()                          # zgrad
    if step % k == 0:
        clip_grad_norm(1.0)
        optimizer.step(); scheduler.step(); zero_grad()
    if step % eval_every == 0: fluency_eval()    # regresión
```

---

## 7. Datos sintéticos y auto-mejora (factible para una persona)

- **Self-Instruct / Alpaca:** sembrar ~175 instrucciones, pedir al profesor (vía `zollama`) que genere miles de pares instrucción-respuesta → corpus SFT.
- **Orca-style:** capturar las **cadenas de razonamiento** del profesor (no sólo la respuesta) para transferir razonamiento.
- **Bucle de auto-mejora gobernado:** generar → filtrar (juez IA / verificadores) → SFT/on-policy → evaluar. **Cuidado** con el colapso de distribución y la amplificación de errores del profesor (monitorizar con eval).

---

## 8. El pipeline completo "de un profesor a Fluency"

```
1. fluency pull qwen2.5:7b              # zollama: descarga profesor a Ollama
2. fluency distill mohawk --stages ABC --teacher ollama:qwen2.5:7b \
        --student configs/fluency-1b.zon --tokens 5B --out fluency-1b-base
3. fluency distill seqkd --teacher ... --student fluency-1b-base --data synth/    # afinado instrucción
4. fluency distill onpolicy --teacher ... --student fluency-1b-base --out fluency-1b-op
5. fluency distill dpo --pairs prefs.jsonl --student fluency-1b-op --out fluency-1b-chat
6. fluency eval --model fluency-1b-chat --suite eval/  # compara vs profesor y baseline (criterio E4)
7. registro: docs/PROFESORES.md + checkpoint versionado
```

Resultado: un modelo **Fluency propio**, subcuadrático, destilado de un Transformer abierto, **producido íntegramente por código Zig**, listo para servir como motor (Capa 5) y actuar de agente (Capa 4).

[→ Siguiente: 05 — Runtime agéntico](05-runtime-agentico.md)
