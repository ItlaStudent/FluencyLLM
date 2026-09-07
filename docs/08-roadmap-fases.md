# 08 — Roadmap por fases

[← Plan Maestro](../PLAN-MAESTRO.md) · [← 07 Entrenamiento vía IDEs](07-entrenamiento-via-ides.md) · [→ 09 Estructura y tooling](09-estructura-tooling.md)

8 fases incrementales. Cada fase entrega algo **funcional y testeado**, y tiene un **gate-check** (criterio de avance) antes de pasar a la siguiente. No hay fechas absolutas (depende de la dedicación); el orden y las dependencias sí son firmes.

> **Principio:** "camina antes de correr". Cada fase deja un producto usable, no un esqueleto. Se valida contra referencias reales (modelos en Ollama) en cuanto es posible.

> **Estado de avance:** este documento define los objetivos (inviolables). El progreso real, la evidencia de cada gate-check y los próximos pasos concretos viven en [ESTADO-FASES.md](ESTADO-FASES.md) (documento vivo). Resumen a 2026-05-30: **F0, F1, F2, F3 al 100%** (gate-checks E2/E3/E4 cumplidos y verificados); F4 ~25%, F5 ~70%.

---

## F0 — Fundaciones y gobernanza

**Objetivo:** repo Zig que compila, con gobernanza activa y la base numérica.

**Entregables:**
- `build.zig` + `build.zig.zon` (Zig fijado ≥ 0.16.0), pasos `build`, `test`, `gate`, `eval`, `bench`.
- **Constitución + gates** (`tools/gate_purity.zig`, `gate_constitution.zig`, `gate_quality.zig`, `gate_budgets.zig`) funcionando en pre-commit y CI. Hash de integridad fijado.
- Archivos de gobernanza por-IDE en su sitio.
- `util` (json, bytes, hash sha256, rng con semilla, crypto), `ztensor` (tensores, dtypes, **bf16 emulado**, mmap), `zkernel` base (`matmul`, `rmsnorm`, `softmax`, `silu`, `rope`) con `*_ref` + tests de igualdad numérica.
- `parallelFor` sobre `std.Io`.

**Gate-check F0:** `zig build gate && zig build test` verde; `matmul` SIMD coincide con `matmul_ref` dentro de tolerancia; gate de pureza demostrado (un PR de prueba con un `.py` es rechazado).

---

## F1 — Inferencia de modelos existentes (caminar)

**Objetivo:** cargar un Transformer abierto y generar texto correcto, en CPU, 100 % Zig.

**Entregables:**
- `zload`: lector GGUF (header/KV/tensor-info) + dequant F32/F16/Q8_0/Q4_0/Q4_K. Lector safetensors.
- `ztok`: BPE byte-level + SentencePiece unigram (cubre Llama/Qwen/Mistral). Test de igualdad de IDs vs referencia.
- `zarch` mínimo: bloque Transformer (GQA + RoPE + KV-cache + SwiGLU + RMSNorm), sampler (greedy/temp/top-p).
- `zserve` mínimo: `POST /v1/chat/completions` (no streaming) + `GET /v1/models`.

**Gate-check F1 (criterio E2):** cargar un modelo pequeño (p. ej. Qwen2.5-0.5B GGUF) y reproducir la salida greedy de Ollama para varios prompts (tolerancia de logits). Esto **valida el camino de profesor 100 % Zig** para F3.

---

## F2 — Núcleo subcuadrático (la identidad del proyecto)

**Objetivo:** mixers subcuadráticos + bloque híbrido + decodificación restringida.

**Entregables:**
- Kernels: `linear_attn_chunkwise`, **`scan_chunkwise` (SSD/Mamba-2)**, `attn_swa`. Todos con `*_ref` y test de estabilidad en secuencia larga.
- Mixers tras vtable: `GLA` → `Mamba2` → `SWA/GQA`. Bloque híbrido con patrón configurable en `model.zon`.
- Estado dual de inferencia (KV-cache + estado recurrente), `stepDecode` O(1) memoria.
- Sampler con **decodificación restringida por gramática** (json-schema→gramática) para tool-calls válidos.
- Streaming SSE en `zserve`.

**Gate-check F2 (criterio E3):** un modelo híbrido inicializado aleatoriamente corre prefill+decode; la **memoria de decodificación es constante** respecto a tokens generados en capas SSM; `scan_chunkwise` coincide con la recurrencia `*_ref` token-a-token.

---

## F3 — Aprendizaje (el corazón diferencial)

**Objetivo:** destilar un profesor Transformer en un Fluency subcuadrático, todo en Zig.

**Entregables:**
- `zgrad`: autodiff reverse-mode con VJP para el op-set + mixers (backward de `scan`/`delta`/`attn`).
- `ztrain`: bucle con **AdamW** y **Muon**, precisión mixta, acumulación, clipping, schedules, pipeline de datos mmap.
- `zdistill`: **MOHAWK A/B/C**, KD logits, Seq-KD; profesor vía `zollama` y vía `zload`.
- `fluency eval`: suite de evaluación + comparación baseline/profesor.
- Primer **`Fluency-100M`/`Fluency-350M`** destilado de un profesor 1-3B.

**Gate-check F3 (criterio E4):** un Fluency destilado alcanza ≥ X % de la calidad del profesor en la suite propia con ≤ 5B tokens; entrenamiento reproducible con semilla; sin leaks.

---

## F4 — Runtime agéntico

**Objetivo:** Fluency actúa como agente con herramientas, memoria y seguridad.

**Entregables:**
- `zagent`: `Registry` de tools (vtable), parser Hermes/OpenAI, bucle ReAct + presupuesto + interrupción, plan-and-execute, reflexión con verificador externo.
- Tools nativas: `fs_*` (workspace), `bash` (sandbox), `http_fetch` (allowlist), `search`, `memory_*`, `delegate`, `finish`.
- `zmemory`: store propio en Zig (vectores + BM25) híbrido (ADR-0009, sin SQLite/C); skills como memoria procedural.
- **`ztui` + CLI `fluency`**: TUI interactiva (chat/agente, streaming, diffs aprobables, sesiones). Invocable desde CMD como `fluency`. Ver [12](12-cli-y-tui.md).
- Seguridad Art. IV: sandbox, secretos ChaCha20-Poly1305, deny-by-default, auditoría.
- Afinado de Fluency para tool-use (datos de trayectorias).

**Gate-check F4 (criterio E5):** completa un set de tareas multi-herramienta con tool-calls válidos (restringidos), respeta presupuesto, sin `panic` en la ruta de tools, sin leaks, secretos nunca en claro.

---

## F5 — Protocolos y embebido

**Objetivo:** ZSA hablando todos los protocolos del ecosistema y embebible.

**Entregables:**
- `zrpc` núcleo JSON-RPC 2.0 + framings.
- `zmcp` cliente (tools externas → registro) + servidor (tools propias + `sampling`).
- Host de hooks + skills; empaquetado como **plugin** MCP (`plugin.json`, `.mcp.json`).
- `zserve` completo (OpenAI + estilo Ollama).
- `libfluency` (C-ABI) + `fluency.wasm`.
- (opcional) `zlsp`.

**Gate-check F5 (criterio E7):** un host MCP usa una tool de ZSA y ZSA usa una tool MCP externa; `libfluency` responde desde Node y `fluency.wasm` desde navegador.

---

## F6 — Integración con el ecosistema

**Objetivo:** ZSA como motor drop-in para cualquier app compatible con OpenAI/Ollama.

**Entregables:**
- Cualquier app compatible con OpenAI/Ollama apuntando a ZSA cambiando sólo la URL base.
- Apps embebidas consumiendo `libfluency`/`fluency.wasm`.
- Guía de migración + perfiles de modelo (`Fluency-1B` como motor principal).
- Telemetría/observabilidad básica.

**Gate-check F6 (criterio E6):** las apps compatibles funcionan con ZSA cambiando sólo la URL base; latencia/calidad dentro de lo aceptable para sus casos de uso.

---

## F7 — Evolución (correr)

**Objetivo:** subir el techo de rendimiento y capacidad **sin romper la pureza**.

**Entregables (priorizables por ADR):**
- **Backend GPU vía Zig** (SPIR-V/PTX/AMDGCN) tras la vtable `Backend` — a medida que el backend del compilador madure. **Nunca** CUDA/Triton.
- Mixers adicionales: **Gated-DeltaNet/-2**, **RWKV-7**, **SparseTopK** (eval del enfoque "subq.ai").
- **MoE** en el FFN; cuantización propia para inferencia; speculative decoding y speculative tool-calls.
- `Fluency-3B`+; on-policy a mayor escala; A2A para multi-agente.
- Multimodalidad (visión) como línea de I+D.

**Gate-check F7:** cada incorporación pasa los gates, mejora una métrica de eval sin regresión, y respeta Art. I (si toca GPU, es Zig→GPU).

---

## Dependencias entre fases

```
F0 ──▶ F1 ──▶ F2 ──▶ F3 ──▶ F4 ──▶ F5 ──▶ F6
                 │            │      │
                 └────────────┴──────┴──▶ F7 (evolución continua, tras F3+)
```
- F3 (aprendizaje) depende de F1 (cargar profesor) y F2 (alumno subcuadrático).
- F4 (agéntica) necesita un Fluency usable (F3) y sampler restringido (F2).
- F6 necesita serving (F5) y un modelo bueno (F3/F4).
- F7 es continua una vez hay base entrenable.

## Hitos "demo-ables" (para mantener motivación y validar pronto)

1. **Fin F1:** "ZSA generó texto correcto de un Qwen, en Zig puro, en mi CPU."
2. **Fin F2:** "decodificación con memoria constante — gráfica RSS plana."
3. **Fin F3:** "Fluency-350M destilado de Qwen responde coherente; lo entrené yo, en Zig."
4. **Fin F4:** "Fluency usó `bash` y `fs` para completar una tarea sola."
5. **Fin F5:** "cualquier host MCP-compatible usa ZSA como servidor MCP; cualquier app lo embebe por WASM."
6. **Fin F6:** "Cualquier app compatible con Ollama corre sobre ZSA en vez de Ollama."

[→ Siguiente: 09 — Estructura y tooling](09-estructura-tooling.md)
