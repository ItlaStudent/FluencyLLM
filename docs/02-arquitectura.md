# 02 — Arquitectura

[← Plan Maestro](../PLAN-MAESTRO.md) · [← 01 Visión](01-vision-alcance.md) · [→ 03 Núcleo subcuadrático](03-nucleo-subcuadratico.md)

Este documento describe **cómo está construido ZSA**: sus 7 capas, el mapa de módulos Zig, el flujo de datos, y las decisiones de cómputo (CPU/SIMD ahora, GPU-vía-Zig después) que se derivan de la Constitución.

---

## 1. Vista de 10.000 pies

ZSA es una **pila por capas**. Cada capa depende sólo de las inferiores y expone interfaces estables (vtable en runtime y/o `comptime`) para que la capa superior —y los puntos de extensión— evolucionen sin tocar el núcleo (Art. VII).

```
┌─────────────────────────────────────────────────────────────────────┐
│ CAPA 6 · GOBERNANZA Y EVOLUCIÓN                                       │
│   gates CI (Zig) · registro de modelos · eval harness · versionado   │
├─────────────────────────────────────────────────────────────────────┤
│ CAPA 5 · PROTOCOLOS Y SERVING                                         │
│   zrpc (JSON-RPC 2.0) · zserve (OpenAI API+SSE / Ollama-style)        │
│   zmcp (MCP cliente+servidor) · zlsp · host de hooks/skills/plugins   │
│   fluency CLI + ztui (TUI) · libfluency (C-ABI) · wasm │
├─────────────────────────────────────────────────────────────────────┤
│ CAPA 4 · RUNTIME AGÉNTICO  (zagent)                                   │
│   registro de tools · ReAct/plan-execute · memoria · sub-agentes      │
│   seguridad deny-by-default · sandbox · auditoría                     │
├─────────────────────────────────────────────────────────────────────┤
│ CAPA 3 · APRENDIZAJE  (ztrain · zdistill)                             │
│   autodiff(zgrad) · optimizadores · KD/MOHAWK/on-policy/DPO           │
│   pipeline de datos · profesores (zollama/zload)                      │
├─────────────────────────────────────────────────────────────────────┤
│ CAPA 2 · ARQUITECTURA DEL MODELO  (zarch)                             │
│   bloque híbrido · mixers {Mamba-2, GatedDeltaNet, RWKV-7, GLA,       │
│   atención SWA/GQA, sparse top-k} · sampler · estado dual (KV+rec.)   │
├─────────────────────────────────────────────────────────────────────┤
│ CAPA 1 · FORMATOS E INTEROP DE MODELOS                                │
│   zload (GGUF/safetensors) · ztok (tokenizadores) · zollama (cliente) │
├─────────────────────────────────────────────────────────────────────┤
│ CAPA 0 · FUNDACIONES  (puro Zig, sin terceros)                        │
│   ztensor (tensores, dtypes, mmap) · zkernel (SIMD @Vector+comptime)  │
│   allocators · paralelismo std.Io · util (json, bytes, hash, rng)     │
└─────────────────────────────────────────────────────────────────────┘
            ▲ backend de cómputo intercambiable: CPU-SIMD (hoy) │ GPU-Zig SPIR-V/PTX (evolución) │ WASM-SIMD
```

---

## 2. Capa 0 — Fundaciones (puro Zig)

La base numérica y de sistema. **Sin una línea de terceros.**

### 2.1 `ztensor` — tensores
- Tensor N-dimensional con *shape*, *strides*, dtype y vista (slice) sobre un buffer. Soporta vistas no-propietarias (para mmap) y propietarias (arena).
- **dtypes:** `f32`, `f16`, **`bf16` (emulado: ver §2.4)**, `f64`, enteros de ancho arbitrario para cuantización (`i8`, `i4`, `u4`…), y los formatos de bloque cuantizado (Q4_0, Q4_K, Q6_K… definidos en [04 §GGUF]).
- **Backing:** `mmap` de archivos de pesos (read-only, zero-copy), arena para activaciones efímeras, pool para KV/estado recurrente.

### 2.2 `zkernel` — kernels SIMD
- Operaciones: `matmul` (con micro-kernel *tiled* y empaquetado), `rmsnorm`, `layernorm`, `softmax`, `rope`, activaciones (`silu`, `gelu`), `scan` asociativo (para SSM), `gather`/`top-k` (para sparse).
- Implementadas con **`@Vector`** + **`comptime`**: el tamaño de tile, el ancho del vector y el dtype se especializan en tiempo de compilación (`inline for`, funciones que devuelven tipos). Es uno de los mejores encajes de Zig (la investigación confirma que `@Vector` puro **supera a `llama2.c`** sin BLAS).
- Cada kernel tiene una **referencia escalar lenta y legible** (`*_ref`) y un test de igualdad numérica (Art. V.2).

### 2.3 Paralelismo
- **Importante (Zig 0.16):** `std.Thread.Pool` fue **eliminado** (PR #30557, dic-2025). El paralelismo de datos (repartir filas de matmul / cabezas de atención entre núcleos) se hace con el **modelo `std.Io`** (`io.async` / `Io.Group` sobre el backend `Io.Threaded`). Se encapsula en un helper `zkernel.parallelFor` para no esparcir la API nueva por todo el código.

### 2.4 bf16 emulado (decisión forzada por #3148)
- Zig **no** tiene `bf16` nativo (propuesta #3148 abierta). `ztensor.bf16` lo representa como `u16` con conversiones:
  - `bf16→f32`: ensanchar = `@as(u32, x) << 16` reinterpretado como `f32` (la parte alta del f32).
  - `f32→bf16`: truncar con redondeo round-to-nearest-even.
- No hay aritmética vectorial bf16 por HW; los cómputos se hacen en f32 y se almacena/transfiere en bf16. Documentado en ADR-0003.

### 2.5 Utilidades
- `util.json` (parser/serializador JSON propio — necesario para JSON-RPC, GGUF metadata-as-json, configs), `util.bytes` (lectura LE/BE, mmap helpers), `util.hash` (sha256 para GGUF/manifests y para el hash constitucional), `util.rng` (PRNG con semilla para determinismo, Art. V.3), `util.crypto` (ChaCha20-Poly1305 para secretos, Art. IV.3 — implementación Zig pura; `std.crypto` ya la provee).

> **Nota:** `std.crypto`, `std.json`, `std.http` existen en stdlib Zig; se usan preferentemente. Donde stdlib no alcance (p. ej. SSE, framing JSON-RPC), se implementa en `util`/`zrpc`. Esto **no** viola la pureza: stdlib es Zig.

---

## 3. Capa 1 — Formatos e interoperabilidad

Permite **leer el mundo** (modelos ajenos) sin incorporar su código.

- **`zload`** — lectores/escritores de **GGUF** (header, KV metadata, tensor info, dequant de bloques Q*) y **safetensors** (u64 len + header JSON + datos). Detalles de formato y fórmulas de dequant en [04](04-aprendizaje-destilacion.md).
- **`ztok`** — tokenizadores reconstruidos desde GGUF (`tokenizer.ggml.*`) o `tokenizer.json`: BPE byte-level (GPT-2/tiktoken: Llama-3, Qwen), SentencePiece unigram (Llama-1/2, Mistral) con Viterbi, WordPiece (BERT). Manejo correcto de tokens especiales y del regex de pre-tokenización (causa #1 de "carga bien pero genera basura").
- **`zollama`** — cliente HTTP del demonio Ollama (`/api/pull`, `/api/chat`, `/api/generate`, `/api/embed`, `/api/show`, `/api/tags`). Sirve para usar modelos descargables **como profesores** y para introspección. Implementado sobre `std.http` + `util.json`.

Estas tres piezas son el puente "aprender de otros LLM" del lado de la *entrada*.

---

## 4. Capa 2 — Arquitectura del modelo (`zarch`)

El corazón. Detalle completo en **[03 — Núcleo subcuadrático](03-nucleo-subcuadratico.md)**; aquí, la forma arquitectónica:

- **Bloque híbrido unificado.** Cada capa = `x += Mixer(Norm(x)); x += FFN/MoE(Norm(x))`. Sólo cambia el `Mixer`.
- **Mixers (vtable `Mixer.VTable`):** `forwardPrefill(seq)`, `stepDecode(state, token)`, `initState()`, `paramCount()`. Implementaciones:
  - Subcuadráticos (prioridad): **Mamba-2/SSD**, **Gated-DeltaNet**, **RWKV-7**, **GLA**.
  - Atención (compat): **SWA** (ventana deslizante), **GQA** full, con **RoPE** + **KV-cache**.
  - **Sparse top-k** (experimental, "estilo subq.ai"): enrutado de queries a bloques de claves seleccionados + atención exacta sobre el subconjunto.
- **Estado dual de inferencia:** las capas de atención llevan KV-cache (acotado por SWA); las subcuadráticas llevan **estado recurrente de tamaño fijo**. El runtime avanza ambos por token.
- **Sampler:** greedy, temperatura, top-p, top-k, repetición; y **decodificación restringida** por gramática (estilo GBNF) para garantizar tool-calls/JSON válidos (clave para la agéntica — ver [05](05-runtime-agentico.md)).
- **Config de modelo** (`model.zon`): nº de capas, qué mixer en cada capa (el "patrón de hibridación"), dims, vocab, etc. Cambiar el patrón es editar datos, no código.

---

## 5. Capa 3 — Aprendizaje (`zgrad`, `ztrain`, `zdistill`)

El puente "aprender de otros LLM" del lado del *entrenamiento*. Detalle en **[04](04-aprendizaje-destilacion.md)**.

- **`zgrad`** — autodiff reverse-mode (cinta de operaciones + reglas VJP) para el conjunto mínimo de ops: matmul, add, softmax, norm, activaciones, embedding, cross-entropy, y los mixers (scan SSM, etc.).
- **`ztrain`** — bucle de entrenamiento: optimizadores **AdamW** y **Muon** (este último ~50 % menos memoria de optimizador, usado en GLM-4.5/Kimi), precisión mixta (cómputo bf16/f32, copia maestra f32), acumulación de gradiente, clipping, schedules (warmup+coseno/WSD). Pipeline de datos con corpus tokenizado empaquetado en `.bin` mmap.
- **`zdistill`** — destilación: KD por logits (KL+T), secuencia (Seq-KD), **on-policy** (KL inversa), **DPO/ORPO** (preferencias sin reward model), y **MOHAWK** (cruzada Transformer→subcuadrático). El profesor se consulta vía `zollama` o vía inferencia propia con `zload`.

---

## 6. Capa 4 — Runtime agéntico (`zagent`)

Detalle en **[05](05-runtime-agentico.md)**. Resumen estructural (patrón inspirado en nullclaw + hermes-agent):

- **Registro de tools** (vtable `Tool.VTable`: `execute/name/description/parameters_json`), auto-registro, despacho sin `panic`.
- **Bucle** ReAct (think→tool_calls→tool_responses) con **presupuesto** de iteraciones/tokens e interrupción; modo plan-and-execute con *re-plan gate*; reflexión con **verificador externo** (no auto-crítica).
- **Memoria** híbrida: contexto (working) + **store propio en Zig** (embeddings + índice invertido BM25, ADR-0009; sin SQLite/C) fusionados, + skills como **memoria procedural**.
- **Sub-agentes** con contexto aislado y **privilegios acotados** (sin delegación recursiva por defecto).
- **Seguridad** deny-by-default, sandbox del SO, secretos cifrados, auditoría (Art. IV).

---

## 7. Capa 5 — Protocolos y serving

Detalle en **[06](06-interop-protocolos.md)**. El hallazgo clave: **un único núcleo JSON-RPC 2.0 sirve MCP y LSP**, y el mismo servidor HTTP sirve la API OpenAI y el transporte MCP Streamable-HTTP.

- **`zrpc`** — códec JSON-RPC 2.0 + framings: newline-delimited (MCP stdio), `Content-Length` (LSP), HTTP-POST/SSE (MCP HTTP + OpenAI API).
- **`zserve`** — API **OpenAI-compatible** (`/v1/chat/completions` con streaming SSE y `tools`, `/v1/models`, `/v1/embeddings`) y **estilo Ollama** (`/api/chat`, `/api/generate`, `/api/embed`). Esto es lo que vuelve a ZSA un **drop-in** para cualquier app/IDE compatible.
- **`zmcp`** — **cliente** MCP (consumir tool servers, ofrecer `sampling` con nuestro modelo) y **servidor** MCP (exponer nuestras tools).
- **`zlsp`** — servidor LSP opcional (reusa `zrpc`).
- **Host de extensiones** — cargadores de hooks (eventos + contrato JSON), skills (`SKILL.md` + progressive disclosure), plugins.
- **CLI + TUI (`fluency`)** — `ztui`: front-end de terminal TUI (`fluency` abre la TUI; `fluency serve` el servidor; `fluency run` one-shot). Ver [12](12-cli-y-tui.md).
- **Embebido** — `libfluency` (C-ABI: `export fn` con tipos C-ABI) y `fluency.wasm` (`wasm32-freestanding`/`wasi`, SIMD128).

---

## 8. Capa 6 — Gobernanza y evolución

Detalle en [07](07-entrenamiento-via-ides.md), [09](09-estructura-tooling.md), [10](10-riesgos-decisiones.md).
- **Gates** `tools/gate_*.zig` (pureza, constitución, calidad, presupuestos) — el enforcement del Art. IX.
- **Eval harness** (`fluency eval`) — suites de calidad y regresión, comparación contra baseline/profesor.
- **Registro de modelos** — checkpoints versionados, metadatos de procedencia (`docs/PROFESORES.md`), migración de formatos.

---

## 9. Flujos de datos principales

### 9.1 Inferencia (servir como motor)
```
cliente (app/IDE) ─HTTP /v1/chat/completions─▶ zserve
   └▶ ztok.encode ─▶ zarch.prefill(prompt) [mixers] ─▶ sampler(restringido)
        ─▶ zarch.stepDecode×N (estado dual) ─▶ ztok.decode ─▶ SSE stream ─▶ cliente
   (si hay tools) ─▶ zagent bucle ReAct ─▶ tool/MCP ─▶ continúa
```

### 9.2 Aprendizaje (destilar un profesor)
```
profesor (Ollama o GGUF) ──oráculo──▶ zdistill
   ├ MOHAWK A: alinear matriz de mezcla (Frobenius)
   ├ MOHAWK B: alinear estados ocultos (L2)
   └ MOHAWK C: transferir pesos + KD logits (KL+T)   ──▶ checkpoint Fluency versionado
        (zgrad backward · ztrain optimizador Muon/AdamW · datos mmap)
```

### 9.3 Embebido (apps no-Zig)
```
host no-Zig ─C-ABI/WASM─▶ libfluency: fluency_load(model) · fluency_generate(prompt,opts,callback) · fluency_free
```

---

## 10. Decisión de cómputo: CPU/SIMD ahora, GPU-vía-Zig después

Esta es la decisión arquitectónica más delicada y se deriva **directamente de la Constitución (Art. I.4)**. Registrada en **ADR-0002**.

| Backend | Estado | Pureza | Uso en ZSA |
|---|---|---|---|
| **CPU + `@Vector` SIMD** | Maduro en Zig; probado competitivo | 100 % Zig | **Default inmediato** (inferencia y entrenamiento pequeño). |
| **WASM + SIMD128** | Maduro (`wasm32`) | 100 % Zig | Inferencia en navegador. Ancho 128-bit. |
| **GPU vía backend Zig** (SPIR-V/PTX/AMDGCN) | **Experimental** (~50 % tests Vulkan) | 100 % Zig (compila Zig→GPU) | **Evolución (Fase 7)**, a medida que el backend madure. Detrás de la misma vtable de backend. |
| ~~CUDA/cuBLAS/ROCm/Metal/Triton/XLA~~ | Maduro | **❌ otro lenguaje** | **PROHIBIDO** por Art. I.3/I.4. |

**Consecuencia honesta y asumida:** sin CUDA/BLAS, el entrenamiento masivo es inviable a corto plazo. Por eso la estrategia es **modelos pequeños + destilación eficiente** (MOHAWK necesita ~3-5B tokens, no billones) y **kernels CPU SIMD bien hechos**. El rendimiento "grande" es un objetivo de evolución que llegará por el backend GPU **de Zig**, no por romper la pureza.

El **backend de cómputo es una vtable** (`Backend.VTable`: `matmul`, `scan`, `softmax`…), de modo que CPU-SIMD, WASM-SIMD y (futuro) GPU-Zig son intercambiables sin tocar `zarch`/`zkernel` callers. Esto materializa la evolucionabilidad del Art. VII.

---

## 11. Dependencias entre módulos (resumen)

```
util ─◀ ztensor ─◀ zkernel ─◀ zarch ─◀ {ztrain, zdistill, zagent, zserve}
                         ▲           ▲
                      zgrad ─────────┘ (entrenamiento)
zload ─◀ zarch (cargar pesos)     zrpc ─◀ {zserve, zmcp, zlsp}
ztok ─◀ {zarch, zserve}           zollama ─◀ {zdistill, zagent}
ztui ─◀ fluency (CLI/TUI)         {zagent, zarch, zserve} ─◀ ztui (la TUI los usa)
```
Regla: **las flechas no se invierten** (sin dependencias circulares). El núcleo (`util`→`zarch`) no conoce a las capas superiores.

[→ Siguiente: 03 — Núcleo subcuadrático](03-nucleo-subcuadratico.md)
