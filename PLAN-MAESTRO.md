# Zig Super Agent (ZSA) — Plan Maestro

> **Un motor de IA (inferencia + entrenamiento + agéntica) escrito 100 % en Zig (≥ 0.16.0), con prioridad arquitectónica subcuadrática y compatibilidad Transformer, diseñado como motor independiente que puede servir como cerebro, diferenciador, herramienta, LLM o potenciador de cualquier aplicación que necesite IA local.**

- **Producto / modelo / CLI:** **Fluency** (se invoca como `fluency` desde la terminal, con TUI — ver [doc 12](docs/12-cli-y-tui.md))
- **Codename interno del repo:** Zig Super Agent (ZSA)
- **Estado del documento:** Plan v1.1 — 2026-05-29
- **Lenguaje:** Zig ≥ 0.16.0 — **exclusivo y obligatorio** (ver [Constitución](CONSTITUCION.md))
- **Licencia:** **Apache License 2.0** (decidida por el autor). Implica elegir modelos profesores con licencia que permita redistribuir pesos derivados bajo Apache 2.0 — preferir Apache-2.0/MIT (ver [04](docs/04-aprendizaje-destilacion.md) y [10](docs/10-riesgos-decisiones.md))

---

## 0. Cómo leer este plan

Este plan es un conjunto de documentos. **Empieza por la [Constitución](CONSTITUCION.md): es la capa de reglas permanentes e inviolables** que cualquier IDE, agente o modelo debe respetar al modificar el proyecto. El resto desarrolla el "qué", "cómo" y "cuándo".

| # | Documento | Qué contiene |
|---|-----------|--------------|
| 📜 | [CONSTITUCION.md](CONSTITUCION.md) | **Reglas permanentes e inviolables.** Mecanismos de enforcement. Lo primero que lee cualquier agente. |
| 🤖 | [AGENTS.md](AGENTS.md) | Punto de entrada para colaboradores y automatizaciones. Remite a la Constitución. |
| 01 | [Visión y alcance](docs/01-vision-alcance.md) | Propósito, no-objetivos, criterios de éxito, casos de uso. |
| 02 | [Arquitectura](docs/02-arquitectura.md) | Las 7 capas del sistema, mapa de módulos, flujo de datos, decisión CPU/GPU 100 % Zig. |
| 03 | [Núcleo subcuadrático](docs/03-nucleo-subcuadratico.md) | Arquitectura del modelo: híbrido SSM/atención, mixers (Mamba-2, Gated-DeltaNet, RWKV-7), kernels, numérica. |
| 04 | [Aprendizaje y destilación](docs/04-aprendizaje-destilacion.md) | Cómo aprende de otros LLM (GGUF/Ollama) 100 % en Zig: MOHAWK, KD, on-policy, DPO/ORPO; entrenamiento. |
| 05 | [Runtime agéntico](docs/05-runtime-agentico.md) | Tool-calling, bucle ReAct, planificación, memoria, sub-agentes, seguridad (estilo hermes-agent + nullclaw). |
| 06 | [Interoperabilidad y protocolos](docs/06-interop-protocolos.md) | MCP (cliente+servidor), hooks, skills, plugins, API OpenAI-compatible, C-ABI/WASM embebido. |
| 07 | [Entrenamiento vía IDEs](docs/07-entrenamiento-via-ides.md) | Cómo agentes de IDE entrenan/mejoran el modelo dentro de los límites constitucionales. |
| 08 | [Roadmap por fases](docs/08-roadmap-fases.md) | 8 fases, hitos, entregables, criterios de avance (gate-checks). |
| 09 | [Estructura y tooling](docs/09-estructura-tooling.md) | Layout del repo, `build.zig`, testing, gates de CI escritos en Zig. |
| 10 | [Riesgos y decisiones](docs/10-riesgos-decisiones.md) | Riesgos técnicos, trade-offs y ADRs (registro de decisiones de arquitectura). |
| 12 | [CLI `fluency` y TUI](docs/12-cli-y-tui.md) | Cómo se invoca desde CMD y su interfaz de terminal. **Convención de nombres canónica.** |
| 11 | [Referencias](docs/11-referencias.md) | Todas las fuentes verificadas (papers, specs, repos) con URLs. |

---

## 1. Resumen ejecutivo

### 1.1 Qué es ZSA

ZSA es un **motor de modelo de lenguaje + runtime agéntico**, escrito íntegramente en Zig, pensado para ejecutarse **localmente** (CPU, edge, navegador vía WASM, o GPU vía backends Zig a futuro) y servir como **motor de IA intercambiable** para cualquier aplicación que lo consuma. No es "otro wrapper de Ollama": es un motor propio que (a) **carga y ejecuta** modelos abiertos (GGUF/safetensors), (b) **aprende de ellos** destilando su conocimiento a un modelo propio de arquitectura subcuadrática, y (c) **actúa de forma agéntica** (herramientas, planificación, memoria) exponiéndose por las mismas APIs que ya consumen los IDEs y cualquier cliente compatible con OpenAI/Ollama.

### 1.2 Las cinco propiedades obligatorias (del encargo) y cómo se cumplen

| Requisito del encargo | Cómo lo resuelve el plan | Documento |
|---|---|---|
| **100 % Zig, ningún otro lenguaje** | Regla constitucional #1 con enforcement mecánico en CI (escrito en Zig). CPU+SIMD puro; GPU sólo vía backends Zig (SPIR-V/PTX/AMDGCN), nunca CUDA/Triton/Python. | [Constitución](CONSTITUCION.md), [02](docs/02-arquitectura.md) |
| **Aprender de otros LLM, lo aprendido 100 % en Zig** | Destilación cruzada Transformer→subcuadrático (**MOHAWK**), KD por logits, on-policy y secuencias. El profesor se consulta como oráculo (vía nuestro propio motor GGUF o vía Ollama HTTP); **todo el código de aprendizaje y los pesos resultantes son Zig**. | [04](docs/04-aprendizaje-destilacion.md) |
| **Capacidades agénticas avanzadas** (estilo hermes-agent, nullclaw) | Runtime agéntico con registro de herramientas (vtable), bucle ReAct + presupuesto, plan-and-execute, reflexión con verificador externo, sub-agentes con privilegios acotados, memoria híbrida SQLite-BLOB + FTS5. | [05](docs/05-runtime-agentico.md) |
| **Auto-entrenamiento desde modelos descargables / Ollama** | Cliente Ollama (`/api/pull`, `/api/chat`, `/api/embed`) + lector GGUF nativo → el "profesor" alimenta el pipeline de destilación. Generación de datos sintéticos (self-instruct/Orca). | [04](docs/04-aprendizaje-destilacion.md) |
| **Entrenable vía CLI/IDEs** | Archivo de gobernanza (`AGENTS.md`), skills/comandos `fluency train`/`fluency distill`, y gates de CI que rechazan cualquier cambio que viole la Constitución. | [07](docs/07-entrenamiento-via-ides.md) |
| **Lineamientos permanentes e inviolables** | La [Constitución](CONSTITUCION.md): reglas numeradas + **enforcement mecánico** (no sólo texto) + hash de integridad + protección en CODEOWNERS/CI. | Constitución |
| **Compatible Transformer, prioridad subcuadrática** | Bloque híbrido unificado: el *mixer* es intercambiable (SSM/lineal/sparse/atención). Por defecto ~80-90 % capas subcuadráticas + ~10-20 % atención de ventana deslizante. | [03](docs/03-nucleo-subcuadratico.md) |
| **Compatible con MCP, hooks, skills, plugins** | Núcleo JSON-RPC 2.0 único → cliente y servidor MCP; cargador de hooks/skills/plugins (host). | [06](docs/06-interop-protocolos.md) |

### 1.3 Decisión arquitectónica central: el bloque híbrido

La investigación es concluyente: **los modelos subcuadráticos puros chocan contra un muro** en recuperación exacta e in-context learning, y **casi todos los modelos subcuadráticos de producción 2025-2026 son híbridos** (Jamba, Nemotron-H, Qwen3-Next, MiniMax, Samba). El patrón ganador:

```
bloque_i = x + Mixer_i(Norm(x));  x = x + MLP/MoE(Norm(x))
```

donde `Mixer_i` es **intercambiable**:
- **Subcuadrático (prioridad, ~80-90 % de capas):** Mamba-2 (SSD), Gated-DeltaNet, RWKV-7, GLA.
- **Atención (compatibilidad, ~10-20 %):** atención de ventana deslizante (mantiene el coste acotado), GQA, RoPE, KV-cache.
- **Sparse "estilo subq.ai" (experimental):** atención dispersa enrutada top-k (un tipo de mixer más, a evaluar; **no** se ancla el diseño a las cifras no verificadas de subq.ai).

Esto cumple literalmente "prioridad subcuadrática + compatibilidad Transformer" y es la ruta de **menor riesgo y probada en producción**.

### 1.4 Cómo "aprende de otros LLM" sin escribir una línea fuera de Zig

El mecanismo es **destilación cruzada de arquitectura** (Transformer → subcuadrático), validada por los papers **MOHAWK** (`arXiv:2408.10189`), **Mamba-in-the-Llama** (`arXiv:2408.15237`) y **Llamba** (`arXiv:2502.14458`):

1. Cargamos un modelo profesor abierto (p. ej. Llama/Qwen/Mistral) desde GGUF/safetensors **con nuestro lector Zig**, o lo consultamos vía Ollama.
2. **Etapa A — Orientación de matrices:** alineamos la matriz de mezcla de la atención del profesor con la matriz semiseparable de nuestro SSM (Frobenius).
3. **Etapa B — Alineación de estados ocultos:** igualamos las salidas por bloque (L2).
4. **Etapa C — Transferencia de pesos + KD por logits:** copiamos embeddings/MLP/norms del profesor y destilamos la distribución de salida (KL + temperatura).

Con **~3-5 mil millones de tokens** (1 % de lo que costaría entrenar desde cero) se obtiene un alumno subcuadrático competitivo. **Todo el código que hace esto es Zig; los pesos resultantes son nuestros.**

### 1.5 La honestidad sobre la GPU (consecuencia de "100 % Zig")

- **Inferencia CPU pura en Zig es viable y competitiva hoy** (`llama2.zig` supera a `llama2.c`). Es la base inmediata.
- **bf16 no es tipo nativo en Zig** (propuesta #3148 abierta) → se emula como `u16` + conversión (la parte alta de un f32). Decisión documentada.
- **GPU de alto rendimiento en Zig puro es experimental** (backend SPIR-V ~50 % de tests). Para respetar "ningún otro lenguaje", ZSA **no** usará CUDA/ROCm/Metal/Triton/XLA. La GPU llegará vía los **backends propios de Zig** (Zig→SPIR-V/PTX/AMDGCN) conforme maduren — siguen siendo *código Zig*. Hasta entonces, el rendimiento grande se obtiene por **modelos pequeños + destilación + SIMD CPU**, no por fuerza bruta.

Esta es una elección deliberada: **pureza del lenguaje por encima de velocidad máxima de entrenamiento**. El roadmap lo asume (modelos edge primero).

### 1.6 Prior art que aceleramos (no reinventamos)

| Componente | Referencia en Zig a estudiar/portar | Nota |
|---|---|---|
| Runtime agéntico en Zig | **nullclaw** (Zig 0.16.0, 678 KB) | Patrón vtable-por-subsistema, memoria SQLite-BLOB, seguridad deny-by-default. |
| Inferencia desde cero | **llama2.zig** (cgbur), **zllm** (0.16+Metal), **llm.zig** | Referencias de kernels SIMD y bucle de inferencia. |
| Lectura de pesos | **safetensors-zig**, **tokenizers-zig**, **ggml-zig** | Bloques de carga reutilizables (auditar madurez/versión Zig). |
| Patrón agéntico (diseño) | **hermes-agent** (Python, referencia conceptual) | Registro de tools, skills como memoria procedural, formato Hermes XML. |

### 1.7 Resumen del roadmap (detalle en [08](docs/08-roadmap-fases.md))

| Fase | Objetivo | Entregable clave |
|------|----------|------------------|
| **F0** | Fundaciones | `build.zig`, Constitución, gates CI en Zig, `ztensor`+`zkernel` (matmul/softmax/rmsnorm SIMD). |
| **F1** | Inferencia | Cargar GGUF + tokenizer + correr un Transformer/SSM existente en CPU. API OpenAI-compatible mínima. |
| **F2** | Núcleo subcuadrático | Mixers Mamba-2 / Gated-DeltaNet / RWKV-7 + bloque híbrido + sampler con decodificación restringida. |
| **F3** | Aprendizaje | `zgrad` (autodiff) + `ztrain` (AdamW/Muon) + `zdistill` (MOHAWK + KD) sobre un profesor Ollama. |
| **F4** | Agéntica | `zagent`: tools, ReAct, memoria, sub-agentes, seguridad. |
| **F5** | Protocolos | MCP cliente+servidor, hooks/skills/plugins host, C-ABI + WASM. |
| **F6** | Integración ecosistema | ZSA como motor drop-in para cualquier app compatible con OpenAI/Ollama (drop-in de Ollama). |
| **F7** | Evolución | GPU vía Zig, MoE, escalado, eval continua, auto-mejora gobernada. |

### 1.8 Nomenclatura (nombres canónicos — ver [doc 12](docs/12-cli-y-tui.md))

> CLI/producto/modelo = **Fluency** (comando `fluency`). Codename interno del repo = ZSA. Módulos internos:

```
fluency        CLI + TUI                        libfluency  librería C-ABI (símbolos fluency_*)
zcore          orquestación del motor           fluency.wasm build WebAssembly
 ├─ ztensor    tensores N-dim, dtypes, mmap      ztok       tokenizadores (BPE/SPM/WP)
 ├─ zkernel    kernels SIMD (@Vector+comptime)   zload      GGUF/safetensors I/O
 ├─ zgrad      autodiff (cinta reverse-mode)     zollama    cliente Ollama HTTP
 ├─ zarch      bloque híbrido + mixers           zrpc       núcleo JSON-RPC 2.0
 ├─ ztrain     bucle de entrenamiento            zmcp       MCP cliente + servidor
 ├─ zdistill   destilación (MOHAWK/KD/DPO)        zserve     API OpenAI-compatible + SSE
 ├─ zagent     runtime agéntico                  zlsp       servidor LSP (opcional)
 └─ ztui       TUI de terminal
```

Modelos producidos: familia **Fluency** (p. ej. `Fluency-100M`, `Fluency-350M`, `Fluency-1B`).

---

## 2. Principios rectores (resumen; normativos en la Constitución)

1. **Pureza de lenguaje.** Todo el código de primera parte es Zig. Cero dependencias de otro lenguaje en el árbol de build.
2. **Subcuadrático primero.** El coste por token en decodificación e inferencia de contexto largo debe ser sub-O(N²). La atención es la excepción acotada, no la regla.
3. **Local y embebible.** Debe correr sin nube, exponerse como C-ABI/WASM, y servir de cerebro a otras apps.
4. **Evolucionable por diseño.** Mixers, kernels, tokenizadores y backends son intercambiables tras interfaces estables (vtable/comptime).
5. **Aprende, no copia.** Destila conocimiento de otros modelos a pesos propios; no reempaqueta binarios ajenos.
6. **Agéntico y seguro.** Deny-by-default, sandbox, secretos cifrados, todo en bucle con presupuesto y verificación externa.
7. **Gobernanza mecánica.** Las reglas no se cumplen "por buena fe": se cumplen porque el CI en Zig rechaza lo que las viola.

---

## 3. Próximos pasos inmediatos

1. Revisar y **ratificar la [Constitución](CONSTITUCION.md)** (es la pieza que vuelve "inviolables" los lineamientos).
2. Confirmar nombre del proyecto/modelo y licencia (ver [10](docs/10-riesgos-decisiones.md)).
3. Arrancar **Fase F0** siguiendo [08-roadmap](docs/08-roadmap-fases.md) y [09-estructura](docs/09-estructura-tooling.md).
4. Conectar el primer profesor (un modelo en Ollama) para validar el pipeline de destilación end-to-end en pequeño.

> Todo número de versión, formato y API citado en este plan está verificado contra fuentes primarias listadas en [11-referencias](docs/11-referencias.md) (fecha de corte: 2026-05-29).
