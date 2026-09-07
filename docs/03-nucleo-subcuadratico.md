# 03 — Núcleo subcuadrático (arquitectura del modelo)

[← Plan Maestro](../PLAN-MAESTRO.md) · [← 02 Arquitectura](02-arquitectura.md) · [→ 04 Aprendizaje](04-aprendizaje-destilacion.md)

Este documento define **la arquitectura de la familia de modelos Fluency**: por qué híbrida, qué mixers, qué kernels hay que escribir, y la numérica delicada. Cumple el mandato "prioridad subcuadrática + compatibilidad Transformer" (Art. II) de la forma **probada en producción**.

---

## 1. Por qué híbrido (y no subcuadrático puro)

La evidencia 2024-2026 es contundente:

- Los modelos subcuadráticos **puros** (Mamba, RWKV, atención lineal) son excelentes en coste pero **flaquean en recuperación exacta e in-context learning** (las "induction heads" que hacen copiar/recordar con precisión).
- **Casi todos los modelos subcuadráticos de producción son híbridos**: intercalan **una fracción pequeña de capas de atención** entre muchas capas SSM/lineales y **recuperan la calidad Transformer** manteniendo el coste bajo.

| Modelo | Receta | Ratio atención |
|---|---|---|
| Jamba (AI21) | Mamba + Transformer + MoE | ~1:7 |
| Nemotron-H (NVIDIA) | Mamba-2 + MLP + atención | **~8 %** (4 de 52 capas en el 8B) |
| Samba (Microsoft) | Mamba + ventana deslizante | intercalado |
| Qwen3-Next | atención lineal + softmax + MoE | 25 % softmax |
| MiniMax-01/M1 | Lightning (lineal) + softmax | ~12,5 % (7:1) |

**Decisión ZSA (ADR-0004):** el modelo por defecto es híbrido con **~80-90 % de capas subcuadráticas** y **~10-20 % de atención de ventana deslizante**, colocadas no en los extremos. Esto satisface literalmente "prioridad subcuadrática" (II.3: ≥ 75 % subcuadrático) y "compatibilidad Transformer".

> **Sobre subq.ai:** su enfoque expresado es **atención dispersa adaptativa al contenido** (top-k + clustering jerárquico) sobre modelos base Transformer. Es una dirección legítima, pero **no tiene paper ni código públicos y sus cifras no están verificadas**. ZSA lo trata como **un tipo de mixer más a evaluar** (`SparseTopK`, §4.5), no como el ancla del diseño (Art. II.4).

---

## 2. El bloque unificado

Todas las capas comparten esqueleto; sólo cambia el *mixer*:

```
entrada x  (residual stream, dim = d_model)
   │
   ├─ h = Norm(x)                 # RMSNorm (preferido)
   ├─ x = x + Mixer_i(h)          # ← intercambiable: SSM | lineal | RWKV | atención | sparse
   ├─ h2 = Norm(x)
   └─ x = x + FFN_i(h2)           # MLP (SwiGLU) o MoE
salida x
```

- **Norm:** RMSNorm (más barato y estable que LayerNorm; estándar en LLMs modernos).
- **FFN:** SwiGLU por defecto; **MoE** opcional (capacidad sin coste de cómputo proporcional — evolución).
- **Patrón de hibridación:** un vector en `model.zon`, p. ej. para 24 capas: `[S,S,S,A,S,S,S,S,S,A,S,S,S,S,S,A,S,S,S,S,S,A,S,S]` donde `S`=subcuadrático, `A`=atención SWA. Editar el patrón es cambiar **datos**, no código.

### Interfaz `Mixer` (vtable)
```zig
pub const Mixer = struct {
    pub const VTable = struct {
        // prefill: procesa toda la secuencia (forma chunk-paralela / matmul-rica)
        forwardPrefill: *const fn (self: *anyopaque, io: Io, h: Tensor, out: Tensor) Error!void,
        // decode: un token, avanzando el estado (forma recurrente, O(1) memoria)
        stepDecode: *const fn (self: *anyopaque, io: Io, state: *State, token_h: Tensor, out: Tensor) Error!void,
        initState: *const fn (self: *anyopaque, alloc: Allocator, batch: usize) Error!State,
        // para autodiff en entrenamiento:
        backward: *const fn (self: *anyopaque, io: Io, grad_out: Tensor, ctx: *Tape) Error!void,
    };
    ptr: *anyopaque,
    vtable: *const VTable,
};
```
Añadir un mixer nuevo = implementar la vtable y registrarlo. **El núcleo no se toca** (Art. VII.1). La dualidad **prefill paralelo / decode recurrente** es la esencia de todos los subcuadráticos.

---

## 3. La gran idea común: dualidad recurrencia↔paralelo

Todos los mixers subcuadráticos comparten una estructura matemática:

- En **entrenamiento/prefill** se expresan en **forma chunk-paralela** (rica en matmul, aprovecha SIMD/cores): se trocea la secuencia en *chunks* (típico 64-256 tokens), se calcula la parte **intra-chunk en paralelo** y se propaga el **estado entre chunks**.
- En **inferencia/decode** se expresan como **recurrencia**: estado de tamaño fijo `S_t = f(S_{t-1}, x_t)`, salida `y_t = g(S_t, x_t)`. **Memoria constante por token, sin KV-cache.**

Esto es lo que hace que sean O(N) en vez de O(N²), y es **la razón por la que son ideales para correr local** (la ventaja asesina: memoria de decodificación constante). El kernel central que hay que dominar es el **chunkwise scan**.

---

## 4. Los mixers (qué implementar y en qué orden)

### 4.1 Mamba-2 / SSD (prioridad #1 — el caballo de batalla)
- **Idea:** SSM selectivo. Estado latente `h_t = Ā_t·h_{t-1} + B̄_t·x_t`, salida `y_t = C_t·h_t`, con `B,C,Δ` **dependientes del input** (selección de qué recordar/olvidar por contenido).
- **SSD (State-Space Duality, Mamba-2):** reformula el SSM como producto de **matrices semiseparables** → cómputo **chunkwise rico en matmul**, 2-8× más rápido que Mamba-1.
- **Por qué primero:** mejor relación calidad/coste, base de Nemotron-H/Jamba, abundante literatura. Ref: `arXiv:2405.21060`, `arXiv:2312.00752`.
- **Kernel:** chunkwise SSD (intra-chunk matmul + estado inter-chunk). Estado en **fp32** por estabilidad (§6).

### 4.2 Gated-DeltaNet / Gated-DeltaNet-2 (prioridad #2 — recall)
- **Idea:** atención lineal con **regla delta** (actualización tipo fast-weights: escribe el *error* predicho-objetivo, no el valor crudo) + **gating** (olvido adaptativo). Mejora drásticamente la **recuperación asociativa**, la mayor debilidad de los lineales.
- **Gated-DeltaNet-2** (NVIDIA, mayo-2026) desacopla *erase* y *write* en dos gates → estado del arte a su escala; pensado como mixer recurrente dentro de un bloque Transformer, con variante híbrida de ventana deslizante.
- **Kernel:** chunkwise con **representación WY** (productos de Householder) para paralelizar la recurrencia delta; el backward necesita VJP gate-aware. Ref: `arXiv:2412.06464`, `arXiv:2406.06484`.
- **Licencia:** el código de referencia de NVIDIA es **no comercial** — **no se copia**; se reimplementa en Zig desde el paper (Art. I/III).

### 4.3 RWKV-7 "Goose" (prioridad #3 — alternativa fuerte)
- **Idea:** RNN que entrena en paralelo. **Time-mixing** (recurrencia con decay aprendido + token-shift) reemplaza la atención O(N²); **channel-mixing** hace de FFN. RWKV-7 generaliza la regla delta con gating vectorial y *learning-rate* en contexto; teóricamente reconoce todos los lenguajes regulares manteniéndose paralelizable.
- **Por qué incluirlo:** ecosistema de pesos abiertos amplio (buenos profesores), y es un punto de comparación valioso. Ref: `arXiv:2503.14456`.

### 4.4 GLA (Gated Linear Attention) — base lineal de referencia
- **Idea:** atención lineal `φ(Q)·(φ(K)ᵀV)` con gate dependiente del input. Equivale a un RNN con estado matricial. La plantilla "FlashLinearAttention" chunkwise nace aquí. Útil como mixer simple y bien entendido para validar el framework. Ref: `arXiv:2312.06635`.

### 4.5 Atención (compatibilidad Transformer)
- **SWA (Sliding-Window Attention):** atención completa pero limitada a una ventana → **mantiene el coste y la KV-cache acotados**. Es la forma preferida para las capas de atención del híbrido (II.3).
- **GQA full:** atención completa con *grouped-query* (menos cabezas KV) + **RoPE** + **KV-cache**. Necesaria para **cargar y correr modelos Transformer existentes** tal cual (compatibilidad, y para servir de profesor con nuestro propio motor).

### 4.6 SparseTopK (experimental — "estilo subq.ai")
- **Idea:** para cada query, puntuar contra claves, quedarse con el **top-k** más relevante y hacer **atención exacta sólo sobre ese subconjunto** (coste ≈ O(N·k), k≪N). Opcionalmente clustering jerárquico.
- **Estado:** mixer **a prototipar y medir** contra los anteriores; **no** es base del diseño. Es el kernel "menos commoditizado" (gather + block-sparse) y el más difícil.

**Orden de implementación recomendado:** GLA (validar framework) → Mamba-2 (caballo de batalla) → SWA/GQA (compat + profesores) → Gated-DeltaNet (recall) → RWKV-7 → SparseTopK (I+D).

---

## 5. Kernels a construir (Capa 0 `zkernel`)

| Kernel | Para | Dificultad | Nota |
|---|---|---|---|
| `matmul` *tiled* + pack | todo | media | base de todo; SIMD `@Vector`, comptime tile. |
| `rmsnorm`, `softmax`, `silu/gelu`, `rope` | todo | baja | fusionar cuando se pueda. |
| **`scan_chunkwise` (SSD)** | Mamba-2 | **alta** | intra-chunk matmul + estado inter-chunk; backward con recompute. |
| `delta_wy_chunkwise` | Gated-DeltaNet | **alta** | Householder/WY; VJP gate-aware en backward. |
| `linear_attn_chunkwise` | GLA/RWKV | media-alta | normalizador/gating en lugar de denominador crudo. |
| `attn_swa` + `kv_cache` | SWA/GQA | media | flash-style por bloques; RoPE. |
| `topk_gather` + `block_sparse_attn` | SparseTopK | **muy alta** | selección + gather + atención por bloques. |
| `xent_fused` | entrenamiento | baja | cross-entropy fundido. |

Cada kernel: implementación SIMD + **referencia escalar `*_ref`** + test de igualdad numérica (Art. V.2). El **chunkwise scan estable y rápido con backward correcto** es lo más difícil de todo el proyecto — se prioriza tiempo ahí.

---

## 6. Numérica delicada (dónde se rompen estos modelos)

La investigación marca cuatro trampas; el diseño las contempla explícitamente:

1. **Estabilidad del estado.** La recurrencia puede explotar/desvanecerse. Parametrizar `A` para radio espectral ≤ 1: diagonal con parte real negativa vía `softplus`/`exp`, init estilo HiPPO. Tests de estabilidad en secuencias largas.
2. **Normalización de la atención lineal.** El denominador `1/(φ(q)·Σφ(k))` puede desbordar. Preferir **gating explícito + normalización de salida** (GroupNorm/RMSNorm en el readout) en vez del denominador crudo.
3. **Precisión.** Las recurrencias son **más sensibles** que la atención a baja precisión. **El estado se mantiene en fp32** aunque los matmul corran en bf16. (Documentado: bf16 emulado, §02 2.4.)
4. **Corrección en los bordes de chunk.** La descomposición intra/inter-chunk es la fuente #1 de bugs sutiles. **Validar siempre contra una recurrencia de referencia lenta** (un `*_ref` token-a-token) dentro de tolerancia.

---

## 7. Tamaños objetivo de la familia Fluency

Acorde al no-objetivo de "no frontera desde cero" y a la realidad CPU:

| Modelo | Params | Uso | Cómo se obtiene |
|---|---|---|---|
| `Fluency-30M` | ~30M | smoke tests, CI, WASM demo | entrenamiento corto / destilación |
| `Fluency-100M` | ~100M | edge, embebido ligero | destilación de profesor 1-3B |
| `Fluency-350M` | ~350M | tareas locales generales | destilación + on-policy |
| `Fluency-1B` | ~1B | motor principal para apps | destilación de profesor 7-8B + DPO/ORPO |
| `Fluency-3B`+ | ~3B+ | evolución (requiere GPU-Zig) | Fase 7 |

El "salto de calidad" no viene de más parámetros, sino de **mejor destilación** (MOHAWK + on-policy) de profesores fuertes. Ver [04](04-aprendizaje-destilacion.md).

---

## 8. Compatibilidad Transformer concreta

Para "cargar y correr un Transformer existente":
1. `zload` lee el GGUF/safetensors y su config (capas, dims, vocab, RoPE θ, GQA heads).
2. `zarch` instancia un modelo **todo-atención** (mixers `GQA`/`SWA`) con esos pesos.
3. `ztok` reconstruye el tokenizer exacto.
4. Se valida la salida contra una referencia (p. ej. los logits del mismo modelo en Ollama) — esto **también valida** que podemos usarlo como profesor con nuestro propio motor (ruta 100 % Zig de destilación).

Así, "compatibilidad Transformer" no es teórica: es el camino para que Fluency **aprenda de** Transformers reales.

[→ Siguiente: 04 — Aprendizaje y destilación](04-aprendizaje-destilacion.md)
