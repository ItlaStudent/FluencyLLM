# 10 — Riesgos y decisiones (ADRs)

[← Plan Maestro](../PLAN-MAESTRO.md) · [← 09 Estructura](09-estructura-tooling.md) · [→ 11 Referencias](11-referencias.md)

Riesgos técnicos con mitigación, y el registro de decisiones de arquitectura (ADRs) que fijan las elecciones clave. Cada ADR es vinculante hasta que otro lo supersede (Art. VIII).

---

## 1. Riesgos técnicos y mitigaciones

| # | Riesgo | Impacto | Probabilidad | Mitigación |
|---|--------|---------|--------------|------------|
| R1 | **Sin CUDA/BLAS, entrenamiento lento** (consecuencia de Art. I) | Alto | Cierta | Modelos pequeños + **destilación MOHAWK** (~5B tokens, no billones); kernels SIMD bien hechos; GPU-vía-Zig en F7. Es un trade-off **aceptado**, no un bug. |
| R2 | **bf16 no nativo en Zig** (#3148) | Medio | Cierta | Emulación `ztensor.bf16` (u16+conversión); cómputo en f32; estado SSM en f32. ADR-0003. |
| R3 | **API de Zig inestable entre minors** (0.14→0.15→0.16 rompieron ArrayList, I/O, async, `@cImport`) | Medio | Alta | **Fijar versión** en `build.zig.zon`; encapsular APIs volátiles (`parallelFor` sobre `std.Io`); presupuesto de porting al subir. |
| R4 | **`std.Thread.Pool` eliminado**; API `std.Io` nueva y poco documentada | Medio | Cierta | Helper propio `parallelFor`; ejemplos online están obsoletos → no copiarlos. |
| R5 | **GPU pura Zig inmadura** (~50 % tests Vulkan) | Medio | Alta | No bloquea F0-F6 (todo CPU/WASM); GPU es F7 y opcional, tras la vtable `Backend`. |
| R6 | **Kernel chunkwise scan: difícil y propenso a bugs sutiles** | Alto | Alta | Test obligatorio contra recurrencia `*_ref` token-a-token; empezar por GLA (más simple) antes de Mamba-2; estado en f32. |
| R7 | **Calidad del modelo destilado insuficiente** | Alto | Media | Híbrido (recupera recall); on-policy + DPO; profesores fuertes; eval continua con umbral (E4); subir tamaño si hace falta. |
| R8 | **Tokenizer incompatible → "carga bien, genera basura"** | Medio | Media | Replicar regex/byte-map exactos; test de igualdad de IDs vs referencia (F1). |
| R9 | **Repos Zig de referencia targetean Zig viejo (0.11)** | Bajo | Cierta | Usarlos como *referencia conceptual*, no dependencia; reimplementar para 0.16. |
| R10 | **Madurez de libs Zig de terceros (safetensors-zig, etc.) incierta** | Bajo | Media | Auditar antes de depender; por defecto reimplementar (mantiene control + pureza). |
| R11 | **Licencias de profesores/datasets** (open source Apache 2.0) | Alto (legal) | Media | Producto Apache 2.0 ⇒ sólo profesores con licencia que permita redistribuir pesos derivados bajo Apache 2.0: **preferir Apache-2.0/MIT** (Qwen2.5, Mistral), evitar NC y licencias con restricción de salidas/naming (Llama, Gemma). `docs/PROFESORES.md` obligatorio con campo de licencia (Art. III.3, [04](04-aprendizaje-destilacion.md) §1). |
| R12 | **Un colaborador intenta "optimizar" rompiendo la pureza** | Alto | Media | Gates mecánicos (Art. IX) + CODEOWNERS + prioridad constitucional sobre la tarea. |
| R13 | **Alcance enorme para una persona** | Alto | Alta | Fases con entregables usables; cada fase es valiosa por sí sola; el flujo de trabajo asistido desde la terminal ayuda dentro de barreras (07). |
| R14 | **subq.ai resulta no reproducible / "AI Theranos"** | Bajo | — | El diseño **no** depende de subq.ai (Art. II.4); su enfoque es sólo un mixer experimental opcional (SparseTopK). |
| R15 | **Decodificación restringida (gramática) compleja de implementar** | Medio | Media | Versión simple primero (JSON balanceado); máscara precomputada después; degradar a validación-post si falla. |

---

## 2. Decisiones abiertas (requieren confirmación del autor)

Estas **no** son suposiciones silenciosas: se listan para que el autor decida. Hay un default razonable propuesto.

| # | Decisión | Default propuesto | Alternativas |
|---|----------|-------------------|--------------|
| D1 | Nombre del proyecto/modelo | ✅ **DECIDIDO:** producto/modelo/CLI = **Fluency** (comando `fluency`); codename interno del repo = ZSA | — |
| D2 | Licencia | ✅ **DECIDIDO: Apache License 2.0.** Implica profesores con licencia que permita redistribuir pesos derivados bajo Apache 2.0 → preferir Apache-2.0/MIT (ver [04](04-aprendizaje-destilacion.md) §1) | — |
| D3 | Profesor inicial para F3 | un Qwen2.5 pequeño (multilingüe, licencia permisiva) | Llama-3.2, Mistral, etc. (verificar licencia de destilación) |
| D4 | ¿Ollama como dependencia operativa o sólo opcional? | **Opcional** (ruta pura preferida; Ollama acelera arranque) | requerir Ollama para destilar |
| D5 | Idioma de código/comentarios | Español (autor) o inglés | mixto consistente por archivo |
| D6 | Proveedor de CI | a elección | GitHub Actions, Forgejo/Codeberg CI, local |

---

## 3. Registro de Decisiones de Arquitectura (ADRs)

> Formato resumido. Cada ADR se versiona en `docs/adr/NNNN-titulo.md`. Un ADR no puede anular la Constitución (Art. VIII.2).

### ADR-0001 — Zig exclusivo, versión fijada ≥ 0.16.0
- **Contexto:** el encargo exige 100 % Zig; la API de Zig rompe entre minors.
- **Decisión:** todo el producto es Zig; se fija una versión concreta en `build.zig.zon`; APIs volátiles se encapsulan.
- **Consecuencias:** porting periódico al subir de versión; máxima pureza y control. (Art. I)

### ADR-0002 — Cómputo: CPU-SIMD ahora, GPU-vía-Zig después; nunca CUDA/BLAS
- **Contexto:** rendimiento vs pureza (Art. I.3/I.4).
- **Decisión:** backend CPU `@Vector` por defecto; WASM-SIMD para navegador; GPU sólo vía backends Zig (SPIR-V/PTX/AMDGCN) en F7; **prohibido** CUDA/ROCm/Metal/Triton/XLA/BLAS. Backend tras vtable.
- **Consecuencias:** entrenamiento limitado a modelos pequeños a corto plazo (R1); identidad y portabilidad preservadas. **Aceptado.**

### ADR-0003 — bf16 emulado
- **Contexto:** bf16 no es tipo nativo (#3148).
- **Decisión:** `u16` + conversión (alta de f32); cómputo en f32; estado SSM en f32.
- **Consecuencias:** sin aritmética bf16 por HW; correcto y portable. (Art. I, §02/§03)

### ADR-0004 — Arquitectura híbrida subcuadrático+atención por defecto
- **Contexto:** subcuadrático puro flaquea en recall; producción usa híbridos.
- **Decisión:** ~80-90 % capas subcuadráticas (Mamba-2/GDN/RWKV-7), ~10-20 % atención de ventana deslizante; patrón configurable en `model.zon`; ≥ 75 % subcuadrático verificado por gate (II.3).
- **Consecuencias:** calidad cercana a Transformer con coste subcuadrático; satisface Art. II literalmente.

### ADR-0005 — Aprender por destilación (MOHAWK + KD + on-policy + DPO), no reempaquetar
- **Contexto:** "aprender de otros LLM, 100 % Zig".
- **Decisión:** destilación cruzada de arquitectura como método principal; profesor vía `zollama` o `zload`; pesos resultantes propios; licencias registradas.
- **Consecuencias:** viable en ~5B tokens; cumple Art. I y III.

### ADR-0006 — Gobernanza con enforcement mecánico
- **Contexto:** "lineamientos permanentes e inviolables por cualquier colaborador o automatización".
- **Decisión:** Constitución + gates en Zig (pureza/constitución/calidad/presupuestos) + CODEOWNERS + hash de integridad + enmienda sólo humana.
- **Consecuencias:** las reglas no dependen de la buena fe del agente; violarlas es mecánicamente imposible de mergear. (Art. IX/X)

### ADR-0007 — Un núcleo JSON-RPC 2.0 para MCP/LSP; HTTP para OpenAI/MCP-HTTP
- **Contexto:** MCP y LSP comparten JSON-RPC 2.0; OpenAI API y MCP-HTTP comparten servidor HTTP.
- **Decisión:** `zrpc` único + framings intercambiables; `zserve` HTTP compartido.
- **Consecuencias:** menos código, más superficies de compatibilidad. (Art. VI, [06](06-interop-protocolos.md))

### ADR-0008 — Memoria agéntica híbrida (vector + BM25), sin DB externa
- **Contexto:** footprint mínimo (estilo nullclaw) + pureza.
- **Decisión:** memoria híbrida (embeddings + índice invertido BM25) fusionada por coseno + BM25; tras vtable de proveedores.
- **Consecuencias:** ligereza; sin dependencia vectorial pesada. El store concreto lo fija ADR-0009.

### ADR-0009 — Store de memoria: implementación propia en Zig (✅ DECIDIDO)
- **Contexto:** nullclaw usa SQLite (requiere libc/binding C ⇒ chocaría con Art. I). Hay que elegir store.
- **Decisión:** **store embebido escrito 100 % en Zig** (estructura simple tipo LSM/B-tree + índice invertido + almacén de vectores), **sin SQLite ni C**.
- **Consecuencias:** más trabajo de implementación, pero preserva la pureza (Art. I) y mantiene el binario diminuto. Reimplementar sólo lo mínimo necesario.

### ADR-0010 — CLI `fluency` + TUI interactivo (✅ DECIDIDO)
- **Contexto:** el autor quiere invocar Fluency desde la terminal, con TUI interactivo.
- **Decisión:** ejecutable único `fluency`; modo TUI interactivo por defecto (módulo `ztui`), más `run`/`serve`/`mcp-server`/`distill`/etc. La librería TUI, si se usa externa, debe ser **Zig pura** (evaluar `libvaxis`) y aprobada por ADR; si no, TUI propia. Manejo de terminal cruda aislado en `src/platform/`. Ver [12](12-cli-y-tui.md).
- **Consecuencias:** experiencia interactiva desde la terminal; nueva superficie de UI a mantener; entra en F4.

---

## 4. Métricas de salud del proyecto (revisar cada fase)

- ✅ `zig build gate && zig build test` siempre verde en `main`.
- 📉 Latencia/token y RSS dentro de `budgets.zon`.
- 📈 Calidad de Fluency vs profesor en `fluency eval` (tendencia al alza).
- 🧪 Cobertura: todo módulo público y todo kernel con test.
- 🔒 0 secretos en claro; 0 leaks; 0 archivos no-Zig.

[→ Siguiente: 11 — Referencias](11-referencias.md)
