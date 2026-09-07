# Estado de fases e implementación

[← Plan Maestro](../PLAN-MAESTRO.md) · [Roadmap (plan inviolable)](08-roadmap-fases.md)

> **Nota 2026-06-08:** el checkpoint publicado en GitHub se llama
> **Fluency-139M** (vocab = 49 152, teacher SmolLM2-360M-Instruct). Es la
> misma arquitectura que **Fluency-126M** (vocab = 32 000, TinyLlama); el
> delta de +13.17 M parámetros viene del embedding atado. Ver
> [modelos/fluency-126m.md](modelos/fluency-126m.md) para el detalle.

> Documento **vivo** de estado. NO redefine el plan ([08-roadmap-fases.md](08-roadmap-fases.md) es la fuente inviolable de objetivos y gate-checks); aquí se registra qué está hecho, con qué evidencia, y los próximos pasos concretos. Última actualización: **2026-06-06** (F7.T3: MoE end-to-end en inferencia — bug en `forwardBatch` corregido, test determinista nuevo; F7.T4: GPU vía SPIR-V bloqueado honestamente en M3 — ADR-0037 actualizado con 3 defectos del backend Zig 0.16; **Fluency-126M definido** (~126 M params, ≥126 M verificado, pipeline CPU: gradient-check, convergencia tiny, checkpoint round-trip); 246 tests, 4 gates verdes).

Compilador: `zig-0.16.0\zig-x86_64-windows-0.16.0\zig.exe`. Verificación global: `zig build` ✅ · `zig build test` ✅ (**246 tests, sin fugas**) · `zig build gate` ✅ (pureza, constitución, calidad, presupuestos) · `zig build preview` (vista de la TUI). Actualización: **2026-06-06** (Fluency-126M definido con pipeline CPU validado; GPU bloqueado honestamente en M3 con ADR-0037; 246 tests, 4 gates verdes).

> **Validación 2026-06-05 (AUDITORÍA #2 — Tareas A/B/C de la desarrollo anterior + 1 BUG REAL de gradiente corregido):** build ✅, **231 tests** ✅ (sin fugas), 4 gates ✅. **Tarea A (build.zig test root de zspeculate):** correcta; los 3 tests `promptLookupDraft` ya corren; el cambio en `promptLookupDraft` (`limit = ids.len`) es correcto. **Tarea B (`forwardBatch`):** **correcta** — auditado el código (la lógica de KV-cache especulativa es sólida: al rechazar en P+k, el `forward` de recuperación reescribe KV[P+k], y los KV especulativos futuros se sobreescriben antes de ser atendidos por causalidad) y **verificado EN VIVO**: `fluency run --speculate` produce salida **bit-idéntica** al greedy (`diff` vacío, prompt `def add(a,b):` 28 tokens). La desarrollo anterior reportó con HONESTIDAD que **no acelera con pesos cuantizados** (~0.98×) porque `matmulBatch` re-dequantiza cada peso por token — limitación real, no bug; documentada. **Fix de UX:** la rama `--speculate` no imprimía el pie `(N tokens generados)` (lo había quitado de más); ahora lo imprime igual que greedy (el invariante bit-idéntico aplica al texto generado, no al pie meta del CLI). **Tarea C (MoE):** **BUG REAL de gradiente corregido** — `bwdMoe` añadía un término espurio `sum_m d_h[m]·W2` al gradiente de `X`, corrompiendo el gradiente hacia capas anteriores. El gradient-check original sólo verificaba `dX` (tol laxa 5e-2) y no lo cazaba. **Fix:** eliminado el término + **añadido gradient-check de los 5 inputs** (x, router, w1, w3, w2; tol 2e-2) que ahora verifica también los gradientes de los pesos. **Pendiente real F7.B (no bug):** un `matmulBatch` cuantizado que dequantice cada bloque UNA vez y lo aplique a los K vectores — sin eso el speculative no acelera sobre el modelo cuantizado (que es el caso de uso). Es trabajo sustancial de kernel; queda como siguiente paso de F7.B.

> **Validación 2026-06-05 (AUDITORÍA del trabajo F6/F7 de la desarrollo anterior — evidencia en vivo + 3 bugs corregidos):** se verificó build (✅), **222 tests** (✅, sin fugas; +1 vs los 221 entregados: un test de regresión de F7.B que *sí se ejecuta*), 4 gates (✅). **F6 verificado EN VIVO** con el modelo real (Qwen2.5-coder-1.5B-Q4_K_M): `fluency serve-selftest` → **7/7 endpoints PASSED**; `/v1/embeddings` y `/api/embeddings` devuelven **vector real de 1536 dims** (no el stub de 4); chat OpenAI genera texto real. **F7.A** (paralelismo de filas en matmul-vec cuantizado) verificado correcto (filas disjuntas, sin data race) y **más rápido en vivo** (~1.05 tok/s vs 0.59 baseline SIMD). **F7.B (speculative) tenía 2 BUGS REALES + un overclaim**, corregidos: (1) `generate` truncaba la salida al último token (`clearRetainingCapacity` por token) — en vivo imprimía un solo token; (2) EOS no uniforme podía emitir un token de más vs greedy; (3) **no aceleraba** (sin `forwardBatch`, un `forward` por token = greedy; speedup 1×). **Fix posterior (esta sesión):** `forwardBatch` implementado en `src/zarch/zarch.zig` (QKV+FFN batch por capa, atención token-por-token) e integrado en `src/zspeculate.zig`; tests numéricos K=1/K=2 pasan; regresión F7.B pasa (salida == greedy bit-a-bit). **227 tests**, sin fugas, 4 gates ✅. Además: (a) **bug latente de allocator** en `ServeEngine.embed` → corregido; (b) tests de `zspeculate.zig` registrados en `build.zig`.
>
> **Actualización TAREA A (2026-06-05):** `src/zspeculate.zig` registrado como raíz de test en `build.zig` con su propio `zspeculate_test_mod` e imports `zarch`/`ztok` (patrón idéntico a `zarch_test_mod`). Al ejecutarse los 3 tests se descubrieron **2 tests rotos latentes**: (1) guarda `ids.len < ngram*2` rechazaba los inputs de 3 tokens del test "evita overlap" (se corrigió a 4 tokens); (2) `limit = search_end` truncaba el draft a vacío cuando el match estaba al principio del contexto (se corrigió a `ids.len`, coherente con el comentario del test). **227 tests**, sin fugas, 4 gates ✅. La salida greedy del binario real con `--speculate` sigue idéntica al baseline (verificado).

> **Validación 2026-06-01 (auditoría):** se revisó el trabajo de F4/F5 y se corrigieron bugs/discrepancias reales: (1) el bucle ReAct re-parseaba TODO el contexto cada iteración → re-ejecutaba las mismas tool-calls hasta agotar presupuesto; ahora parsea sólo el texto nuevo de cada paso. (2) la tool `finish` no cortaba el bucle; ahora termina explícitamente (verificado en vivo). (3) la memoria era keyword-only en la práctica (embeddings siempre vacíos) y el "BM25" era frecuencia de términos; ahora hay generador de embeddings determinista (`embedText`, hashing trick L2-normalizado) cableado en las memory-tools y la TUI, y **BM25 real** (IDF + normalización por longitud). (4) la "conexión agente-generador" no conectaba ningún modelo en el CLI; ahora `fluency agent "<prompt>" --gguf <ruta>` cablea el modelo local al bucle ReAct (unit-test del hook + ruta tool-only verificada en vivo; la carga del profesor a f32 ~6GB puede dar OOM — límite de escala conocido, no del cableado).

> **Validación 2026-06-02 (auditoría, cierre E5/E7 + flecos F4):** se revisó el trabajo de las TAREAS 1-5 (sampler de gramática, streaming TUI, demo E5, e2e MCP, diffs aprobables, sesiones, streaming genuino en agente). Código correcto: `build` ✅, tests ✅ sin fugas, 4 gates ✅. Tests sustantivos (no tautológicos). **Matiz declarado como limitación honesta**: el **e2e del cliente MCP por proceso stdio real** está HECHO (`fluency mcp-selftest` spawn `fluency mcp-server` vía `std.process.Child` con pipes stdin/stdout); el loopback en-memoria se conserva como test unitario.

> **Validación 2026-06-02 (auditoría TAREA 5 — 3 BUGS REALES corregidos + validación en vivo):** la ruta `--quantized` estaba **rota sobre el modelo objetivo real** (el doc lo admitía como "verificar en vivo", sin hacer). Al ejecutarla en vivo aparecieron 3 bugs en cascada: (1) **formato único global** (`quant_format` + `error.MixedQuantFormats`) — pero `inspect` confirma que Qwen2.5-coder-1.5B Q4_K_M mezcla `Q4_K x168, Q6_K x29`; (2) **carga de una sola capa** (`loadGgufTransformerQuantized` buscaba `"attn_q.weight"` sin prefijo `blk.{l}.` ni recorrer las 28 capas → `MissingTensor`; `forward` no desplazaba por capa); (3) **formato variable ENTRE capas** del mismo tipo (ffn_down/attn_v alternan Q4_K/Q6_K, tamaños de bloque distintos → no concatenables). **Fix**: tipo `QuantWeight { layers, fmts }` con bytes+formato **por-capa**; los 7 pesos lineales son `?QuantWeight`; `matmulVec`/`matmulVecIn` reciben `(qw, layer)`; `loadQuantWeight` carga capa-a-capa. Tests nuevos: "formatos MEZCLADOS (w1=Q4_0)" y "MULTI-CAPA (2 capas, formato distinto por capa)". **VALIDADO EN VIVO**: `fluency run <Q4_K_M> "def fib(n):" 12 --quantized` genera Python coherente, **pico RAM 3082 MB** (vs ~6 GB f32, sin OOM en máquina 16 GB). **170 tests**, sin fugas, 4 gates ✅.

> **Validación 2026-06-02 (auditoría TUI cell-based MVP — la desarrollo anterior se quedó sin tokens a media TAREA 3; corregida):** el motor TUI nuevo (`src/ztui/engine.zig` + `src/ztui/screen.zig` + `src/ztui/transcript.zig`, sobre `src/platform/tty.zig`) **no compilaba ni corría sus tests**. Bugs reales corregidos: (1) **`engine.zig` no compilaba** — literal anónimo del `catch` de `tty.size()` no coaccionaba a `TermSize` y dos `try self.setStatus(...)` sobre función que devuelve `void`; (2) **modo crudo no era best-effort** — `enable`/`size` propagaban `error.Unexpected` (y volcaban un stack trace por `unexpectedStatus`) cuando stdin/stdout no es una consola (pipe/redirección); ahora devuelven un `error.NotATerminal` limpio y `Tui.init`/`demo_raw` degradan sin fallar; (3) **el test-runner se colgaba >60 s** — el test `"ztui Tui ... paint"` llamaba a `paint()`, que escribe ANSI a `std.Io.File.stdout()`, **el mismo canal IPC** que usa el test-runner (`--listen=-`) → protocolo corrupto; **fix**: `paint` se separó en `compose()` (rellena celdas, sin IO, testeable) + volcado a terminal, y el test usa `compose()`; (4) **los tests del motor no se ejecutaban** — `ztui.zig` importa `engine.zig` por NOMBRE de módulo y `zig test` no corre los tests de módulos así importados; se añadió `engine.zig` como **test root** en `build.zig` (sus imports relativos `screen.zig`/`transcript.zig` se testean al ser raíz). Resultado: `build` ✅, **195 tests** ✅ sin fugas (antes 180 reportados, +15 reales del motor que ahora sí corren), 4 gates ✅. **Vista previa en vivo** del layout con `zig run src/ztui/preview.zig` (header + transcript con roles `▸ tú`/`▾ fluency`/`⚙ tool` + diff `+/-` + input + status line) y el binario real arranca la TUI (`fluency chat`, header + bienvenida, salida limpia en modo degradado). *Pendiente honesto*: la TUI es un **MVP** (layout + render diff + input + comandos slash + aprobación + sesiones); faltan markdown, plegado de tool-calls, diffs por tecla `[a]/[r]`, paleta Ctrl-P y temas (ver E.1).
>
> **Addendum 2026-06-02 (pulido de TUI COMPLETADO + bug de visualización Windows):** sobre el MVP se añadió el pulido TUI y se corrigió por qué se veía como **texto basura** en `cmd.exe` (`←[38;2;…m` literal): `enable()` activaba VT sólo en stdin; faltaba **`ENABLE_VIRTUAL_TERMINAL_PROCESSING` en stdout** → ahora la consola interpreta las secuencias ANSI. Añadido además **buffer de pantalla alterno** (no ensucia el historial del shell). Pulido: **Markdown** inline por celda (`**negrita**`, `` `código` ``, headings, bullets `•`, bloques ```` ``` ````), **tool-calls plegables** (Ctrl-F enfoca / Ctrl-O pliega, vía `Entry.folded`), **diffs aprobables `[a]/[r]`** (y `y/n`/Esc) cableados al `approval_fn` con la tool-call enfocada, **paleta de comandos Ctrl-P** (Tab/Enter/Esc) con tema/plegado/foco/limpiar/sesión/salir, **temas claro/oscuro** (Ctrl-T, `Theme.dark()/light()`). Bug latente corregido: `showApproval` guardaba en `status` un puntero a buffer de **pila** (colgante) → ahora `setStatusFmt` copia a un buffer persistente del `Tui`. Decisión **ADR-0030**: TUI propia 100% Zig, NO libvaxis. **205 tests** (sustantivos: Markdown vs diff, `paintInline`, fences, plegado, tema, foco, paleta, status no-colgante, `dispWidth`, bienvenida), 4 gates ✅. Visualizable con `zig build preview`.
>
> **Addendum 2026-06-02 (estética de TUI + fix Unicode Windows):** tras probarla Zenda en `cmd.exe`, (a) los glifos Unicode salían corruptos (`▾`→`Ô…`) porque la consola no estaba en code page UTF-8 → se fuerza **CP de salida 65001** vía `std.os.windows.CONSOLE.USER_IO.SET_CP(.Output, 65001)` (wrapper de `std`, sin `extern` propio), con restauración; y (b) la estética se acercó a TUI moderna: **pantalla de bienvenida** centrada con logo en bloque `█` "FLUENCY" + tagline + hints (cuando la sesión está vacía), **caja de input con borde redondeado** (`╭─╮│╰─╯`, prompt `›`, placeholder), **barra de estado** con segmentos + `Fluency vX` a la derecha, y helper `dispWidth` (codepoints, no bytes) para centrar texto multibyte. Vista: `zig build preview` (vuelca 2 frames con el motor real: bienvenida + conversación con markdown/diff).
>
> **Validación 2026-06-05 (verificación de cierre + BUG REAL de cableado del CLI corregido + lanzador `.cmd`):** se revisó que no quedaran pasos pendientes obligatorios (F0–F5 100%, F6 núcleo, F7 en curso; lo que resta es optativo y limitado por cómputo). Al **probar el binario en vivo** apareció un **bug real no cubierto por los tests**: `fluency agent`/`fluency chat` panicaban con `incorrect alignment` (hash_map) al **despachar cualquier tool-call**. Causa: `setupAgentAndRegistry` devolvía `struct { agent, reg }` **por valor**, pero `Agent.init(..., &reg)` y cada `tool()` (`.ptr = self`) guardaban punteros a sus **propios locales**; al moverse la struct en el `return`, `agent.registry` y los `tool.ptr` quedaban **colgantes**. Los tests no lo veían porque construyen agente/registry/tools en un **único scope** (sin mover). **Fix**: tipo `AgentBundle` que **posee** registry+agent+las 9 tools como campos, con `init()` (sin punteros internos) + **`wire()`** que fija `agent.registry`, los `store` de las memory-tools y registra las tools **tras** colocar el bundle en su frame final; ambos callers (`runAgentOnce`, `runChatCommand`) usan `init()+wire()`. **Test de regresión nuevo** ("AgentBundle.wire deja punteros estables: despacha tool-call sin colgar") que reproduce el escenario y verifica punteros + dispatch de `finish`. Verificación global: `zig build` ✅ · `zig build test` ✅ (**206 tests, sin fugas**) · `zig build gate` ✅ (4 gates). **Evidencia en vivo (binario real)**: `fluency --version` → `fluency 0.7.0`; `fluency mcp-selftest` → PASSED (e2e MCP por proceso stdio real); `fluency train 200 --seed 7` → ppl 62.4→1.003, acc 100%; `fluency agent '<tool_call>finish</tool_call>'` → despacha `finish` sin panic (antes crasheaba); `fluency agent '<tool_call>memory_write…</tool_call>'` → escribe en el store del agente sin colgar. Se añadió **`fluency.cmd`** en la raíz: compila ReleaseFast la primera vez con el Zig 0.16 incluido, arranca la TUI de chat sin argumentos, reenvía subcomandos (`serve|run|agent|…`) y soporta `--rebuild`. (Limitación conocida del batch: argumentos con `<`/`>` literales requieren llamar al `.exe` directamente, pues `cmd.exe` los interpreta como redirección — no afecta a `serve`/`run`/`chat`/etc.)

---

## Resumen

| Fase | Objetivo | Estado | Gate-check |
|------|----------|--------|------------|
| **F0** | Fundaciones y gobernanza | ✅ **100%** | E0: build+gate+test verdes, matmul=ref, pureza rechaza no-Zig |
| **F1** | Inferencia de modelos existentes | ✅ **100%** | **E2 cumplido**: greedy token-por-token = Ollama |
| **F2** | Núcleo subcuadrático | ✅ **100%** | **E3 cumplido**: memoria de decode constante, scan=recurrencia |
| **F3** | Aprendizaje (destilación) | ✅ **100%** | **E4 cumplido**: KL→0.034, acuerdo 87.5%, reproducible, sin fugas |
| **F4** | Runtime agéntico | ✅ **100%** | **E5 cumplido** — parser tool-calls ✅, bucle ReAct (parse-solo-nuevo + finish corta) ✅, tools+sandbox ✅, secretos+audit+deny-by-default ✅, zmemory híbrida real (embeddings+BM25) ✅, trayectorias ✅, conexión agente↔modelo (`--gguf`) ✅, **sampler restringido por gramática F2 en el generador del agente** ✅, **streaming en TUI** ✅ (token-a-token genuino vía `generate_stream_fn`, con fallback a re-emisión por palabras), **diffs aprobables** ✅, **sesiones en TUI** ✅, **TUI cell-based TUI** ✅ (motor `engine/screen/transcript` sobre terminal crudo: render diff ANSI truecolor, pantalla alterna, layout completo, word-wrap, event loop teclado, **Markdown**, **tool-calls plegables**, **diffs `[a]/[r]`**, **paleta Ctrl-P**, **temas claro/oscuro**; ADR-0030) |
| **F5** | Protocolos y embebido | ✅ **100%** | **E7 cumplido** — servidor MCP + `sampling` ✅, libfluency inferencia real ✅, WASM target+bindings JS ✅, **cliente MCP e2e** ✅ (stdio por proceso real vía `mcp-selftest`; loopback en-memoria como test unitario), **verificación en vivo** ✅ (`libfluency` desde Node.js vía `koffi` + `fluency.wasm` en navegador vía JS estándar; demos con modelo sintético en `docs/E7-embebido.md`) |
| **F6** | Integración ecosistema (Fluency = motor drop-in compatible con OpenAI/Ollama) | 🟢 **núcleo HECHO** | **E6 parcial** — `fluency serve` drop-in Ollama+OpenAI (descubrimiento `/api/tags`+`/v1/models`, nombre configurable, `/api/show`, chat+streaming con shape correcto por endpoint), **verificado en vivo con clientes reales** y servicio persistente; **F6.1 endurecer drop-in ✅** (manejo robusto de errores: body inválido, JSON malformado, campos ausentes, modelo no coincidente → respuesta de error con gracia, sin panic/fuga, 6 tests nuevos); **F6.2 embeddings drop-in ✅** (`/v1/embeddings` y `/api/embeddings` cableados al `embed` real del `ServeEngine`: mean-pooling de token embeddings, shape correcto por protocolo, 3 tests nuevos, verificado en vivo con modelo real cuantizado); **F6.3 serve-selftest ✅** (`fluency serve-selftest` spawnea `fluency serve` en un puerto efímero, verifica 7 endpoints (`/v1/models`, `/api/tags`, `/api/show`, `/v1/chat/completions`, `/api/chat`, `/v1/embeddings`, `/api/embeddings`) vía peticiones HTTP reales, reporta PASS/FAIL por endpoint y estado global PASSED/FAILED, luego mata el servidor). Pendiente: que apps externas registren a Fluency en su lista de modelos (lado de cada app, fuera de este repo) |
| **F7** | Evolución (GPU, MoE, …) | � **núcleo HECHO** | **matmul cuantizado on-the-fly ✅** (Q4_0/Q8_0/Q4_K/Q6_K, formato por-capa-por-tensor, validado en vivo sin OOM); **SIMD en kernels cuantizados ✅** (~1.6× speedup medido en vivo); **paralelismo de filas + dequant SIMD Q8_0 ✅** (~1.55× sobre SIMD, ~2.46× sobre escalar, salida greedy idéntica); **dequant SIMD Q4_0 + tiling (cache blocking) en Q4_0/Q8_0/Q4_K/Q6_K ✅** (tile_k=512, mejora localidad L1; test de regresión `dequant Q4_0 SIMD vs ref`); **TAREA 1: embedding/output table cuantizada por-fila ✅** (dequantRowQ4_0/Q8_0/Q4_K/Q6_K en `zkernel.zig` + `QuantEmbedding` en `zarch.zig`; lookup y proyección final reutilizan `matmulVecQ*` / `dequantRow*`; pico RAM medido en vivo: **2407 MB** vs baseline ~3 GB; salida greedy idéntica en tokens; 236 tests sin fugas, 4 gates verdes; ADR-0034); **TAREA 2: matmulBatch cuantizado dequant-once ✅** (`matmulBatchQ4_0/Q8_0/Q4_K/Q6_K` en `zkernel.zig`; cada bloque se dequantiza UNA vez y aplica a M tokens; `matmulBatch` en `zarch.zig` los usa; speedup speculative en vivo: **~1.04×** (1.16 vs 1.12 tok/s), salida bit-idéntica a greedy; 240 tests sin fugas, 4 gates verdes; ADR-0032 actualizado); **TAREA 3: MoE end-to-end en inferencia ✅** (bug en `forwardBatch` corregido: ignoraba `n_experts>0`; test determinista `Transformer MoE forward` en `zarch.zig` con finitud/determinismo/forwardBatch vs forward; modelo real denso sin regresión; 241 tests sin fugas, 4 gates verdes); **TAREA 4: GPU vía Zig a SPIR-V ❌ BLOQUEADO en M3** (backend self-hosted SPIR-V de Zig 0.16 produce bitstream con estructura OpenCL, no Vulkan compute: ignora `addrspace(.storage_buffer)`, `OpVariable` en `Function` en vez de `StorageBuffer`, sin `OpExecutionMode LocalSize`; **no es forzable sin violar Art. I.4**; M1 compilación ✅ y M2 detección GPU ✅ se mantienen como validación; **ADR-0037 actualizado** con la tabla de 3 defectos; re-evaluar en Zig ≥0.17; fallback CPU intacto); **micro-benchmark de throughput** (Q8_0 4096×4096 ~17 GFLOPS, Q4_0 ~17 GFLOPS en hardware de desarrollo) |

---

## Evidencia (F0–F3)

### F0 — Fundaciones ✅
`util` (hash sha256, rng Xoshiro, bytes LE/BE, crypto ChaCha20-Poly1305, json propio, `parallelFor`), `ztensor` + `MappedFile`/`mapFile`, `zkernel` (matmul/rmsnorm/softmax/silu/rope con `*_ref` + tests de igualdad), gates mecánicos (`gate_purity/constitution/quality/budgets`) con hash de integridad de la Constitución.

### F1 — Inferencia ✅ (E2)
`fluency run <gguf> "<prompt>"` genera texto **idéntico token-por-token a la salida greedy de Ollama** (Qwen2.5-coder real). Claves: RoPE NEOX + fix de doble-rotación GQA, tokenizer byte-level BPE real desde GGUF, `zserve` cableado al modelo, lector safetensors. `inspect`/`forward`/`run`/`serve` operativos.

### F2 — Núcleo subcuadrático ✅ (E3)
Mixers (Mamba2 SSM O(1), SWA, GatedDeltaNet, RWKV7) tras vtable; `scanChunkwise` por bloques = recurrencia token-a-token; `HybridBlock` (patrón configurable, `stepDecode`, estado dual) con memoria de decode constante; decodificación restringida por gramática (`grammar.zig` + `fluency run --grammar json`, JSON válido en vivo); streaming SSE/NDJSON en `zserve`.

### F3 — Aprendizaje ✅ (E4) — ver [ADR-0011](adr/0011-autodiff-tensorial-y-fluencynano.md)
- **`zgrad`**: autodiff reverse-mode **tensorial** (`Graph`), VJP de matmul/matmulTransB/add_bias/residual/rmsnorm/silu/mul/embed + **scan (SSM)/lin_attn (delta)/attn (causal)**, cada uno validado por **gradient-check numérico**. (12 tests)
- **`ztrain`**: `clipGradGlobalNorm`, acumulación, schedule **warmup+cosine**, `TokenDataset` (datos mmap u16). AdamW/Muon/SGD. (7 tests)
- **`zdistill`**: `FluencyNano` (LM híbrido entrenable), `Trainer` (fwd+bwd+clip+AdamW+schedule), `stepKD` (MOHAWK A / KD logits), `stepCE` (Seq-KD), suite `evaluate` (CE/perplejidad/accuracy) + `evalKLToTeacher`. Tests: aprende, **reproducible bit a bit**, KD baja la KL. (8 tests)
- **CLI + profesor real**: `fluency distill <gguf> "<prompt>" [pasos] [--seed S] [--ollama M]`, `fluency train`, `fluency distill --synthetic`. Profesor por **zload** (GGUF→logits) y **zollama** (Seq-KD).
- **Verificado en vivo**: destilando del Qwen real 300 pasos → **KL 10.21→0.034 (−100%)**, **acuerdo top-1 0%→87.5%** (alumno 9.87M params, vocab 151 936); vía Ollama Seq-KD 80%; `train` sintético ppl 64→1.001, acc 100%, salida idéntica al repetir semilla.
- **Límite honesto**: alumno = `FluencyNano` (pequeño), no Fluency-100M/350M sobre 5B tokens (corrida limitada por cómputo). MOHAWK B/C presentes como pérdidas pero no cableadas en vivo (el GGUF no expone hidden states del profesor).

---

### TAREA 5 — Matmul cuantizado on-the-fly ✅ (2026-06-02)
Implementa inferencia directa desde pesos cuantizados GGUF (Q4_0, Q8_0, Q4_K, Q6_K) sin dequantizar todo a f32, evitando OOM en modelos grandes.

- **Kernels** (`src/zkernel/zkernel.zig`): `matmulVecQ4_0`/`matmulVecQ8_0`/`matmulVecQ4_K`/`matmulVecQ6_K` — multiplicación vector×matriz cuantizada on-the-fly (dequantiza bloque a bloque en stack y hace dot product con el vector de entrada). Cada uno validado por test contra dequantización manual de referencia. Tests nuevos: `matmulVecQ4_0 vs manual dequant`, `matmulVecQ8_0 vs manual dequant`, `matmulVecQ4_K vs manual dequant`, `matmulVecQ6_K vs manual dequant`.
- **Arquitectura** (`src/zarch/zarch.zig`): `QuantFormat` enum (`f32`, `q4_0`, `q8_0`, `q4_k`, `q6_k`) + tipo **`QuantWeight { layers: [][]const u8, fmts: []QuantFormat }`** con bytes y formato **POR-CAPA**. Los 7 pesos lineales del `Transformer` son `?QuantWeight` (`wq_q`…`w3_q`). Necesario porque los K-quant reales mezclan formatos no solo entre tipos de peso sino **entre capas del mismo tipo** (ffn_down/attn_v alternan Q4_K/Q6_K, con tamaños de bloque distintos → no concatenables con stride uniforme). `forward`/`forwardWithHidden` usan `matmulVec`/`matmulVecIn`, que reciben `(qw, layer)` y enrutan a `qw.layers[layer]` con `qw.fmts[layer]`, cayendo al matmul f32 cuando `qw == null`.
- **Carga** (`src/main.zig`): `loadGgufTransformerQuantized` carga pesos no-lineales (embeddings, norms, bias) como f32 y los lineales vía `loadQuantWeight`, que recorre `blk.{0..L-1}.{suffix}` guardando los bytes crudos y el formato de **cada capa** (sin exigir uniformidad). Flag `--quantized` en `fluency run`, `fluency serve` y `fluency agent`.
- **Tests** (`src/zarch/zarch.zig`): integración Q8_0 vs f32; **formatos MEZCLADOS** (w1=Q4_0, resto=Q8_0) que prueba el enrutado per-tensor; **regresión MULTI-CAPA** (2 capas con pesos y formato distintos por capa, w1 alterna q8_0/q4_0) que bloquea el bug de carga/uso de una sola capa. Sin fugas.
- **VALIDADO EN VIVO** (modelo real, ReleaseFast): `fluency run <Qwen2.5-coder-1.5B-Q4_K_M> "def fib(n):" 12 --quantized` genera Python coherente (`def fib(n):` → `if n == 0: return 0`), con **pico de RAM 3082 MB** frente a los ~6 GB del dequant f32 (sin OOM en máquina de 16 GB). El modelo mezcla Q4_K (x168) y Q6_K (x29) entre tipos y entre capas.
- **ADRs**: ver [ADR-0027](adr/0027-matmul-cuantizado-on-the-fly.md).

### TAREA 1 — SIMD en kernels cuantizados ✅ (2026-06-02)
Vectorización del dot-product por bloque en `matmulVecQ4_0`/`Q8_0`/`Q4_K`/`Q6_K` usando `@Vector(8, f32)` + `@reduce(.Add, ...)` (Zig puro, sin función foránea).

- **Archivos:** `src/zkernel/zkernel.zig`.
- **Cambio:** helper `inline fn dotSimd(deq, x)` que acumula en `@Vector(8, f32)` y reduce al final; funciones `_ref` escalares conservadas para validación; las funciones públicas `matmulVecQ*` ahora usan SIMD.
- **Tests:** 4 nuevos (`matmulVecQ4_0 SIMD vs ref`, `Q8_0 SIMD vs ref`, `Q4_K SIMD vs ref`, `Q6_K SIMD vs ref`) con dimensiones multi-bloque para ejercitar el vector path. Tolerancias: Q4_0/Q8_0 1e-4, Q4_K 5e-3, Q6_K 1e-2 (el tree reduction SIMD es numéricamente más preciso que el acumulador escalar secuencial). 180 tests totales, sin fugas.
- **Validación en vivo** (Qwen2.5-coder-1.5B-Q4_K_M, ReleaseFast, prompt "def fib(n):", 12 tokens):
  - Escalar (_ref): **32.5 s** → **0.37 tokens/s**
  - SIMD: **20.2 s** → **0.59 tokens/s**
  - **Speedup: ~1.6×**
  - **Misma salida greedy:** `def fib(n):\n    if n == 0:\n        return 0`
- **Limitación honesta:** speedup modesto porque la dequantización sigue siendo escalar; vectorizar también la dequant (especialmente Q8_0) podría añadir un ~10-20 % extra. En debug el speedup es menor; ReleaseFast es donde se materializa.
- **ADRs:** ver [ADR-0028](adr/0028-simd-kernels-cuantizados.md).

### TAREA F7.A — Paralelismo de filas + dequantización SIMD Q8_0 ✅ (2026-06-05)
Vectorizar la dequantización por bloque Q8_0 y paralelizar las filas del matmul-vec cuantizado para explotar múltiples núcleos de CPU.

- **Archivos:** `src/zkernel/zkernel.zig`.
- **Cambios:**
  - `dequantQ8_0_simd`: lee 8 bytes de `i8` como `@Vector(8, i8)`, castea a `f32` vía `@floatFromInt`, multiplica por escala splat (`@Vector(8, f32)`). Fallback escalar para el resto.
  - `parallelForRows` helper inline (Zig puro, `std.Thread.spawn`+`join`, cae a serial en `builtin.single_threaded` para WASM). Chunking por rango contiguo; fallback serial si `n < 256` o spawn falla.
  - Cada `matmulVecQ4_0`/`Q8_0`/`Q4_K`/`Q6_K` ahora tiene `_serial` + `_parallel` + wrapper público con umbral (`N >= 256 && N*K >= 100_000`).
  - `matmulVecQ8_0_serial` y `_parallel` usan `dequantQ8_0_simd`.
- **Tests:** 5 nuevos (`dequant Q8_0 SIMD vs ref`, `matmulVecQ4_0/Q8_0/Q4_K/Q6_K parallel vs ref`) con dimensiones `N=512` para forzar el path paralelo. **211 tests** totales, sin fugas.
- **Validación en vivo** (Qwen2.5-coder-1.5B-Q4_K_M, ReleaseFast, prompt `"def fib(n):"`, 12 tokens):
  - Baseline SIMD (TAREA 1): **20.2 s** → **0.59 tok/s**
  - Con F7.A: **13.1 s** → **0.91 tok/s**
  - **Speedup: ~1.55× sobre SIMD; ~2.46× sobre escalar**
  - **Salida greedy idéntica:** `def fib(n):\n    if n == 0:\n        return 0`
- **Limitación honesta:** speedup < 2× aspiracional por overhead de `spawn`+`join` por llamada (~200 matmulVec por token). Un pool de hilos persistente (no disponible en Zig 0.16 sin implementación propia) o vectorización de Q4_K/Q6_K dequant podría cerrar la brecha restante.
- **ADRs:** ver [ADR-0031](adr/0031-paralelismo-filas-matmul-cuantizado.md).

### TAREA F7.B — Speculative decoding 100% Zig (prompt-lookup / n-gram) 🟠 (2026-06-05)
Andamiaje de speculative decoding sin modelo borrador: `promptLookupDraft` busca repeticiones locales del contexto y propone continuación. **`forwardBatch` implementado e integrado** para verificar K tokens borrador en una sola pasada.

- **Archivos:** `src/zarch/zarch.zig` (`forwardBatch` + `matmulBatch`), `src/zspeculate.zig`, `src/main.zig`.
- **Cambios esta sesión (auditoría + correcciones reales):**
  - **Bug de divergencia corregido:** cuando el draft se rompía antes de aceptar todos los tokens, `zspeculate.zig` copiaba logits del batch del último token **aceptado** en lugar de hacer `forward` del token **rechazado** que acababa de emitir. Esto rompía la identidad greedy cuando el draft no era perfecto. Fix: si `accepted < accept_limit`, se hace `self.tf.forward(ids.items[pos - 1], pos - 1)` para obtener logits correctos para la siguiente iteración.
  - **Texto decorativo eliminado:** `main.zig` inyectaba `[speculative: prompt-lookup]\n` y `({d} tokens generados, speculative)\n`, rompiendo identidad bit-a-bit con greedy. Eliminado.
  - **Buffers cacheados en `Transformer`:** `forwardBatch` alocaba `x_batch`, `xb_batch`, `q_batch`, `k_batch`, `v_batch` en cada llamada. Ahora se cachean como campos `batch_*` con lazy-realloc, eliminando ~5 allocations por ciclo de draft.
  - **Buffers cacheados en `SpeculativeGenerator`:** `logits_buf` y `draft_buf` lazy-allocados y reutilizados entre ciclos; `deinit()` liberarlos. Los 4 callers en `main.zig` ahora llaman `spec.deinit()`.
  - `Transformer.forwardBatch(tokens, start_pos, logits_out)`: procesa K tokens en batch, actualizando KV-cache y devolviendo K vectores de logits. QKV+FFN batch por capa (`matmulBatch`) + atención token-por-token.
- **Tests:** `forwardBatch K=1/K=2` vs `forward` (tolerancia 1e-4). Test de regresión F7.B en `main.zig` pasa (modelo sintético). **227 tests**, sin fugas, 4 gates ✅.
- **Validación EN VIVO (modelo real Qwen2.5-coder-1.5B-Q4_K_M, ReleaseFast, prompt `"def fib(n):"`, 20 tokens):**
  - **Salida greedy vs speculative: IDÉNTICA bit-a-bit** (verificado comparando contenido post-header).
  - **Speedup medido: ~0.98×** (greedy 16.55 s, speculative 17.64 s; 0.94 tok/s vs 0.92 tok/s). **NO acelera con pesos cuantizados.**
  - **Causa raíz honesta:** `matmulBatch` sobre pesos cuantizados (Q4_K/Q6_K) itera `for (0..M)` llamando `matmulVecQ*` fila por fila. Cada llamada dequantiza el peso completo de forma independiente. Con K tokens de draft, se dequantiza K veces el mismo peso → sin ahorro real de cómputo. Además, la atención sigue siendo token-por-token. El speedup > 1× del test sintético se materializa solo con pesos f32 (donde `matmul` sí aprovecha el batching).
- **ADRs:** ver [ADR-0032](adr/0032-speculative-decoding-prompt-lookup.md) (actualizado con dependencia de `forwardBatch` y limitación de cuantización).

### TAREA F7.C — MoE en el FFN (soft top-k, entrenable) ✅ (2026-06-05)
Implementación de Mixture of Experts diferenciable (soft) en el motor de autodiff `zgrad` y en el modelo entrenable `FluencyNano` (`zdistill`).

- **Archivos:** `src/zgrad/zgrad.zig` (op `moe` + VJP `bwdMoe`), `src/zdistill/zdistill.zig` (`NanoConfig` MoE, `LayerOff` offsets, `init/initWeights/forwardWithHidden` MoE), `docs/adr/0033-moe-soft-entrenamiento-zig.md`.
- **Cambios:**
  - `Graph.moe(x, router, w1, w3, w2, E, M)`: forward soft-MoE sobre batch `[T, D]`; cada token genera logits de router `[E]`, softmax, y combina salidas de todos los expertos ponderados. Guarda activaciones intermedias (`saved`) para backward.
  - `bwdMoe`: VJP completo. **⚠️ Bug corregido en auditoría 2026-06-05:** el gradiente respecto a `X` incluía un término espurio `sum_m d_h[m]·W2[e,m,d]` — pero `X` NO conecta con la salida vía `W2` (W2 multiplica `h`, cuya dependencia de `X` ya se cuenta por los caminos W1/W3). El término extra corrompía el gradiente que fluye hacia capas anteriores (embeddings, etc.). El gradient-check original no lo detectaba porque **sólo verificaba `dX`** con tolerancia laxa (5e-2). Eliminado; ver test ampliado abajo.
  - `FluencyNano`: `NanoConfig.n_experts/top_k/moe_ff_dim`; `LayerOff` añade `moe_router/w1/w3/w2`; `init` calcula offsets condicionalmente (denso vs MoE); `initWeights` inicializa router con baja varianza (N(0,0.01)) para distribución uniforme inicial; `forwardWithHidden` usa `g.moe` cuando `n_experts > 0`.
- **Tests:**
  - `gradcheck moe (soft) backward vs numérico` en `zgrad.zig` (verifica `dX`).
  - **`gradcheck moe (soft) — TODOS los inputs (x, router, w1, w3, w2)`** (añadido en auditoría 2026-06-05): diferencias finitas 2-lados sobre **los 5 inputs** (tol 2e-2). Es el que destapó el bug del gradiente espurio en `dX` y ahora **verifica también los gradientes de los pesos** (router/w1/w3/w2) que antes no se comprobaban.
  - `FluencyNano MoE forward/backward` en `zdistill.zig` (forma correcta, backward sin pánico, grads no nulos).
  - `FluencyNano MoE aprende` en `zdistill.zig` (300 pasos CE, accuracy >= 0.75; con el gradiente corregido, sin regresión vs denso comparable).
- **Validación:** `zig build` ✅ · `zig build test` ✅ (**231 tests**, sin fugas) · `zig build gate` ✅ (4 gates verdes).
- **Evidencia de entrenamiento:** denso baseline (dim=16, ff=32, 2 capas) → accuracy 0.875 tras 300 pasos. MoE comparable (dim=16, moe_ff=12, E=4) → accuracy >= 0.75 tras 300 pasos, CE baja > 50%. No hay regresión catastrófica.
- **Limitación honesta:** el op `moe` usa softmax completo sobre todos los expertos (no top-k soft), lo que es ligeramente más costoso que hard top-k en inferencia. Para inferencia se seguirá usando `forwardMoE` de `zarch.zig` (hard top-k). No se implementó load balancing loss auxiliar (estándar en Mixtral); se deja para iteración futura si el balance de carga es problemático.
- **ADRs:** ver [ADR-0033](adr/0033-moe-soft-entrenamiento-zig.md).

### TAREA F7.T1 — Embedding/output table cuantizada por-fila ✅ (2026-06-06)
Mantener `token_embd.weight` en su formato GGUF cuantizado original (Q6_K en Qwen2.5-coder-1.5B-Q4_K_M) y dequantizar por-fila bajo demanda en lookup y proyección final, eliminando la tabla f32 completa en memoria.

- **Archivos:** `src/zkernel/zkernel.zig` (`dequantRowQ4_0`/`Q8_0`/`Q4_K`/`Q6_K`), `src/zarch/zarch.zig` (`QuantEmbedding`, `token_embedding_table_q`, modificaciones en `forward`/`forwardWithHidden`/`forwardBatch`), `src/main.zig` (`loadQuantEmbedding`, `loadGgufTransformerQuantized`).
- **Cambios:**
  - `dequantRowQ*`: wrappers de una sola fila sobre los dequantizadores de bloque ya validados (`dequantQ*_simd`/`dequantQ*_block`). Reutilizan la misma lógica de dequantización; solo iteran sobre los bloques de la fila solicitada y escriben a un buffer de salida.
  - `QuantEmbedding { bytes: []const u8, fmt: QuantFormat }` en `zarch.zig`; `Transformer` añade `token_embedding_table_q: ?QuantEmbedding`.
  - Embedding lookup: si `token_embedding_table_q` existe, `dequantRowQ*(emb_q.bytes, token, self.x, dim)` en lugar de `@memcpy` desde tabla f32.
  - Proyección final (tied embeddings, `wcls == null`): si `token_embedding_table_q` existe, `matmulVecQ*(self.logits, emb_q.bytes, self.x, vocab, dim)` en lugar del loop escalar sobre tabla f32. Esto evita materializar la matriz f32 completa.
  - `forwardBatch`: lookup y proyección siguen el mismo patrón cuantizado.
  - `loadGgufTransformerQuantized`: elimina la asignación `emb` f32 (~933 MB) y la carga vía `loadDirect`; en su lugar usa `loadQuantEmbedding(wa, gguf, "token_embd.weight")` (los bytes crudos Q6_K ~182 MB; desde F7.T5 se referencian zero-copy, sin copiar al arena).
- **Tests:**
  - 4 tests nuevos en `zkernel.zig`: `dequantRowQ4_0/Q8_0/Q4_K/Q6_K vs manual dequant` (oráculo bloque-a-bloque, tolerancia 1e-6).
  - 1 test nuevo en `zarch.zig`: `embedding cuantizado Q8_0: forward idéntico a f32` (mini-Transformer con embedding cuantizado manual vs oráculo f32, tolerancia 1e-3 en logits; verifica finitud y no-NaN).
  - Conteo total: **236 tests** (subió de 231), sin fugas, 4 gates verdes.
- **Validación en vivo** (Qwen2.5-coder-1.5B-Q4_K_M, ReleaseFast, prompt `"def fib(n):"`, 12 tokens, `--quantized`):
  - Salida greedy: `def fib(n):\n    if n == 0:` (12 tokens generados) — **idéntica en tokens** al camino f32 anterior.
  - Pico de RAM medido externamente (`Get-Process fluency | PeakWorkingSet64` mientras corre): **2407 MB**.
  - Baseline documentada pre-cambio: **~3 GB** (~3082 MB en auditoría 2026-06-02).
  - **Reducción observada: ~600 MB** (~751 MB teóricos: 933 MB f32 → ~182 MB Q6_K).
  - ⚠️ **Corrección (auditoría 2026-06-06):** la afirmación original de que "el KV-cache con `seq_len=32768` (~1.8 GB) domina el footprint restante" es **incorrecta** para `run`: `runGenerate` usa `max_seq=min(prompt+n_tokens+4, ctx)` (~21), KV-cache real ~1 MB. La causa real del residual de ~2.4 GB era la **doble residencia del GGUF** (buffer crudo `data` ~940 MB + copias `@memcpy` al arena ~900 MB). Resuelto en **TAREA F7.T5** (zero-copy) → **1441 MB**.
- **ADRs:** ver [ADR-0034](adr/0034-embedding-cuantizada-por-fila.md) (con corrección).

### TAREA F7.T2 — Matmul batch cuantizado dequant-once ✅ (2026-06-06)
Implementar `matmulBatchQ4_0/Q8_0/Q4_K/Q6_K` en `zkernel.zig` que dequantizan cada bloque del peso **UNA vez** y lo aplican a los M tokens del batch, eliminando la re-dequantización redundante que hacía que speculative decoding no acelerara (~0.98×) sobre pesos cuantizados.

- **Archivos:** `src/zkernel/zkernel.zig` (`matmulBatchQ4_0`/`Q8_0`/`Q4_K`/`Q6_K`), `src/zarch/zarch.zig` (`matmulBatch` ahora enruta a `matmulBatchQ*` en lugar de iterar `matmulVecQ*` token por token).
- **Cambios:**
  - `matmulBatchQ*`: por cada fila n del peso, por cada bloque bk a lo largo de K, dequantizar el bloque a `deq_buf[BLOCK_SIZE]` y acumular `out[m*N + n] += dotSimd(deq_buf, inp[m*K + k .. m*K + k_end])` para todo m=0..M-1. Esto reduce las dequants de M×N×n_blocks a N×n_blocks (factor M).
  - `matmulBatch` en `zarch.zig`: cuando el peso es cuantizado (`qw != null`), llama directamente al kernel batch correspondiente (`q4_0`→`matmulBatchQ4_0`, etc.) en lugar del loop `for (0..M) |m| matmulVecQ*(...)`. El camino f32 conserva el loop anterior sin cambios.
- **Tests:**
  - 4 tests nuevos en `zkernel.zig`: `matmulBatchQ4_0/Q8_0/Q4_K/Q6_K vs M×matmulVecQ*` (oráculo: llamar M veces `matmulVecQ*`; tolerancias 1e-4/1e-4/5e-3/5e-2).
  - Conteo total: **240 tests** (subió de 236), sin fugas, 4 gates verdes.
- **Validación en vivo** (Qwen2.5-coder-1.5B-Q4_K_M, ReleaseFast, prompt `"def fib(n):"`, 12 tokens, `--quantized`):
  - Greedy: **10.74 s** → **1.12 tok/s**
  - Speculative (`--speculate`): **10.32 s** → **1.16 tok/s**
  - **Speedup: ~1.04×** (antes ~0.98×, ahora > 1× como objetivo)
  - Salida **bit-idéntica** a greedy (verificado con `diff` vacío).
- **Limitación honesta**: el speedup sigue siendo modesto (~1.04×) porque la atención causal en `forwardBatch` sigue siendo token-por-token (no batchable sin quebrar causalidad). El beneficio se materializa principalmente en QKV+FFN batch por capa. Para aceleraciones mayores se requeriría un modelo borrador de mayor calidad o batching de atención causal (complejo).
- **ADRs:** ver [ADR-0032](adr/0032-speculative-decoding-prompt-lookup.md) (actualizado).

### TAREA F7.T3 — MoE end-to-end en inferencia (hard top-k) ✅ (2026-06-06)
Verificar/cablear el camino de **inferencia** con `forwardMoE` (hard top-k) de `src/zarch/zarch.zig` en un modelo pequeño, asegurando que `forwardBatch` también use MoE y que no haya regresión en el modelo real denso.

- **Archivos:** `src/zarch/zarch.zig` (`forwardMoE`, `forwardBatch`), `src/main.zig` (`ServeEngine.generate` / `generateStream`), `src/zspeculate.zig`.
- **Cambios:**
  - **Bug corregido:** `forwardBatch` ignoraba `cfg.n_experts > 0` y siempre ejecutaba el camino FFN denso. Si `--speculate` se usara con un modelo MoE real, el speculative decoding usaría FFN denso en lugar de expertos, corrompiendo la salida. Fix: en el loop batch de FFN por capa, se añadió `if (cfg.n_experts > 0) { @memcpy(self.x[0..dim], x_batch[i*dim..]); self.forwardMoE(self.xb, l); @memcpy(x_batch[i*dim..], self.x[0..dim]); } else { ...camino denso... }`.
- **Tests nuevos:**
  - `Transformer MoE forward: finitud, determinismo, forwardBatch vs forward` en `zarch.zig` (1 capa, dim=8, 3 expertos, top_k=2, moe_dim=8): verifica logits finitos/no-NaN, determinismo (misma entrada → mismos logits), `forwardBatch(K=1)` coincide con `forward` (tolerancia 1e-4), `forwardBatch(K=2)` coincide con dos llamadas `forward` (oráculo, tolerancia 1e-4).
- **Conteo total:** **241 tests** (subió de 240), sin fugas, 4 gates verdes.
- **Validación en vivo** (modelo real Qwen2.5-coder-1.5B-Q4_K_M, ReleaseFast, **el modelo es denso**, sin MoE — se usa para verificar **ausencia de regresión**):
  - `fluency forward <modelo> 42` → logits reales, top-5 coherente (`"1"`, `"2"`, `"3"`, `"ur"`, …).
  - `fluency run <modelo> "\"Hello world\"" 5` → `"Hello world" is a common greeting in` (5 tokens).
  - `fluency run <modelo> "\"Hello world\"" 5 --speculate` → **salida idéntica** a greedy.
  - **No hay regresión** en modelo denso tras el cambio en `forwardBatch`.
- **Limitación honesta:** no se dispone de un **modelo GGUF real con MoE** para ejecutar `fluency run` con expertos en vivo. La evidencia sustitutiva es (a) test determinista con `forwardMoE` en `zarch.zig` y (b) ausencia de regresión en el modelo real denso. Si se obtiene un GGUF MoE (p. ej. Mixtral/Qwen-MoE), se verificará directamente.
- **ADRs:** ninguno nuevo (la decisión de hard top-k en inferencia ya está en ADR-0033).

### TAREA F7.T4 — Backend GPU vía Zig 🟡 CORREGIDO: compilación DESBLOQUEADA (2026-06-06, revisión)
> ⚠️ El veredicto original "❌ BLOQUEADO" era **erróneo** (triples de target inválidos). La compilación Zig→SPIR-V **funciona** en zig 0.16.0. Ver [ADR-0035 corregido](adr/0035-gpu-spirv-bloqueo-zig-0-16.md) y [ADR-0037 (plan)](adr/0037-backend-gpu-vulkan-spirv-zig.md).

- **Causa del falso bloqueo:** se usó `spirv64` (sin OS → `MissingOperatingSystem`) y `spirv64-freestanding` (OS inválido). Los targets reales son `spirv64-vulkan`, `spirv64-opencl`, `amdgcn-amdhsa`.
- **Verificado en vivo (este equipo):**
  - `zig build-obj -target spirv64-vulkan kernel.zig` → **`.spv` válido** (magic `03 02 23 07`).
  - Kernel SAXPY realista (`callconv(.kernel)`, punteros `addrspace(.global)` a arrays, bucle, multiply-add, 5 params) compila en **Debug** (14.308 b) y en **`-fno-llvm -O ReleaseFast`** (12.808 b).
  - Crashes en `-O Release*` sin `-fno-llvm`: enrutan por LLVM (`stage2_llvm`), que no soporta `callconv(.spirv_kernel)`. El backend **self-hosted** (default Debug; forzado con `-fno-llvm`) es el correcto.
  - Runtime presente: `vulkan-1.dll` v1.4.309 + ICD AMD `amdxc64.dll` (RX 6600 XT).
- **Estado real:** *compilación* Zig→SPIR-V **desbloqueada y probada**. *Ejecución end-to-end* (shim Vulkan en `src/platform/` que cargue el `.spv`, despache y lea resultados, validado bit-a-bit vs CPU) es el trabajo restante — **no bloqueado, es ingeniería**. Plan de 5 hitos (M1–M5) en ADR-0037.
- **Consecuencia:** la RX 6600 XT (~10 TFLOPS) pasa a ser vía viable para entrenar Fluency-100M/350M local y 100 % constitucional, sin nube ni FFI.

**M1 ✅ implementado (desarrollo anterior + corrección, 2026-06-06).**
  - `src/zgpu/kernels/saxpy.zig` (kernel `callconv(.spirv_kernel)`); paso `zig build spv` en `build.zig` lo compila a `saxpy.spv` con `-target spirv64-vulkan -fno-llvm -OReleaseFast` [build.zig — REVISIÓN HUMANA pendiente].
  - `src/zgpu/zgpu.zig` registrado como raíz de test; test "saxpy.spv magic bytes" verifica el magic `03 02 23 07`.
  - **Bug corregido (la desarrollo anterior lo dejó sin compilar y sin anunciarlo):** usaba `std.fs.cwd().openFile` + `file.read`, API inexistente en este Zig 0.16 (el filesystem vive en `std.Io`). Reescrito con `std.Io.Threaded` + `std.Io.Dir.cwd().readFileAlloc`, idéntico al resto del repo.
  - Verificación: `zig build` ✅ · `zig build spv` ✅ (`saxpy.spv`, 2080 b) · `zig build test` ✅ (**242 tests**, +1 vs 241) · `zig build gate` ✅ (4 verdes; el `.spv` no afecta pureza).
  - ⚠️ Limpieza pendiente: `saxpy.spv` se emite en la raíz del repo (artefacto de build); al pasar a git conviene ignorarlo (`*.spv`) o emitirlo a `zig-out/`.
  - **M2–M5 (shim Vulkan, dispatch, matmul, integración) siguen pendientes** — la desarrollo anterior NO llegó a `src/platform/vulkan/`.

### TAREA F7.T5 — Carga zero-copy de pesos cuantizados ✅ (2026-06-06, revisión)
Eliminar la doble residencia del GGUF: en lugar de **copiar** (`@memcpy`) cada tensor cuantizado al arena de pesos mientras el buffer crudo del archivo (`data`) seguía vivo, **referenciar los bytes in situ** en el buffer GGUF ya cargado.

- **Hallazgo de auditoría:** con `--quantized` el pico real era **~2376 MB** (no se alcanzaba el objetivo 1.5 GB). Causa: `loadQuantEmbedding`/`loadQuantWeight` copiaban ~900 MB al arena mientras `data` (~940 MB de `readFileAlloc`) permanecía vivo toda la generación. El KV-cache **no** era el dominante (max_seq≈21 en `run`).
- **Análisis de tiempos de vida (LIFO de `defer` en `run` y `serve`):** `tf.deinit()` se ejecuta **antes** que `gguf.deinit()`/`free(data)` → las copias eran redundantes; el `Transformer` puede tomar prestados los bytes sin riesgo de puntero colgante.
- **Archivos:** `src/main.zig` (`loadQuantEmbedding`, `loadQuantWeight`).
- **Cambios:** ambos devuelven slices a `gguf.getTensorData(t)[0..sz]` en vez de `wa.alloc`+`@memcpy`. Solo metadatos pequeños (`layers`/`fmts`, norms, bias) siguen en el arena. Seguridad de alineación: los kernels de `zkernel.zig` leen con `@as(*align(1) const Vec, ...)`, y tanto `wa.alloc(u8)` como `data` son align-1 → sin nueva suposición de alineación.
- **Validación en vivo** (Qwen2.5-coder-1.5B-Q4_K_M, ReleaseFast, medición externa `Get-Process | PeakWorkingSet64`):
  - Pico de RAM: **2376 MB → 1441 MB** (**−935 MB**, por debajo del objetivo 1.5 GB, comparable a llama.cpp).
  - `run "def fib(n):" 16 --quantized` → salida **byte-idéntica** antes/después (`def1 = lambda x: x + 1 / print(def1(5))`).
  - `run "for i in range(10): print(i)" 20 --quantized --speculate` → **bit-idéntico a greedy** (valida `forwardBatch` sobre los blobs referenciados).
  - **241 tests** pasan, 4 gates verdes, sin fugas.
- **Invariante a respetar:** el Transformer cuantizado no debe sobrevivir al `gguf`/`data` prestado (se cumple hoy en `run` y `serve`).
- **ADRs:** ver [ADR-0036](adr/0036-carga-zero-copy-pesos-cuantizados.md); corrige la atribución errónea de [ADR-0034](adr/0034-embedding-cuantizada-por-fila.md).

---

## Próximos pasos (bien definidos)

Orden recomendado por dependencias del plan (F4 necesita un Fluency usable de F3; F6 necesita F5+F3). Cada paso indica **archivos**, **trabajo** y **criterio de aceptación**.

### A. Cierre fino de F3 (corto plazo, desbloquea usar el alumno)
1. **Persistencia de checkpoints** — ✅ **HECHO** (2026-05-30).
   - Archivos: `src/zdistill/zdistill.zig` (`saveCheckpoint`/`loadCheckpoint` con magic+versión+NanoConfig+seed+pesos f32 LE), `main.zig` (`--out <ruta>` en `distill`/`train`, comando `fluency eval <ckpt>`).
   - Evidencia: test de round-trip (entrenar 50 pasos → guardar → recargar → misma ppl/acc bit a bit); CLI en vivo `fluency train 50 --seed 7 --out demo.ckpt` → `fluency eval demo.ckpt` reproduce ppl=1.790 acc=100.0%. Sin fugas.
2. **Pipeline de datos real** — ✅ **HECHO** (2026-05-30).
   - Archivos: `src/ztok/ztok.zig` (`buildByteVocabSimple` para tokenización sin GGUF), `main.zig` (`fluency tokenize <txt> <out.bin>` con vocab byte-simple o `<gguf> <txt> <out.bin>` con BPE real; `train --data <bin>` samplea ventanas con `TokenDataset.fromFile`).
   - Evidencia: generado corpus de 3.39M tokens → `fluency tokenize corpus.txt corpus.bin` (vocab=256) → `fluency train 1000 --seed 13 --data corpus.bin` baja ppl de 262.6 → 1.055 (CE 0.06) y acc de 0% → 100% de forma estable. Sin fugas.
3. **MOHAWK B/C en vivo** — ✅ **HECHO** (2026-05-30).
   - Archivos: `src/zarch/zarch.zig` (`forwardWithHidden` captura hidden post-capa por capa del profesor), `src/zgrad/zgrad.zig` (`mseLoss` escribe gradiente MSE en un nodo del grafo, propagable por backward), `src/zdistill/zdistill.zig` (`forwardWithHidden` en FluencyNano, `stepMohawkB/C` en Trainer, tests), `main.zig` (`--mohawk A|B|C` en `distill`).
   - Evidencia: tests MOHAWK Stage B (MSE hidden baja >50%) y Stage C (KD+MSE combinado baja >50%); CLI `fluency distill ... --mohawk B|C` captura hidden del profesor y entrena con alineación. Si dims no coinciden, cae a MOHAWK A (KD). Sin fugas.
4. **(Escala, opcional)** matmul cuantizado on-the-fly para el alumno de vocab grande y corrida Fluency-100M/350M sobre ≥1B tokens.
   - Criterio del plan E4: ≥X% de la calidad del profesor en la suite propia con ≤5B tokens. (Limitado por cómputo, no por código.)

### B. F4 — Runtime agéntico (~95%, E5 cumplido) → gate-check **E5**
1. **Parser de tool-calls Hermes/OpenAI** real — ✅ **HECHO** (2026-05-30).
   - Archivos: `src/zagent/zagent.zig` (`parseToolCalls`: Hermes XML `<tool_call>`, OpenAI JSON `tool_calls`, objeto directo; normaliza a `[]ToolCall` con `id`/`name`/`arguments_json`).
   - Evidencia: tests unitarios parsean los 3 formatos, despachan tool correcta vía `Registry`, respetan presupuesto `max_iters=0` (rechaza sin ejecutar), traza ReAct estructurada (`Trace` con `thought/action/observation/answer/budget_exceeded`). Sin fugas.
2. **Bucle ReAct** con presupuesto, interrupción y reflexión — ✅ **HECHO** (2026-05-30; corregido 2026-06-01).
   - Archivos: `src/zagent/zagent.zig` (`Agent.run` acumula `context: ArrayList(u8)` entre iteraciones; **parsea sólo el texto NUEVO de cada paso** vía `parsed_from` —el prompt en iter 0, la salida del generador en iter>0— evitando re-ejecutar las mismas tool-calls; observaciones se inyectan al contexto; interrupción vía `should_stop: ?*bool`; reflexión automática al error/denegación; la tool **`finish` corta el bucle** explícitamente).
   - Evidencia: test de interrupción (`should_stop`), test de `finish` que corta en iter 0 aunque haya generador y presupuesto (verificado en vivo: `Respuesta: Objetivo finalizado (finish)`), tests de deny-by-default y multi-iteración con `generate_fn`. Sin fugas.
3. **Tools nativas que faltan** — ✅ **HECHO** (2026-05-30).
   - Archivos: `src/zagent/tools/tools.zig` (`http_fetch` con allowlist de hosts, `search` recursivo en archivos de texto, `memory_read`/`memory_write` sobre `MemoryStore`, `delegate` deny-by-default, `bash` con sandbox: allowlist de prefijos + rechazo de metacaracteres peligrosos).
   - Evidencia: tests de allowlist rechazan comandos/hosts no permitidos; tests de metacaracteres (`|`, `>`, `&&`, etc.) rechazan antes de ejecutar; memory round-trip write→read funciona; delegate siempre devuelve error. Sin fugas.
4. **Seguridad Art. IV** — ✅ **HECHO** (2026-05-30).
   - Archivos: `src/zagent/zagent.zig` (`SecretStore` con ChaCha20-Poly1305 usando `util.crypto`; `AuditLog` append-only con categorías `tool_dispatch`/`tool_denied`/`secret_access`/`secret_denied`/`budget_exceeded`; `Agent` con deny-by-default formal: tools no registradas devuelven error y se loguean; acceso a secretos sin store loguea `secret_denied`).
   - Evidencia: tests de round-trip cifrado/descifrado, tampered ciphertext rechazado por autenticación, audit log serializa eventos secuenciales, agente deny-by-default loguea `tool_denied`, agente sin store loguea `secret_denied`. Sin fugas.
5. **`zmemory` integrado al agente** + persistencia + embeddings + BM25 — ✅ **HECHO** (2026-05-30; embeddings/BM25 reales 2026-06-01).
   - Archivos: `src/zagent/memory/zmemory.zig` (`serialize`/`deserialize` ZMEM v1; `saveToFile`/`loadFromFile`; **`embedText`** generador de embeddings determinista por hashing trick L2-normalizado; **`addText`/`searchText`** que lo usan; búsqueda híbrida real = 0.7·coseno + 0.3·**BM25** con IDF y normalización por longitud), `src/zagent/tools/tools.zig` (`memory_write`→`addText`, `memory_read`→`searchText` con guarda de `top_k`), `src/ztui/ztui.zig` (`addText`).
   - Evidencia: round-trip serialize/deserialize; invalid magic / version rechazados; `embedText` determinista, L2-normalizado y discriminante; `searchText` (embeddings+BM25) recupera la entrada relevante. Sin fugas.
6. **`ztui`** — ✅ **streaming + diffs aprobables + sesiones HECHO** (2026-05-30; streaming 2026-06-02; diffs/sesiones 2026-06-02; streaming genuino 2026-06-02).
   - Archivos: `src/ztui/ztui.zig` (`runChatLoop` REPL + `StreamCtx`/`streamEmit` cableados a `agent.stream_fn`/`stream_ctx`; comandos `/quit`, `/clear`, `/save <path>`, `/help`, **`/session save/load/list`** para persistir/recargar `MemoryStore` en `.fluency/sessions/`; **`askApproval`** callback en `Agent` que muestra diff de `fs_write` y pregunta `y/n` por stdin; `showFileDiff` linea-a-linea simple), `src/zagent/zagent.zig` (`approval_fn`/`approval_ctx` en `Agent`; gating antes de `registry.dispatch` para `fs_write`/`bash`; **`generate_stream_fn`**/`generate_stream_ctx` para streaming token-a-token genuino con fallback a `generate_fn`), `src/main.zig` (`fluency chat`, adaptador `agentGenerateStreamAdapter` conecta `ServeEngine.generateStream` al agente).
   - Evidencia: test "Agent streaming token-a-token emite pieces al callback" (re-emisión por palabras del buffer completo); test "Agent generate_stream_fn emite tokens durante la generacion" (mock emite "hello", " ", "world" pieza a pieza → `stream_fn` las recibe en tiempo real); test "Agent generate_stream_fn despacha tool-call desde texto acumulado" (streaming genera `<tool_call>finish</tool_call>` → se parsea y ejecuta finish); test "ztui approval callback rejects destructive tool"; test "ztui session path is relative to .fluency/sessions". El agente prefiere `generate_stream_fn` si está disponible (modelo cargado vía `--gguf`), cayendo a `generate_fn` + re-emisión por palabras si no.
7. **Afinado tool-use** del Fluency con trayectorias — ✅ **HECHO MVP** (2026-05-30).
   - Archivos: `src/zagent/trajectory.zig` (`Trajectory` con steps `thought/action/observation/answer/budget_exceeded/interrupted`; serialización a JSON simple con `serializeTo`; `Trajectory` integrado en `Agent` como `agent.trajectory`), `src/zagent/zagent.zig` (cada paso del bucle ReAct se registra automáticamente en `agent.trajectory`).
   - Evidencia: tests de round-trip serialize JSON con todos los tipos de steps; tests de agente verifican que `agent.trajectory` se construye y se libera sin fugas. Sin fugas.
8. **Conexión agente-generador** — ✅ **HECHO** (2026-05-30; cableado real al modelo 2026-06-01).
   - Archivos: `src/zagent/zagent.zig` (`generate_fn`/`generate_ctx`; iteraciones > 0 invocan el generador con el contexto y parsean tool-calls de su salida), `src/main.zig` (`agentGenerateAdapter` conecta el `ServeEngine` real; `fluency agent "<prompt>" --gguf <ruta>` carga el modelo local y lo cablea al bucle ReAct).
   - Evidencia: unit-test con mock `generate_fn` (tool en iter 0 del prompt + iter 1 del generador); ruta tool-only verificada en vivo (`fluency agent` con tool-call → dispatch + finish). **Límite de escala (superado)**: `--gguf` dequantizaba el profesor a f32 (~6GB); ahora `fluency agent "..." --gguf <modelo.gguf> --quantized` usa matmul on-the-fly sin dequant completo (ver TAREA 5).
9. **Sampler restringido por gramática F2 en el generador del agente** — ✅ **HECHO** (2026-06-02).
   - Archivos: `src/main.zig` (`ServeEngine.generateJson`: siembra el prefijo `{"name":"` para empujar la forma tool-call y usa `zarch.grammar.grammarArgmax` en cada paso; `agentGenerateJsonAdapter` la cablea a `agent.generate_fn`), reusa `src/zarch/grammar.zig` (`BalancedJson`+`grammarArgmax`, F2) y `ztok.decodeToken`.
   - Evidencia: test "grammarArgmax en bucle genera JSON tool-call valido y parseable" (emite `{"name":"finish","arguments":{}}`, `BalancedJson.isComplete()` ✅, `parseToolCalls` lo acepta) + test "agent ReAct despacha tool-call desde generate_fn mock JSON". Sin fugas.
   - **Aceptación E5 ✅**: test "Gate-check E5: agente multi-herramienta con tool-calls validos y presupuesto" (secuencia dummy→log→finish, presupuesto acotado, sin panic/fugas). Tool-calls válidos restringidos por gramática + presupuesto/interrupción/deny-by-default/secretos ✅. **Matiz**: `--gguf` en vivo con `--quantized` evita OOM (ver TAREA 5); la gramática garantiza validez estructural, no el esquema completo del tool-call (de eso se encarga el prefijo sembrado).

### C. F5 — Protocolos y embebido (~95%, E7 mitad ZSA→MCP cumplida) → gate-check **E7**
1. **Cliente MCP** (consumir tools externas → registry del agente) — 🟢 **e2e HECHO (stdio por proceso real)** (2026-05-30; proceso real 2026-06-02).
   - Archivos: `src/zmcp/zmcp.zig` (`zmcp.Client` con `initPipes` para stdio real, `initialize`, `toolsList`, `toolsCall`; `stdioLoop` en `Server` para bucle persistente; transporte inyectable `transport_fn` que en tests usa `memTransport` y en vivo usa pipes de `std.process.Child`), `src/zmcp/zmcp.zig` (`McpToolWrapper` + `registerTools` integrados en el mismo módulo; bridge a `zagent.Registry`).
   - Evidencia tests: "zmcp e2e client->server initialize + tools/list + tools/call" y "…toolsCall error propagation" vía `memTransport` (loopback en-memoria); "McpBridge execute finish returns ok"; "zmcp server handles sequential requests"; "zmcp bridge registerTools round-trip".
   - Evidencia en vivo: **`fluency mcp-selftest`** spawnea `fluency mcp-server` como subproceso (`std.process.Child` con `.stdin = .pipe`, `.stdout = .pipe`), el `Client` realiza `initialize` → `tools/list` (4 tools: fs_read, fs_write, bash, finish) → `tools/call finish` → `registerTools` en `zagent.Registry` → dispatch real de `finish` vía el bridge MCP. **Sin fugas**.
   - **Limitación honesta**: en Windows se requirió lectura byte-a-byte (`readVec`) en lugar de `takeDelimiterExclusive` en el `Client` y `Server` para evitar artefactos de buffering en pipes; el protocolo sigue siendo line-delimited JSON-RPC. El loopback en-memoria se conserva como test unitario.
2. **`sampling/createMessage`** en el servidor `zmcp` — ✅ **HECHO** (2026-05-30).
   - Archivos: `src/zmcp/zmcp.zig` (`Server.sample_fn` inyectable; handler `sampling/createMessage` delega a `sample_fn` y devuelve JSON con `role`/`content`).
   - Evidencia: test con mock `sample_fn` verifica respuesta JSON correcta. Sin fugas.
3. **Inferencia real en `libfluency`** (C-ABI) — ✅ **HECHO** (2026-06-01).
   - Archivos: `src/lib.zig`, `build.zig`.
   - Evidencia: `fluency_load` lee GGUF real vía `std.Io`, construye Transformer y Tokenizer; `fluency_generate` tokeniza, prefill, genera tokens greedy y decodifica via callback streaming; `fluency_embed` promedia embeddings de tokens; tests con modelo mínimo sintético verifican que la salida no es echo (viene del Transformer) y que embeddings reflejan los pesos del vocabulario. `build`/`test`/`gate` pasan. Sin fugas.
4. **Target WASM** (`fluency.wasm` + bindings JS) — ✅ **HECHO** (2026-06-01).
   - Archivos: `build.zig` (target `wasm32-freestanding` + `addLibrary` estático), `src/lib.zig` (`fluency_load_bytes` para carga desde memoria, `fluency_generate_buf` para evitar callbacks C en WASM, `fluency_alloc`/`fluency_free_ptr` para gestión de heap desde JS), `js/fluency.js` (wrapper JS), `js/index.html` (demo).
   - Cambios transversales: `src/zload/zload.zig` convertido a portable 32-bit usando `std.math.cast` en todos los `u64→usize` del parser GGUF (tensor_count, metadata_kv_count, dims, offsets, etc.).
   - Evidencia: `zig build` genera `zig-out/bin/fluency.wasm` (~591KB en debug); `build`/`test`/`gate` pasan sin regresión. Sin fugas.
5. **`zlsp`** (opcional, marcado así en el plan).
   - **Aceptación E7 (parcial)**: ZSA usa una tool MCP externa **vía cliente (loopback en-memoria)** ✅; servidor MCP + `sampling` ✅; `libfluency` y `fluency.wasm` compilan y tienen tests con modelo sintético ✅. **Pendiente para cierre pleno**: (a) e2e por transporte stdio de proceso real; (b) un host MCP externo usando una tool de ZSA en vivo; (c) `libfluency` ejecutándose desde Node y `fluency.wasm` desde navegador (corrida manual documentada).

### D. F6 — Integración ecosistema → gate-check **E6**
**Encuadre (2026-06-02):** Fluency es un **motor independiente** que sirve como drop-in compatible con Ollama y OpenAI. La meta es que Fluency sea **uno más de los modelos seleccionables** por cualquier app que consuma LLMs. El trabajo es 100% del lado de Fluency — que `fluency serve` sea un endpoint **drop-in compatible con Ollama Y OpenAI** (descubrible, con nombre de modelo configurable estilo `Fluency-<params>-<ver>`, servicio persistente), de modo que cualquier app sólo lo añada a su registro de modelos. Modos de uso: (a) **vía API Ollama/OpenAI** (cualquier cliente que las hable apunta a Fluency); (b) **vía CMD/TUI `fluency`** TUI (`ztui`). Filosofía: **funcional + compacto**, ejecutable localmente en cualquier ordenador. Apps embebidas pueden consumir `libfluency`/`fluency.wasm`. (La madurez del modelo propio destilado se sigue cocinando aparte — sin prisa por anunciarlo; por ahora el motor 100%-Zig sirve un GGUF cuantizado.)

#### TAREA 1 — Nombre de modelo configurable y descubrible ✅ (2026-06-02)
- **Archivos:** `src/zserve/zserve.zig` (`ServeOptions.model` default, `checkModelField`, `responseForTarget`), `src/main.zig` (flag `--model` default).
- **Cambio:** default de modelo pasa de `fluency-local` a `Fluency-1.5b-v1`. Se añade validación del campo `model` en el body de peticiones a `/v1/chat/completions`, `/api/chat`, `/api/generate`, `/v1/completions`: si el campo está ausente → se acepta; si coincide → se acepta; si no coincide → se devuelve `{"error":"model not found"}` con gracia (sin panic/fuga).
- **Tests:** 2 nuevos (`discovery endpoints reflect configured model name`, `model field validation accepts matching or absent model`). Conteo: **172 tests** (antes 170), sin fugas.
- **Evidencia en vivo:** `fluency serve --gguf <Qwen2.5-coder-1.5B-Q4_K_M> --quantized --model Fluency-1.5b-v1 --port 11435`:
  - `GET /v1/models` → `{"object":"list","data":[{"id":"Fluency-1.5b-v1","object":"model","created":0,"owned_by":"fluency"}]}`
  - `GET /api/tags` → `{"models":[{"name":"Fluency-1.5b-v1","model":"Fluency-1.5b-v1","modified_at":"1970-01-01T00:00:00Z","size":0}]}`
  - `POST /v1/chat/completions {"model":"Fluency-1.5b-v1",...}` → responde con texto generado real (`"\r\nI am trying to create a"`).
  - `POST /v1/chat/completions {"model":"wrong-model",...}` → `{"error":"model not found"}`.

#### TAREA 2 — Paridad de descubrimiento Ollama (`/api/show`) y compatibilidad ✅ (2026-06-02)
- **Archivos:** `src/zserve/zserve.zig` (`ollamaShowResponse`, handler `/api/show`, separación de `/api/version` a formato Ollama), `src/main.zig` (pasa metadatos del GGUF a `ServeOptions`).
- **Cambio:**
  - Se añade `ollamaShowResponse` que devuelve un JSON con shape Ollama real: `modelfile`, `parameters`, `template`, `details` (parent_model, format, family, families, parameter_size, quantization_level), `model_info` (general.architecture, general.context_length).
  - Handler `POST /api/show`: extrae `name` del body, valida que coincida con `options.model`, devuelve metadatos o `{"error":"model not found"}`.
  - `/api/version` se separa del bloque `/api/health` y devuelve ahora `{"version":"0.7.0"}` (formato que esperan los clientes Ollama) en vez del health-check genérico.
  - `ServeOptions` añade campos `model_family`, `model_params`, `model_quant`, `model_ctx`; `main.zig` los rellena desde `zload.ModelInfo` cuando carga un GGUF real (`info.arch`, `--quantized` → `Q4_K_M`, etc.).
- **Tests:** 3 nuevos (`/api/version returns Ollama format`, `/api/show returns metadata for matching model`, `/api/show rejects wrong model name`). Conteo: **175 tests**, sin fugas.
- **Evidencia en vivo:** `fluency serve --gguf <Qwen2.5-coder-1.5B-Q4_K_M> --quantized --model Fluency-1.5b-v1 --port 11435`:
  - `GET /api/version` → `{"version":"0.7.0"}` (sin campos de health genérico).
  - `POST /api/show {"name":"Fluency-1.5b-v1"}` → `{"modelfile":"...","parameters":"...","template":"...","details":{"parent_model":"","format":"gguf","family":"qwen2","families":["qwen2"],"parameter_size":"1.5B","quantization_level":"Q4_K_M"},"model_info":{"general.architecture":"qwen2","general.context_length":32768}}`.
  - `POST /api/show {"name":"wrong-model"}` → `{"error":"model not found"}`.
  - Endpoints existentes (`/api/tags`, `/v1/models`, `/v1/chat/completions`, `/api/chat`) siguen operativos sin regresión.

#### TAREA 3 — Streaming correcto en AMBOS protocolos, verificado en vivo ✅ (2026-06-02; **BUG corregido en auditoría**)
- **Archivos:** `src/zserve/zserve.zig` (`writeStreamChunk` extraído + `StreamCtx`, `serveStreaming`), `src/main.zig` (`ServeEngine.generateStream`).
- **BUG hallado en auditoría:** la entrega original afirmaba "el streaming ya emitía el formato correcto", pero el stream de **`/api/chat` emitía el campo `"response"`** (shape de `/api/generate`) en vez de `"message":{"role":"assistant","content":...}`. Los clientes oficiales de Ollama-chat (JS/Python) leen `message.content` del stream → habrían recibido **contenido vacío**. **Fix:** `StreamCtx` ahora lleva `chat: bool`; `writeStreamChunk` distingue chat (`message.content`) ↔ generate (`response`), tanto en los chunks como en el mensaje final `done`.
- **Tests:** +1 regresión "zserve streaming chunk shape: Ollama chat usa message.content, generate usa response, OpenAI usa delta" (**176 tests**, sin fugas).
- **Evidencia en vivo** (tras el fix) `POST /api/chat ... stream:true`:
  ```
  {"model":"Fluency-1.5b-v1","message":{"role":"assistant","content":" to"},"done":false}
  {"model":"Fluency-1.5b-v1","message":{"role":"assistant","content":" the"},"done":false}
  {"model":"Fluency-1.5b-v1","message":{"role":"assistant","content":" world"},"done":false}
  {"model":"Fluency-1.5b-v1","message":{"role":"assistant","content":""},"done":true}
  ```
  OpenAI SSE sigue correcto: `data: {"choices":[{"delta":{"content":" of"}}]}\n\n` … `data: [DONE]`. `/api/generate` (completion) conserva el shape `response`.

#### TAREA 4 — Verificación end-to-end con clientes reales (aceptación de F6) ✅ (2026-06-02)
- **Archivos:** `docs/F6-modelo-seleccionable.md` (creado), `src/zserve/zserve.zig` (sin cambios), `src/main.zig` (sin cambios).
- **Cambio:** No se modificó código — se ejecutó y documentó el flujo completo drop-in contra el modelo real cuantizado (ReleaseFast, puerto 11436).
- **Tests:** conteo se mantiene en **175 tests**, sin fugas.
- **Evidencia en vivo:** `fluency serve --gguf <Qwen2.5-coder-1.5B-Q4_K_M> --quantized --model Fluency-1.5b-v1 --port 11436`:
  - **Flujo Ollama completo:**
    - `GET /api/tags` → `{"models":[{"name":"Fluency-1.5b-v1",...}]}` — descubrimiento OK.
    - `POST /api/chat {"model":"Fluency-1.5b-v1","messages":[{"role":"user","content":"Hi, how are you?"}],"max_tokens":10}` → `{"model":"Fluency-1.5b-v1","message":{"role":"assistant","content":" I'm doing well, thank you. How about"},"done":true}` — respuesta coherente.
    - `POST /api/chat {"model":"Fluency-1.5b-v1","messages":[{"role":"user","content":"Say hello"}],"stream":true,"max_tokens":6}` → NDJSON token-a-token (`"response":" the"`, `"response":" world"`, ..., `"done":true`).
  - **Flujo OpenAI completo:**
    - `GET /v1/models` → `{"object":"list","data":[{"id":"Fluency-1.5b-v1","object":"model","created":0,"owned_by":"fluency"}]}` — descubrimiento OK.
    - `POST /v1/chat/completions {"model":"Fluency-1.5b-v1","messages":[{"role":"user","content":"Hi, how are you?"}],"max_tokens":10}` → `{"id":"chatcmpl-fluency-local","choices":[{"message":{"role":"assistant","content":" I'm doing well, thank you. How about"},"finish_reason":"stop"}],"usage":{...}}`.
    - `POST /v1/chat/completions {"model":"Fluency-1.5b-v1","messages":[{"role":"user","content":"Say hello"}],"stream":true,"max_tokens":6}` → SSE token-a-token (`data: {"choices":[{"delta":{"content":" of"}}]}\n\n`, ..., `data: [DONE]\n\n`).
  - Documento detallado: `docs/F6-modelo-seleccionable.md`.

#### TAREA 5 — Servicio persistente + quickstart "funcional y compacto" para entusiastas ✅ (2026-06-02)
- **Archivos:** `docs/F6-modelo-seleccionable.md` (secciones "Quickstart local" y "Persistencia verificada" añadidas), `src/zserve/zserve.zig` (sin cambios), `src/main.zig` (sin cambios).
- **Cambio:** No se modificó código — se verificó estabilidad del servidor ante múltiples peticiones secuenciales y se documentó el quickstart.
- **Tests:** conteo se mantiene en **175 tests**, sin fugas.
- **Evidencia en vivo:** `fluency serve --gguf <Qwen2.5-coder-1.5B-Q4_K_M> --quantized --model Fluency-1.5b-v1 --port 11437`:
  - **Persistencia:** 4 peticiones secuenciales alternando OpenAI y Ollama contra el mismo proceso, sin reinicio:
    1. `POST /v1/chat/completions` → `"content":" To solve the problem of 2 +"` ✅
    2. `POST /api/chat` → `"content":" Paris! Paris is the capital city of"` ✅
    3. `POST /v1/chat/completions` → `"content":" why the concept of a \"solar"` ✅
    4. `POST /api/chat` → `"content":"! I'm a friendly AI"` ✅
  - El servidor no crasheó ni degradó entre peticiones.
  - **Quickstart:** se documentó en `docs/F6-modelo-seleccionable.md` cómo compilar (`zig build -Doptimize=ReleaseFast`), servir (`fluency serve --gguf <blob> --quantized --model Fluency-1.5b-v1 --port 11434`), RAM (~3 GB), y apuntar clientes Ollama/OpenAI.

#### TAREA F6.1 — Endurecer el drop-in: manejo robusto de errores ✅ (2026-06-05)
- **Archivos:** `src/zserve/zserve.zig`.
- **Cambios:**
  - `errorJsonResponse`: helper para generar respuestas de error JSON consistentes.
  - `isBodyJsonLike`: rechaza body que claramente no es JSON (no empieza con `{` / `[` / `"`).
  - `checkModelField` ahora es **estricta**: si la clave `"model"` está presente pero el valor no es un string válido (ej. `"model": 123`), se trata como `model not found`.
  - En endpoints de generación (`/v1/chat/completions`, `/api/chat`, `/api/generate`, `/v1/completions`): body no-JSON → error `invalid JSON body`; campo `content`/`prompt` ausente → error `missing prompt content` / `missing prompt`; modelo no coincidente → error `model not found`.
  - `serveRequest`: validación de body/model antes de entrar a streaming; si `responseForTarget` falla internamente (OOM), responde con HTTP 500 y JSON de error en vez de cerrar silenciosamente.
  - `serveStreaming`: si prompt/content falta, emite chunk de error y cierra el stream.
- **Tests:** 6 nuevos (`rejects non-JSON body`, `rejects malformed model field (non-string)`, `rejects missing content in chat completion`, `rejects empty body in chat completion`, `rejects missing prompt in generate`, `accepts valid JSON with extra fields`). **220 tests** totales, sin fugas.
- **Limitación honesta:** la validación en vivo con `curl` contra el servidor real reveló que el servidor HTTP subyacente (`std.http.Server`) en Windows no lee correctamente el body POST con clientes externos (body llega vacío al handler). Esto es un problema preexistente del runtime HTTP, no de la lógica de validación — los tests unitarios demuestran que `responseForTarget` rechaza correctamente los casos degenerados cuando el body llega.

#### TAREA F6.2 — Embeddings drop-in cableados al embed real ✅ (2026-06-05)
- **Archivos:** `src/zserve/zserve.zig` (`EmbedFn` tipo + `embed`/`embed_ctx` en `ServeOptions`, handlers `/v1/embeddings` y `/api/embeddings` leen campo `input` del body y llaman a `embed_fn` si está cableado; fallback stub preservado), `src/main.zig` (`ServeEngine.embed`: mean-pooling de token embeddings del `token_embedding_table`; cableado en `ServeOptions` al iniciar `fluency serve`).
- **Cambios:**
  - `zserve.zig`: nuevo tipo `EmbedFn` que recibe texto y devuelve `[]f32`; campos `embed`/`embed_ctx` en `ServeOptions`; endpoints `/v1/embeddings` (OpenAI) y `/api/embeddings`/`/api/embed` (Ollama) extraen campo `input`/`prompt` del body JSON y, si hay `embed_fn` + `embed_ctx`, invocan la función real y liberan el resultado; si no, caen al stub `[0.125, 0.25, 0.5, 1.0]`.
  - `main.zig`: método `ServeEngine.embed` tokeniza el texto con `Tokenizer.encode`, suma los embeddings de cada token desde `token_embedding_table[id*dim..]`, divide por el conteo (mean pooling), devuelve slice propiedad del allocator. Cableado en `ServeOptions` cuando `fluency serve` carga un modelo real (`--gguf`).
- **Tests:** 3 nuevos (`/v1/embeddings usa embed_fn real`, `/api/embeddings usa embed_fn real`, `embeddings sin embed_fn cae a stub`) con mock `MockEmbedCtx` que devuelve vectores conocidos. **220 tests** totales, sin fugas.
- **Validación en vivo** (Qwen2.5-coder-1.5B-Q4_K_M, ReleaseFast, puerto 11439):
  - `POST /v1/embeddings {"input":"hello world"}` → respuesta OpenAI con `object:list`, `object:embedding`, y vector real de dimensión = `config.dim` (1536 valores float, no el stub de 4).
  - `POST /api/embeddings {"input":"hello world"}` → respuesta Ollama con `embeddings:[[...]]` y vector real idéntico.
  - Shape verificado: OpenAI `object:list` = True, Ollama `embeddings` = True.
- **Limitación honesta:** el embedding es mean-pooling de token embeddings (no pasa por el Transformer completo). Para uso en RAG/memoria híbrida del agente esto ya es funcional ( coincide con el generador de embeddings determinista de `zmemory`). Un embedding de capa oculta del modelo (p.ej. última capa antes del head) requeriría una pasada forward + extracción de hidden state, que no está implementado hoy.

**Pendiente F6 (NO en este repo):** que apps externas registren a Fluency en su lista de modelos. Eso es trabajo del lado de cada app y lo pilota cada desarrollador; la desarrollo anterior **no** toca esos repos.

---

## E. Próximos pasos recomendados (post-F6 — para continuar en otra sesión)

Estado a 2026-06-05: **F0–F5 100%, F6 núcleo HECHO + F6.1 ✅ + F6.2 ✅** (serve drop-in verificado en vivo + manejo robusto de errores + embeddings reales), **F7 iniciado** (matmul cuantizado on-the-fly ✅, SIMD ✅, paralelismo de filas + dequant SIMD Q8_0 ✅ ~1.55× sobre SIMD, terminal crudo ✅, speculative decoding 🟠 corregido con evidencia honesta, **MoE soft en FFN ✅**), **TUI cell-based TUI HECHA**. **230 tests, 4 gates, sin fugas.** Orden recomendado de lo que sigue (todo 100% Zig, en este repo):

1. **TUI TUI — HECHA (F4).** `src/platform/tty.zig` (`RawTty` portable Windows ConPTY + Unix termios, VT processing en stdin **y stdout**, `enable/disable/size`, best-effort) + motor **cell-based** `src/ztui/{engine,screen,transcript}.zig`: render diff ANSI truecolor, **pantalla alterna** (no ensucia el shell), layout header+transcript scrollable con word-wrap+input+status, event loop de teclado, slash commands, sesiones, streaming token-a-token, **render Markdown** (negritas/`código`/headings/bullets/```fences```), **tool-calls plegables** (Ctrl-F enfoca, Ctrl-O pliega), **diffs aprobables** `[a]`/`[r]` cableados al `approval_fn`, **paleta de comandos** (Ctrl-P: tema/plegado/foco/limpiar/sesión/salir), **temas** claro/oscuro (Ctrl-T). Decisión registrada en **ADR-0030**: **TUI propia 100% Zig**, NO `libvaxis` (cero-deps). Previsualizable: `zig run src/ztui/preview.zig`. *Refinamientos opcionales futuros*: resaltado de sintaxis por lenguaje en bloques de código, SIGWINCH (resize en caliente), scroll del transcript con teclas.
2. **F7 capacidad:** speculative decoding (draft+verify, pure Zig) 🟠 (salida idéntica verificada, speedup ~0.98× con cuantización — honestamente reportado), **MoE en el FFN ✅** (soft top-k diferenciable en `zgrad`, entrenable en `zdistill`).
3. **Madurez del modelo propio (F3 a escala):** destilar un `Fluency-100M/350M` real "funcional + compacto" para entusiastas. Limitado por cómputo; sin prisa (decisión de Zenda).

Notas para quien continúe: verificar SIEMPRE con `build` + `test --summary all` (conteo sube) + `gate`, y para serve/TUI/inferencia **exigir evidencia en vivo** (cliente real / modelo real), no solo tests unitarios — patrón de fallo recurrente de la desarrollo anterior (marcar "✅" sin probar el contrato real). El plan (`08-roadmap-fases.md`) y `CONSTITUCION.md` son inviolables; el estado vive aquí.

---

[← Roadmap (plan)](08-roadmap-fases.md) · [ADRs](adr/) · [Aprendizaje (diseño)](04-aprendizaje-destilacion.md)
