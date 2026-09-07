# 05 — Runtime agéntico (`zagent`)

[← Plan Maestro](../PLAN-MAESTRO.md) · [← 04 Aprendizaje](04-aprendizaje-destilacion.md) · [→ 06 Interop y protocolos](06-interop-protocolos.md)

Capacidades agénticas avanzadas al nivel de **hermes-agent** (NousResearch, referencia conceptual) y **nullclaw** (Zig 0.16.0, referencia de implementación directa). Todo en Zig, deny-by-default (Art. IV).

---

## 1. Filosofía

Un agente es un **bucle**: el modelo razona, decide usar herramientas, observa resultados y repite hasta terminar. ZSA implementa este bucle con tres obsesiones tomadas de la mejor prior art:

- De **hermes-agent**: registro de herramientas auto-poblado, **skills como memoria procedural**, delegación a sub-agentes con privilegios acotados, presupuesto de iteraciones, formato de tool-call Hermes XML, trayectorias en JSONL (que además sirven para fine-tuning).
- De **nullclaw**: **todo es una vtable** (tools, memoria, canales, backends) → swappable y diminuto; memoria híbrida SQLite-BLOB (sin DB vectorial pesada); seguridad deny-by-default; **nunca `panic`**, siempre `Result`; arranque en milisegundos.

---

## 2. Registro y despacho de herramientas

```zig
pub const Tool = struct {
    pub const VTable = struct {
        name:           *const fn (*anyopaque) []const u8,
        description:    *const fn (*anyopaque) []const u8,
        parameters_json:*const fn (*anyopaque) []const u8,   // JSON Schema
        execute:        *const fn (*anyopaque, Io, args: Json, *Sandbox) Error!ToolResult,
    };
    ptr: *anyopaque, vtable: *const VTable,
};
```
- **Auto-registro:** los tools se registran en un `Registry` en init; añadir uno no toca el bucle.
- **Regla de oro (Art. IV.2):** `execute` **nunca** hace `panic`; valida `args` contra su schema y devuelve `ToolResult{ok|error}`. Los errores de ejecución se devuelven al modelo como observación (`isError:true`) para que se autocorrija — no abortan el proceso.
- **Tools nativas iniciales:** `fs_read`/`fs_write` (acotadas al workspace), `bash` (sandbox), `http_fetch` (allowlist), `search`, `memory_*`, `delegate`, `finish`. Las tools de MCP se **fusionan en el mismo registro** (ver [06](06-interop-protocolos.md)).

---

## 3. Formato de tool-call

Soporta los dos formatos dominantes y los normaliza:

- **OpenAI `tool_calls`** (entrada/salida de la API `zserve`): array `{id, type:function, function:{name, arguments}}` donde `arguments` es **string JSON**.
- **Hermes XML** (lo que emiten muchos modelos abiertos, y lo que Fluency aprenderá a emitir):
  ```
  <tool_call>{"name":"bash","arguments":{"command":"zig build test"}}</tool_call>
  ... runtime responde ...
  <tool_response>{"tool_call_id":"...","name":"bash","content":"All tests passed"}</tool_response>
  ```
  con razonamiento en `<think>...</think>`. `arguments` es objeto JSON parseado, **no** string doblemente codificado.

El parser: extrae tags → parsea JSON → **valida tipos contra el schema** antes de ejecutar. Las trayectorias se persisten en JSONL estilo ShareGPT (roles system/human/gpt/tool) — reutilizables como datos de fine-tuning (cierra el ciclo con [04](04-aprendizaje-destilacion.md)).

---

## 4. Decodificación restringida (tool-calls siempre válidos)

Para **garantizar** que el modelo emita JSON/tool-calls válidos, el sampler de `zarch` soporta **decodificación restringida por gramática** (estilo GBNF): se enmascaran a probabilidad cero los tokens que violarían la gramática en cada paso. Un conversor `json-schema → gramática` genera la gramática desde `parameters_json` de cada tool.
- Beneficio comprobado: reduce >95 % los errores de tool-use.
- Diseño: máscara precomputada por estado (autómata de pila) para overhead casi nulo (idea de XGrammar; se implementa una versión Zig).
- Temperatura baja para tool-use fiable.

---

## 5. El bucle agéntico

```zig
while (iter < budget.max_iters and budget.tokens_left() > 0 and !interrupted) {
    const out = try model.generate(ctx, .{ .grammar = active_grammar, .temp = low });
    const calls = parseToolCalls(out);              // Hermes/OpenAI
    if (calls.len == 0) break;                       // turno final (finish_reason=stop)
    for (calls) |call| {
        const res = registry.dispatch(call, &sandbox); // nunca panic
        ctx.appendToolResponse(call.id, res);
    }
    iter += 1;
}
```

Patrones soportados (configurable por tarea):
- **ReAct** (default): razonar→actuar→observar. Adaptable, un call por paso.
- **Plan-and-Execute:** un planificador fuerte genera el plan; ejecución más barata; **re-plan gate** cada K pasos o ante baja confianza. Mejor para tareas largas (coste ≈ 1×fuerte + N×barato).
- **Reflexión con verificador externo:** la autocrítica del *mismo* modelo repite sus puntos ciegos → se usa **verificación externa** (correr tests, type-check, validar schema) en lugar de auto-crítica pura.

**Guardas (Art. IV):** presupuesto de iteraciones/tokens, flag de interrupción, criterio de parada (`finish`/sin tool-calls). Señales de salud (nº de sub-agentes 3-6, tokens monótonos) para detectar bucles atascados.

---

## 6. Memoria (híbrida, ligera)

Taxonomía CoALA: **working / episódica / semántica / procedural**.

- **Working:** la ventana de contexto + scratchpad. Auto-compactación cuando crece (estilo nullclaw `agent.zig`).
- **Largo plazo (`zmemory`, vtable):** **store propio escrito en Zig** (ADR-0009) — no SQLite/C, para no romper la pureza (Art. I). **Truco de footprint (inspirado en nullclaw):** índice invertido (keyword/BM25) + embeddings como vectores en un store embebido propio, **fusionados** por similitud coseno + BM25 con pesos configurables (p. ej. 0.7 vector / 0.3 keyword). Sin DB vectorial ni SQLite externos. Proveedores alternativos tras la misma vtable (Markdown, LRU, archivo, ninguno).
- **Procedural = skills:** procedimientos que el agente crea y reutiliza. **Inyectados como mensajes de usuario, no en el system prompt** (truco de hermes-agent para preservar el prompt caching). Uso registrado en sidecar JSON.
- **Embeddings:** producidos por el propio ZSA (`/v1/embeddings`) o por un proveedor configurable.

---

## 7. Sub-agentes (delegación)

- `delegate(goal)` lanza un sub-agente con **contexto aislado**.
- **Privilegios acotados (Art. IV.5):** un sub-agente **no** puede, por defecto, delegar de nuevo (sin fork-bombs recursivos) ni tocar herramientas de gestión, salvo rol `orchestrator` explícito.
- Modos: objetivo único (el padre espera el resumen) o batch paralelo (cap por `max_concurrent_children`), sobre `std.Io.Group`.

---

## 8. Seguridad (deny-by-default, Art. IV)

| Mecanismo | Implementación |
|---|---|
| Permisos | Negados por defecto; allowlists explícitas (FS workspace-only, red por host, exec por binario). |
| Sandbox | Detección del SO; límites de recursos (mem/CPU/disco/subprocesos) siempre; bloqueo de null-byte y escape por symlink. |
| Secretos | Cifrado autenticado (ChaCha20-Poly1305, `std.crypto`) + keyfile local. Nunca en texto plano/logs/repo. |
| Gateway | Bind `127.0.0.1` por defecto; exposición pública requiere acción humana + token. |
| Auditoría | Log firmado de acciones sensibles con retención. |
| Tools | Nunca `panic`; validación de input; sanitización. |

---

## 9. Exponerse como cerebro agéntico

- **API:** `zagent` se expone vía `zserve` (`/v1/chat/completions` con `tools`) — otras apps lo usan como "modelo con herramientas".
- **MCP:** consume tools externas (cliente) y expone las suyas (servidor) — ver [06](06-interop-protocolos.md).
- **A2A (Agent-to-Agent, evolución):** `/.well-known/agent-card.json` + JSON-RPC `/a2a` (como nullclaw) para interoperar con otros agentes.

---

## 10. Qué copiar y qué evitar de la prior art

**Copiar:** vtable-por-subsistema (nullclaw), memoria SQLite-BLOB híbrida (nullclaw), skills-como-memoria-procedural inyectadas como user-msg (hermes-agent), delegación con privilegios acotados (hermes-agent), presupuesto de iteraciones (hermes-agent), deny-by-default + secretos cifrados (nullclaw).

**Evitar:** el footgun de Zig de **devolver una vtable que apunta a un struct temporal** (puntero colgante) — el *caller* debe poseer el struct (local o heap). nullclaw documenta esto como su trampa principal; nuestros tests de leaks + revisión lo vigilan.

[→ Siguiente: 06 — Interoperabilidad y protocolos](06-interop-protocolos.md)
