# 06 — Interoperabilidad y protocolos

[← Plan Maestro](../PLAN-MAESTRO.md) · [← 05 Runtime agéntico](05-runtime-agentico.md) · [→ 07 Entrenamiento vía IDEs](07-entrenamiento-via-ides.md)

Cómo ZSA cumple "compatible con MCP, hooks, skills, plugins y todo lo que los IDEs soportan" (Art. VI) y cómo se vuelve un **motor drop-in** para cualquier app compatible.

---

## 1. El hallazgo que simplifica todo: un núcleo JSON-RPC 2.0

MCP y LSP usan **el mismo wire format (JSON-RPC 2.0)** y **el mismo handshake `initialize` con negociación de capacidades**; sólo difieren en el **framing** y el **namespace de métodos**. La API OpenAI y el transporte MCP Streamable-HTTP viven sobre **el mismo servidor HTTP**. Por tanto:

> **`zrpc` (un solo núcleo JSON-RPC 2.0) + framings intercambiables sirve a la vez: cliente MCP, servidor MCP y servidor LSP.** Y el servidor HTTP sirve a la vez la API OpenAI y MCP-HTTP.

```
zrpc
 ├─ tipos: Request{id,method,params} · Response{id,result|error} · Notification{method,params} · Error{code,message,data}
 ├─ framing newline-delimited     → MCP stdio
 ├─ framing Content-Length        → LSP
 ├─ framing HTTP POST + SSE       → MCP Streamable-HTTP  +  API OpenAI
 └─ dispatch table por método (entrante) + correlador id (saliente) + timeouts/cancel
```

---

## 2. API OpenAI-compatible (`zserve`) — máxima prioridad de integración

Es **lo que vuelve a ZSA usable por todo el ecosistema** (cualquier cliente compatible con OpenAI/Ollama). Endpoints:

- **`POST /v1/chat/completions`** (el principal): request con `model`, `messages[]` (roles system/user/assistant/tool), `tools[]`+`tool_choice`, `temperature`, `max_tokens`, `stream`, `response_format` (incl. `json_schema`). 
  - Respuesta no-streaming: `choices[].message` con `content` o `tool_calls[]` (¡`arguments` es **string JSON**!), `finish_reason ∈ stop|length|tool_calls`, `usage`.
  - **Streaming (SSE):** `Content-Type: text/event-stream`, chunks `data: {choices:[{delta:{content|tool_calls}}]}`, terminar con `data: [DONE]`. Tool-calls se streamean por fragmentos de `arguments`.
- **`GET /v1/models`** → `{object:list, data:[{id,object:model,...}]}`. Los IDEs lo llaman para poblar su lista; **el `id` debe coincidir** o dan 404.
- **`POST /v1/embeddings`** → `{data:[{embedding:[...]}], usage}`. Para indexado/recuperación.
- (Opcional) `POST /v1/completions` (legacy, autocompletado/FIM).

**Compatibilidad estilo Ollama** (para apps que ya usan Ollama): `POST /api/chat`, `/api/generate`, `/api/embed`, `GET /api/tags`. Con esto, **cualquier app compatible con Ollama apunta a ZSA cambiando sólo la URL base** (criterio de éxito E6).

> Detalle pragmático: aceptar (e ignorar) `Authorization: Bearer` arbitrario; tolerar campos no soportados (`logit_bias`, `n`…) como hace Ollama. Los IDEs lo toleran.

---

## 3. MCP (Model Context Protocol) — cliente y servidor

Versión estable objetivo: **`2025-11-25`** (default sin header HTTP: `2025-03-26`). Roadmap `2026-07-28` (stateless core, Tasks, MCP Apps) **no es estable** — no construir contra él aún.

### 3.1 Transportes
- **stdio** (local, primario): subproceso; JSON-RPC **newline-delimited** por stdin/stdout (sin newlines embebidos); stderr = logs.
- **Streamable HTTP** (remoto): un único endpoint con POST (cliente→servidor; `Accept: application/json, text/event-stream`; responde JSON único o abre SSE) y GET (SSE servidor→cliente). Headers `MCP-Session-Id`, `MCP-Protocol-Version`. Validar `Origin`, bind localhost.

### 3.2 Handshake
`initialize` (cliente manda `protocolVersion`+`capabilities`+`clientInfo`) → respuesta del servidor (`capabilities`+`serverInfo`) → `notifications/initialized`.

### 3.3 `zmcp` cliente (consumir tools externas)
- Implementa `initialize`, `tools/list`, `tools/call` (mínimo viable) y opcional `resources/*`, `prompts/*`; suscribirse a `*_list_changed`.
- **Las tools MCP se fusionan en el `Registry` de `zagent`** ([05](05-runtime-agentico.md)) → el agente usa tools nativas y MCP por el mismo camino.
- **Ofrecer `sampling/createMessage`**: cuando un servidor MCP pide generar, **ZSA responde con su propio modelo** (Fluency) → ZSA es el "modelo" de cualquier servidor MCP.

### 3.4 `zmcp` servidor (exponer las capacidades de ZSA)
- Implementa `initialize` + `tools/list`/`tools/call` devolviendo `content[]` (text/image/resource). Honra los MUST de seguridad (Origin, localhost, validación). stdio primero, HTTP después.
- Así, ZSA expone sus herramientas (y, vía sampling, su modelo) a cualquier host MCP.

### 3.5 Primitivas (resumen de métodos)
`tools/list|call`, `resources/list|read|subscribe` (+templates), `prompts/list|get`, `sampling/createMessage`, `roots/list`, `elicitation/create` (form/url), utilidades (ping, progress, cancel, logging, completion, paginación).

---

## 4. Hooks, Skills, Plugins (ZSA como **host** y como **paquete**)

> Importante: hooks/skills/plugins **no son protocolos de wire**; son **convenciones de filesystem/JSON** del host. ZSA los soporta de dos maneras: (a) como **host** que los carga; (b) empaquetándose **como plugin** de cualquier host compatible.

### 4.1 Hooks (host)
- ZSA implementa un **bucle de hooks** modelado en el de hosts MCP estándar (el más completo): eventos de ciclo de vida (`SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `Stop`, etc.).
- Contrato: el hook recibe JSON por stdin (`session_id`, `cwd`, `hook_event_name`, `tool_name`, `tool_input`…) y responde por stdout (exit 0 + JSON: `decision`, `permissionDecision`, `updatedInput`, `additionalContext`…; exit 2 = bloqueo). Tipos: `command`, `http`, `mcp_tool`, `prompt`, `agent`.
- Configurables en `settings.json` con `matcher` por nombre de tool. Esto da a ZSA **observabilidad y control** sobre su propia ejecución agéntica.

### 4.2 Skills (host)
- Una skill = directorio con **`SKILL.md`** (frontmatter YAML `name`+`description` + cuerpo Markdown) + archivos opcionales.
- **Progressive disclosure:** nivel 1 sólo metadata (`name`/`description`, ~100 tokens) en el system prompt; nivel 2 el cuerpo se lee cuando es relevante; nivel 3 recursos/scripts se cargan/ejecutan bajo demanda.
- ZSA escanea `~/.fluency/skills/` y `.fluency/skills/`, inyecta metadata y carga el cuerpo cuando el modelo lo decide. (Es la misma idea que las skills de `zagent`, §05 §6.)

### 4.3 Plugins (paquete)
- Un plugin de hosts MCP estándar = `plugin.json` + `skills/`, `commands/`, `agents/`, `hooks/hooks.json`, `.mcp.json`, `.lsp.json`.
- **ZSA se distribuye como plugin**: empaqueta su servidor MCP (`.mcp.json` apuntando a `fluency mcp-server`), comandos (`fluency distill/train/eval`) y skills. Así un usuario de hosts MCP estándar instala ZSA y obtiene el motor + sus tools.

---

## 5. LSP (opcional, `zlsp`)
Reusa `zrpc` con framing `Content-Length`. Útil si se quiere ofrecer features tipo "language server" (p. ej. completado asistido por Fluency dentro de un IDE). Baja prioridad; el núcleo compartido lo hace casi gratis.

---

## 6. Embebido: C-ABI y WASM (Art. VI.3)

Para que **apps no-Zig** usen ZSA sin adoptar Zig:

### 6.1 `libfluency` (C-ABI)
```c
// firmas C-ABI estables (tipos planos: punteros, ints, floats, callbacks)
FluencyModel* fluency_load(const char* model_path);
int       fluency_generate(FluencyModel*, const char* prompt, const FluencyOpts*, FluencyTokenCb, void* user);
int       fluency_embed(FluencyModel*, const char* text, float* out, int out_len);
void      fluency_free(FluencyModel*);
```
- En Zig: `export fn` con `callconv(.c)` y `extern struct`. Build con `addLibrary`/`-dynamic`. Consumible desde Node (N-API/ffi), Python (ctypes), C#, etc.

### 6.2 `fluency.wasm`
- `wasm32-freestanding` (o `wasi`), funciones exportadas con `export`/`-rdynamic`, SIMD128 para inferencia en navegador. Mantener firmas exportadas en tipos C-ABI simples.

---

## 7. Tabla: qué construir para cada compatibilidad

| Requisito IDE/ecosistema | Pieza ZSA | Prioridad |
|---|---|---|
| Ser modelo backend de IDEs y apps | `zserve` (OpenAI + Ollama-style) | **F1/F5 — máxima** |
| Consumir tools del ecosistema | `zmcp` cliente | F5 |
| Exponer tools/modelo a hosts | `zmcp` servidor (+ sampling) | F5 |
| Control de ejecución | host de hooks | F5 |
| Capacidades empaquetadas | skills host + plugin pkg | F5 |
| Embeber en apps no-Zig | `libfluency` C-ABI + `fluency.wasm` | F5/F6 |
| Language-server features | `zlsp` | opcional |

[→ Siguiente: 07 — Entrenamiento vía IDEs](07-entrenamiento-via-ides.md)
