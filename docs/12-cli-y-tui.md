# 12 — CLI `fluency` y TUI

[← Plan Maestro](../PLAN-MAESTRO.md) · [← 06 Interop](06-interop-protocolos.md) · [→ 11 Referencias](11-referencias.md)

El modelo y el producto se llaman **Fluency**. Este documento define cómo se **invoca desde la terminal** (CMD/PowerShell/bash) y cómo es su **interfaz de terminal (TUI)** interactiva. Todo 100 % Zig (Art. I).

---

## 0. Convención de nombres (canónica)

> Esta sección manda sobre cualquier nombre anterior en el plan.

| Concepto | Nombre |
|---|---|
| **Producto / modelo / CLI** | **Fluency** |
| **Comando de terminal** | **`fluency`** |
| Familia de modelos | `Fluency-30M`, `Fluency-100M`, `Fluency-350M`, `Fluency-1B`, `Fluency-3B`… |
| Librería C-ABI | `libfluency` (símbolos `fluency_*`) |
| Build WASM | `fluency.wasm` |
| **Codename interno del repo/proyecto** | **Zig Super Agent (ZSA)** — solo nombre interno; el usuario ve "Fluency" |

> Donde otros documentos del plan digan `zsa` (comando) o `SubZ` (modelo), **léase `fluency` / `Fluency`**. Los nombres de módulos internos (`zcore`, `ztensor`, `zarch`, etc.) no cambian: son internos, no los ve el usuario.

---

## 1. Se invoca desde CMD

`fluency` es un **ejecutable único** (`fluency.exe` en Windows, `fluency` en Linux/macOS) que, instalado en el `PATH`, se llama desde cualquier terminal. Modos:

```sh
fluency                      # abre la TUI interactiva (chat + agente) en el directorio actual
fluency "arregla este bug"   # atajo: abre la TUI ya con ese primer mensaje
fluency run "<prompt>"       # one-shot NO interactivo (imprime y sale; pipeable)
fluency serve                # modo "cerebro": servidor OpenAI/Ollama-compatible + MCP
fluency mcp-server           # arranca como servidor MCP (para que un IDE lo consuma)
```

> Nota de plataforma: en Windows 11 (tu entorno) la TUI usa la consola moderna (Windows Terminal / ConPTY). El manejo de terminal cruda va en `src/platform/` (shim de SO, permitido por Art. I.3 con ADR), de modo que el resto es Zig portable.

### Distribución / instalación
- Binario único, sin runtime: se copia al `PATH` (o `winget`/`scoop` en Windows, `brew`/script en Unix — a definir).
- Cross-compilado desde una sola máquina (fortaleza de Zig) a Windows/Linux/macOS/ARM.

---

## 2. Subcomandos completos

```
fluency                       Abre la TUI interactiva (default).
fluency run <prompt>          Ejecuta una tarea agéntica de una sola vez (no interactivo).
fluency chat                  Alias explícito de la TUI de chat.
fluency serve [--port N]      Servidor cerebro: /v1/chat/completions (+stream, tools),
                              /v1/models, /v1/embeddings, /api/chat (estilo Ollama).
fluency mcp-server [--stdio|--http]   Expone Fluency como servidor MCP a IDEs.
fluency mcp add <url|cmd>     Registra un servidor MCP que Fluency consumirá (cliente).
fluency models                Lista los modelos Fluency locales y su info.
fluency pull <modelo>         Descarga un modelo profesor (vía Ollama) para destilar.
fluency distill <subcmd> ...  mohawk | seqkd | onpolicy | dpo  (ver doc 04).
fluency train <config.zon>    Entrenamiento desde cero / continuación.
fluency eval --suite eval/    Evaluación contra baseline/profesor.
fluency gate                  Corre los gates constitucionales (alias de `zig build gate`).
fluency --version | --help    Metadatos.
```

Todos los subcomandos comparten config (`~/.fluency/config.zon` global + `.fluency/` por proyecto) y sesiones persistentes.

---

## 3. La TUI (cómo se visualiza)

La TUI es una aplicación de terminal **cell-based** (rejilla de celdas con color/estilo), renderizada con secuencias ANSI sobre la terminal en modo crudo. Layout:

```
┌ Fluency · Fluency-1B · sesión: refactor-auth ·  3.2k tok · ⏎ enviar ─────────────┐
│                                                                                  │
│  ▸ tú: refactoriza el login para usar el nuevo AuthService                       │
│                                                                                  │
│  ▾ fluency:  Voy a localizar el login y el AuthService.                          │
│     ⚙ tool: search("login")                          ▸ 4 resultados              │
│     ⚙ tool: fs_read("src/auth/login.ts")             ▸ 120 líneas               │
│     Propongo este cambio:                                                        │
│     ┌ diff src/auth/login.ts ───────────────────────────────────────────────┐   │
│     │ - const u = await db.users.find(...)                                    │   │
│     │ + const u = await authService.login(creds)                             │   │
│     └─────────────────────────────────────────────────────────────────────────┘ │
│     [a] aceptar  [r] rechazar  [e] editar                                        │
│                                                                                  │
├──────────────────────────────────────────────────────────────────────────────── │
│ > escribe tu mensaje…                                              (Ctrl-J nueva)│
└─ modo: AGENTE ─ /help comandos ─ Ctrl-C salir ─ Ctrl-P paleta ───────────────────┘
```

**Componentes:**
- **Header:** producto, modelo activo, sesión, contador de tokens, hint de envío.
- **Transcript scrollable:** mensajes con **render Markdown** (negritas, listas, bloques de código con resaltado), **llamadas a herramientas plegables/expandibles**, **diffs** con aceptar/rechazar, spinners durante la generación con streaming token a token.
- **Input multilínea** con historial, autocompletado de `/comandos` y rutas de archivo.
- **Status line:** modo (chat/agente), atajos, estado (pensando/usando herramienta/esperando confirmación).
- **Paleta de comandos** (Ctrl-P): cambiar modelo, abrir sesión, alternar modo, etc.
- **Temas** (claro/oscuro/alto contraste).

**Modos:**
- **Chat:** conversación pura (sin tocar el sistema).
- **Agente:** el bucle `zagent` (ReAct + tools) con **confirmación humana** para acciones sensibles (deny-by-default, Art. IV) — la TUI muestra diffs/comandos y pide aprobar.

---

## 4. Arquitectura del CLI/TUI (módulos)

```
fluency (src/main.zig)            # dispatch de subcomandos
 └─ ztui (src/ztui/)              # motor TUI: render de celdas, layout, input, eventos, markdown, diff
      ├─ usa zagent  (bucle agéntico, tools, memoria)   ← doc 05
      ├─ usa zarch   (inferencia/streaming del modelo)  ← doc 03
      └─ usa zserve  (cuando corre `fluency serve`)     ← doc 06
```

- **`ztui`** es un módulo nuevo (Capa 5). Renderiza sobre terminal cruda; el manejo de TTY específico de SO vive en `src/platform/`.
- La TUI **no** mete lógica de modelo: orquesta `zagent`/`zarch` y pinta. Sesiones se guardan vía el store de memoria propio en Zig (doc 05, ADR-0009).

### Dependencia de TUI (decisión en ADR-0010)
- **Opción A (preferida si es pura y madura):** usar **`libvaxis`** (librería TUI **escrita en Zig**, cell-based, soporte Windows/Unix) como dependencia `build.zig.zon`. Al ser **Zig puro**, es admisible bajo Art. I.2 **con ADR** (`gate_purity` verifica que toda dependencia sea Zig). 
- **Opción B:** **TUI propia mínima** en `ztui` (ANSI + modo crudo + render de celdas) — máxima pureza/control, más trabajo.
- **Decisión:** evaluar `libvaxis` en F4; si su madurez/versión-Zig encaja, adoptarla por ADR; si no, TUI propia. En ningún caso una librería TUI de otro lenguaje (ncurses/termbox C ⇒ prohibido).

---

## 5. Encaje con el resto del plan

| Pieza | Relación |
|---|---|
| Motor de otras apps | `fluency serve` expone las APIs de [doc 06](06-interop-protocolos.md); cualquier app compatible apunta ahí. |
| Agéntica | La TUI es el front-end de `zagent` ([doc 05](05-runtime-agentico.md)); modo agente con confirmación humana. |
| Entrenar/destilar | `fluency distill/train/eval/pull` ([doc 04](docs/04-aprendizaje-destilacion.md)); también invocables por CLI/IDEs ([doc 07](07-entrenamiento-via-ides.md)). |
| Gobernanza | `fluency gate` corre el enforcement (Art. IX). |
| Roadmap | TUI llega en **F4** (sobre `zagent`); `serve`/MCP en **F5**. Ver [doc 08](08-roadmap-fases.md). |

---

## 6. Resumen

`fluency` se escribe en la terminal, abre una **TUI de agente** con chat, herramientas y diffs aprobables, y también funciona como **cerebro servidor** (`fluency serve`) y **servidor MCP** para IDEs. Todo en Zig; la única pieza dependiente de SO (terminal cruda) queda aislada en `src/platform/`. La librería TUI, si se usa una externa, debe ser **Zig pura** (p. ej. `libvaxis`) y aprobada por ADR.

[→ Volver al Plan Maestro](../PLAN-MAESTRO.md)
