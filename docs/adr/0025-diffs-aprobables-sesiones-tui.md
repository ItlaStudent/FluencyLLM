# ADR-0025: Diffs aprobables y sesiones en la TUI

## Contexto

La TUI (`fluency chat`) ya tenía REPL básico con streaming por palabras, persistencia de memoria (`/save <path>`), y comandos `/quit`/`/clear`/`/help`. Los dos flecos pendientes de F4 eran: (a) pedir confirmación humana antes de ejecutar tool-calls destructivas (`fs_write`, `bash`), mostrando el cambio propuesto; (b) gestión de sesiones persistentes (`/session save/load/list`).

## Decisión

1. **Approval callback en `Agent`**: añadir `approval_fn: ?ApprovalFn` y `approval_ctx: ?*anyopaque` al struct `Agent` (`src/zagent/zagent.zig`). Antes de `registry.dispatch`, si la tool es `fs_write` o `bash` y existe `approval_fn`, se invoca el callback. Si devuelve `false`, la tool-call se rechaza con error "rechazada por el usuario" y se inyecta al contexto ReAct como reflexión.
   - Firma: `fn(ctx: ?*anyopaque, io: std.Io, tool_name: []const u8, args_json: []const u8) anyerror!bool`
   - Esto mantiene la lógica de aprobación desacoplada del agente core: la TUI (o cualquier frontend) decide cómo pedir confirmación.

2. **Diff visual en la TUI**: el callback `askApproval` en `src/ztui/ztui.zig` parsea `args_json` para `fs_write`, lee el archivo actual vía `std.Io.Dir.cwd().readFileAlloc`, y muestra un diff línea-a-línea simple (`-` old / `+` new). Para `bash`, muestra el comando propuesto. Luego lee `y/n` de stdin con un reader temporal.

3. **Sesiones persistentes**: comandos `/session save <nombre>`, `/session load <nombre>`, `/session list` en `runChatLoop`. Las sesiones se guardan en `.fluency/sessions/<nombre>` usando el formato ZMEM existente (`MemoryStore.saveToFile`/`loadFromFile`). Se añade `ensureSessionDir` para crear `.fluency/sessions/` vía `std.Io.Dir.cwd().createDirPath`.

## Consecuencias

- **Positivas**: el agente ahora tiene gating de aprobación configurable por frontend; la TUI puede usarse en modo agente con confianza sobre mutaciones; las sesiones permiten retomar contexto entre ejecuciones.
- **Negativas**: el diff es un algoritmo línea-a-línea naive (no LCS/unified diff completo), suficiente para MVP pero no comparable a `git diff`. La aprobación requiere stdin interactivo; en pipes no interactivos las tool-calls destructivas se rechazan por default (EndOfStream → `false`).

## Evidencia

- `zig build` ✅
- `zig build test --summary all` → **161/161 tests passed**, sin fugas
- `zig build gate` ✅ (4 gates OK)
- Tests nuevos:
  - "ztui approval callback rejects destructive tool": mock rechaza `fs_write` → agente recibe "rechazada por el usuario"
  - "ztui approval callback accepts destructive tool": `finish` no requiere aprobación → ejecuta normalmente
  - "ztui session path is relative to .fluency/sessions": ruta de sesión correcta
  - "ztui command parsing": `/session save/load/list` reconocidos

## Archivos tocados

- `src/zagent/zagent.zig` (`ApprovalFn`, campos `approval_fn`/`approval_ctx` en `Agent`, gating en `Agent.run`)
- `src/ztui/ztui.zig` (`askApproval`, `showFileDiff`, `/session save/load/list`, `sessionDir`, `sessionPath`, `ensureSessionDir`)
- `build.zig` (root de test `ztui_test_mod` para ejecutar tests del módulo)
- `docs/ESTADO-FASES.md` (actualización estado F4)
