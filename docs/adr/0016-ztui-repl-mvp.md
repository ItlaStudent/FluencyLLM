# ADR-0016 — TUI REPL básica (MVP de interfaz interactiva)

## Estado
Aceptado (2026-05-30)

## Contexto
F4 requiere una TUI interactiva. El documento [12-cli-y-tui.md](12-cli-y-tui.md) describe una TUI completa con celdas, layouts, markdown render, diffs aprobables, etc. Construir eso de cero es un proyecto de varias semanas. Se necesita un MVP funcional que el usuario pueda invocar desde CMD hoy.

## Decisión
Implementar un **REPL interactivo mínimo** (`ztui.runChatLoop`) como MVP:
1. **Lectura de stdin** línea a línea vía `std.Io.File.Reader.interface.takeDelimiterExclusive('\n')`.
2. **Loop**: imprime prompt `> `, lee input, lo pasa a `Agent.run`, muestra resultado.
3. **Persistencia implícita**: cada mensaje del usuario y cada respuesta del agente se guardan en `agent.memory` (MemoryStore).
4. **Comandos slash**:
   - `/quit` — salir del loop.
   - `/clear` — limpiar entradas de memoria.
   - `/save <path>` — serializar memoria a archivo ZMEM v1.
   - `/help` — mostrar ayuda.
5. **CLI integration**:
   - `fluency` (sin args) → entra en chat loop por defecto.
   - `fluency chat` → entra en chat loop explícito.
   - `fluency agent "<prompt>"` → one-shot (sin cambios).
6. **Refactorización**: extracción de `setupAgentAndRegistry` en `main.zig` para compartir inicialización entre `runAgentOnce` y `runChatCommand`.

## Consecuencias
- No hay modo crudo de terminal (raw mode), ANSI complejo, ni render de celdas. Es un REPL de texto plano.
- No hay streaming token-a-token ni diffs aprobables. Se añadirán en iteraciones futuras.
- El historial de conversación vive en `agent.memory` y puede persistirse con `/save`.

## Alternativas consideradas
- **TUI completa propia**: demasiado trabajo para esta iteración; el MVP REPL demuestra la arquitectura.
- **libvaxis**: dependencia externa Zig; se evaluará en iteración futura si madura para 0.16.

## Referencias
- `src/ztui/ztui.zig`: `runChatLoop`.
- `src/main.zig`: `setupAgentAndRegistry`, `runAgentOnce`, `runChatCommand`.
- `docs/12-cli-y-tui.md`: especificación completa de TUI a largo plazo.
