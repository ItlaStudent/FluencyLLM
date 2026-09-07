# ADR-0017 — Bucle ReAct multi-iteración con interrupción y reflexión

## Estado
Aceptado (2026-05-30)

## Contexto
F4 requiere un bucle ReAct real que itere múltiples veces: pensamiento → acción → observación → siguiente pensamiento. La implementación anterior de `Agent.run` ejecutaba un único paso (parseaba `prompt`, ejecutaba tools, y hacía `break` inmediato). Esto no era un bucle ReAct; era un dispatch one-shot.

Además, el roadmap requiere:
- Interrupción externa (usuario puede detener al agente).
- Reflexión cuando una tool falla (para que el modelo reconsidere).
- Plan-and-execute (fase de planificación antes de la ejecución).

## Decisión
1. **Bucle real multi-iteración**:
   - `Agent.run` ahora mantiene un `context: std.ArrayList(u8)` que acumula el prompt del usuario + observaciones de cada tool.
   - En cada iteración se parsean tool-calls de `context.items` (no del prompt original).
   - Las observaciones se inyectan al contexto en formato `[Resultado de <tool>: <content>]`.
   - El loop itera hasta `max_iters` o hasta que `parseToolCalls` devuelva `len == 0`.
   - Se eliminó el `break` arbitrario después de la primera iteración.
2. **Interrupción**:
   - `Agent` tiene campo `should_stop: ?*bool`.
   - Al inicio de cada iteración, si `should_stop` apunta a `true`, el agente añade `Answer: Interrumpido por el usuario.` y sale.
   - Esto permite que la TUI (u otro hilo) detenga el agente sin `panic`.
3. **Reflexión**:
   - Si una tool devuelve `is_error = true`, se añade al contexto: `[Reflexión: ocurrió un error, revisar estrategia]`.
   - Si una tool es denegada por deny-by-default, se añade: `[Reflexión: herramienta <name> denegada por deny-by-default]`.
   - Estas reflexiones son visibles para el modelo en la siguiente iteración (cuando esté conectado).
4. **Plan-and-execute**:
   - Por ahora, la fase de planificación es implícita: el prompt del usuario define el plan.
   - En el futuro, una fase `Agent.plan` generará una lista de steps antes de entrar al bucle ReAct.

## Consecuencias
- El bucle ya puede iterar múltiples veces cuando el contexto contiene nuevos tool-calls (generados por el modelo).
- Sin modelo conectado, el bucle sigue comportándose como one-shot para prompts simples (compatibilidad hacia atrás).
- El contexto crece en memoria proporcionalmente al número de iteraciones. En el futuro puede truncarse o resumirse.

## Referencias
- `src/zagent/zagent.zig`: `Agent.run`, `Agent.should_stop`, acumulación de `context`.
