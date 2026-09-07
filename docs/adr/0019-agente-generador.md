# ADR-0019 — Conexión del agente al generador de texto (inferencia)

## Estado
Aceptado (2026-05-30)

## Contexto
F4 requiere que el agente pueda generar texto (y por tanto tool-calls) usando el modelo de inferencia (zarch). Hasta ahora, `Agent.run` solo parseaba tool-calls del prompt del usuario, pero no generaba nuevo texto. Para que el bucle ReAct sea realmente autónomo, el agente debe poder pedirle al modelo "¿qué hago ahora?" a partir del contexto acumulado.

## Decisión
1. **Callback inyectable `generate_fn`**:
   - `Agent` tiene dos nuevos campos:
     - `generate_fn: ?*const fn (ctx: ?*anyopaque, prompt: []const u8, allocator: std.mem.Allocator) anyerror![]u8`
     - `generate_ctx: ?*anyopaque`
   - Esto permite inyectar cualquier generador de texto (zarch, Ollama, mock para tests) sin acoplar `zagent` a `zarch`.
2. **Uso en `Agent.run`**:
   - Iteración 0: parsea tool-calls del prompt original (sin cambios).
   - Iteraciones > 0: si `generate_fn` está configurado, se llama al generador con `context.items` (todo el historial acumulado). El texto generado se inyecta al contexto y se registra como `thought` en trace/trajectory.
   - Luego se parsean tool-calls del contexto ampliado.
   - Si no hay `generate_fn`, el agente se comporta como one-shot (compatible con tests existentes).
3. **Formato de generación**:
   - El generador debe devolver texto que `parseToolCalls` pueda parsear (Hermes XML `<tool_call>`, OpenAI JSON `{"tool_calls":[...]}`, u objeto directo).
   - El contexto acumulado puede contener texto libre mezclado con JSON; `parseToolCalls` ya soporta buscar `<tool_call>` dentro de texto arbitrario.

## Consecuencias
- El agente ahora puede operar en modo autónomo multi-paso cuando está conectado a un modelo.
- Sin generador configurado, el comportamiento es idéntico al anterior (compatibilidad total con tests).
- El generador recibe todo el contexto acumulado, permitiendo que el modelo "vea" observaciones anteriores.
- En el futuro, `main.zig` puede inyectar `zarch.generate` o `zollama.chat` como `generate_fn`.

## Referencias
- `src/zagent/zagent.zig`: `Agent.generate_fn`, `Agent.generate_ctx`, uso en `Agent.run`.
- `src/zagent/zagent.zig` test: mock `generate_fn` que devuelve `<tool_call>` y verifica ejecución en iteración 1.
