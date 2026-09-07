# ADR-0012 — Parser de tool-calls y traza ReAct estructurada

## Estado
Aceptado (2026-05-30)

## Contexto
F4 requiere un runtime agéntico capaz de interpretar tool-calls emitidos por el modelo. El código previo usaba `std.mem.indexOf` hardcodeado sobre el prompt, lo cual no escala a múltiples herramientas ni soporta formatos reales. Se necesita:
- Parser de tool-calls en los formatos dominantes (Hermes XML y OpenAI JSON).
- Estructura de traza ReAct que persista pasos (pensamiento, acción, observación) sin strings sueltos.
- Presupuesto de iteraciones (budget) estricto en el bucle.

## Decisión
1. **Normalización a `ToolCall`**: todo parser converge a `struct { id, name, arguments_json }`. El `Registry` despacha por `name` + `arguments_json` (ya funcionaba así).
2. **Parser multi-formato en `zagent.zig`**:
   - Prioridad: Hermes XML (`<tool_call>...</tool_call>`) → OpenAI JSON (`{"tool_calls":[...]}`) → objeto directo (`{"name":"...","arguments":{...}}`).
   - Usa `util.json` (parser JSON propio del proyecto, Art. I) — no `std.json` para mantener consistencia con F0.
3. **`Trace` estructurada**: `union(enum)` con `thought`, `action`, `observation`, `answer`, `budget_exceeded`. Cada paso tiene dueño de memoria vía allocator; `serializeTo` produce texto ReAct legible.
4. **Budget**: el bucle `while (iter < max_iters)` garantiza parada; si se agota, se registra `budget_exceeded` en la traza. No hay panic en la ruta de tools (Art. IV.2).

## Consecuencias
- Los tests pueden inyectar tool-calls como texto JSON y verificar despacho real sin necesidad de un modelo.
- El parser no valida schemas (eso es responsabilidad de cada `Tool.execute`).
- Cuando el agente use un modelo real, el `prompt` se reemplazará por la salida del modelo; `parseToolCalls` seguirá funcionando sin cambios.

## Alternativas consideradas
- Usar `std.json.parseFromSlice` para tipar fuertemente: descartado porque `util.json` ya es la infraestructura del proyecto y no depende de stdlib internals.
- Parser PEG/regex: descartado por complejidad innecesaria; los formatos dominantes son JSON anidado.

## Referencias
- `src/zagent/zagent.zig`: `parseToolCalls`, `Trace`, `Agent.run`.
- `src/util/json.zig`: parser/serializador propio.
- `docs/05-runtime-agentico.md`: diseño de ReAct y formatos.
