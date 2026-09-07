# ADR-0026: Streaming token-a-token genuino en el bucle del agente

## Contexto

El bucle ReAct del agente (`Agent.run`) ya soportaba `generate_fn`: un callback que devuelve el texto completo generado, y luego `Agent.run` lo trocea por palabras/ espacios y re-emite al `stream_fn` del agente. Esto funcionaba pero era **re-emisión post-facto**: el modelo decodifica todo el texto primero, y solo después se emiten las piezas.

`ServeEngine.generateStream` (camino `serve`/SSE) ya emitía tokens genuinos a medida que se decodificaban, pero no estaba conectado al agente. La TAREA 3 era cerrar esta brecha.

## Decisión

1. **Añadir `generate_stream_fn` a `Agent`** (`src/zagent/zagent.zig`): callback con firma:
   ```zig
   fn(ctx: ?*anyopaque, prompt: []const u8, allocator: std.mem.Allocator, max_tokens: usize,
      emit_ctx: ?*anyopaque, emit: *const fn(emit_ctx: ?*anyopaque, piece: []const u8) anyerror!void) anyerror!void
   ```
   El generador emite piezas a través del callback `emit`; NO acumula ni devuelve buffer. El agente acumula el texto localmente mediante un wrapper `ArrayList(u8)` que también reenvía cada pieza al `stream_fn` del agente (si existe).

2. **Preferencia en `Agent.run`**: en iteraciones > 0, si `generate_stream_fn` está presente, se usa. Si no, se cae al `generate_fn` existente con re-emisión por palabras. Esto garantiza compatibilidad hacia atrás: los callers que solo setean `generate_fn` siguen funcionando exactamente igual.

3. **Adaptador en `main.zig`** (`agentGenerateStreamAdapter`): puente entre `ServeEngine.generateStream` (que usa `zserve.EmitFn` con `*anyopaque`) y la firma del agente (con `?*anyopaque`). El adaptador crea un wrapper local que hace el cast de tipos y reenvía cada pieza.

4. **Cableado automático en `runAgentOnce`**: cuando se carga un modelo `--gguf`, se setean tanto `generate_fn` como `generate_stream_fn` (ambos apuntan al mismo `ServeEngine`), dando al agente la capacidad de streaming genuino en vivo.

## Consecuencias

- **Positivas**: el streaming del agente ahora es token-a-token genuino cuando hay modelo cargado; la TUI recibe las piezas a medida que el modelo decodifica, no después. Se elimina la limitación honesta declarada en `ESTADO-FASES.md`.
- **Negativas**: ninguna. El fallback a `generate_fn` preserva compatibilidad con todos los tests existentes.

## Evidencia

- `zig build` ✅
- `zig build test --summary all` → **163/163 tests passed**, sin fugas
- `zig build gate` ✅ (4 gates OK)
- Tests nuevos:
  - "Agent generate_stream_fn emite tokens durante la generacion": mock emite "hello", " ", "world" como piezas separadas → `stream_fn` las recibe en orden, texto acumulado = "hello world"
  - "Agent generate_stream_fn despacha tool-call desde texto acumulado": streaming genera `<tool_call>finish</tool_call>` → se parsea, despacha `finish` y se verifica `fin.call_count == 1`

## Archivos tocados

- `src/zagent/zagent.zig` (`generate_stream_fn`/`generate_stream_ctx` en `Agent`, wrapper de acumulación/emisión en `Agent.run`, tests)
- `src/main.zig` (`agentGenerateStreamAdapter`, cableado en `runAgentOnce`)
- `docs/ESTADO-FASES.md` (actualización F4 a 100%, eliminación de matiz de streaming)
