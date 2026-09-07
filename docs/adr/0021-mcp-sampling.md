# ADR-0021 — Handler sampling/createMessage en servidor MCP

## Estado
Aceptado (2026-05-30)

## Contexto
El servidor MCP de ZSA anunciaba la capacidad `sampling: {}` en `initializeResult`, pero no tenía handler para `sampling/createMessage`. Los hosts MCP pueden usar esta capacidad para pedirle al servidor que genere texto (ej. completar una conversación).

## Decisión
1. **Campo inyectable `sample_fn`**:
   - `Server.sample_fn: ?SampleFn` donde `SampleFn = fn(ctx: ?*anyopaque, messages_json: []const u8, allocator) ![]u8`.
   - `Server.sample_ctx: ?*anyopaque` para pasar contexto opaco (modelo, config, etc.).
2. **Handler `sampling/createMessage`**:
   - Recibe params JSON con `messages`, `modelPreferences`, etc.
   - Serializa params a JSON y llama a `sample_fn`.
   - Devuelve resultado JSON-RPC con formato MCP: `{role: "assistant", content: {type: "text", text: "..."}}`.
   - Si no hay `sample_fn` configurado, devuelve texto fallback `"sampling not configured"`.

## Consecuencias
- ZSA puede servir como backend de generación para hosts MCP.
- El generador es inyectable: puede ser `zarch.generate`, `zollama.chat`, o mock para tests.
- No hay acoplamiento entre zmcp y el motor de inferencia.

## Referencias
- `src/zmcp/zmcp.zig`: `Server.sample_fn`, `createMessage`.
