# ADR-0015 — Integración de zmemory al agente con persistencia binaria

## Estado
Aceptado (2026-05-30)

## Contexto
F4 requiere que el agente mantenga memoria entre interacciones. `zmemory.zig` ya tenía `MemoryStore` con búsqueda híbrida (embeddings + keywords), pero:
- No estaba integrado al `Agent` (solo disponible como puntero externo en tools).
- No tenía persistencia a disco.
- El CLI `fluency agent` no registraba las tools `memory_read`/`memory_write`.

## Decisión
1. **Persistencia binaria en `MemoryStore`**:
   - Formato propio **ZMEM v1** little-endian:
     - Magic: `"ZMEM"` (4 bytes)
     - Versión: `u32` LE
     - N entries: `u32` LE
     - Por entry: `text_len: u32` LE, text bytes, `emb_len: u32` LE, emb bytes (`f32` LE × emb_len)
   - `serialize`/`deserialize` trabajan en memoria (buffer `[]u8`), desacoplados de I/O.
   - `saveToFile`/`loadFromFile` usan `std.Io.Dir.cwd().writeFile` / `readFileAlloc`.
   - `deserialize` limpria entradas previas antes de cargar (reemplazo, no merge).
2. **Integración en `Agent`**:
   - `Agent` ahora incluye `memory: MemoryStore` como campo valor (inicializado en `init`).
   - `Agent.deinit()` libera el memory store.
   - Las tools `MemoryReadTool`/`MemoryWriteTool` se conectan a `&agent.memory` en el CLI.
3. **CLI `fluency agent`**:
   - Registra las 9 tools nativas: `fs_read`, `fs_write`, `bash`, `finish`, `memory_read`, `memory_write`, `http_fetch`, `search`, `delegate`.
   - Memory tools apuntan al `MemoryStore` del agente creado.

## Consecuencias
- El formato ZMEM es simple y forward-compatible (magic + version upfront).
- No se usa JSON para persistencia de embeddings (más compacto y rápido con binario).
- `Agent` ahora tiene `deinit`; el caller debe invocarlo (ya se hace en `main.zig` con `defer`).

## Alternativas consideradas
- JSON para persistencia: descartado por overhead de strings y falta de tipado binario para f32.
- Merge en lugar de reemplazo en `deserialize`: descartado por simplicidad; el caller puede hacer merge manual si lo necesita.

## Referencias
- `src/zagent/memory/zmemory.zig`: `serialize`, `deserialize`, `saveToFile`, `loadFromFile`.
- `src/zagent/zagent.zig`: `Agent.memory`, `Agent.deinit`.
- `src/main.zig`: registro de todas las tools nativas con `agent.memory`.
