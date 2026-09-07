# ADR-0013 — Tools nativas con sandbox y allowlist

## Estado
Aceptado (2026-05-30)

## Contexto
F4 requiere un conjunto completo de herramientas nativas para que el agente pueda interactuar con el sistema de forma controlada. El código previo tenía `fs_read`, `fs_write`, `bash` y `finish`, pero `bash` carecía de restricciones de seguridad y faltaban `http_fetch`, `search`, `memory_*` y `delegate`.

## Decisión
1. **BashTool con sandbox explícito**:
   - Allowlist de prefijos permitidos (`zig build test`, `zig version`, `dir `, `echo `, etc.).
   - Rechazo de metacaracteres peligrosos (`|`, `>`, `<`, `&`, `;`, `$`, `` ` ``, `\n`, `\r`) antes de ejecutar.
   - Si no pasa validación, devuelve `ToolResult{ is_error: true }` sin llegar a `spawn`.
2. **HttpFetchTool con allowlist de hosts**:
   - Extracción de host desde la URL (`extractHost`).
   - Allowlist cerrada: `127.0.0.1`, `localhost`, `api.github.com`, `raw.githubusercontent.com`.
   - Usa `std.http.Client` (ya presente en `zollama`).
3. **SearchTool**:
   - Búsqueda recursiva en archivos de texto del workspace (exts `.zig`, `.md`, `.txt`, etc.).
   - Profundidad máxima 2; salta `.git` y `.zig-cache`.
4. **MemoryReadTool / MemoryWriteTool**:
   - Wrapper sobre `zmemory.MemoryStore` (keyword + cosine similarity).
   - Requieren puntero a `MemoryStore` en el struct; si es null, devuelven error.
5. **DelegateTool**:
   - Stub que siempre rechaza (Art. IV.5 deny-by-default).

## Consecuencias
- Cada tool valida sus propios argumentos y devuelve errores estructurados; nunca `panic`.
- Los tests que involucran ejecución real de red o shell no se ejecutan en CI para evitar flakes; los tests de allowlist cubren la lógica de seguridad sin side-effects.
- `tools.zig` no puede ser root de test porque importa `zagent.zig` relativamente (`../zagent.zig`), lo cual Zig 0.16 rechaza cuando el archivo es root. Los tests viven en `zagent.zig`.

## Alternativas consideradas
- Regex/PEG para validar comandos bash: descartado por complejidad innecesaria; lista de metacaracteres es suficiente.
- Allowlist de hosts por wildcard (ej. `*.github.com`): descartado por simplicidad; lista cerrada de hosts explícitos es más segura y auditable.

## Referencias
- `src/zagent/tools/tools.zig`: implementación de todas las tools.
- `src/zagent/zagent.zig`: tests de allowlist y sandbox.
- `docs/05-runtime-agentico.md`: diseño de tools nativas.
