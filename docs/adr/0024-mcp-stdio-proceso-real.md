# ADR-0024: Cliente MCP por transporte stdio de proceso real

## Contexto

El cliente MCP (`zmcp.Client`) ya soportaba transporte inyectable (`transport_fn`) usado en tests con `memTransport` (loopback en-memoria). La tarea era cerrar el matiz "loopback" declarado en `ESTADO-FASES.md`: hacer que el cliente se comunique con un proceso `fluency mcp-server` real vía pipes stdin/stdout, ejecutando `initialize`, `tools/list`, `tools/call` y registrando las tools en `zagent.Registry`.

## Decisión

1. **Añadir `initPipes` a `Client`**: permite inyectar un `std.Io.File.Writer` (stdin del child) y un `std.Io.File.Reader` (stdout del child) en lugar de los defaults de consola.
2. **Hacer `stdioLoop` persistente en `Server`**: bucle `while(true)` que lee líneas JSON-RPC del stdin y escribe respuestas a stdout con `flush` explícito, en lugar del modo single-shot anterior.
3. **Reemplazar `takeDelimiterExclusive` por lectura byte-a-byte (`readVec`) en `Client.request` y `Server.stdioLoop`**: en Windows, `takeDelimiterExclusive` con pipes de `std.process.Child` produce artefactos de buffering que devuelven líneas vacías o corrupción entre requests consecutivos. La lectura explícita byte-a-byte elimina el problema manteniendo el mismo protocolo line-delimited.
4. **Integrar `McpToolWrapper` y `registerTools` directamente en `zmcp.zig`**: elimina el archivo `bridge.zig` separado que causaba conflictos de módulo y simplifica la API pública.
5. **Comando `fluency mcp-selftest`**: spawn del propio ejecutable como `mcp-server` subproceso, ejercitando el ciclo completo e2e en vivo.

## Consecuencias

- **Positivas**: el e2e por proceso real ya no es una limitación honesta; el loopback en-memoria se conserva como test unitario rápido y aislado.
- **Negativas**: la lectura byte-a-byte es menos eficiente que `takeDelimiterExclusive` (más syscalls), pero es aceptable para un protocolo de control (JSON-RPC de bajo volumen). En sistemas POSIX puede que `takeDelimiterExclusive` funcione bien; se deja documentado en el código como alternativa.

## Evidencia

- `zig build` ✅
- `zig build test --summary all` → **157/157 tests passed**, sin fugas
- `zig build gate` ✅ (4 gates OK)
- `fluency mcp-selftest` ejecutado en vivo: spawn → initialize → tools/list (4 tools) → tools/call finish → bridge a Registry → dispatch real → PASSED, sin fugas

## Archivos tocados

- `src/zmcp/zmcp.zig` (`initPipes`, `stdioLoop` con `readVec`, `McpToolWrapper`, `registerTools`, `request` pública, `flush` en escritura)
- `src/zmcp/bridge.zig` (eliminado, contenido movido a `zmcp.zig`)
- `src/main.zig` (modo bucle `mcp-server`, comando `mcp-selftest`, tests nuevos)
- `docs/ESTADO-FASES.md` (actualizado conteo y estado)
