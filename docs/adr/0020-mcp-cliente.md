# ADR-0020 — Cliente MCP (stdio) para consumir tools externas

## Estado
Aceptado (2026-05-30)

## Contexto
F5 requiere que ZSA pueda consumir tools de servidores MCP externos (como filesystem, fetch, etc.) y registrarlas en el `zagent.Registry` del agente. El protocolo MCP usa JSON-RPC 2.0 line-delimited sobre stdio.

## Decisión
1. **Cliente MCP (`zmcp.Client`)**:
   - Conexión vía `std.Io.File` (stdin del proceso servidor / stdout del cliente).
   - Métodos: `initialize`, `toolsList`, `toolsCall`.
   - `toolsList` devuelve `[]ClientTool` (nombre, descripción, schema).
   - `toolsCall` envía `tools/call` y devuelve `ToolResult` (content, is_error).
   - Comunicación JSON-RPC line-delimited (cada mensaje termina en `\n`).
2. **Bridge (`zmcp.bridge`)**:
   - `McpToolWrapper`: envuelve un `zmcp.Client` y un nombre de tool; implementa la vtable de `zagent.Tool`.
   - `registerTools`: lista tools del servidor MCP y las registra en `zagent.Registry`.
   - Cada tool externa aparece como tool nativa del agente; el dispatch se traduce a `client.toolsCall`.
3. **Sin dependencia circular**:
   - `zmcp` importa `zagent` (unidireccional).
   - `zagent` no sabe de MCP.

## Consecuencias
- ZSA puede consumir cualquier servidor MCP estándar.
- Las tools MCP se integran transparentemente en el bucle ReAct del agente.
- El caller debe mantener vivo el `Client` y los wrappers mientras el agente esté activo.

## Referencias
- `src/zmcp/zmcp.zig`: `Client`, `ClientTool`, `toolsList`, `toolsCall`.
- `src/zmcp/bridge.zig`: `McpToolWrapper`, `registerTools`.
