# ADR-0007 — Un núcleo JSON-RPC 2.0 para MCP/LSP; HTTP para OpenAI/MCP-HTTP

- **Contexto:** MCP y LSP comparten JSON-RPC 2.0; OpenAI API y MCP-HTTP comparten servidor HTTP.
- **Decisión:** `zrpc` único + framings intercambiables; `zserve` HTTP compartido.
- **Consecuencias:** menos código, más superficies de compatibilidad. (Art. VI, [06](06-interop-protocolos.md))
