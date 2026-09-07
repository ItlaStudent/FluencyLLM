# ADR-0008 — Memoria agéntica híbrida (vector + BM25), sin DB externa

- **Contexto:** footprint mínimo (estilo nullclaw) + pureza.
- **Decisión:** memoria híbrida (embeddings + índice invertido BM25) fusionada por coseno + BM25; tras vtable de proveedores.
- **Consecuencias:** ligereza; sin dependencia vectorial pesada. El store concreto lo fija ADR-0009.
