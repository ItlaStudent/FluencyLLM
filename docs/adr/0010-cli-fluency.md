# ADR-0010 — CLI `fluency` + TUI

- **Contexto:** el autor quiere invocar Fluency desde la terminal con una TUI interactiva.
- **Decisión:** ejecutable único `fluency`; modo TUI interactivo por defecto (módulo `ztui`), más `run`/`serve`/`mcp-server`/`distill`/etc. La librería TUI, si se usa externa, debe ser **Zig pura** (evaluar `libvaxis`) y aprobada por ADR; si no, TUI propia. Manejo de terminal cruda aislado en `src/platform/`.
- **Consecuencias:** experiencia interactiva desde la terminal; nueva superficie de UI a mantener; entra en F4.
