# ADR-0003 — bf16 emulado

- **Contexto:** bf16 no es tipo nativo en Zig (#3148).
- **Decisión:** `u16` + conversión (alta de f32); cómputo en f32; estado SSM en f32.
- **Consecuencias:** sin aritmética bf16 por HW; correcto y portable. (Art. I, §02/§03)
