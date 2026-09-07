# ADR-0002 — Cómputo: CPU-SIMD ahora, GPU-vía-Zig después; nunca CUDA/BLAS

- **Contexto:** rendimiento vs pureza (Art. I.3/I.4).
- **Decisión:** backend CPU `@Vector` por defecto; WASM-SIMD para navegador; GPU sólo vía backends Zig (SPIR-V/PTX/AMDGCN) en F7; **prohibido** CUDA/ROCm/Metal/Triton/XLA/BLAS. Backend tras vtable.
- **Consecuencias:** entrenamiento limitado a modelos pequeños a corto plazo (R1); identidad y portabilidad preservadas. **Aceptado.**
