# ADR-0004 — Arquitectura híbrida subcuadrático+atención por defecto

- **Contexto:** subcuadrático puro flaquea en recall; producción usa híbridos.
- **Decisión:** ~80-90 % capas subcuadráticas (Mamba-2/GDN/RWKV-7), ~10-20 % atención de ventana deslizante; patrón configurable en `model.zon`; ≥ 75 % subcuadrático verificado por gate (II.3).
- **Consecuencias:** calidad cercana a Transformer con coste subcuadrático; satisface Art. II literalmente.
