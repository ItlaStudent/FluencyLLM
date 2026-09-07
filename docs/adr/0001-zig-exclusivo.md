# ADR-0001 — Zig exclusivo, versión fijada ≥ 0.16.0

- **Contexto:** el encargo exige 100 % Zig; la API de Zig rompe entre minors.
- **Decisión:** todo el producto es Zig; se fija una versión concreta en `build.zig.zon`; APIs volátiles se encapsulan.
- **Consecuencias:** porting periódico al subir de versión; máxima pureza y control. (Art. I)
