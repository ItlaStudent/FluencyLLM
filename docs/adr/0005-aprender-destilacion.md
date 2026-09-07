# ADR-0005 — Aprender por destilación (MOHAWK + KD + on-policy + DPO), no reempaquetar

- **Contexto:** "aprender de otros LLM, 100 % Zig".
- **Decisión:** destilación cruzada de arquitectura como método principal; profesor vía `zollama` o `zload`; pesos resultantes propios; licencias registradas.
- **Consecuencias:** viable en ~5B tokens; cumple Art. I y III.
