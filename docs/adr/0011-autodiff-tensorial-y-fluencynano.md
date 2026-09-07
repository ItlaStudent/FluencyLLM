# ADR-0011 — Autodiff tensorial, FluencyNano y rutas de profesor (F3)

- **Estado:** Aceptado — 2026-05-30 (implementa F3; gate-check E4 cumplido).
- **Contexto:** F3 exige aprender por destilación 100 % en Zig (Art. I, III). Hacían falta tres piezas inexistentes o de juguete: (1) autodiff capaz de derivar el op-set de un LM híbrido incluyendo los mixers subcuadráticos; (2) un alumno entrenable real; (3) consultar un profesor real por las dos rutas del plan (zload puro y zollama pragmático). El `zarch` de inferencia (F1) está optimizado y no es diferenciable; reusarlo para backward habría sido frágil.

- **Decisión:**
  1. **Autodiff a nivel de tensor en `zgrad`** (`Graph`): cinta de nodos con VJP por operación; activaciones en un arena (sin fugas); los parámetros del modelo son **hojas que aliasan** slices externos y reciben su gradiente en un buffer del arena. Op-set con backward: matmul, `matmulTransB` (embeddings atados), add_bias, residual, rmsnorm, silu, mul, embed, **scan (SSM diagonal Mamba-2), lin_attn (delta/atención lineal) y attn (atención causal multi-cabeza)**. Cada VJP se valida por **gradient-check numérico** (diferencias finitas) como test permanente.
  2. **`FluencyNano`** (en `zdistill`): LM híbrido **entrenable** (embed → bloques {rmsnorm→mixer→rmsnorm→SwiGLU} con residual → salida atada), mixer alternando **attn ↔ SSM scan** por capa. Pesos en **un buffer plano** (optimizer/clipping triviales). El forward se escribe con ops de `zgrad`, así el backward es automático. Es deliberadamente pequeño ("camina antes de correr"): la maquinaria es la entregable, no el tamaño.
  3. **Dos rutas de profesor**, ambas Zig de nuestro lado: **zload** (GGUF real ejecutado con `zarch` → logits por posición → KD por logits, MOHAWK Stage A) y **zollama** (Ollama genera continuación → Seq-KD por cross-entropy). El profesor es **agnóstico** para `zdistill`: el orquestador (`main`) calcula logits/targets y los inyecta, de modo que `zdistill` no arrastra dependencias pesadas.
  4. **`ztrain`** aporta lo que faltaba para un bucle real: `clipGradGlobalNorm`, acumulación, schedule **warmup+cosine**, y `TokenDataset` (pipeline de datos mmap u16 LE).

- **Consecuencias:**
  - Aprendizaje por destilación **real y reproducible** en Zig puro, verificado: del Qwen2.5-coder real, KL(prof‖alumno) 10.21→0.034 y acuerdo top-1 0%→87.5% en 300 pasos; reproducible bit a bit con semilla; sin fugas. (Art. I, III; gate-check E4.)
  - **Límite asumido:** el alumno demostrado es `FluencyNano`, no un Fluency-100M/350M sobre 5B tokens — eso es una corrida limitada por cómputo, no por el código. MOHAWK **B/C** (alineación de hidden states) quedan como funciones de pérdida; cablearlas en vivo requiere exponer los hidden del profesor en `zarch` (trabajo futuro).
  - Coste: el forward del alumno sobre vocab grande (151 936) es O(T·V·D); aceptable a esta escala, pero motiva el matmul cuantizado on-the-fly y checkpoints a disco (ver ESTADO-FASES, próximos pasos).
