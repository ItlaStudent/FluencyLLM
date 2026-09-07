# F7 — Evolución: Design Document

> **Basado en:** PLAN-MAESTRO.md §1.7, docs/08-roadmap-fases.md §F7, docs/02-arquitectura.md §10, docs/03-nucleo-subcuadratico.md §4-6, CONSTITUCION.md Art. I/II/VII
> **Objetivo:** Subir el techo de rendimiento y capacidad sin romper la pureza (Art. I/II)
> **Gate-check F7:** cada incorporación pasa los gates, mejora una métrica de eval sin regresión, y respeta Art. I

## 1. Prerrequisito: Arreglar Gates de CI

**Contexto (Art. IX):** Los 4 gates (`gate_purity`, `gate_constitution`, `gate_quality`, `gate_budgets`) están rotos por cambios de API en Zig 0.16:
- `for` sobre arrays const requiere sintaxis con índice explícito (Zig 0.16 cambió la semántica de tuplas)
- `ChildProcess.Term` usa `.exited` (minúscula) no `.Exited`
- `AddressSpace.of()` cambió a `.limited()`

**Decisión:** Corregir la sintaxis para Zig 0.16 en los 4 archivos `tools/gate_*.zig`. Sin cambios de lógica.

## 2. Primer ADR/FEATURE: MoE en el FFN

**Contexto (docs/03-nucleo-subcuadratico.md §2, §5):** El plan especifica que el bloque unificado es:
```
x = x + Mixer(Norm(x))
x = x + FFN/MoE(Norm(x))    ← "MoE opcional (evolución)"
```
La arquitectura (ADR-0004, doc 02 §4) contempla MoE como evolución ("MoE en el FFN" es entregable explícito de F7).

**Decisión:** Implementar Mixture of Experts como reemplazo configurable del FFN single (SwiGLU):

- **Router (gating):** Linear(d_model → n_experts) + softmax + top-k (k=2 por defecto, k=1 mínimo)
- **Expertos:** Cada experto es un SwiGLU MLP completo (reusa la estructura FFN existente)
- **Load balancing:** Pérdida auxiliar (z-loss) para distribución uniforme de tokens entre expertos
- **Config:** `n_experts: u32`, `top_k: u32`, `expert_dim: u32` (por defecto = d_ffn / n_experts) en `model.zon`
- **Integración:** Nueva struct `MoEFFN` en `zarch/mixers/moe.zig` tras la interfaz existente. Si `n_experts > 0`, el bloque usa MoEFFN en vez de single FFN.
- **Tests:** Verificar que router distribuye tokens entre expertos, que la salida combinada es correcta, que no hay dispersión de memoria

**Ubicación en árbol:** `src/zarch/mixers/moe.zig` (siguiendo docs/09-estructura-tooling.md §1)

**Consecuencias:** Aumento de capacidad del modelo sin aumento proporcional de FLOPs (sólo top-k expertos activos por token). Incremento lineal de footprint en parámetros y memoria.

## 3. Segundo FEATURE: Mixers subcuadráticos reales

**Contexto (docs/03-nucleo-subcuadratico.md §4, §6):** El plan detalla los mixers a implementar. Actualmente Mamba-2, Gated-DeltaNet y RWKV-7 son passthrough stubs sin pesos aprendibles. El orden de implementación recomendado (§4, último párrafo): GLA (ya hecho en ref) → Mamba-2 → SWA/GQA (existe) → Gated-DeltaNet → RWKV-7 → SparseTopK (I+D).

### 3.1 Gated-DeltaNet (prioridad #2 según §4.2)

**Arquitectura (§4.2, §5):** Atención lineal con regla delta + gating. Kernel chunkwise con representación WY.

**Implementación:**
- Pesos: `W_q`, `W_k`, `W_v`, `W_gate`, `delta_bias`, `beta` (inicializados desde config)
- Estado: `S: [d_state, d_model]f32` (matriz de covarianza), `norm: [d_model]f32`
- `forwardPrefill`: chunkwise con WY representation (docs/03 §4.2, §5 tabla)
- `stepDecode`: Recurrencia delta: `S = decay*S + delta*kron(k, v)`, salida = `S^T q`
- Referencia escalar `*_ref` para validación (Art. V.2, docs/03 §6.4)

### 3.2 RWKV-7 "Goose" (prioridad #3 según §4.3)

**Arquitectura (§4.3):** RNN con decay aprendido + token-shift. Puede reconocer lenguajes regulares completamente.

**Implementación:**
- Pesos: `W_k`, `W_v`, `W_r`, `W_g1`, `W_g2`, `time_decay`, `time_faaaa` (docs/03 §4.3)
- Time-mixing: recurrencia con decay vectorial + LR en contexto
- Channel-mixing: MLP (reusa SwiGLU existente)
- `forwardPrefill`: forma paralelizable (scan chunkwise)
- `stepDecode`: recurrencia O(1) por token

**Tests para ambos mixers:** 
- `forwardPrefill` coincide con `stepDecode` secuencial (docs/03 §6.4)
- Estabilidad numérica en secuencias largas de 4096+ tokens
- Sin leaks de memoria

### 3.3 Mejora de Mamba-2 existente

El stub actual de `mamba2.zig` debe reemplazarse con implementación real:
- Pesos: `W_x`, `W_b`, `W_c`, `W_dt` con init HiPPO para A (docs/03 §4.1, §6.1)
- Kernel SSD con chunkwise scan (docs/03 §5, kernel `scan_chunkwise` ya tiene ref)
- Estado recurrente en f32 por estabilidad (docs/03 §6.3)

## 4. Tercer FEATURE: Cuantización para inferencia

**Contexto (PLAN-MAESTRO §1.5, docs/03 §6.3):** El plan menciona cuantización como entregable de F7. `zload` ya define los tipos Q4_0, Q8_0 pero `tensorTypeSize` devuelve 0 para ellos.

**Implementación:**
- Completar `tensorTypeSize` en `zload` para Q4_0 (6 bytes/bloque), Q8_0 (9 bytes/bloque)
- Implementar dequant: `dequantQ4_0(block, out_f32)` y `dequantQ8_0(block, out_f32)` en `zkernel`
- Implementar `matmul_q4_0` y `matmul_q8_0`: descomprime inline en registros SIMD y hace matmul sin materializar el tensor completo
- Tests: dequant coincide con referencia conocida (formato GGUF estándar), matmul cuantizado coincide con matmul f32 dentro de tolerancia

## 5. Plan de implementación

### Iteración 1: Gates
1. Fix `gate_purity.zig` - for loop syntax
2. Fix `gate_constitution.zig` - for loop syntax
3. Fix `gate_quality.zig` - `.Exited` → `.exited` + for syntax
4. Fix `gate_budgets.zig` - `.of()` → `.limited()` + for syntax
5. Verificar: `zig build gate`

### Iteración 2: MoE
1. Crear `ADR-0011-moe-ffn.md`
2. Implementar `zarch/mixers/moe.zig`: `MoEFFN` struct, `router` linear, expertos SwiGLU
3. Integrar en `zarch.zig`: el bloque elige entre FFN single y MoE según config
4. Tests: router distribuye, forward produce output correcto, backward (cuando exista)
5. Verificar: `zig build gate && zig build test`

### Iteración 3: Mixers
1. Crear `ADR-0012-mixers-subcuadraticos.md`
2. Implementar Gated-DeltaNet: `moe.zig` renombrar a su propio archivo o crear `gdn.zig` real
3. Implementar RWKV-7: `rwkv7.zig` real
4. Reemplazar stub de Mamba-2 con implementación real
5. Tests para cada mixer: prefill = decode secuencial, estabilidad larga secuencia

### Iteración 4: Cuantización
1. Crear `ADR-0013-cuantizacion-inferencia.md`
2. Completar `zload.tensorTypeSize` para Q4_0/Q8_0
3. Implementar dequant en `zkernel`
4. Implementar `matmul_q4_0` y `matmul_q8_0`
5. Tests contra referencia f32

## 6. Dependencias

- Gates: independiente, puede ir primero
- MoE: depende de `zarch` (existe), no depende de mixers ni cuantización
- Mixers: independientes entre sí (cada uno implementa la vtable)
- Cuantización: depende de `zload` (existe) y `zkernel` (existe)

## 7. Verificación (gate-check)

Cada iteración debe pasar:
- `zig build gate` (gates arreglados)
- `zig build test` (sin leaks)
- Mejora métrica de eval (latencia, calidad) sin regresión
- Art. I: 100% Zig, sin nuevas dependencias externas
