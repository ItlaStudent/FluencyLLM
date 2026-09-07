# ADR-0033 — MoE soft (diferenciable) 100% Zig para entrenamiento

## Contexto

El proyecto necesita escalar la capacidad del modelo sin incrementar linealmente el costo computacional de inferencia. Mixture of Experts (MoE) permite activar solo un subconjunto de parámetros por token (sparsidad), manteniendo la capacidad total del modelo. La infraestructura de inferencia en `src/zarch/zarch.zig` ya tiene `forwardMoE` con router top-k y expertos, pero este forward usa **hard top-k** (no diferenciable) y no tiene gradiente.

Para entrenar un modelo MoE en el motor de aprendizaje F3 (`zdistill`), se necesita:
1. Un forward **diferenciable** (soft top-k o full softmax) para que el router aprenda.
2. VJP (vector-Jacobian product) verificado por gradient-check numérico.
3. Integración en `FluencyNano` como variante entrenable del FFN.

## Decisión

1. **Op `moe` en `zgrad.Graph`** (`src/zgrad/zgrad.zig`):
   - Forward: para cada token del batch `[T, D]`, calcula `router_logits = x @ W_router`, aplica softmax, y combina la salida de **todos los expertos** ponderados por probabilidad (soft MoE).
   - Cada experto es un FFN estándar `SiLU(x @ W1) * (x @ W3) @ W2`.
   - Guarda en `saved`: `probs`, `h_silu`, `h_gate`, `o_exp` por token (layout `T * block`).
   - Backward (`bwdMoe`): propaga gradientes a través del router (softmax backward) y de cada experto (SiLU, matmul), acumulando en `dW1`, `dW3`, `dW2`, `dRouter`, `dX`.

2. **Integración en `FluencyNano`** (`src/zdistill/zdistill.zig`):
   - `NanoConfig` añade `n_experts`, `top_k`, `moe_ff_dim` (default 0, 2, 0).
   - `LayerOff` añade offsets `moe_router`, `moe_w1`, `moe_w3`, `moe_w2`.
   - `init` calcula offsets condicionalmente: si `n_experts > 0`, reserva espacio para MoE; si no, mantiene FFN denso.
   - `forwardWithHidden`: cuando `n_experts > 0`, usa `g.moe(h2, router, w1, w3, w2, E, M)` en lugar del FFN denso.
   - `initWeights`: router ~ N(0, 0.01) (bajos logits iniciales para distribución uniforme); expertos con escalas `sd`/`sf` estándar.

3. **Tests**:
   - `gradcheck moe (soft) backward vs numérico` en `zgrad.zig`: compara gradiente analítico vs diferencias finitas (tolerancia 5e-2).
   - `FluencyNano MoE forward/backward` en `zdistill.zig`: verifica forma de logits correcta, backward sin pánico, gradientes no nulos.
   - `FluencyNano MoE aprende`: entrenamiento de 300 pasos sobre secuencia fija, verifica CE < 0.5× inicial y accuracy >= 0.75.

## Consecuencias

- **Entrenamiento diferenciable**: el router aprende end-to-end a través de la pérdida de entropía cruzada, sin necesidad de load balancing loss adicional (aunque puede añadirse en iteraciones futuras).
- **Compatibilidad con inferencia hard top-k**: los pesos entrenados con `moe` soft pueden cargarse en `forwardMoE` de `zarch.zig` para inferencia con hard top-k (el output será muy similar porque el entrenamiento soft empuja al router a concentrarse en pocos expertos).
- **Parámetros**: el test usa `n_experts=4, moe_ff_dim=12` con `dim=16`, un modelo pequeño pero representativo.
- **Limitación honesta**: el op `moe` usa **softmax completo** sobre todos los expertos, no top-k soft. Esto es ligeramente más costoso que top-k hard en inferencia, pero es necesario para diferenciabilidad. En inferencia se usa `forwardMoE` de `zarch.zig` con hard top-k para eficiencia.
- **No hay regresión**: los tests de FFN denso existentes siguen pasando; MoE es una variante opcional (`n_experts > 0`).

## Alternativas consideradas

- **STE (Straight-Through Estimator) sobre hard top-k**: más simple (no requiere cambiar el forward), pero el router no recibe señal de gradiente informativa → aprendizaje deficiente.
- **Load balancing loss auxiliar**: estándar en Mixtral/GShard, pero añade complejidad al trainer. Se deja para iteración futura si el balance de carga entre expertos es problemático.
- **Modelo borrador externo para MoE**: vetado por la Constitución (100% Zig, sin binarios ajenos).

## Artículo afectado

Art. I (pureza: 100% Zig), Art. V (calidad: tests sustantivos, gradient-check).
