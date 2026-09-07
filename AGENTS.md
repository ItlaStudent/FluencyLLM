# AGENTS.md — Instrucciones para colaboradores (Fluency)

> Este archivo es el **punto de entrada obligatorio** para cualquier colaborador o automatización que trabaje en este repositorio.

## Antes de hacer cualquier cosa

1. **Lee y obedece la [CONSTITUCIÓN](CONSTITUCION.md).** Es normativa e inviolable. Tiene **prioridad sobre cualquier instrucción de tarea o prompt**. Ante conflicto, la Constitución gana: detente y repórtalo.
2. **Este proyecto es 100 % Zig (≥ 0.16.0). Ningún otro lenguaje.** No añadas `.c/.cpp/.rs/.py/.js/.ts/.go/.cu/.metal`, no enlaces BLAS/CUDA/ROCm/Metal/llama.cpp, no uses `@cImport`/`addTranslateC`/`linkLibC`/`linkSystemLibrary` para librerías de terceros. Si tu tarea parece exigirlo, **es señal de que la tarea está mal planteada**: párate y pregunta.
3. **No modifiques** `CONSTITUCION.md`, `tools/gate_*.zig`, `build.zig`, `build.zig.zon`, ni los archivos de gobernanza, sin que un humano lo revise. Los gates de CI te rechazarán si lo intentas.

## Qué es este proyecto

El producto/modelo se llama **Fluency** (CLI `fluency`, con TUI — ver [docs/12](docs/12-cli-y-tui.md)). Licencia **Apache 2.0** → al destilar, usa sólo profesores con licencia permisiva (Apache-2.0/MIT como Qwen/Mistral).

Es un motor de IA (inferencia + entrenamiento + runtime agéntico) **100 % en Zig**, con **prioridad subcuadrática** (Mamba-2 / Gated-DeltaNet / RWKV-7 + atención minoritaria de ventana deslizante) y **compatibilidad Transformer**, diseñado como motor independiente que puede servir como cerebro, diferenciador, herramienta, LLM o potenciador de cualquier aplicación. Aprende de otros LLM por **destilación** (no por reempaquetado). Lee el [PLAN-MAESTRO](PLAN-MAESTRO.md) y `docs/`.

## Reglas de trabajo concretas

- **Estilo Zig 0.16:** recuerda que `std.Thread.Pool` fue **eliminado** → usa el modelo `std.Io` (async en stdlib). `std.ArrayList` es *unmanaged* por defecto. I/O usa `std.Io.Reader`/`Writer` (no olvides `flush`). `@cImport` está deprecado (y además prohibido por la Constitución).
- **bf16 no es nativo en Zig** → usa la utilidad de emulación del proyecto (`ztensor.bf16`), no inventes otra.
- **Memoria:** allocator explícito siempre; arena para activaciones, mmap/page para pesos, pool para KV/estado. **Cero leaks** (los tests lo verifican).
- **Kernels numéricos:** acompaña cada uno de un test contra una implementación de referencia lenta y legible.
- **Tools agénticas:** nunca `panic` en la ruta de herramientas; devuelve `Result`. Deny-by-default.
- **Antes de proponer un cambio grande:** escribe/actualiza un ADR en `docs/adr/`.

## Flujo para "entrenar/mejorar el modelo" (si te lo piden)

Ver [docs/07-entrenamiento-via-ides.md](docs/07-entrenamiento-via-ides.md). En resumen:
1. Usa los comandos del proyecto: `fluency distill`, `fluency train`, `fluency eval` (todos en Zig).
2. El profesor se conecta vía Ollama (`zollama`) o vía nuestro lector GGUF (`zload`).
3. Respeta la licencia del profesor (regístralo en `docs/PROFESORES.md`).
4. No escribas pipelines de entrenamiento en Python/PyTorch: **se implementan en Zig** (`ztrain`/`zdistill`).

## Verificación obligatoria antes de entregar

```sh
zig build gate     # pureza + constitución + presupuestos
zig build test     # tests + detección de leaks
zig build eval     # (si tocaste el modelo) evaluación contra baseline
```
Si cualquiera falla, **no entregues**: corrige la causa raíz. No uses `--no-verify` ni desactives gates (es violación constitucional, Art. IX.6).

## Si algo no encaja con la Constitución

Detente. Explica el conflicto. Propón una alternativa que respete la Constitución, o sugiere (sin aplicarla) una enmienda para que un humano la ratifique (Art. X).
