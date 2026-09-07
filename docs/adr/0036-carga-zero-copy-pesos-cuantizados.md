# ADR-0036 — Carga zero-copy de pesos cuantizados (referencia in situ al buffer GGUF)

> Nota: el número **0035** lo ocupa el informe de bloqueo del backend GPU SPIR-V
> (`0035-gpu-spirv-bloqueo-zig-0-16.md`), tema distinto. Este ADR completa el 0034.

## Contexto

Tras ADR-0034 (embedding cuantizada), el pico de RAM con `--quantized` para
Qwen2.5-coder-1.5B-Q4_K_M seguía en **~2.4 GB**, lejos del objetivo de ~1.5 GB y
del consumo de llama.cpp (~1–1.5 GB).

Auditoría (2026-06-06) localizó la causa real: **doble residencia del GGUF**.
`loadGgufTransformerQuantized` (`src/main.zig`) y sus auxiliares `loadQuantEmbedding`
/ `loadQuantWeight` **copiaban** (`@memcpy`) cada tensor cuantizado a un arena
(`wa`), mientras el buffer crudo del archivo (`data`, ~940 MB cargado vía
`readFileAlloc`) seguía vivo durante toda la generación. Resultado: ~940 MB (data)
\+ ~900 MB (copias) + estado ≈ 2.4 GB.

Análisis de tiempos de vida (orden LIFO de `defer` en `runGenerate` y en `serve`):

```
defer alloc.free(data)   // se ejecuta el ÚLTIMO
defer gguf.deinit()
defer arena.deinit()
defer tf.deinit()        // se ejecuta el PRIMERO
```

→ `tf` siempre se destruye **antes** que `gguf`/`data`. Por tanto las copias eran
**redundantes**: el `Transformer` puede referenciar los bytes del GGUF *in situ* sin
riesgo de puntero colgante.

## Decisión

**Carga zero-copy**: `loadQuantEmbedding` y `loadQuantWeight` devuelven slices que
apuntan directamente a `gguf.getTensorData(t)[0..sz]` en lugar de copiar a `wa`.
Solo los metadatos pequeños (`layers`/`fmts`, norms, bias) siguen alojándose en el
arena.

Seguridad de alineación verificada: los blobs antes vivían en `wa.alloc(u8, …)`
(align 1) y ahora en `data` de `readFileAlloc` (align 1); **todos** los kernels de
`zkernel.zig` leen vía `@as(*align(1) const Vec, @ptrCast(...))`, así que no existe
suposición de alineación que el cambio pueda romper.

## Consecuencias

- **RAM medida en vivo** (medición externa `Get-Process fluency | PeakWorkingSet64`,
  `run <Qwen2.5-coder-1.5B-Q4_K_M> "def fib(n):" 16 --quantized`):
  - Antes (ADR-0034, con copias): **~2376 MB**
  - Después (zero-copy): **~1441 MB** → **−935 MB**, por debajo del objetivo 1.5 GB.
- **Corrección byte-idéntica**: salida greedy idéntica antes/después
  (`def1 = lambda x: x + 1 / print(def1(5))`); y `--speculate` (que usa
  `forwardBatch` sobre los blobs referenciados) sigue **bit-idéntico a greedy**
  (prompt `for i in range(10): print(i)`, 20 tokens).
- **Tests/gates**: 241 tests pasan, 4 gates verdes, 0 fugas (los blobs referenciados
  no se liberan en `tf.deinit`; los libera `gguf.deinit`/`free(data)` del caller).
- **Invariante de tiempo de vida** (a respetar): el `Transformer` cuantizado NO debe
  sobrevivir al `gguf`/`data` del que toma prestados los pesos. Hoy se cumple en
  `run` y `serve` (ambos mantienen `data` vivo mientras existe `tf`). Si en el futuro
  el modelo debe persistir más que el archivo, habría que volver a copiar o usar mmap
  con propiedad explícita.

## Alternativas consideradas

- **Liberar `data` tras cargar y quedarse solo con las copias**: vetado — el
  tokenizador y `gguf` referencian `data`; liberarlo antes rompería esas referencias.
- **mmap del archivo (en vez de `readFileAlloc` + referencia)**: deseable a futuro
  (alinea con docs/02 "pesos zero-copy vía mmap"), reduciría también la residencia de
  `data` a páginas bajo demanda. Requiere abstracción de mmap 100 % Zig; trabajo
  independiente, no bloqueante.

## Relación

- Completa ADR-0034 (corrige su atribución errónea del residual al KV-cache).
- Avanza el objetivo de footprint mínimo de docs/02 (pesos zero-copy).

## Artículo afectado

Art. I (100 % Zig, sin FFI — `readFileAlloc` + slices nativos), Art. V (validado en
vivo y contra el camino previo), Art. VI (allocator explícito, cero fugas).
