# ADR-0022: Inferencia real en libfluency (C-ABI)

## Contexto
`src/lib.zig` exponía la superficie C-ABI de `libfluency` pero `fluency_generate` solo hacía *echo* del prompt con un prefijo fijo. La TAREA 9 de F5 exige conectar la biblioteca dinámica al motor de inferencia real de F1 (Transformer + Tokenizer cargados desde GGUF).

## Decisión
Reescribir `src/lib.zig` para que `fluency_load` parse un archivo GGUF real, dequantize pesos, construya un `zarch.Transformer` y un `ztok.Tokenizer`, y los guarde en un `LibModel` opaco a C. `fluency_generate` tokeniza el prompt, ejecuta prefill y generación greedy (o con temperatura), decodifica tokens vía callback streaming C, y `fluency_embed` promedia los embeddings de los tokens del texto.

## Detalles técnicos
- `fluency_load` usa `std.Io.Threaded` para leer el archivo GGUF del disco, luego `zload.parseGguf`, `buildTokenizer` y `loadGgufTransformer` (lógica replicada de `main.zig` para autonomía del módulo). Los pesos viven en un `ArenaAllocator` almacenado junto al modelo.
- `FluencyModel` es un `extern struct` opaco para C (`magic: u32, _opaque: usize`) que oculta el puntero a `LibModel`.
- `fluency_generate` respeta `FluencyOpts.max_tokens` y `.temperature`; sampling greedy cuando `temp <= 0`.
- `fluency_embed` devuelve el promedio de `token_embedding_table` para los tokens del input.
- `fluency_free` libera el Transformer, Tokenizer, Arena (pesos), datos GGUF y el propio wrapper.

## Tests
- `fluency_load` con archivo inexistente devuelve `null`.
- Modelo mínimo sintético (vocab=4, dim=8, 1 capa) con pesos configurados para que el Transformer genere un token predecible; se verifica que `fluency_generate` produce salida real (no echo) y que `fluency_embed` refleja los embeddings del vocabulario.

## Consecuencias
- `libfluency` ya es consumible desde Node (N-API/ffi), Python (ctypes), C#, etc. para inferencia real.
- El módulo `libfluency_mod` en `build.zig` requiere imports de `zload`, `ztok`, `zarch`, `zkernel`, `ztensor`.
- El test de `lib.zig` en `build.zig` ya no es standalone; necesita un módulo de test con los mismos imports.

## Verificación
`zig build` ✅, `zig build test` 143 tests ✅, `zig build gate` ✅, sin fugas.
