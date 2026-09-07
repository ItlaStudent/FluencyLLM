# ADR-0023: Target WASM (`fluency.wasm`) + bindings JS mínimos

## Contexto
F5 requiere que `libfluency` sea consumible no solo vía C-ABI desde procesos nativos, sino también desde el navegador vía WebAssembly. Zig soporta `wasm32-freestanding` nativamente.

## Decisión
1. Añadir un target `wasm32-freestanding` en `build.zig` compilando `src/lib.zig` como biblioteca **estática** (`.static`; `.dynamic` genera errores de relocation en `wasm-ld`).
2. Adaptar `src/lib.zig`:
   - `fluency_load` protegido con `comptime` para devolver `null` en wasm (no hay `std.Io` ni sistema de archivos).
   - `fluency_load_bytes(data, len)` para cargar un GGUF desde bytes en memoria (el host JS fetcha el modelo y lo pasa al wasm).
   - `fluency_generate_buf(prompt, max_tokens, temp, out_buf, out_cap)` para evitar callbacks C complejos en WASM; escribe todo el texto generado en un buffer.
   - `fluency_alloc(len)` / `fluency_free_ptr(ptr, len)` para que JS reserve/libere memoria en el heap de wasm.
3. Convertir `src/zload/zload.zig` a portable 32-bit: todo `u64` leído del formato GGUF que se usa como índice o tamaño de array (`usize`) pasa por `std.math.cast(usize, v) orelse error.Overflow`. Esto incluye `tensor_count`, `metadata_kv_count`, `key_len`, `name_len`, `dims`, `offset`, etc.
4. Crear `js/fluency.js`: wrapper JS que instancia el wasm, copia datos al heap, invoca los exports de Zig, y decodifica resultados.
5. Crear `js/index.html`: demo mínimo con `<input type="file">` para cargar un GGUF, `<textarea>` para prompt, y botones para generar/embed.

## Consecuencias
- El artefacto `fluency.wasm` (~591KB en debug) se genera automáticamente con `zig build`.
- `zload.zig` ahora es portable a 32-bit sin cambios semánticos (ningún GGUF real excede 2^32 en sus conteos).
- Los bindings JS no usan WASI ni glue Emscripten; son puro WASM + JS.

## Verificación
`zig build` ✅, `zig build test` 143 tests ✅, `zig build gate` ✅, sin fugas.
