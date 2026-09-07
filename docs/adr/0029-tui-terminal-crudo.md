# ADR-0029: Terminal crudo portable (Windows/Unix) para TUI

## Estado

Aceptado (2026-06-02)

## Contexto

La TUI cell-based de Fluency (doc 12) requiere leer teclas individuales (flechas, Ctrl, Escape) sin esperar Enter, ocultar el eco de caracteres, y conocer el tamaño del terminal. Esto se logra poniendo el terminal en **modo crudo** (raw mode).

Necesitamos una solución que:
- Sea **100 % Zig**, sin librerías C/ncurses/termbox/libvaxis externas (Art. I.3 de la Constitución).
- Funcione en **Windows** (ConPTY/Console API) y **Unix** (termios/ioctl).
- Se restaure automáticamente al salir (incluso en error), para no dejar el terminal inutilizado.
- Sea aislada en `src/platform/` como única pieza dependiente de SO, con tests verificables.

## Decisión

Implementar `src/platform/tty.zig` con una API portable mínima:

- `RawTty.enable(io)` — guarda el modo original y activa raw mode.
- `RawTty.disable(io)` — restaura el modo original.
- `RawTty.deinit(io)` — wrapper seguro para `defer`.
- `size(io)` — devuelve `(rows, cols)` del terminal visible.

### Backend Windows

Usar **exclusivamente** los wrappers de `std.os.windows.CONSOLE.USER_IO` que Zig 0.16 ya expone:

- `GET_MODE` / `SET_MODE` (DWORD) para leer/escribir el modo de la consola.
- `GET_SCREEN_BUFFER_INFO` para obtener `dwWindowSize` (filas × columnas).
- Flags: desactivar `ENABLE_LINE_INPUT` + `ENABLE_ECHO_INPUT` + `ENABLE_PROCESSED_INPUT`; activar `ENABLE_VIRTUAL_TERMINAL_INPUT` (0x0200) para que flechas envíen secuencias VT.

Esto **no usa `extern fn` ni FFI**; invoca `io.operate(.device_io_control → ISSUE_USER_IO)` contra el handle de stdin/stdout del PEB, que `std.Io.File.stdin/stdout()` ya provee.

### Backend Unix (Linux)

Usar `std.posix` (termios) y `std.os.linux` (ioctl):

- `tcgetattr(fd)` / `tcsetattr(fd, .NOW, termios)` para raw mode.
- `ioctl(fd, T.IOCGWINSZ, &winsize)` para tamaño.
- Constantes de `lflag` (`ECHO=8`, `ICANON=2`, `ISIG=1`) y `V.MIN`/`V.TIME` definidas inline para x86_64/aarch64 Linux (garantizado por el target del proyecto).

## Consecuencias

- **Positivas**
  - Cumple pureza 100 % Zig: cero archivos `.c/.h`, cero `extern`, cero librerías terceras.
  - Restauración garantizada vía `defer tty.deinit(io)`.
  - API pequeña y predecible; no arrastra bindings de librería externa.
  - Funciona en Windows Terminal/ConPTY y en cualquier terminal Unix moderno.

- **Negativas**
  - Dos backends que deben mantenerse; cambios en el modelo de consola de Windows o en ioctl de Linux requieren ajustes manuales.
  - `size()` y `enable()` pueden fallar si stdout/stdin no es un terminal (redirección); los callers deben manejar `error.Unexpected` o `error.NotATerminal`.
  - No captura eventos de cambio de tamaño de terminal (SIGWINCH en Unix); eso requeriría un handler de señales adicional.

## Alternativas consideradas

1. **`libvaxis`** (Zig puro, TUI completa): descartada para esta tarea porque es una dependencia externa grande y el objetivo es un primer paso aislado (raw mode). Se puede reconsiderar en ADR futuro para el render completo.
2. **`ncurses` / `termbox`**: descartadas por FFI/C (violación constitucional).
3. **`std.os.windows` syscalls crudos (`NtDeviceIoControlFile`)**: descartada porque `std.os.windows.CONSOLE.USER_IO` ya encapsula el IOCTL necesario con tipos seguros.

## Evidencia

- **Compilación cross-target**: `zig build test` incluye `src/platform/tty.zig` como test root; pasa en native (Windows) y compila para `x86_64-linux-gnu`.
- **Tests**: 3 tests añadidos (round-trip enable/disable, deinit sin crash, size compila); 183 tests totales, sin fugas.
- **Demo en vivo**: `src/platform/demo_raw.zig` ejecutado en consola real muestra `Raw mode ON` y lee teclas sin Enter; en CI/pipe cae a modo buffered con mensaje explicativo.
- **Gate**: `zig build gate` pasa (pureza, constitución, calidad, presupuestos).

## Notas de implementación

- `RawTty` guarda estado por-OS vía `switch (builtin.os.tag)` en un campo `state` opaco.
- Los tests crean un `std.Io.Threaded` temporal para obtener un `Io`; esto es aceptable porque los tests no requieren async real.
- El demo usa `std.Io.File.readStreaming` directamente en lugar de `File.Reader` para evitar complejidad de buffering intermedio.
