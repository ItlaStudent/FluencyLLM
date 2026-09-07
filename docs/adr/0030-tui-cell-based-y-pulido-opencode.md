# ADR-0030: TUI cell-based propia (100% Zig) + pulido de TUI

## Estado

Aceptado (2026-06-02)

## Contexto

Sobre el terminal crudo de [ADR-0029](0029-tui-terminal-crudo.md), la doc 12 pide una TUI **cell-based** moderna: header, transcript scrollable, input multilínea, status line, render Markdown, tool-calls plegables, diffs aprobables por tecla, paleta de comandos y temas. El [ADR-0010] dejó pendiente decidir entre **TUI propia mínima** vs adoptar **`libvaxis`** (librería TUI escrita en Zig).

El desarrollo anterior implementó el esqueleto del motor (`engine.zig` + `screen.zig` + `transcript.zig`) pero quedó **sin compilar** y con varios bugs; esta ADR registra la decisión y el cierre.

## Decisión

**TUI propia mínima 100% Zig — NO se adopta `libvaxis`.**

Razones:
- **Pureza / cero dependencias** (Art. I.2): aunque `libvaxis` es Zig puro y sería admisible vía ADR, añadir una dependencia externa contradice el principio de cero-deps del proyecto cuando la superficie que necesitamos (rejilla de celdas + diff ANSI + layout + input) es pequeña y ya está escrita.
- **Control total** del render y del modelo de eventos, sin acoplarnos al ciclo de vida de una librería externa ni a su versión de Zig.

### Arquitectura

```
src/platform/tty.zig     RawTty (raw mode) + size()           ← ADR-0029
src/ztui/screen.zig      Screen (rejilla de Cell), render diff ANSI truecolor,
                         Theme (dark/light), Layout
src/ztui/transcript.zig  Transcript: entradas con rol, word-wrap, scroll,
                         render Markdown (negritas/code/headings/bullets/fences),
                         diff coloreado (en tool-calls), plegado
src/ztui/engine.zig      Tui: compose()/paint(), event loop de teclado, slash
                         commands, aprobación, sesiones, foco, paleta, temas
src/ztui/preview.zig     volcado no-interactivo del layout (dev)
```

- **`compose()` vs `paint()`**: `compose()` rellena la rejilla **sin IO** (testeable); `paint()` = `compose()` + `screen.render()` a stdout. Esto evita el bug donde escribir ANSI a `std.Io.File.stdout()` durante un test corrompía el canal IPC del test-runner (`--listen=-`) y lo colgaba.
- **Render diff**: `Screen.render` sólo emite secuencias ANSI para las celdas que cambiaron respecto al frame anterior (`prev`), minimizando el tráfico.
- **Buffer de pantalla alterno** (`\x1b[?1049h`/`l`): la TUI entra en pantalla alterna como vim y restaura el shell intacto al salir.

### Pulido de TUI

- **Markdown** inline por celda: `**negrita**`, `` `código` ``, headings `#/##/###`, bullets `- `/`* ` → `•`, bloques ` ``` ` con estilo de código.
- **Tool-calls plegables**: `Entry.folded`; navegar con **Ctrl-F** (enfocar siguiente tool-call) y plegar/desplegar con **Ctrl-O**.
- **Diffs aprobables**: en el flujo de aprobación se muestra la tool-call/diff (líneas `+/-` coloreadas) y se acepta/rechaza con **`a`/`r`** (también `y`/`n`, Esc=rechazar), cableado al `approval_fn` del agente.
- **Paleta de comandos** (**Ctrl-P**): overlay con tema, plegado, foco, limpiar memoria, guardar sesión y salir; navegación con **Tab**, ejecución con **Enter**, cierre con **Esc**.
- **Temas** claro/oscuro (`Theme.dark()/light()`), alternables con **Ctrl-T** o desde la paleta.

### Estética de TUI

- **Pantalla de bienvenida** centrada cuando la sesión está vacía: logo en bloque `█` "FLUENCY", tagline y hints (igual espíritu que una tarjeta central de bienvenida).
- **Caja de input** con borde redondeado (`╭─╮│╰─╯`), prompt `›` y placeholder; cursor `▏` tras el texto.
- **Barra de estado** con segmentos a la izquierda y `Fluency vX` a la derecha (sin solaparse).
- Helper `dispWidth` (cuenta **codepoints**, no bytes) para centrar correctamente texto con glifos multibyte.

### Fixes críticos de visualización (Windows)

1. El render se veía como texto crudo (`←[38;2;…m`) porque `enable()` sólo activaba VT en **stdin**. Se añadió **`ENABLE_VIRTUAL_TERMINAL_PROCESSING` en stdout** (best-effort, restaurado al salir).
2. Los glifos Unicode (`▸ ▾ ⚙ • █ ╭─│`) salían corruptos (`▾`→`Ô…`) porque la consola interpretaba cada byte UTF-8 como un carácter **CP850**. Se fuerza la **code page de salida a UTF-8 (65001)** vía `std.os.windows.CONSOLE.USER_IO.SET_CP(.Output, 65001)` (mismo wrapper de `std` que `SET_MODE`, **sin `extern` en nuestro código**), guardando y restaurando la CP original.

## Consecuencias

- **Positivas**: 100% Zig, cero deps; control total; render eficiente (diff); experiencia de TUI moderna (markdown, diffs por tecla, paleta, temas); se restaura el shell limpio.
- **Negativas**: mantenemos el motor de render nosotros; sin soporte de SIGWINCH (resize en caliente) todavía; el resaltado de sintaxis de bloques de código es plano (un único color), sin tokenizar por lenguaje.

## Alternativas consideradas

1. **`libvaxis`** (Zig puro): descartada por el principio cero-deps; reconsiderable si la TUI creciera mucho.
2. **ncurses/termbox (C)**: prohibidas por FFI (Art. I).
3. **Mantener el REPL de líneas**: descartado; no cumple la doc 12.

## Evidencia

- `zig build` ✅ · `zig build test --summary all` ✅ **203 tests, sin fugas** · `zig build gate` ✅ (4 gates).
- Tests sustantivos nuevos: clasificación Markdown vs diff, `paintInline` (negrita/código no desbordan ni imprimen marcadores), code-fence marca líneas, entrada plegada produce resumen, toggle de tema, foco+plegado de tool-calls, paleta abre/mueve/cierra + `compose` con overlay, `setStatusFmt` sin puntero colgante.
- **Vista previa en vivo**: `zig run src/ztui/preview.zig` vuelca el layout (header + transcript con roles, headings, bullets, bloque de código, diff en tool-call) en texto plano.
- **Binario real**: `fluency chat` arranca la TUI (pantalla alterna + VT processing) y la restaura al salir.

## Notas

- Bugs reparados al cerrar la TAREA 3 (el desarrollo anterior se quedó sin tokens): no compilaba (`catch` de `size()` sin tipo, `try` sobre `void`); modo crudo no best-effort; cuelgue del test-runner por ANSI a stdout; tests del motor que no corrían (faltaba `engine.zig` como test root); VT processing de salida ausente; status con puntero colgante a buffer de pila.
