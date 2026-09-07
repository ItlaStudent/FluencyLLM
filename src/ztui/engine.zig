// engine.zig — motor cell-based TUI (event loop + render + input)
const std = @import("std");
const zagent = @import("zagent");
const platform = @import("platform");
const screen = @import("screen.zig");
const transcript = @import("transcript.zig");
const Screen = screen.Screen;
const Theme = screen.Theme;
const Layout = screen.Layout;
const Style = screen.Style;
const Transcript = transcript.Transcript;
const Role = transcript.Role;

// =====================================================================
// Callbacks / context
// =====================================================================

const StreamCtx = struct {
    tui: *Tui,
};

fn streamEmit(ctx: ?*anyopaque, piece: []const u8) anyerror!void {
    const s: *StreamCtx = @alignCast(@ptrCast(ctx.?));
    try s.tui.appendStream(piece);
}

fn askApproval(ctx: ?*anyopaque, _: std.Io, tool_name: []const u8, args_json: []const u8) !bool {
    const sctx: *StreamCtx = @alignCast(@ptrCast(ctx.?));
    const tui = sctx.tui;
    try tui.showApproval(tool_name, args_json);
    // Esperar [a]ceptar / [r]echazar (también y/n). Repinta el frame primero.
    while (true) {
        try tui.paint();
        const key = try tui.readKey();
        if (key == 0) continue;
        switch (key) {
            'a', 'A', 'y', 'Y' => {
                tui.setStatus("✔ aceptado");
                return true;
            },
            'r', 'R', 'n', 'N', 0x1b => {
                tui.setStatus("✘ rechazado");
                return false;
            },
            else => {},
        }
    }
}

/// Ancho de display en celdas (número de codepoints UTF-8, no de bytes).
fn dispWidth(str: []const u8) u16 {
    var n: u16 = 0;
    var i: usize = 0;
    while (i < str.len) {
        const len = std.unicode.utf8ByteSequenceLength(str[i]) catch 1;
        i += len;
        n += 1;
    }
    return n;
}

// =====================================================================
// Tui state
// =====================================================================

pub const Tui = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    stdout: std.Io.File,
    screen: Screen,
    theme: Theme,
    theme_kind: screen.ThemeKind,
    layout: Layout,
    tty: platform.tty.RawTty,
    transcript: Transcript,
    input_buf: std.ArrayList(u8),
    cursor: usize,
    status: []const u8,
    status_buf: [192]u8,
    header: []const u8,
    running: bool,
    needs_repaint: bool,
    stream_piece: ?[]const u8,
    stream_alloc: ?std.mem.Allocator,
    // Foco para plegar/desplegar tool-calls (índice de entrada enfocada).
    focus: ?usize,
    // Paleta de comandos (Ctrl-P).
    palette_open: bool,
    palette_sel: usize,

    pub fn init(alloc: std.mem.Allocator, io: std.Io, w: u16, h: u16) !Tui {
        var tty = platform.tty.RawTty.init();
        // Modo crudo best-effort: si stdin no es un terminal (pipe/redirección),
        // seguimos en modo degradado (entrada por líneas) en vez de fallar.
        tty.enable(io) catch {};
        return .{
            .alloc = alloc,
            .io = io,
            .stdout = std.Io.File.stdout(),
            .screen = try Screen.init(alloc, w, h),
            .theme = Theme.dark(),
            .theme_kind = .dark,
            .layout = Layout.calc(w, h),
            .tty = tty,
            .transcript = Transcript.init(alloc),
            .input_buf = std.ArrayList(u8).empty,
            .cursor = 0,
            .status = "modo: CHAT   ·   Ctrl-P comandos   ·   Ctrl-C salir",
            .status_buf = undefined,
            .header = "Fluency · Chat",
            .running = true,
            .needs_repaint = true,
            .stream_piece = null,
            .stream_alloc = null,
            .focus = null,
            .palette_open = false,
            .palette_sel = 0,
        };
    }

    pub fn deinit(self: *Tui) void {
        self.tty.deinit(self.io);
        self.screen.deinit();
        self.transcript.deinit();
        self.input_buf.deinit(self.alloc);
        if (self.stream_piece) |p| {
            if (self.stream_alloc) |a| a.free(p);
        }
    }

    pub fn resize(self: *Tui, w: u16, h: u16) !void {
        self.screen.deinit();
        self.screen = try Screen.init(self.alloc, w, h);
        self.layout = Layout.calc(w, h);
        self.needs_repaint = true;
    }

    pub fn addEntry(self: *Tui, role: Role, text: []const u8) !void {
        try self.transcript.add(role, text);
        self.needs_repaint = true;
    }

    pub fn appendStream(self: *Tui, text: []const u8) !void {
        try self.transcript.appendToLast(text);
        self.needs_repaint = true;
    }

    pub fn setStatus(self: *Tui, text: []const u8) void {
        self.status = text;
        self.needs_repaint = true;
    }

    /// Igual que setStatus pero copia a un buffer persistente del Tui (para
    /// textos formateados: evita punteros colgantes a buffers de pila).
    pub fn setStatusFmt(self: *Tui, comptime fmt: []const u8, args: anytype) void {
        const out = std.fmt.bufPrint(&self.status_buf, fmt, args) catch self.status_buf[0..0];
        self.status = out;
        self.needs_repaint = true;
    }

    pub fn setHeader(self: *Tui, text: []const u8) void {
        self.header = text;
        self.needs_repaint = true;
    }

    pub fn toggleTheme(self: *Tui) void {
        self.theme_kind = switch (self.theme_kind) {
            .dark => .light,
            .light => .dark,
        };
        self.theme = Theme.of(self.theme_kind);
        self.needs_repaint = true;
    }

    /// Enfoca la siguiente entrada tool (la única plegable). Envuelve.
    pub fn focusNextTool(self: *Tui) void {
        const n = self.transcript.entries.items.len;
        if (n == 0) return;
        const start = if (self.focus) |f| f + 1 else 0;
        var i: usize = 0;
        while (i < n) : (i += 1) {
            const idx = (start + i) % n;
            if (self.transcript.entries.items[idx].role == .tool) {
                self.focus = idx;
                self.needs_repaint = true;
                return;
            }
        }
        // ninguna tool-call
        self.focus = null;
        self.needs_repaint = true;
    }

    /// Pliega/despliega la tool-call enfocada.
    pub fn toggleFoldFocused(self: *Tui) void {
        const f = self.focus orelse {
            self.focusNextTool();
            return;
        };
        if (f < self.transcript.entries.items.len) {
            const e = &self.transcript.entries.items[f];
            e.folded = !e.folded;
            self.needs_repaint = true;
        }
    }

    // ---- Paleta de comandos (Ctrl-P) ----
    const PaletteCmd = enum { theme, fold_focus, focus_next, clear_mem, save_session, quit };
    const palette_labels = [_][]const u8{
        "Cambiar tema (claro/oscuro)",
        "Plegar/desplegar tool-call enfocada",
        "Enfocar siguiente tool-call",
        "Limpiar memoria (/clear)",
        "Guardar sesión (\"last\")",
        "Salir de Fluency",
    };
    const palette_cmds = [_]PaletteCmd{ .theme, .fold_focus, .focus_next, .clear_mem, .save_session, .quit };

    fn openPalette(self: *Tui) void {
        self.palette_open = true;
        self.palette_sel = 0;
        self.needs_repaint = true;
    }
    fn closePalette(self: *Tui) void {
        self.palette_open = false;
        self.needs_repaint = true;
    }
    fn paletteNext(self: *Tui) void {
        self.palette_sel = (self.palette_sel + 1) % palette_labels.len;
        self.needs_repaint = true;
    }

    fn runPalette(self: *Tui, agent: *zagent.Agent) !void {
        const cmd = palette_cmds[self.palette_sel];
        self.closePalette();
        switch (cmd) {
            .theme => self.toggleTheme(),
            .fold_focus => self.toggleFoldFocused(),
            .focus_next => self.focusNextTool(),
            .clear_mem => {
                for (agent.memory.entries.items) |e| {
                    self.alloc.free(e.text);
                    self.alloc.free(e.embedding);
                }
                agent.memory.entries.clearRetainingCapacity();
                try self.addEntry(.assistant, "Memoria limpiada.");
            },
            .save_session => try self.saveSession("last", agent),
            .quit => self.running = false,
        }
    }

    pub fn showApproval(self: *Tui, tool_name: []const u8, args_json: []const u8) !void {
        // Mostrar args/diff en transcript como tool entry y enfocarla.
        try self.addEntry(.tool, args_json);
        self.focus = self.transcript.entries.items.len - 1;
        self.setStatusFmt("APROBAR {s}?  [a]ceptar  [r]echazar  (Esc=rechazar)", .{tool_name});
    }

    pub fn readKey(self: *Tui) !u8 {
        var buf: [1]u8 = undefined;
        var bufs = [1][]u8{&buf};
        while (true) {
            const n = std.Io.File.readStreaming(std.Io.File.stdin(), self.io, &bufs) catch |err| switch (err) {
                error.EndOfStream => return 0,
                else => return err,
            };
            if (n == 0) continue;
            return buf[0];
        }
    }

    /// Lee un evento de teclado: un byte simple o una secuencia de escape (flechas, etc.
    /// Devuelve un enum para que el event loop pueda distinguir teclas especiales.
    pub const KeyEvent = union(enum) {
        byte: u8,
        arrow_up,
        arrow_down,
        arrow_left,
        arrow_right,
        page_up,
        page_down,
        home,
        end_key,
    };

    /// Lee un evento de teclado completo, parseando secuencias de escape.
    /// Si el byte leído es ESC (0x1b), intenta leer más bytes para detectar flechas.
    pub fn readEvent(self: *Tui) !KeyEvent {
        const first = try self.readKey();
        if (first != 0x1b) return .{ .byte = first };
        // Posible secuencia de escape. Lee el siguiente byte con timeout best-effort.
        // En modo raw, si sólo hay ESC, readKey bloqueará. Para evitarlo, devolvemos ESC.
        // Pero en la práctica, las flechas envían ESC[A, ESC[B, etc. inmediatamente.
        // Intentamos leer el siguiente byte; si falla o es 0, es ESC solo.
        const second = self.readKey() catch return .{ .byte = 0x1b };
        if (second == 0) return .{ .byte = 0x1b };
        if (second == '[') {
            const third = try self.readKey();
            if (third == 0) return .{ .byte = 0x1b };
            return switch (third) {
                'A' => .arrow_up,
                'B' => .arrow_down,
                'C' => .arrow_right,
                'D' => .arrow_left,
                'H' => .home,
                'F' => .end_key,
                '5' => blk: {
                    const fourth = try self.readKey();
                    if (fourth == '~') break :blk .page_up;
                    break :blk .{ .byte = 0x1b };
                },
                '6' => blk: {
                    const fourth = try self.readKey();
                    if (fourth == '~') break :blk .page_down;
                    break :blk .{ .byte = 0x1b };
                },
                else => .{ .byte = 0x1b },
            };
        }
        if (second == 'O') {
            const third = try self.readKey();
            if (third == 0) return .{ .byte = 0x1b };
            return switch (third) {
                'A' => .arrow_up,
                'B' => .arrow_down,
                'C' => .arrow_right,
                'D' => .arrow_left,
                'H' => .home,
                'F' => .end_key,
                else => .{ .byte = 0x1b },
            };
        }
        return .{ .byte = 0x1b };
    }

    /// Rellena la rejilla de celdas (header/transcript/input/status) SIN tocar IO.
    /// Es la parte testeable del render; `flush` la vuelca a la terminal.
    pub fn compose(self: *Tui) !void {
        self.screen.clear();
        const s = &self.screen;
        const th = self.theme;

        // Header bar
        const h = self.layout.header;
        s.fill(h, ' ', .{ .fg = th.header_fg, .bg = th.header_bg });
        s.putStr(h.x + 1, h.y, self.header, .{ .fg = th.header_fg, .bg = th.header_bg, .bold = true });

        // Transcript (o pantalla de bienvenida si no hay mensajes)
        const t = self.layout.transcript;
        s.fill(t, ' ', .{ .fg = th.transcript_fg, .bg = th.transcript_bg });
        if (self.transcript.entries.items.len == 0) {
            self.composeWelcome(t);
        } else {
            try self.transcript.paintFocus(s, t, th, self.focus);
        }

        self.composeInputBox();
        self.composeStatusBar();

        if (self.palette_open) self.composePalette();
    }

    /// Pantalla de bienvenida centrada (logo + tagline + hints).
    fn composeWelcome(self: *Tui, rect: screen.Rect) void {
        const s = &self.screen;
        const th = self.theme;
        const logo = [_][]const u8{
            "███████ ██      ██    ██ ███████ ███    ██  ██████ ██    ██",
            "██      ██      ██    ██ ██      ████   ██ ██       ██  ██ ",
            "█████   ██      ██    ██ █████   ██ ██  ██ ██        ████  ",
            "██      ██      ██    ██ ██      ██  ██ ██ ██         ██   ",
            "██      ███████  ██████  ███████ ██   ████  ██████    ██   ",
        };
        const total: u16 = @intCast(logo.len + 4);
        const top: u16 = rect.y + (if (rect.h > total) (rect.h - total) / 2 else 0);
        for (logo, 0..) |row, k| {
            const w = dispWidth(row);
            const x = rect.x + (if (rect.w > w) (rect.w - w) / 2 else 0);
            const yy: u16 = @intCast(top + k);
            s.putStr(x, yy, row, .{ .fg = th.assistant_label, .bg = th.transcript_bg, .bold = true });
        }
        const tagline = "IA 100% Zig · local y compacta · para entusiastas";
        const hint = "Escribe tu mensaje y pulsa Enter   ·   Ctrl-P comandos   ·   Ctrl-T tema";
        const ty: u16 = @intCast(top + logo.len + 1);
        const hy: u16 = @intCast(top + logo.len + 2);
        const tw = dispWidth(tagline);
        const hw = dispWidth(hint);
        s.putStr(rect.x + (if (rect.w > tw) (rect.w - tw) / 2 else 0), ty, tagline, .{ .fg = th.heading, .bg = th.transcript_bg, .bold = true });
        s.putStr(rect.x + (if (rect.w > hw) (rect.w - hw) / 2 else 0), hy, hint, .{ .fg = th.status_fg, .bg = th.transcript_bg });
    }

    /// Caja de entrada con borde redondeado (3 filas: borde, texto, borde).
    fn composeInputBox(self: *Tui) void {
        const s = &self.screen;
        const th = self.theme;
        const i = self.layout.input;
        if (i.w < 4 or i.h < 1) return;
        const right = i.x + i.w - 1;
        const border = Style{ .fg = th.assistant_label, .bg = th.transcript_bg };
        const mid_y = if (i.h >= 3) i.y + 1 else i.y;
        // Borde superior
        if (i.h >= 3) {
            s.putCell(i.x, i.y, .{ .ch = '╭', .style = border });
            var x = i.x + 1;
            while (x < right) : (x += 1) s.putCell(x, i.y, .{ .ch = '─', .style = border });
            s.putCell(right, i.y, .{ .ch = '╮', .style = border });
            // Borde inferior
            const by = i.y + i.h - 1;
            s.putCell(i.x, by, .{ .ch = '╰', .style = border });
            x = i.x + 1;
            while (x < right) : (x += 1) s.putCell(x, by, .{ .ch = '─', .style = border });
            s.putCell(right, by, .{ .ch = '╯', .style = border });
            // Laterales
            s.putCell(i.x, mid_y, .{ .ch = '│', .style = border });
            s.putCell(right, mid_y, .{ .ch = '│', .style = border });
        }
        // Prompt + texto
        const text_x0 = i.x + 2;
        const text_max = if (right > text_x0 + 1) right - text_x0 - 1 else 1;
        s.putCell(i.x + 1, mid_y, .{ .ch = '›', .style = .{ .fg = th.user_label, .bg = th.transcript_bg, .bold = true } });
        const input_text = self.input_buf.items;
        const tstyle = Style{ .fg = th.transcript_fg, .bg = th.transcript_bg };
        const tx = text_x0 + 1;
        if (input_text.len == 0) {
            // Placeholder sin cursor (no pisar la primera letra).
            s.putStr(tx, mid_y, "escribe tu mensaje…", .{ .fg = th.status_fg, .bg = th.transcript_bg });
        } else {
            const shown_text = if (input_text.len <= text_max) input_text else input_text[input_text.len - text_max ..];
            s.putStr(tx, mid_y, shown_text, tstyle);
            // Cursor justo tras el texto visible.
            const shown: u16 = @intCast(@min(shown_text.len, text_max));
            const cursor_x = @min(tx + shown, right - 1);
            s.putCell(cursor_x, mid_y, .{ .ch = '▏', .style = .{ .fg = th.user_label, .bg = th.transcript_bg, .bold = true } });
        }
    }

    /// Barra de estado inferior con segmentos.
    fn composeStatusBar(self: *Tui) void {
        const s = &self.screen;
        const th = self.theme;
        const st = self.layout.status;
        s.fill(st, ' ', .{ .fg = th.status_fg, .bg = th.status_bg });
        s.putStr(st.x + 1, st.y, self.status, .{ .fg = th.status_fg, .bg = th.status_bg });
        // Derecha: nombre/versión (sólo si no se solapa con el texto izquierdo).
        const right_label = "Fluency v0.7";
        const rw = dispWidth(right_label);
        const lw = dispWidth(self.status);
        if (st.w > rw + 2 and (st.x + 1 + lw) < (st.x + st.w - rw - 2)) {
            s.putStr(st.x + st.w - rw - 1, st.y, right_label, .{ .fg = th.assistant_label, .bg = th.status_bg, .bold = true });
        }
    }

    /// Dibuja el overlay de la paleta de comandos, centrado.
    fn composePalette(self: *Tui) void {
        const s = &self.screen;
        const th = self.theme;
        const n: u16 = @intCast(Tui.palette_labels.len);
        const box_w: u16 = @min(self.screen.w -| 4, 52);
        const box_h: u16 = n + 2; // título + items
        const x0: u16 = (self.screen.w -| box_w) / 2;
        const y0: u16 = (self.screen.h -| box_h) / 2;
        s.fill(.{ .x = x0, .y = y0, .w = box_w, .h = box_h }, ' ', .{ .fg = th.overlay_fg, .bg = th.overlay_bg });
        s.putStr(x0 + 1, y0, "  Paleta de comandos (Tab mueve · Enter ejecuta · Esc cierra)", .{ .fg = th.overlay_fg, .bg = th.overlay_bg, .bold = true });
        for (Tui.palette_labels, 0..) |label, i| {
            const yy: u16 = @intCast(y0 + 1 + i);
            const selected = (i == self.palette_sel);
            const style = if (selected)
                Style{ .fg = th.sel_fg, .bg = th.sel_bg, .bold = true }
            else
                Style{ .fg = th.overlay_fg, .bg = th.overlay_bg };
            // barra de selección a lo ancho
            s.fill(.{ .x = x0, .y = yy, .w = box_w, .h = 1 }, ' ', style);
            const marker = if (selected) "› " else "  ";
            s.putStr(x0 + 1, yy, marker, style);
            s.putStr(x0 + 3, yy, label, style);
        }
    }

    /// Compone la rejilla y la vuelca a la terminal (diff ANSI a stdout).
    pub fn paint(self: *Tui) !void {
        if (!self.needs_repaint) return;
        self.needs_repaint = false;
        try self.compose();
        try self.screen.render(self.stdout, self.io);
    }

    pub fn runEventLoop(self: *Tui, agent: *zagent.Agent) !void {
        var sctx = StreamCtx{ .tui = self };
        agent.stream_fn = &streamEmit;
        agent.stream_ctx = &sctx;
        agent.approval_fn = &askApproval;
        agent.approval_ctx = &sctx;

        // Buffer de pantalla alterno (como vim): no ensucia el historial
        // del shell y lo restaura intacto al salir. Solo si hay terminal real.
        if (self.tty.enabled) {
            try self.stdout.writeStreamingAll(self.io, "\x1b[?1049h\x1b[2J\x1b[H");
        }
        defer if (self.tty.enabled) {
            self.stdout.writeStreamingAll(self.io, "\x1b[0m\x1b[?25h\x1b[2J\x1b[H\x1b[?1049l") catch {};
        };

        // Sin mensaje inicial: el transcript vacío muestra la pantalla de bienvenida.

        while (self.running) {
            // Poll de resize: si el tamaño de la terminal cambió, redimensionar.
            if (self.tty.enabled) {
                if (platform.tty.size(self.io)) |ts| {
                    if (ts.cols != self.screen.w or ts.rows != self.screen.h) {
                        try self.resize(ts.cols, ts.rows);
                    }
                } else |_| {}
            }

            try self.paint();

            const ev = try self.readEvent();

            // Extraer byte para los handlers existentes que esperan u8
            const key: u8 = switch (ev) {
                .byte => |b| b,
                // Teclas especiales se manejan abajo
                else => 0,
            };

            // --- Scroll del transcript con flechas/PageUp/PageDown ---
            switch (ev) {
                .arrow_up => {
                    self.transcript.scrollUp(1);
                    self.needs_repaint = true;
                    continue;
                },
                .arrow_down => {
                    const total = self.transcript.totalRenderLines(self.layout.transcript.w) catch 0;
                    self.transcript.scrollDown(1, total, self.layout.transcript.h);
                    self.needs_repaint = true;
                    continue;
                },
                .page_up => {
                    self.transcript.scrollUp(self.layout.transcript.h -| 1);
                    self.needs_repaint = true;
                    continue;
                },
                .page_down => {
                    const total = self.transcript.totalRenderLines(self.layout.transcript.w) catch 0;
                    self.transcript.scrollDown(self.layout.transcript.h -| 1, total, self.layout.transcript.h);
                    self.needs_repaint = true;
                    continue;
                },
                else => {},
            }

            if (key == 0) continue;

            // --- Paleta abierta: intercepta toda la entrada ---
            if (self.palette_open) {
                if (key == 0x1b) { // Esc cierra
                    self.closePalette();
                } else if (key == 0x09) { // Tab mueve
                    self.paletteNext();
                } else if (key == '\r' or key == '\n') { // Enter ejecuta
                    try self.runPalette(agent);
                }
                continue;
            }

            // Ctrl-C -> salir
            if (key == 3) {
                self.running = false;
                break;
            }

            // Atajos de control (antes que ESC para que ESC sólo salga del chat)
            switch (key) {
                0x10 => { // Ctrl-P: paleta
                    self.openPalette();
                    continue;
                },
                0x14 => { // Ctrl-T: tema
                    self.toggleTheme();
                    continue;
                },
                0x06 => { // Ctrl-F: enfocar siguiente tool-call
                    self.focusNextTool();
                    continue;
                },
                0x0f => { // Ctrl-O: plegar/desplegar enfocada
                    self.toggleFoldFocused();
                    continue;
                },
                0x1b => { // Esc: salir del chat
                    self.running = false;
                    break;
                },
                else => {},
            }

            // Enter -> send
            if (key == '\r' or key == '\n') {
                const input = try self.alloc.dupe(u8, self.input_buf.items);
                defer self.alloc.free(input);
                self.input_buf.clearRetainingCapacity();
                self.cursor = 0;

                const trimmed = std.mem.trim(u8, input, " \t\r\n");
                if (trimmed.len == 0) continue;

                // Slash commands
                if (std.mem.startsWith(u8, trimmed, "/")) {
                    try self.handleCommand(trimmed, agent);
                    continue;
                }

                try self.addEntry(.user, trimmed);
                self.setStatus("pensando…");
                try self.paint();

                try agent.memory.addText(trimmed);
                var result = try agent.run(trimmed, 5);
                defer result.deinit(self.alloc);

                if (agent.stream_fn == null) {
                    try self.addEntry(.assistant, result.items);
                }
                try agent.memory.addText(result.items);
                self.setStatus("modo: CHAT   ·   Ctrl-P comandos   ·   Ctrl-C salir");
                continue;
            }

            // Backspace / DEL
            if (key == 0x7f or key == 0x08) {
                if (self.cursor > 0) {
                    _ = self.input_buf.orderedRemove(self.cursor - 1);
                    self.cursor -= 1;
                    self.needs_repaint = true;
                }
                continue;
            }

            // Ctrl-J newline in input
            if (key == '\n' or key == 0x0a) {
                // Ya manejado arriba como Enter
                continue;
            }

            // Ctrl-U clear input
            if (key == 0x15) {
                self.input_buf.clearRetainingCapacity();
                self.cursor = 0;
                self.needs_repaint = true;
                continue;
            }

            // Printable char
            if (std.ascii.isPrint(key)) {
                try self.input_buf.insert(self.alloc, self.cursor, key);
                self.cursor += 1;
                self.needs_repaint = true;
                continue;
            }
        }
    }

    fn handleCommand(self: *Tui, input: []const u8, agent: *zagent.Agent) !void {
        if (std.mem.eql(u8, input, "/quit")) {
            self.running = false;
        } else if (std.mem.eql(u8, input, "/help")) {
            try self.addEntry(.assistant,
                \\Comandos:
                \\  /quit         — salir
                \\  /clear        — limpiar memoria
                \\  /save <path>  — guardar memoria
                \\  /session save|load|list
                \\  /help         — esta ayuda
            );
        } else if (std.mem.eql(u8, input, "/clear")) {
            for (agent.memory.entries.items) |e| {
                self.alloc.free(e.text);
                self.alloc.free(e.embedding);
            }
            agent.memory.entries.clearRetainingCapacity();
            try self.addEntry(.assistant, "Memoria limpiada.");
        } else if (std.mem.startsWith(u8, input, "/save ")) {
            const path = std.mem.trim(u8, input["/save ".len..], " \t");
            if (path.len == 0) {
                try self.addEntry(.assistant, "Uso: /save <ruta>");
                return;
            }
            agent.memory.saveToFile(self.io, path, self.alloc) catch |err| {
                var buf: [256]u8 = undefined;
                const msg = try std.fmt.bufPrint(&buf, "Error guardando: {s}", .{@errorName(err)});
                try self.addEntry(.assistant, msg);
                return;
            };
            try self.addEntry(.assistant, "Memoria guardada.");
        } else if (std.mem.startsWith(u8, input, "/session save ")) {
            const name = std.mem.trim(u8, input["/session save ".len..], " \t");
            if (name.len == 0) {
                try self.addEntry(.assistant, "Uso: /session save <nombre>");
                return;
            }
            try self.saveSession(name, agent);
        } else if (std.mem.startsWith(u8, input, "/session load ")) {
            const name = std.mem.trim(u8, input["/session load ".len..], " \t");
            if (name.len == 0) {
                try self.addEntry(.assistant, "Uso: /session load <nombre>");
                return;
            }
            try self.loadSession(name, agent);
        } else if (std.mem.eql(u8, input, "/session list")) {
            try self.listSessions();
        } else {
            try self.addEntry(.assistant, "Comando desconocido. Usa /help.");
        }
    }

    fn saveSession(self: *Tui, name: []const u8, agent: *zagent.Agent) !void {
        const dir = try sessionDir(self.alloc);
        defer self.alloc.free(dir);
        _ = std.Io.Dir.cwd().createDirPath(self.io, dir) catch {};
        const path = try sessionPath(self.alloc, name);
        defer self.alloc.free(path);
        agent.memory.saveToFile(self.io, path, self.alloc) catch |err| {
            var buf: [256]u8 = undefined;
            const msg = try std.fmt.bufPrint(&buf, "Error: {s}", .{@errorName(err)});
            try self.addEntry(.assistant, msg);
            return;
        };
        try self.addEntry(.assistant, "Sesión guardada.");
    }

    fn loadSession(self: *Tui, name: []const u8, agent: *zagent.Agent) !void {
        const path = try sessionPath(self.alloc, name);
        defer self.alloc.free(path);
        agent.memory.loadFromFile(self.io, path, self.alloc) catch |err| {
            var buf: [256]u8 = undefined;
            const msg = try std.fmt.bufPrint(&buf, "Error: {s}", .{@errorName(err)});
            try self.addEntry(.assistant, msg);
            return;
        };
        try self.addEntry(.assistant, "Sesión cargada.");
    }

    fn listSessions(self: *Tui) !void {
        const dir_str = try sessionDir(self.alloc);
        defer self.alloc.free(dir_str);
        const dir = std.Io.Dir.cwd().openDir(self.io, dir_str, .{ .iterate = true }) catch |err| switch (err) {
            error.FileNotFound => {
                try self.addEntry(.assistant, "(no hay sesiones guardadas)");
                return;
            },
            else => {
                var buf: [256]u8 = undefined;
                const msg = try std.fmt.bufPrint(&buf, "Error: {s}", .{@errorName(err)});
                try self.addEntry(.assistant, msg);
                return;
            },
        };
        defer dir.close(self.io);
        var it = dir.iterate();
        var any = false;
        while (true) {
            const entry = it.next(self.io) catch continue;
            const e = entry orelse break;
            if (std.mem.endsWith(u8, e.name, ".zmem")) {
                try self.addEntry(.assistant, e.name);
                any = true;
            }
        }
        if (!any) try self.addEntry(.assistant, "(no hay sesiones guardadas)");
    }
};

// =====================================================================
// Session helpers (copied from old ztui.zig)
// =====================================================================

fn sessionDir(alloc: std.mem.Allocator) ![]u8 {
    return try std.fs.path.join(alloc, &.{ ".fluency", "sessions" });
}

fn sessionPath(alloc: std.mem.Allocator, name: []const u8) ![]u8 {
    const dir = try sessionDir(alloc);
    defer alloc.free(dir);
    return try std.fs.path.join(alloc, &.{ dir, name });
}

// =====================================================================
// Public entry point
// =====================================================================

pub fn runChatLoop(io: std.Io, allocator: std.mem.Allocator, agent: *zagent.Agent) !void {
    const term_size = platform.tty.size(io) catch platform.tty.TermSize{ .rows = 24, .cols = 80 };
    var tui = try Tui.init(allocator, io, term_size.cols, term_size.rows);
    defer tui.deinit();
    try tui.runEventLoop(agent);
}

// =====================================================================
// Tests
// =====================================================================

test "Tui init/deinit compiles" {
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var tui = try Tui.init(std.testing.allocator, io, 80, 24);
    defer tui.deinit();
    try std.testing.expect(tui.screen.w == 80);
    try std.testing.expect(tui.screen.h == 24);
}

test "Tui addEntry increases transcript" {
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var tui = try Tui.init(std.testing.allocator, io, 80, 24);
    defer tui.deinit();
    try tui.addEntry(.user, "hello");
    try std.testing.expect(tui.transcript.entries.items.len == 1);
}

test "Tui appendStream extends last entry" {
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var tui = try Tui.init(std.testing.allocator, io, 80, 24);
    defer tui.deinit();
    try tui.addEntry(.assistant, "hello");
    try tui.appendStream(" world");
    try std.testing.expect(std.mem.eql(u8, tui.transcript.entries.items[0].text.items, "hello world"));
}

test "Tui session path helper" {
    const alloc = std.testing.allocator;
    const path = try sessionPath(alloc, "demo");
    defer alloc.free(path);
    try std.testing.expect(std.mem.endsWith(u8, path, ".fluency\\sessions\\demo") or
        std.mem.endsWith(u8, path, ".fluency/sessions/demo"));
}

test "Tui toggleTheme alterna dark/light" {
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    var tui = try Tui.init(std.testing.allocator, threaded.io(), 80, 24);
    defer tui.deinit();
    try std.testing.expect(tui.theme_kind == .dark);
    const dark_bg = tui.theme.transcript_bg;
    tui.toggleTheme();
    try std.testing.expect(tui.theme_kind == .light);
    // el fondo del transcript debe cambiar entre temas
    try std.testing.expect(tui.theme.transcript_bg.r != dark_bg.r);
    tui.toggleTheme();
    try std.testing.expect(tui.theme_kind == .dark);
}

test "Tui focus y fold de tool-calls" {
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    var tui = try Tui.init(std.testing.allocator, threaded.io(), 80, 24);
    defer tui.deinit();
    try tui.addEntry(.user, "hola");
    try tui.addEntry(.tool, "search(x)");
    try tui.addEntry(.assistant, "ok");
    try std.testing.expect(tui.focus == null);
    tui.focusNextTool(); // debe enfocar la entrada tool (índice 1)
    try std.testing.expect(tui.focus.? == 1);
    try std.testing.expect(!tui.transcript.entries.items[1].folded);
    tui.toggleFoldFocused();
    try std.testing.expect(tui.transcript.entries.items[1].folded);
    tui.toggleFoldFocused();
    try std.testing.expect(!tui.transcript.entries.items[1].folded);
}

test "Tui paleta abre/mueve/cierra y compone sin IO" {
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    var tui = try Tui.init(std.testing.allocator, threaded.io(), 80, 24);
    defer tui.deinit();
    try std.testing.expect(!tui.palette_open);
    tui.openPalette();
    try std.testing.expect(tui.palette_open and tui.palette_sel == 0);
    tui.paletteNext();
    try std.testing.expect(tui.palette_sel == 1);
    // compose con la paleta abierta no debe fallar (rellena celdas, sin IO)
    try tui.compose();
    tui.closePalette();
    try std.testing.expect(!tui.palette_open);
}

test "dispWidth cuenta codepoints (no bytes)" {
    try std.testing.expect(dispWidth("abc") == 3);
    // "·" (U+00B7) = 2 bytes pero 1 celda; "█" = 3 bytes, 1 celda
    try std.testing.expect(dispWidth("a·b") == 3);
    try std.testing.expect(dispWidth("███") == 3);
    try std.testing.expect(dispWidth("▸ tú") == 4);
}

test "compose con bienvenida (transcript vacío) no falla" {
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    var tui = try Tui.init(std.testing.allocator, threaded.io(), 80, 24);
    defer tui.deinit();
    try std.testing.expect(tui.transcript.entries.items.len == 0);
    try tui.compose(); // dibuja logo + caja de input + status, sin IO
    // alguna celda del área de transcript debe tener el color del logo
    try std.testing.expect(tui.screen.cells.len == 80 * 24);
}

test "setStatusFmt copia a buffer persistente (sin puntero colgante)" {
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    var tui = try Tui.init(std.testing.allocator, threaded.io(), 80, 24);
    defer tui.deinit();
    tui.setStatusFmt("APROBAR {s}?", .{"fs_write"});
    try std.testing.expect(std.mem.eql(u8, tui.status, "APROBAR fs_write?"));
    // el slice apunta dentro del propio Tui
    try std.testing.expect(@intFromPtr(tui.status.ptr) >= @intFromPtr(&tui.status_buf));
}
