// tty.zig — terminal crudo portable (Windows / Unix)
// Aislado en src/platform/ como única pieza dependiente de SO (permitido Art. I.3).
const std = @import("std");
const builtin = @import("builtin");

pub const TermSize = struct {
    rows: u16,
    cols: u16,
};

/// Estado de terminal cruda; guarda el modo original para restaurar.
pub const RawTty = struct {
    enabled: bool = false,
    state: State = undefined,

    const State = switch (builtin.os.tag) {
        .windows => struct {
            original_mode: std.os.windows.DWORD,
            original_mode_out: std.os.windows.DWORD = 0,
            out_changed: bool = false,
            original_cp_out: std.os.windows.UINT = 0,
            cp_changed: bool = false,
        },
        else => struct {
            original_termios: std.posix.termios,
        },
    };

    pub fn init() RawTty {
        return .{};
    }

    /// Habilita modo crudo en stdin (sin eco, tecla-a-tecla, VT input).
    /// Si stdin no es un terminal, puede devolver `error.Unexpected`.
    pub fn enable(self: *RawTty, io: std.Io) !void {
        if (self.enabled) return;
        switch (builtin.os.tag) {
            .windows => {
                const stdin = std.Io.File.stdin();
                var get_mode = std.os.windows.CONSOLE.USER_IO.GET_MODE;
                // Si stdin no es una consola (redirigido/pipe), GET_MODE falla con
                // INVALID_HANDLE: lo tratamos como "no es terminal" sin volcar traza.
                switch (get_mode.operate(io, stdin) catch return error.NotATerminal) {
                    .SUCCESS => {},
                    else => return error.NotATerminal,
                }
                self.state.original_mode = get_mode.Data;
                // Desactivar LINE_INPUT, ECHO_INPUT, PROCESSED_INPUT;
                // activar VIRTUAL_TERMINAL_INPUT (secuencias VT para flechas).
                const ENABLE_PROCESSED_INPUT: std.os.windows.DWORD = 0x0001;
                const ENABLE_LINE_INPUT: std.os.windows.DWORD = 0x0002;
                const ENABLE_ECHO_INPUT: std.os.windows.DWORD = 0x0004;
                const ENABLE_VIRTUAL_TERMINAL_INPUT: std.os.windows.DWORD = 0x0200;
                const new_mode = (self.state.original_mode & ~@as(std.os.windows.DWORD, ENABLE_PROCESSED_INPUT | ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT)) | ENABLE_VIRTUAL_TERMINAL_INPUT;
                var set_mode = std.os.windows.CONSOLE.USER_IO.SET_MODE(new_mode);
                switch (set_mode.operate(io, stdin) catch return error.NotATerminal) {
                    .SUCCESS => {},
                    else => return error.NotATerminal,
                }
                // CRÍTICO para que la consola INTERPRETE las secuencias ANSI de
                // salida (cursor, color truecolor) en vez de imprimirlas en crudo:
                // activar ENABLE_VIRTUAL_TERMINAL_PROCESSING en stdout. Sin esto la
                // TUI se ve como "←[38;2;...m" literal. Best-effort: si falla, no
                // abortamos el modo crudo de entrada.
                self.state.out_changed = false;
                const stdout = std.Io.File.stdout();
                var get_out = std.os.windows.CONSOLE.USER_IO.GET_MODE;
                if (get_out.operate(io, stdout)) |status| {
                    if (status == .SUCCESS) {
                        self.state.original_mode_out = get_out.Data;
                        const ENABLE_PROCESSED_OUTPUT: std.os.windows.DWORD = 0x0001;
                        const ENABLE_VIRTUAL_TERMINAL_PROCESSING: std.os.windows.DWORD = 0x0004;
                        const out_mode = get_out.Data | ENABLE_PROCESSED_OUTPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING;
                        var set_out = std.os.windows.CONSOLE.USER_IO.SET_MODE(out_mode);
                        if (set_out.operate(io, stdout)) |s2| {
                            if (s2 == .SUCCESS) self.state.out_changed = true;
                        } else |_| {}
                    }
                } else |_| {}
                // Code page de salida UTF-8 (65001): sin esto, cmd.exe interpreta
                // cada byte de un glifo UTF-8 (▸ ▾ ⚙ • ┌─│) como un carácter CP850
                // distinto y se ve corrupto. Best-effort + restaurar al salir.
                self.state.cp_changed = false;
                var get_cp = std.os.windows.CONSOLE.USER_IO.GET_CP(.Output);
                if (get_cp.operate(io, stdout)) |st| {
                    if (st == .SUCCESS) {
                        self.state.original_cp_out = get_cp.Data.CodePage;
                        var set_cp = std.os.windows.CONSOLE.USER_IO.SET_CP(.Output, 65001);
                        if (set_cp.operate(io, stdout)) |st2| {
                            if (st2 == .SUCCESS) self.state.cp_changed = true;
                        } else |_| {}
                    }
                } else |_| {}
            },
            else => {
                const fd = std.Io.File.stdin().handle;
                self.state.original_termios = std.posix.tcgetattr(fd) catch return error.NotATerminal;
                var raw = self.state.original_termios;
                // En Zig 0.16 `tc_lflag_t` es un packed struct de flags (Linux y macOS).
                raw.lflag.ECHO = false;
                raw.lflag.ICANON = false;
                raw.lflag.ISIG = false;
                raw.cc[@intFromEnum(std.posix.V.MIN)] = 0;
                raw.cc[@intFromEnum(std.posix.V.TIME)] = 0;
                try std.posix.tcsetattr(fd, .NOW, raw);
            },
        }
        self.enabled = true;
    }

    /// Restaura el modo original guardado en `enable`.
    pub fn disable(self: *RawTty, io: std.Io) void {
        if (!self.enabled) return;
        switch (builtin.os.tag) {
            .windows => {
                const stdin = std.Io.File.stdin();
                var set_mode = std.os.windows.CONSOLE.USER_IO.SET_MODE(self.state.original_mode);
                _ = set_mode.operate(io, stdin) catch {};
                if (self.state.out_changed) {
                    const stdout = std.Io.File.stdout();
                    var set_out = std.os.windows.CONSOLE.USER_IO.SET_MODE(self.state.original_mode_out);
                    _ = set_out.operate(io, stdout) catch {};
                }
                if (self.state.cp_changed) {
                    const stdout = std.Io.File.stdout();
                    var set_cp = std.os.windows.CONSOLE.USER_IO.SET_CP(.Output, self.state.original_cp_out);
                    _ = set_cp.operate(io, stdout) catch {};
                }
            },
            else => {
                const fd = std.Io.File.stdin().handle;
                _ = std.posix.tcsetattr(fd, .NOW, self.state.original_termios) catch {};
            },
        }
        self.enabled = false;
    }

    /// `disable` + ignora errores; útil en `defer`.
    pub fn deinit(self: *RawTty, io: std.Io) void {
        self.disable(io);
    }
};

/// Obtiene el tamaño visible del terminal (filas, columnas).
/// Usa stdout porque es el dispositivo de terminal.
pub fn size(io: std.Io) !TermSize {
    switch (builtin.os.tag) {
        .windows => {
            const stdout = std.Io.File.stdout();
            var get_info = std.os.windows.CONSOLE.USER_IO.GET_SCREEN_BUFFER_INFO;
            switch (get_info.operate(io, stdout) catch return error.NotATerminal) {
                .SUCCESS => {},
                else => return error.NotATerminal,
            }
            return .{
                .cols = @intCast(get_info.Data.dwWindowSize.X),
                .rows = @intCast(get_info.Data.dwWindowSize.Y),
            };
        },
        .linux => {
            var ws: std.posix.winsize = undefined;
            const fd = std.Io.File.stdout().handle;
            const r = std.os.linux.ioctl(fd, std.os.linux.T.IOCGWINSZ, @intFromPtr(&ws));
            if (std.os.linux.errno(r) != .SUCCESS) return error.NotATerminal;
            return .{
                .cols = ws.col,
                .rows = ws.row,
            };
        },
        else => @compileError("tty.size: plataforma no soportada"),
    }
}

// =====================================================================
// Tests
// =====================================================================

test "RawTty init y round-trip enable/disable" {
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var t = RawTty.init();
    // Si no hay terminal (redirigido/pipe, p.ej. bajo el test runner), skip con éxito.
    t.enable(io) catch {
        try std.testing.expect(!t.enabled);
        return;
    };
    defer t.deinit(io);
    try std.testing.expect(t.enabled);
    t.disable(io);
    try std.testing.expect(!t.enabled);
}

test "RawTty deinit sin enable no rompe" {
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var t = RawTty.init();
    t.deinit(io);
    try std.testing.expect(!t.enabled);
}

test "size compila en el target actual" {
    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    // size puede fallar si stdout no es terminal; solo verificamos que compile.
    _ = size(io) catch {};
}
