@echo off
setlocal enabledelayedexpansion
REM ===========================================================================
REM  Lanzador de Fluency  -  motor de IA local 100%% Zig, subcuadratico y agentico.
REM
REM  Uso:
REM    fluency.cmd                  -> arranca la TUI de chat
REM    fluency.cmd <subcomando> ... -> reenvia al binario (serve, run, agent, ...)
REM    fluency.cmd --rebuild        -> recompila (ReleaseFast) y arranca el chat
REM
REM  Subcomandos del binario: serve | run | agent | chat | distill | train |
REM                           eval | tokenize | inspect | forward | mcp-server |
REM                           mcp-selftest | gate | --version
REM
REM  La primera vez compila solo (ReleaseFast) usando el Zig 0.16 incluido.
REM ===========================================================================

set "ROOT=%~dp0"
set "ZIG=%ROOT%zig-0.16.0\zig-x86_64-windows-0.16.0\zig.exe"
set "BIN=%ROOT%zig-out\bin\fluency.exe"

if not exist "%ZIG%" (
    echo [fluency] No se encontro el compilador Zig en "%ZIG%".
    echo [fluency] Coloca el toolchain de Zig 0.16 en esa ruta y reintenta.
    exit /b 1
)

REM --- Recompilacion forzada con --rebuild ---
if /i "%~1"=="--rebuild" (
    echo [fluency] Recompilando ^(ReleaseFast^)...
    pushd "%ROOT%"
    "%ZIG%" build -Doptimize=ReleaseFast
    set "RC=!errorlevel!"
    popd
    if not "!RC!"=="0" (
        echo [fluency] Fallo la compilacion.
        exit /b !RC!
    )
    "%BIN%" chat
    exit /b !errorlevel!
)

REM --- Compilacion automatica la primera vez ---
if not exist "%BIN%" (
    echo [fluency] Compilando por primera vez ^(ReleaseFast^)...
    pushd "%ROOT%"
    "%ZIG%" build -Doptimize=ReleaseFast
    set "RC=!errorlevel!"
    popd
    if not "!RC!"=="0" (
        echo [fluency] Fallo la compilacion.
        exit /b !RC!
    )
)

REM --- Arranque ---
if "%~1"=="" (
    "%BIN%" chat
) else (
    "%BIN%" %*
)
exit /b %errorlevel%
