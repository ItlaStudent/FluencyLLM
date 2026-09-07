# 07 — Entrenamiento y evolución vía IDEs (gobernado)

[← Plan Maestro](../PLAN-MAESTRO.md) · [← 06 Interop](06-interop-protocolos.md) · [→ 08 Roadmap](08-roadmap-fases.md)

Cómo **colaboradores y herramientas** ayudan a entrenar/mejorar el modelo Fluency **sin poder violar la Constitución**. Este es el requisito de "entrenable vía IDEs" + "lineamientos permanentes e inviolables" unidos.

---

## 1. El modelo mental

Un agente de IDE es un colaborador poderoso pero **no confiable por defecto**: hay que **darle el contexto correcto** y **encerrarlo en barreras mecánicas**. ZSA logra ambas cosas:

1. **Contexto:** cada colaborador/herramienta lee su archivo de gobernanza (`AGENTS.md`), que remite a la [Constitución](../CONSTITUCION.md) y al plan.
2. **Barreras:** los **gates de CI escritos en Zig** (Art. IX) rechazan cualquier cambio que viole las reglas — sin importar qué agente o modelo lo haya hecho. *Las reglas no dependen de la buena fe del agente; dependen de la máquina.*

```
Agente IDE ──lee──▶ AGENTS.md ──▶ CONSTITUCION.md (prioridad sobre la tarea)
     │
     └─propone cambio──▶ pre-commit (gates) ──▶ CI (gates) ──▶ merge
                              │ falla ⇒ rechazo + razón     │ revisión humana (CODEOWNERS) para zonas sensibles
```

---

## 2. Lo que un agente de IDE PUEDE hacer (tareas de entrenamiento)

Todo a través de los comandos Zig del proyecto (no escribiendo pipelines en Python):

| Tarea | Comando | Qué hace el agente |
|---|---|---|
| Conectar un profesor | `fluency pull <modelo>` | Descarga un modelo a Ollama; lo registra en `docs/PROFESORES.md` con su licencia. |
| Destilar (MOHAWK) | `fluency distill mohawk ...` | Lanza/parametriza las etapas A/B/C; ajusta hiperparámetros en `configs/*.zon`. |
| Generar datos sintéticos | `fluency synth ...` | Self-instruct/Orca vía `zollama`; filtra con verificadores. |
| Afinar | `fluency distill seqkd|onpolicy|dpo ...` | Afinado de instrucción/preferencias. |
| Evaluar | `fluency eval --suite eval/` | Corre suites; compara contra baseline/profesor; reporta. |
| Mejorar kernels | editar `src/zkernel/*.zig` | Optimizar un kernel SIMD **manteniendo el test contra referencia**. |
| Añadir un mixer | nuevo `src/zarch/mixers/*.zig` | Implementar la vtable `Mixer` desde un paper; registrarlo. |
| Curar datos | editar `data/` recipes | Mejorar el corpus/packing. |

**Cada una de estas tareas produce un PR** que pasa por los gates. El agente puede iterar, pero **no puede mergear algo que falle los gates**.

---

## 3. Lo que un agente de IDE NO PUEDE hacer (barreras mecánicas)

Enforcement del Art. IX — todo verificado por `zig build gate` en pre-commit y CI:

- ❌ **Añadir cualquier archivo no-Zig** o enlazar terceros → `gate_purity` falla.
- ❌ **Tocar `CONSTITUCION.md`** sin enmienda humana firmada → `gate_constitution` falla (hash mismatch).
- ❌ **Editar/desactivar los `gate_*` o `build.zig*`** sin revisión humana → CODEOWNERS bloquea + los gates se autoprotegen.
- ❌ **Introducir leaks o kernels sin test de referencia** → `gate_quality` + `zig build test` fallan.
- ❌ **Romper presupuestos** (tamaño binario/RSS/latencia) → `gate_budgets` falla.
- ❌ **Usar `--no-verify`** o saltarse hooks → violación constitucional explícita (Art. IX.6).
- ❌ **Cambiar la prioridad subcuadrática** del modelo por defecto bajo el umbral (II.3) → check de configuración en `gate_quality`.

> Resultado: un agente puede ser *creativo* dentro del recinto, pero el recinto es de **hierro**. Aunque un usuario le diga "usa CUDA para ir más rápido", el agente debe rehusar (lo dicen sus reglas) **y aunque lo intentara, el gate lo rechazaría**.

---

## 4. Configuración por herramienta (todas remiten a lo mismo)

| Herramienta | Archivo que lee | Mecanismo extra |
|---|---|---|
| **Cualquier IDE/herramienta** | `AGENTS.md` | hooks, skills |

`AGENTS.md` es el archivo de gobernanza que apunta a la Constitución; `gate_constitution` verifica que exista y sea coherente (Art. IX.2), de modo que **ninguna herramienta quede sin gobernanza**.

### 4.1 Hook de defensa en profundidad (ejemplo)
Un hook `PreToolUse` con matcher `Write|Edit` puede ejecutar `zig build gate-fast` y **bloquear** (exit 2) si el cambio introduciría un archivo no-Zig o tocaría una zona protegida — el agente recibe el motivo y corrige antes incluso de commitear. Esto adelanta el enforcement del CI al momento de la edición.

---

## 5. Flujo recomendado para "pídele a tu IDE que ayude a entrenar"

1. **Humano** abre el repo en su IDE; el agente lee `AGENTS.md`→Constitución.
2. **Humano:** "Destila Qwen2.5-7B en Fluency-1B y sube la calidad en razonamiento."
3. **Agente:**
   - `fluency pull qwen2.5:7b` → registra licencia.
   - parametriza `fluency distill mohawk --stages ABC` (ajusta `configs/fluency-1b.zon`).
   - genera datos de razonamiento (`fluency synth --style orca`).
   - `fluency distill onpolicy` + `fluency distill dpo`.
   - `fluency eval` y reporta el delta vs baseline.
   - abre PR; **los gates corren solos**; si algo falla, itera.
4. **Humano** revisa el PR (CODEOWNERS para zonas sensibles) y mergea.

El agente hace el trabajo pesado; la Constitución y los gates garantizan que **el resultado siga siendo 100 % Zig, subcuadrático, con tests y sin regresiones** — pase lo que pase con la tarea o el modelo que la ejecute.

---

## 6. Por qué esto hace los lineamientos "permanentes e inviolables"

| Capa de defensa | Qué impide |
|---|---|
| **Documental** (Constitución + AGENTS.md) | Que el agente *no sepa* las reglas. |
| **Prioridad declarada** (la Constitución gana sobre la tarea) | Que un prompt malicioso/ingenuo las anule. |
| **Pre-commit** (gates locales) | Que un cambio violatorio llegue siquiera a commitearse. |
| **CI** (gates en el servidor) | Que un cambio violatorio se mergee. |
| **CODEOWNERS** (revisión humana) | Que se toquen las zonas críticas sin un humano. |
| **Auto-protección** (gates vigilan a los gates + hash de la Constitución) | Que se desactive el propio enforcement. |
| **Enmienda sólo humana** (Art. X) | Que una IA cambie las reglas. |

Ninguna capa por sí sola es perfecta; **juntas hacen que violar las reglas sea, en la práctica, imposible para un agente automático** y altamente visible para un humano. Eso es lo que el encargo pide por "inviolable".

[→ Siguiente: 08 — Roadmap por fases](08-roadmap-fases.md)
