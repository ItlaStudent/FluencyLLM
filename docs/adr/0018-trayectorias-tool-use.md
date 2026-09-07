# ADR-0018 — Captura de trayectorias ReAct para afinado de tool-use

## Estado
Aceptado (2026-05-30)

## Contexto
F4 requiere "afinado de Fluency para tool-use (datos de trayectorias)". Para entrenar el modelo a usar herramientas correctamente, necesitamos datos de trayectorias: secuencias de (prompt, pensamiento, acción, observación, respuesta) que el agente genera al interactuar con tools. Estos datos se usarán en F3 (destilación) para fine-tuning del modelo.

## Decisión
1. **Nuevo módulo `trajectory.zig`** en `src/zagent/`:
   - `Trajectory.Step`: tagged union con variantes `thought`, `action`, `observation`, `answer`, `budget_exceeded`, `interrupted`.
   - `Trajectory`: struct con `steps: ArrayList(Step)`, `prompt`, `model`.
   - Métodos `addThought`, `addAction`, `addObservation`, `addAnswer`, `addBudgetExceeded`, `addInterrupted`.
   - `serializeTo` genera JSON simple (formato ZTRAJ) para compatibilidad con pipelines de entrenamiento.
2. **Integración en `Agent`**:
   - Campo `trajectory: ?Trajectory` en `Agent`.
   - En `Agent.run`, se inicializa una nueva `Trajectory` al inicio (liberando la anterior si existe).
   - Cada paso del bucle ReAct se registra automáticamente en `agent.trajectory`.
   - `Agent.deinit` libera la trayectoria si existe.
3. **Uso futuro**:
   - Las trayectorias JSON se guardarán a disco con `/save` en la TUI o automáticamente post-ejecución.
   - `zdistill` consumirá estas trayectorias como datos de entrenamiento para fine-tuning de tool-use.

## Consecuencias
- El agente ahora produce datos estructurados de cada ejecución, listos para entrenamiento.
- El formato JSON es simple y puede parsearse con cualquier librería estándar.
- `Agent.deinit` ahora libera tanto `memory` como `trajectory`.

## Referencias
- `src/zagent/trajectory.zig`: `Trajectory`, `Step`, `serializeTo`.
- `src/zagent/zagent.zig`: `Agent.trajectory`, registro automático en `run`.
