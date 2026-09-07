# 01 — Visión y Alcance

[← Volver al Plan Maestro](../PLAN-MAESTRO.md)

## 1. Visión

Construir **un motor de IA local, independiente y de código abierto**, escrito **íntegramente en Zig**, tan ligero y embebible que pueda vivir dentro de cualquier app —de escritorio, servidor, móvil (Capacitor) o navegador (WASM)— sin imponer dependencias pesadas ni nube. Un motor que **no sólo ejecuta** modelos, sino que **aprende** de los mejores modelos abiertos y **actúa** como agente autónomo. Diseñado para servir como cerebro, diferenciador, herramienta, LLM o potenciador de cualquier proyecto que necesite IA local.

La apuesta de diseño es doble y deliberada:
- **Pureza de lenguaje (Zig 100 %)** como valor identitario, aun a costa de no usar CUDA/BLAS. Esto da binarios diminutos, arranque en milisegundos, cross-compilación trivial y cero "caja negra" ajena.
- **Subcuadrático primero** como apuesta de futuro: coste lineal en contexto, estado de tamaño fijo en decodificación, ideal para correr local y en edge donde el KV-cache cuadrático de los Transformers ahoga la memoria.

## 2. Propósito (el "para qué")

ZSA es un **motor de IA independiente** que puede servir como cerebro, diferenciador, herramienta, LLM o potenciador de cualquier aplicación. Su trabajo es:

1. **Servir inferencia** a apps cliente por APIs estándar (OpenAI-compatible, estilo Ollama, MCP, C-ABI, WASM).
2. **Razonar y actuar** de forma agéntica: usar herramientas, planificar tareas multi-paso, recordar, delegar en sub-agentes.
3. **Mejorar con el tiempo** destilando conocimiento de modelos abiertos a su propia familia de modelos subcuadráticos (**Fluency**).

> **Caso de uso ancla:** cualquier app que use **Ollama** o la **API OpenAI** como backend LLM puede apuntar a ZSA cambiando sólo la URL base. ZSA debe poder **sustituir o complementar a Ollama** exponiendo las mismas APIs. Apps embebidas pueden consumir ZSA como generador/razonador embebido (C-ABI o WASM).

## 3. Usuarios y consumidores

| Consumidor | Cómo usa ZSA | Interfaz |
|---|---|---|
| Apps compatibles con OpenAI/Ollama | Backend LLM local | API OpenAI/Ollama-compatible (HTTP) |
| Apps embebidas | Razonador/generador embebido | C-ABI (`libfluency`) o WASM |
| IDEs y herramientas | (a) modelo backend; (b) consumidor de tools MCP | OpenAI API + MCP |
| Agentes/automatizaciones | Cerebro agéntico autónomo | CLI `fluency`, MCP, A2A |
| Desarrolladores/colaboradores | Entrenar/destilar/evaluar modelos | CLI `fluency distill/train/eval` |

## 4. Alcance (qué SÍ hace ZSA)

- **Inferencia** de modelos propios (Fluency) y de modelos abiertos cargados desde **GGUF/safetensors**.
- **Arquitectura híbrida** subcuadrático + atención (intercambiable por capa).
- **Entrenamiento y destilación** 100 % en Zig (autodiff propio), incl. destilación cruzada Transformer→subcuadrático.
- **Tokenización** propia (BPE byte-level, SentencePiece unigram, WordPiece) compatible con modelos populares.
- **Runtime agéntico** completo (tools, planificación, memoria, sub-agentes, seguridad).
- **Servidor de protocolos**: OpenAI API, Ollama-style, MCP (cliente y servidor), LSP (opcional), hooks/skills/plugins (host).
- **Empaquetado embebible**: CLI, librería C-ABI, WASM.
- **Gobernanza mecánica**: gates de CI que garantizan las reglas constitucionales.

## 5. No-objetivos (qué NO hace, al menos no ahora)

- **No** competir en velocidad de entrenamiento con stacks GPU/CUDA: la pureza Zig lo impide por diseño (ver [02](02-arquitectura.md) §GPU). Se compensa con destilación eficiente y modelos pequeños.
- **No** entrenar modelos frontera de cientos de miles de millones de parámetros desde cero. El foco son modelos **edge/pequeños-medianos** (≤ unos pocos B params) obtenidos por destilación.
- **No** reempaquetar ni redistribuir pesos de terceros como si fueran propios (Art. III).
- **No** depender de servicios en la nube para funcionar (la nube es opcional, p. ej. para descargar un profesor).
- **No** ser un IDE ni un chatbot final: es el **motor** debajo de esas cosas.
- **No** soportar (de inicio) multimodalidad pesada (visión/audio generativos); se deja como evolución (Fase 7+).

## 6. Criterios de éxito (medibles)

| # | Criterio | Métrica objetivo (revisable por ADR) |
|---|---|---|
| E1 | Pureza | `zig build gate` pasa: 0 archivos no-Zig, 0 enlaces a terceros. |
| E2 | Inferencia abierta | Carga y corre un modelo GGUF popular (p. ej. un Qwen/Llama pequeño) con salida correcta vs referencia. |
| E3 | Subcuadrático | Memoria de decodificación **constante** respecto a la longitud generada en capas SSM; latencia/token estable. |
| E4 | Aprendizaje | Un Fluency destilado de un profesor alcanza ≥ X % de la calidad del profesor en un set de eval propio, con ≤ 5B tokens. |
| E5 | Agéntica | Completa tareas multi-herramienta de referencia con tool-calls válidos (decodificación restringida) y presupuesto acotado. |
| E6 | Interop | Cualquier app compatible con OpenAI/Ollama funciona apuntando a ZSA en lugar de Ollama, sin cambios de código salvo la URL. |
| E7 | Embebible | `libfluency` (C-ABI) y `fluency.wasm` cargan y responden desde un host no-Zig (Node/navegador). |
| E8 | Ligereza | Binario de inferencia release y RSS pico dentro de los presupuestos de `budgets.zon`. |
| E9 | Evolución | Añadir un mixer nuevo no requiere tocar el núcleo (sólo registrar la vtable). |

## 7. Supuestos y restricciones explícitas

- **Plataforma primaria de desarrollo:** Windows 11 (entorno del autor) + cross-compilación a Linux/macOS/ARM/WASM (fortaleza de Zig).
- **Hardware de entrenamiento:** se asume CPU multinúcleo de consumo y, opcionalmente, GPU vía backends Zig **a futuro**. No se asume cluster CUDA.
- **Profesores:** modelos abiertos descargables (Ollama / HuggingFace) con licencia compatible con destilación.
- **Zig fijado** a una versión concreta en `build.zig.zon` (≥ 0.16.0) por la inestabilidad de la API entre minors (ver [10](10-riesgos-decisiones.md)).

## 8. Glosario rápido

- **Subcuadrático:** arquitectura con coste sub-O(N²) en longitud de secuencia (SSM, atención lineal, RWKV…).
- **Mixer:** el sub-bloque que mezcla información entre tokens (atención, SSM, etc.). En ZSA es intercambiable.
- **Destilación:** transferir conocimiento de un modelo "profesor" a uno "alumno" más pequeño/distinto.
- **MOHAWK:** método de 3 etapas para destilar un Transformer en un SSM (ver [04](04-aprendizaje-destilacion.md)).
- **MCP:** Model Context Protocol — protocolo estándar para que agentes consuman herramientas/datos.
- **GGUF:** formato de archivo de modelo (pesos + metadatos + tokenizer) de la familia ggml/llama.cpp.

[→ Siguiente: 02 — Arquitectura](02-arquitectura.md)
