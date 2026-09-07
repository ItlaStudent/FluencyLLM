# F6 — Fluency como modelo local seleccionable (drop-in Ollama/OpenAI)

> Fecha de verificación: 2026-06-02
> Modelo de prueba: Qwen2.5-coder-1.5B-Q4_K_M (Apache-2.0, GGUF)
> Servidor: `fluency serve --gguf <blob> --quantized --model Fluency-1.5b-v1 --port 11436`
> Compilación: `zig build -Doptimize=ReleaseFast`

## Resumen

Fluency expone un endpoint HTTP drop-in compatible con **Ollama** y **OpenAI**.
Un cliente que habla cualquiera de esos protocolos puede descubrir, seleccionar y chatear con Fluency **sin saber que no es Ollama/OpenAI**.

---

## Flujo Ollama (descubrimiento + chat + streaming)

### 1. Descubrimiento — `GET /api/tags`

```powershell
curl.exe -s http://127.0.0.1:11436/api/tags
```

**Respuesta:**
```json
{"models":[{"name":"Fluency-1.5b-v1","model":"Fluency-1.5b-v1","modified_at":"1970-01-01T00:00:00Z","size":0}]}
```

El modelo `Fluency-1.5b-v1` aparece listado. Los clientes Ollama lo ven como seleccionable.

### 2. Chat sin streaming — `POST /api/chat`

```powershell
curl.exe -s -X POST http://127.0.0.1:11436/api/chat -H "Content-Type: application/json" -d '{"model":"Fluency-1.5b-v1","messages":[{"role":"user","content":"Hi, how are you?"}],"max_tokens":10}'
```

**Respuesta:**
```json
{"model":"Fluency-1.5b-v1","created_at":"1970-01-01T00:00:00Z","message":{"role":"assistant","content":" I'm doing well, thank you. How about"},"done":true}
```

Texto coherente generado por el modelo real.

### 3. Chat con streaming — `POST /api/chat` (`stream:true`)

```powershell
curl.exe -s -N -X POST http://127.0.0.1:11436/api/chat -H "Content-Type: application/json" -d '{"model":"Fluency-1.5b-v1","messages":[{"role":"user","content":"Say hello"}],"stream":true,"max_tokens":6}'
```

**Respuesta (NDJSON, token a token — shape Ollama *chat* con `message.content`):**
```ndjson
{"model":"Fluency-1.5b-v1","message":{"role":"assistant","content":" to"},"done":false}
{"model":"Fluency-1.5b-v1","message":{"role":"assistant","content":" the"},"done":false}
{"model":"Fluency-1.5b-v1","message":{"role":"assistant","content":" world"},"done":false}
{"model":"Fluency-1.5b-v1","message":{"role":"assistant","content":" of"},"done":false}
{"model":"Fluency-1.5b-v1","message":{"role":"assistant","content":" programming"},"done":false}
{"model":"Fluency-1.5b-v1","message":{"role":"assistant","content":""},"done":true}
```
> Nota: `/api/chat` (chat) usa `message.content`; `/api/generate` (completion) usa `response`. Es el distingo que exige el protocolo Ollama (los clientes de chat leen `message.content`).

---

## Flujo OpenAI (descubrimiento + chat + streaming)

### 1. Descubrimiento — `GET /v1/models`

```powershell
curl.exe -s http://127.0.0.1:11436/v1/models
```

**Respuesta:**
```json
{"object":"list","data":[{"id":"Fluency-1.5b-v1","object":"model","created":0,"owned_by":"fluency"}]}
```

### 2. Chat sin streaming — `POST /v1/chat/completions`

```powershell
curl.exe -s -X POST http://127.0.0.1:11436/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"Fluency-1.5b-v1","messages":[{"role":"user","content":"Hi, how are you?"}],"max_tokens":10}'
```

**Respuesta:**
```json
{"id":"chatcmpl-fluency-local","object":"chat.completion","created":0,"model":"Fluency-1.5b-v1","choices":[{"index":0,"message":{"role":"assistant","content":" I'm doing well, thank you. How about"},"finish_reason":"stop"}],"usage":{"prompt_tokens":16,"completion_tokens":37,"total_tokens":53}}
```

### 3. Chat con streaming — `POST /v1/chat/completions` (`stream:true`)

```powershell
curl.exe -s -N -X POST http://127.0.0.1:11436/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"Fluency-1.5b-v1","messages":[{"role":"user","content":"Say hello"}],"stream":true,"max_tokens":6}'
```

**Respuesta (SSE, token a token):**
```sse
data: {"choices":[{"delta":{"content":" of"}}]}

data: {"choices":[{"delta":{"content":" programming"}}]}

data: {"choices":[{"delta":{"content":"!"}}]}

data: [DONE]

```

---

## Resumen de compatibilidad

| Endpoint | Protocolo | Estado |
|----------|-----------|--------|
| `GET /api/tags` | Ollama | ✅ Devuelve modelo listado |
| `POST /api/chat` | Ollama | ✅ Responde con texto coherente |
| `POST /api/chat` (stream) | Ollama | ✅ NDJSON token-a-token (`message.content`) |
| `GET /v1/models` | OpenAI | ✅ Devuelve modelo listado |
| `POST /v1/chat/completions` | OpenAI | ✅ Responde con texto coherente |
| `POST /v1/chat/completions` (stream) | OpenAI | ✅ SSE token-a-token + `[DONE]` |
| `POST /api/show` | Ollama | ✅ Metadatos con shape real Ollama |
| `GET /api/version` | Ollama | ✅ `{"version":"0.7.0"}` |

---

## Quickstart local

### Requisitos
- Windows (x86_64) con ~4 GB de RAM libre.
- Modelo GGUF (p. ej. Qwen2.5-coder-1.5B-Q4_K_M, Apache-2.0).
- Compilador Zig 0.16.0 (vendorizado en `zig-0.16.0/zig-x86_64-windows-0.16.0/zig.exe`).

### 1. Compilar en ReleaseFast
```powershell
zig-0.16.0\zig-x86_64-windows-0.16.0\zig.exe build -Doptimize=ReleaseFast
```

### 2. Servir el modelo cuantizado
```powershell
$model = "$env:USERPROFILE\.ollama\models\blobs\sha256-29d8c98fa6b098e200069bfb88b9508dc3e85586d20cba59f8dda9a808165104"
zig-out\bin\fluency.exe serve --gguf $model --quantized --model Fluency-1.5b-v1 --port 11434
```

- `--quantized` mantiene ~3 GB de RAM (vs ~6 GB sin cuantizar).
- `--model Fluency-1.5b-v1` es el nombre que anunciará en los endpoints de descubrimiento.
- El servidor escucha en `http://127.0.0.1:11434`.

### 3. Apuntar un cliente

**Cliente Ollama (compatible):**
Configura la URL base a `http://127.0.0.1:11434` y selecciona el modelo `Fluency-1.5b-v1`.

**Cliente OpenAI (compatible):**
Configura la URL base a `http://127.0.0.1:11434/v1` y usa el modelo `Fluency-1.5b-v1`.

**Prueba rápida con curl:**
```powershell
# Descubrimiento Ollama
curl.exe -s http://127.0.0.1:11434/api/tags

# Chat Ollama
curl.exe -s -X POST http://127.0.0.1:11434/api/chat -H "Content-Type: application/json" -d '{"model":"Fluency-1.5b-v1","messages":[{"role":"user","content":"Hello"}],"max_tokens":10}'

# Chat OpenAI
curl.exe -s -X POST http://127.0.0.1:11434/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"Fluency-1.5b-v1","messages":[{"role":"user","content":"Hello"}],"max_tokens":10}'
```

---

## Persistencia verificada

Se ejecutaron **4 peticiones secuenciales** contra el mismo proceso servidor sin reinicio, alternando protocolos OpenAI y Ollama:

1. `POST /v1/chat/completions` → `"content":" To solve the problem of 2 +"` ✅
2. `POST /api/chat` → `"content":" Paris! Paris is the capital city of"` ✅
3. `POST /v1/chat/completions` → `"content":" why the concept of a \"solar"` ✅
4. `POST /api/chat` → `"content":"! I'm a friendly AI"` ✅

El servidor no crasheó ni degradó entre peticiones.

---

## Limitaciones honestas

- **Velocidad:** En CPU (sin GPU), el modelo de 1.5B cuantizado tarda ~1–2 s por token en ReleaseFast. Esto es esperado para inferencia CPU pura.
- **Paralelismo:** El servidor usa un único hilo de aceptación; mientras un cliente consume tokens, otro debe esperar. Adecuado para uso local/individual, no para carga concurrente masiva.
- **Modelo servido:** Por ahora se sirve un GGUF cuantizado de terceros (Qwen2.5-coder-1.5B, Apache-2.0). El modelo propio destilado de Fluency sigue madurando aparte.
- **RAM:** Con `--quantized`, el servicio mantiene ~3 GB de RAM para este modelo de 1.5B (vs ~6 GB en modo f32).
- **size: 0** en `/api/tags`: placeholder; no afecta descubrimiento básico.

> **Corrección de auditoría 2026-06-02:** el streaming de `/api/chat` emitía `"response"` (shape de `/api/generate`) en vez de `"message":{"role":"assistant","content":...}`; los clientes oficiales de Ollama-chat lo habrían leído como contenido vacío. Corregido (`writeStreamChunk` distingue chat↔generate) y **verificado en vivo**; test de regresión "zserve streaming chunk shape" añadido (176 tests).
