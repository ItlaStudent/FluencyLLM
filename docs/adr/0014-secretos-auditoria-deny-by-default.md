# ADR-0014 — Secretos cifrados, auditoría y deny-by-default

## Estado
Aceptado (2026-05-30)

## Contexto
Artículo IV de la Constitución exige:
- Secretos nunca en claro en memoria persistente (Art. IV.3).
- Deny-by-default para acciones delegadas/sensibles (Art. IV.5).
- Auditoría de eventos de seguridad (Art. IV.4).

El proyecto ya contaba con `util.crypto` (ChaCha20-Poly1305) pero no había un almacén de secretos integrado al runtime agéntico.

## Decisión
1. **SecretStore (`zagent.SecretStore`)**:
   - Cifra cada secreto con ChaCha20-Poly1305 usando una clave maestra de 32 bytes.
   - El nombre del secreto se usa como datos asociados (AD) del AEAD, ligando el cifrado al contexto.
   - Nonces aleatorios de 12 bytes generados vía `io.random` (modelo `std.Io`).
   - Variante `putWithNonce` para tests determinísticos.
   - `get` descifra y verifica autenticación; rechazo si el ciphertext o tag fueron alterados.
2. **AuditLog (`zagent.AuditLog`)**:
   - Append-only de eventos con secuencia numérica (`seq: u64`).
   - Categorías: `tool_dispatch`, `tool_denied`, `secret_access`, `secret_denied`, `budget_exceeded`.
   - Cada evento lleva descripción y bandera `is_error`.
   - `serializeTo` produce texto legible para inspección.
3. **Deny-by-default formal en `Agent`**:
   - Si una tool no está registrada en `Registry`, `Agent.run` devuelve error y loguea `tool_denied` sin ejecutar nada.
   - `Agent.getSecret` devuelve `error.SecretStoreNotConfigured` si no hay `SecretStore` activo, logueando `secret_denied`.

## Consecuencias
- Los secretos solo existen en plaintext transitoriamente durante `get`, y son liberados por el caller.
- El audit log no tiene persistencia a disco en esta iteración; vive en memoria durante la sesión del agente.
- No se usa `panic` en ninguna ruta de seguridad; todos los errores devuelven `Result` o `error`.

## Alternativas consideradas
- Argon2/KDF para derivar clave de master password: descartado por simplicidad; el master key es responsabilidad del deployer.
- Persistencia del audit log a archivo: descartado para esta iteración; se puede añadir más tarde vía `std.Io` sin cambiar la API.

## Referencias
- `src/zagent/zagent.zig`: `SecretStore`, `AuditLog`, `Agent.getSecret`, `Agent.run`.
- `src/util/crypto.zig`: `seal`/`open` ChaCha20-Poly1305.
