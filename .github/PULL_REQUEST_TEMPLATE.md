## What and why

<!-- What does this PR change, and why? Link related issues (e.g. Closes #123). -->

## Checklist

- [ ] `zig build gate` passes (purity + constitution + quality + budgets)
- [ ] `zig build test` passes with no leaks
- [ ] `zig build eval` run and compared to baseline (only if the model changed)
- [ ] 100% Zig: no other languages, no FFI to third-party libraries
- [ ] New numerical kernels have a test against a slow reference implementation
- [ ] Large changes have an ADR in `docs/adr/`
- [ ] Does **not** touch `CONSTITUCION.md`, `build.zig*` or `tools/gate_*.zig` — or follows the amendment procedure (Constitution, Art. X)
