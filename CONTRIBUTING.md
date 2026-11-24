# Contributing to Fluency

Thanks for your interest in Fluency. Before you spend time on a contribution,
please read this — Fluency is a constitutional project with **hard, non-negotiable rules**.

## TL;DR

1. **100% Zig (Zig 0.16.x; 0.17 not supported yet).** No other languages. No FFI to third-party libraries.
2. **Subquadratic priority.** Transformer compatibility is supported, but
   attention is a minority mixer by design.
3. **Distillation is the only training paradigm.** Respect upstream licenses.
4. **Deny-by-default agentic layer.** Encrypted secrets. Never `panic` on tools.
5. **Don't touch** `CONSTITUCION.md`, `tools/gate_*.zig`, or `build.zig*` without a
   human-approved amendment (Constitution, Article X).
6. **Never use `--no-verify`** on commits.

If your patch can't satisfy these, **don't open the PR** — open an
[ADR](docs/adr/) proposal first.

## Required reading

Before you write a single line:

- [`CONSTITUCION.md`](CONSTITUCION.md) — the project's hard rules. Inviolable.
- [`AGENTS.md`](AGENTS.md) — operational rules for AI/IDE agents working in
  this repo. They apply to humans too.
- [`PLAN-MAESTRO.md`](PLAN-MAESTRO.md) — the master plan and roadmap.
- [`docs/`](docs/) — architecture details, training docs, ADRs.

## Development environment

- **OS:** Windows and Linux (x86_64/aarch64), both checked in CI. GPU (Vulkan) is Windows-only for now; on Linux the engine runs on CPU. macOS does not build yet. Active developer machines are Windows 11 + PowerShell.
- **Toolchain:** Zig 0.16.x exactly (0.17 has breaking `std.Build` changes and does not build Fluency yet). The repo expects the
  toolchain at `zig-0.16.0/zig-x86_64-windows-0.16.0/zig.exe` if you vendorize
  it locally, but you can use a system `zig` on PATH.
- **No extra runtimes.** No Node, no Python, no anything else is needed to build
  or run Fluency. If a script needs another language to *develop* (e.g. a
  release script), it must be Zig too.

### Build & test

```sh
zig build              # build everything
zig build test         # run all unit tests
zig build gate         # run the constitutional gates (purity, quality, budgets, constitution-hash)
```

Both `zig build gate` and `zig build test` MUST be green before you push.

## Branches and commits

- Branch from `main` with a short, descriptive name: `fix/`, `feat/`, `docs/`,
  `chore/`, etc.
- **Signed commits required.** Set up GPG signing locally:
  ```sh
  git config --global commit.gpgsign true
  git config --global user.signingkey <YOUR_KEY_ID>
  ```
- Commit messages: imperative, present tense. Reference the relevant
  doc/section if applicable (e.g. "ztrain: harden softmax against NaN (III.1, Art. V)").
- **No `--no-verify`** to skip hooks. If a hook fails, fix the underlying issue.

## Pull requests

A PR is mergeable when:

1. `zig build gate` and `zig build test` pass locally and in CI.
2. Constitution and AGENTS.md rules are respected.
3. New code has tests with allocator-leak checks where applicable.
4. Public APIs are documented inline.
5. If you touched architecture: there's a matching ADR in `docs/adr/`.
6. At least one CODEOWNER has approved (see `CODEOWNERS`).

## Things you cannot do without an ADR

- Add a new top-level dependency (even a Zig one).
- Add or modify a `build.zig` target.
- Introduce a new file extension other than `.zig`, `.md`, `.zon`, `.txt`,
  `.bin` (datasets/checkpoints — but those are `.gitignored`).
- Change a kernel's reference implementation or numerical-tolerance threshold.
- Add a new GPU backend, runtime feature, or system call.

## Things you cannot do *at all*

- Add `.c`, `.cpp`, `.rs`, `.py`, `.js`, `.ts`, `.go`, `.cu`, `.metal` source files.
- Link BLAS, CUDA, ROCm, Metal, llama.cpp, ggml, PyTorch, ONNX, XLA, or any
  other third-party native library.
- Use `@cImport`, `addTranslateC`, `linkLibC`, or `linkSystemLibrary` for
  third-party code.
- Embed third-party binaries, weights, or kernels in the product.
- Train on data of unknown or illicit provenance.
- Modify `CONSTITUCION.md`, `tools/constitution.sha256`, `tools/gate_*.zig`,
  or `build.zig*` outside the amendment procedure (Art. X).

These rules are enforced by the gates and by the constitution hash. If you
think one of them needs to change, the path is an **amendment**, not a patch.

## Reporting security issues

Do **not** open a public issue for a security vulnerability. Email the project
owner privately (GitHub: [@ItlaStudent](https://github.com/ItlaStudent)) with a
description of the issue and steps to reproduce. We will respond as fast as we
can.

## License of contributions

By submitting a contribution, you agree that it is licensed under the
[Apache License 2.0](LICENSE), the same license as the rest of Fluency. There
is no CLA; the Apache 2.0 grant in §5 of the license is sufficient.

## Code of conduct

Be kind. Be precise. Don't waste reviewer time with PRs you haven't tested.
Disagreements about design go in ADRs and PR comments, not in personal attacks.
