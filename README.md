# Fluency

> **A 100% Zig, subquadratic, agentic local AI engine.**
> Status: **alpha / engine-functional, model-undertrained.** Honest, not aspirational.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Zig](https://img.shields.io/badge/zig-%E2%89%A50.16.0-orange.svg)](https://ziglang.org)
[![Status](https://img.shields.io/badge/status-alpha-yellow.svg)]()

Fluency is a from-scratch AI engine written entirely in [Zig](https://ziglang.org).
It loads Transformer teachers (GGUF), distils them into its own subquadratic
student architecture, and runs everything — tensor math, autodiff, training,
tokenizer, agent runtime — without linking a single byte of foreign code. No
BLAS, no CUDA, no llama.cpp, no PyTorch, no `@cImport`. Just Zig.

## Features

- **Subquadratic mixers:** SSM diagonal scan, linear attention, SparseTopK,
  Gated-DeltaNet-2, plus minority sliding-window attention.
- **Autodiff engine:** forward + backward for matmul, attention, scan, MoE,
  load-balancing loss, and all custom mixers.
- **Quantization:** native Q4_0 and Q8_0 quantization + SIMD dequantization for
  Q4_0, Q4_K, Q6_K, Q8_0. Compatible with GGUF quantized weights.
- **Distillation pipeline:** seq-KD dataset generation, CE/KD/MOHAWK/ORPO losses,
  MoE load-balancing loss (Switch Transformer style).
- **Serving:** OpenAI-compatible (`/v1/chat/completions`, `/v1/embeddings`) and
  Ollama-compatible (`/api/chat`, `/api/embed`) HTTP APIs. Real hidden-state
  embeddings via transformer forward pass.
- **TUI:** terminal chat UI with scroll, resize handling, syntax highlighting,
  themes, command palette.
- **Agent runtime:** deny-by-default tool calling, MCP server, LSP integration.
- **GPU:** SPIR-V kernel support via Vulkan (optional, auto-skips if unavailable).
- **Constitutional governance:** written constitution enforced by CI gates.

## What this is — and isn't

**It is:**

- A working **end-to-end pipeline**: GGUF load → seq-KD dataset → Zig training
  loop with autodiff → checkpoint → greedy sampling → eval.
- A **deliberately constrained** codebase. Every architectural choice is
  written down in the [Constitution](CONSTITUCION.md) and enforced by gates.
- Numerically hardened: NaN/Inf detection in gradients and weights, stable
  log-softmax, RMSNorm with floored variance, stable SiLU, gradient global-norm
  clipping that nulls non-finite tensors.
- Subquadratic by priority. Attention is supported as a *minority mixer*, not
  as the default building block.

**It isn't (yet):**

- A useful model. The published Fluency-139M checkpoint is **undertrained on
  purpose** — small smoke runs over a 50–722 prompt corpus. It demonstrates the
  pipeline; it does not demonstrate strong language ability. Greedy samples
  still mode-collapse. You should not use this checkpoint to do real work.
- A drop-in llama.cpp replacement. We don't have GPU kernels yet, our quantized
  inference is preliminary, and the operator coverage is exactly what the
  in-repo models need — nothing more.
- A research paper. Numbers below are honest, not cherry-picked, and not
  competitive with state-of-the-art models. We will publish a paper when the
  engine deserves one.

## Constitutional rules (the short version)

Fluency is governed by a [written constitution](CONSTITUCION.md) that no
contributor — human or AI — can amend without a human-approved process. The
short version:

1. **100% Zig.** ≥ 0.16.0. No `.c/.cpp/.rs/.py/.js/.ts/.go/.cu/.metal`. No
   `linkSystemLibrary` / `linkLibC` / `@cImport` against third-party native code.
2. **Subquadratic by default**, Transformer-compatible. Attention is the
   exception, not the rule.
3. **Distillation is the only training paradigm.** Teachers must be
   permissively licensed (we track them in [`docs/PROFESORES.md`](docs/PROFESORES.md)).
4. **Deny-by-default agent runtime.** Secrets are encrypted with
   ChaCha20-Poly1305. Tools never `panic`.
5. The Constitution, gates, and `build.zig*` cannot be modified outside the
   amendment procedure. CI verifies the constitution's SHA-256.

If you can't satisfy these rules, you can't ship a PR. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Quick start

```sh
# Build
zig build

# Run the constitutional gates and tests
zig build gate
zig build test
```

### Distill a small model

You'll need a permissively-licensed teacher in GGUF form. We recommend
[SmolLM2-360M-Instruct-GGUF](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF)
(Apache 2.0, Q8_0 ~386 MB):

```sh
# Smoke run: 50 prompts × 100 steps (~2 h on CPU)
fluency distill ./models/smollm2-360m-instruct-q8_0.gguf \
  "Explica el tema de forma clara y útil." \
  100 \
  --prompts ./data/seed_prompts_smoke.txt \
  --dataset ./data/seqkd-smoke.bin \
  --out ./models/fluency-139m-smoke.ckpt
```

The full prompt corpus is in [`data/seed_prompts.txt`](data/seed_prompts.txt) (722 prompts).

### Smoke result (the actual numbers)

| Metric              | Before | After (100 steps, 50 prompts) |
|---------------------|--------|-------------------------------|
| Train cross-entropy | 10.98  | **6.19**                      |
| Train perplexity    | 58 944 | **489**                       |
| Train top-1         | 0.1%   | **8.9%**                      |
| Held-out CE         | —      | **7.85**                      |
| Held-out top-1      | —      | **5.4%**                      |

Held-out > train means **the model generalizes a little**. Top-1 5.4% means
it's still a very weak language model. Greedy decoding mode-collapses
(`504 2365 6354 16438 281 28 28 30 30 30 …`). This is expected for the
training budget — a real run would be orders of magnitude larger.

## Layout

```
src/                 Zig source
  main.zig           CLI entry point (`fluency` binary)
  ztensor/           Tensor primitives (incl. bf16)
  zkernel/           Numerical kernels, quantization/dequantization (Q4_0/Q4_K/Q6_K/Q8_0)
  zgrad/             Autodiff graph engine (attn, scan, sparse_topk, gated_deltanet2, MoE)
  ztrain/            Training loop, schedulers, optimizers, NaN guards
  zdistill/          Distillation pipeline (seq-KD, dataset, loss, FluencyNano)
  zarch/             Architecture/inference integration, mixers (Mamba-2, Gated-DeltaNet, RWKV-7, SWA)
  zgpu/              SPIR-V GPU kernel support (Vulkan)
  ztui/              Terminal UI (chat, scroll, syntax highlighting, themes)
  zserve/            Serving APIs (OpenAI-compatible, Ollama-compatible)
  zagent/            Agent runtime (deny-by-default, tool calling)
  zrpc/              RPC layer
  zmcp/              MCP (Model Context Protocol) server
  zlsp/              LSP integration
  ztok/              Tokenizer
  zload/             GGUF loader
  zollama/           Ollama client for teacher distillation
  platform/          OS / GPU shim (kept tiny by design)
tools/               Constitutional gates and the constitution hash
docs/                Architecture, ADRs, training plan, teacher registry
data/                Prompts (text). Checkpoints are .gitignored.
models/              GGUF teachers and Fluency checkpoints (gitignored).
zig-0.16.0/          Vendored toolchain (gitignored).
```

## Community

Fluency is an independent, open-source project. Community participation is
welcome and encouraged.

- **Bug reports and feature requests:** Open an issue on GitHub.
- **Pull requests:** Read [CONTRIBUTING.md](CONTRIBUTING.md) and the
  [Constitution](CONSTITUCION.md) first. All PRs must pass `zig build gate` and
  `zig build test`.
- **Discussions:** Use GitHub Discussions for questions, ideas, and show-and-tell.

### Contributing

1. Fork the repository.
2. Create a feature branch.
3. Ensure `zig build gate && zig build test` pass.
4. Open a PR with a clear description of what and why.

The Constitution is binding on all contributors — human and AI alike. If your
change requires modifying `CONSTITUCION.md`, `build.zig*`, or `tools/gate_*.zig`,
follow the amendment procedure in [Article X](CONSTITUCION.md).

## Sponsor

If Fluency is useful to you, consider supporting its development:

**GitHub Sponsors:** [@ItlaStudent](https://github.com/sponsors/ItlaStudent)

Sponsorship helps cover compute costs for training runs, development time, and
infrastructure. Every contribution is appreciated.

## License

[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for third-party teacher
attribution.

## Acknowledgements

- The Zig core team for an absurdly good compiler.
- HuggingFaceTB for the SmolLM2 series (Apache 2.0).
- The TinyLlama project (Apache 2.0).
- The GGUF spec authors at llama.cpp.

— *Jesús Rafael Calderón Mejía ([@ItlaStudent](https://github.com/ItlaStudent))*
