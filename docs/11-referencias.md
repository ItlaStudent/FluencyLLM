# 11 — Referencias

[← Plan Maestro](../PLAN-MAESTRO.md) · [← 10 Riesgos y decisiones](10-riesgos-decisiones.md)

Fuentes verificadas que sustentan el plan (corte: **2026-05-29**). Donde una afirmación es de un proveedor y no está verificada independientemente, se marca **[no verificado]**.

---

## A. Zig (lenguaje, ecosistema, ML)

- Descargas/versiones (0.16.0 estable, 2026-04-13): https://ziglang.org/download/
- Notas 0.16.0 (interfaz `std.Io`, async en stdlib, `@cImport` deprecado, `@Vector` endurecido): https://ziglang.org/download/0.16.0/release-notes.html
- Notas 0.15.1 ("Writergate", removal de async keywords, ArrayList unmanaged): https://ziglang.org/download/0.15.1/release-notes.html
- Migración a Codeberg (repo canónico desde 2025-11-26): https://ziglang.org/news/migrating-from-github-to-codeberg/ · https://codeberg.org/ziglang/zig
- **`std.Thread.Pool` eliminado** (PR #30557, merge 2025-12-22): https://codeberg.org/ziglang/zig/pulls/30557
- **bf16 no nativo** (propuesta #3148, abierta): https://github.com/ziglang/zig/issues/3148
- Referencia del lenguaje (vectores, floats, comptime): https://ziglang.org/documentation/0.16.0/
- GPU en Zig (SPIR-V/PTX/AMDGCN, estado experimental): https://alichraghi.github.io/blog/zig-gpu/
- Sistema de build / `build.zig.zon`: https://ziglang.org/learn/build-system/ · https://github.com/ziglang/zig/blob/master/doc/build.zig.zon.md
- ZML (stack de inferencia Zig sobre MLIR/XLA — *no* Zig puro): https://github.com/zml/zml
- Zigrad (framework con entrenamiento, BLAS-dependiente, en reescritura): https://github.com/Marco-Christiani/zigrad
- llama2.zig (cgbur — Zig puro, supera a llama2.c **[claim del autor]**): https://github.com/cgbur/llama2.zig
- zllm (Zig 0.16 + Metal): https://github.com/jiacai2050/zllm
- Suite SMC17 (safetensors-zig, tokenizers-zig, vllm-zig, faiss-zig — auditar madurez): https://github.com/SMC17
- Awesome Zig: https://github.com/zigcc/awesome-zig · Awesome Zig LLM: https://ziggit.dev/t/awesome-zig-llm-a-curated-list-for-ai-ml-in-zig/15287

## B. Arquitecturas subcuadráticas

- **subq.ai** (sitio): https://subq.ai/ · "Introducing SubQ": https://subq.ai/introducing-subq · **[todas sus cifras: no verificadas; sin paper ni código públicos]** — cobertura: VentureBeat https://venturebeat.com/technology/miami-startup-subquadratic-claims-1-000x-ai-efficiency-gain-with-subq-model-researchers-demand-independent-proof
- Mamba (SSM selectivo): `arXiv:2312.00752` https://arxiv.org/abs/2312.00752
- Mamba-2 / State Space Duality (SSD): `arXiv:2405.21060` https://arxiv.org/abs/2405.21060
- Gated Linear Attention (GLA) + FlashLinearAttention: `arXiv:2312.06635` https://arxiv.org/abs/2312.06635
- RetNet: `arXiv:2307.08621` https://arxiv.org/abs/2307.08621
- DeltaNet: `arXiv:2406.06484` https://arxiv.org/abs/2406.06484
- Gated DeltaNet: `arXiv:2412.06464` https://arxiv.org/abs/2412.06464
- Gated DeltaNet-2 (NVIDIA, 2026-05-24; código **no comercial**): https://github.com/NVlabs/GatedDeltaNet
- RWKV-7 "Goose": `arXiv:2503.14456` https://arxiv.org/abs/2503.14456 · historia RWKV: https://wiki.rwkv.com/basic/architecture.html
- Hyena (long-conv): `arXiv:2302.10866` https://arxiv.org/abs/2302.10866
- Híbridos: Jamba `arXiv:2403.19887` · Samba `arXiv:2406.07522` · Nemotron-H `arXiv:2504.03624` (https://research.nvidia.com/labs/adlr/nemotronh/) · MiniMax-M1 `arXiv:2506.13585`
- flash-linear-attention (referencia conceptual de kernels e híbridos; Python/Triton — **no se enlaza, se reimplementa en Zig**): https://github.com/fla-org/flash-linear-attention
- Guía SSM-Transformer hybrids: https://n1o.github.io/posts/ssm-transformer-hybrids-guide/

## C. Agéntica

- **hermes-agent** (NousResearch, referencia conceptual): https://github.com/NousResearch/hermes-agent · formato de tool-call / function-calling: https://github.com/NousResearch/Hermes-Function-Calling · formato de trayectoria: https://hermes-agent.nousresearch.com/docs/developer-guide/trajectory-format
- **nullclaw** (Zig 0.16.0, referencia de implementación directa): https://github.com/nullclaw/nullclaw · https://nullclaw.io/ · writeup: https://www.marktechpost.com/2026/03/02/meet-nullclaw-the-678-kb-zig-ai-agent-framework-running-on-1-mb-ram-and-booting-in-two-milliseconds/
- Patrones (ReAct / Plan-and-Execute / Reflexión): https://dev.to/gabrielanhaia/react-plan-and-execute-or-reflection-the-three-agent-patterns-every-engineer-needs-in-2026-355p · Plan-and-Act `arXiv:2503.09572`
- Memoria de agentes (CoALA, mem0): https://mem0.ai/blog/what-is-ai-agent-memory
- Decodificación restringida (XGrammar-2): https://blog.mlc.ai/2026/05/04/xgrammar-2-fast-customizable-structured-generation · GBNF (llama.cpp grammars): https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md
- Structured outputs reducen errores de tool-use **[claim AWS]**: https://aws.amazon.com/blogs/machine-learning/structured-outputs-with-amazon-nova-a-guide-for-builders/
- Speculative tool calls: `arXiv:2512.15834`

## D. Formatos, interop y destilación

- **GGUF spec**: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md · overview HF: https://huggingface.co/docs/hub/en/gguf
- Estructuras de cuantización (`ggml-common.h`) y dequant (`ggml-quants.c`); esquemas de bits: https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes
- **safetensors**: https://github.com/huggingface/safetensors · https://huggingface.co/docs/safetensors/index
- **Ollama** API: https://github.com/ollama/ollama/blob/main/docs/api.md · Modelfile: https://docs.ollama.com/modelfile · compat OpenAI: https://docs.ollama.com/api/openai-compatibility
- Tokenizers: BPE desde cero https://sebastianraschka.com/blog/2025/bpe-from-scratch.html · SentencePiece `arXiv:1808.06226` · WordPiece/Unigram (HF) https://huggingface.co/learn/llm-course/en/chapter6/7
- **MOHAWK (Transformer→SSM)**: `arXiv:2408.10189` (código ref. `goombalab/phi-mamba` — reimplementar en Zig)
- **Mamba-in-the-Llama**: `arXiv:2408.15237` (`jxiw/MambaInLlama`)
- **Llamba** (destilación a Mamba vía MOHAWK): `arXiv:2502.14458`
- Destilación: Logit Standardization `arXiv:2403.01427` · On-Policy Distillation (Thinking Machines): https://thinkingmachines.ai/blog/on-policy-distillation/ · survey https://pmc.ncbi.nlm.nih.gov/articles/PMC12634706/
- Optimizadores: Muon https://kellerjordan.github.io/posts/muon/ · AdaMuon `arXiv:2507.11005` · NorMuon `arXiv:2510.05491`
- Preferencias: DPO https://openreview.net/pdf?id=HPuSIXJaa9 · ORPO

## E. Protocolos y extensiones

- **MCP** (spec 2025-11-25): https://modelcontextprotocol.io/specification/2025-11-25 (lifecycle, transports, tools, resources, prompts, sampling, roots, elicitation en sub-rutas) · repo: https://github.com/modelcontextprotocol/modelcontextprotocol · roadmap 2026-07-28 **[RC, no estable]**: https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/
- Hooks MCP (~27 eventos, **version-dependiente**): ver spec MCP para eventos de lifecycle del host
- Skills (Agent Skills, SKILL.md, progressive disclosure): convención estándar de skills con metadatos en SKILL.md y progressive disclosure de capacidades
- Plugins MCP: ver spec MCP para mecanismos de empaquetado y registro de plugins
- API OpenAI-compatible (de-facto; streaming/tool_calls): https://developers.openai.com/api/reference/ · (corroborado por Ollama/vLLM)
- Conexión de clientes a modelos locales: cualquier cliente compatible con OpenAI/Ollama puede consumir Fluency
- LSP 3.17 (JSON-RPC + Content-Length): https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/

---

## Notas de fiabilidad

- **Fuentes primarias** (papers arXiv, specs oficiales, repos, release notes) son las **vinculantes**. Blogs/prensa se usan como corroboración secundaria.
- **subq.ai**: tratar todas sus cifras como **no verificadas**; el diseño de ZSA **no** depende de ellas (Art. II.4).
- **Claims de rendimiento** de terceros (p. ej. "Nx más rápido que…") son de sus autores salvo verificación propia.
- **Detalles que conviene revalidar al implementar**: indexado exacto de sub-escalas en K-quants Q2_K/Q3_K (vs `ggml-quants.c` vigente); nombre exacto de `GeneralPurposeAllocator`/`DebugAllocator` en la versión Zig fijada; ecuaciones MOHAWK contra el PDF final; roster de eventos de hooks contra la versión del host MCP en uso.

---

[← Volver al Plan Maestro](../PLAN-MAESTRO.md)
