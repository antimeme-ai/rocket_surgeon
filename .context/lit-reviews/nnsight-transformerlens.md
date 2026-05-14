---
topic: nnsight, TransformerLens, and nnterp — interception frameworks for transformer internals
status: draft
created: 2026-05-14
sources: nnsight docs/blog, TransformerLens docs/GitHub, nnterp paper, Neel Nanda writings
---

# nnsight + TransformerLens + nnterp: Lit Review

The two dominant approaches to intercepting transformer internals, and the emerging bridge between them.

## nnsight (NDIF Team, Northeastern)

### Core Architecture
- **Deferred execution with thread-based synchronization**: enter `with model.trace()`, code captured via Python AST parsing, executed in separate worker thread
- **Intervention Graph**: portable, serializable, bipartite DAG decoupling experimental design from runtime
- Tensors not directly available — `.save()` required to persist outside context
- Thread sync via PyTorch hooks: accessing `.output`/`.input` blocks until value arrives from forward pass thread

### Key Abstractions
- **Module Properties**: `.output`, `.input`, `.inputs` — main thread waits for values
- **Execution Contexts**: `model.trace()` (single pass), `model.generate()` (autoregressive), `model.session()` (grouped), `model.scan()` (shape inference), `model.edit()` (persistent mods)
- **Control Flow**: `.stop()` for early termination, `SkipException` to skip downstream layers
- **Envoys**: proxy objects for remote model execution — unified local/remote API

### Multi-GPU
- Full vLLM tensor parallelism integration (v0.6)
- Ray-based distributed execution, async streaming
- NVLink: ~0.92x scaling per card; PCIe: 0.70-0.78x
- All-reduce after every transformer layer during inference — communication dominates with PCIe
- NDIF remote infra amortizes model loading costs

### Performance (v0.6)
- Setup cost: ~210µs (down from ~1,100µs)
- Per-intervention cost: ~34µs
- Overhead constant regardless of model size, negligible for forward passes taking seconds
- Source extraction/AST parsing now cached

### Critical Limitation
**Must access modules in execution order** — accessing layer 5 before layer 2 causes deadlock. This is the primary footgun.

### API Ergonomics
- Works with ANY PyTorch model (not just transformers)
- Supports vision, multimodal, diffusion models
- v0.6: error messages point to user code, MCP server integration
- Serializes custom functions by value for remote execution

## TransformerLens (Neel Nanda)

### Core Architecture
- **Re-implements transformer architectures from scratch** to add HookPoints at every activation site
- Inspired by Anthropic's internal Garcon tool
- HookPoint: dummy module (identity by default) wrapping intermediate activations

### HookPoint Abstraction
- Named hierarchically: `blocks.{i}.hook_resid_pre`, `blocks.{i}.attn.hook_z`, etc.
- PyTorch's native hooks wrapped in quality-of-life abstraction
- `run_with_hooks()`: temporary hooks for single forward pass
- `run_with_cache()`: capture all activations into cache dict

### Design Philosophy
"Keep the gap between having an experiment idea and seeing the results as small as possible" — optimized for research velocity and exploratory analysis.

### Strengths
- **Standardized naming** across 50+ model variants
- Rich cache system with metadata for precise patching
- Activation patching with clean API
- Huge community: 50+ published papers
- Exploratory-friendly — works in Colab

### Critical Limitations
- **Limited to transformer LLMs**: no vision, diffusion, other architectures
- **No multi-GPU support**: single-GPU inference only
- **Re-implementation overhead**: each new architecture needs manual implementation
- **Model parity concerns**: re-implementations may not exactly match HuggingFace
- **Not suitable for 70B+ models** without significant memory engineering
- **Hook global state**: hooks persist until explicitly removed, run_with_hooks masks this

### Footguns
- Prepends BOS token by default (prepend_bos=False to disable)
- Hooks added but not removed persist alongside fixed versions
- Multiple forward passes accumulate hook overhead
- Re-implementation can diverge from HuggingFace numerically

## nnterp (Emerging Bridge)

### What it solves
Bridges TransformerLens consistency and nnsight generality. Lightweight wrapper around nnsight mapping diverse HuggingFace conventions to standardized names.

### Key Innovation
- Module renaming: GPT-2's `transformer.h` -> `layers`; LLaMA's `model.layers` stays `layers`
- I/O accessors: `model.layers_output[5]` handles tensor/tuple differences
- Standardized methods: logit lens, patchscope, activation steering work identically across 50+ models
- Validation tests catch implementation bugs

### Tradeoff
"Sanity checks rather than formal correctness guarantees." Attention probability access is "very implementation sensitive" and may break with new HuggingFace releases or Flash Attention.

## Comparative Analysis

| Dimension | nnsight | TransformerLens | nnterp |
|---|---|---|---|
| Scope | Any PyTorch model | Transformer LLMs only | Transformer LLMs via nnsight |
| Approach | Wraps existing models | Re-implements architectures | Thin layer on nnsight |
| Naming | Model-specific | Standardized | Standardized (maps HF) |
| Remote Execution | Yes (NDIF) | No | Yes (via nnsight) |
| Multi-GPU | vLLM integration | No | Yes (via nnsight+vLLM) |
| Performance | 210µs setup, 34µs/intervention | Low overhead | Low overhead |
| Maturity | Production (v0.6) | Stable | Early (ICLR 2025) |

## OpenAI Transformer Debugger
Worth noting: OpenAI released an interactive debugger combining automated interpretability with SAEs. Allows surgical intervention with immediate visualization. Traces connections for circuit discovery. Research-focused UI, not designed for programmatic/LLM use. **This is closest to rocket_surgeon's concept but lacks multi-GPU, LLM-native interface, and step-through semantics.**

## Design Lessons for rocket_surgeon

### From nnsight
1. **Deferred execution enables serialization/remote execution** but has execution-order footgun
2. **Step-through debugger naturally enforces execution order** — this is a feature, not a bug
3. **vLLM integration for multi-GPU** is proven path
4. **Module property access (.output, .input)** is clean API pattern

### From TransformerLens
1. **Standardized naming across architectures** is essential for ergonomics
2. **Explicit activation caching** better than hook global state for read operations
3. **run_with_hooks() scoped pattern** prevents hook leaks
4. **Research velocity** matters — short feedback loops

### From nnterp
1. **Adaptive naming layer** is the right compromise between consistency and flexibility
2. **Don't re-implement models** — wrap them with standardized access

### Hybrid Recommendation
- Capture interventions declaratively (like nnsight) for serialization/reproducibility
- Execute eagerly (like TransformerLens) for step-through debugging UX
- Standardize naming (like nnterp) without re-implementing architectures
- Step-through interface naturally enforces execution order (nnsight's footgun becomes a feature)

## Sources

- github.com/ndif-team/nnsight, nnsight.net
- nnsight 0.6 blog post (2026/02)
- arxiv 2407.14561 (NNsight and NDIF paper)
- github.com/TransformerLensOrg/TransformerLens
- transformerlensorg.github.io/TransformerLens
- arxiv 2511.14465 (nnterp paper)
- github.com/Butanium/nnterp
- github.com/openai/transformer-debugger
- neelnanda.io/mechanistic-interpretability
