---
topic: pyvene, inseq, and circuitsvis — intervention, attribution, and visualization tools
status: draft
created: 2026-05-14
sources: pyvene docs/paper, inseq docs/paper, CircuitsVis repo, Anthropic circuit tracing
---

# pyvene + inseq + CircuitsVis: Lit Review

Intervention frameworks, attribution methods, and visualization tools in the mechanistic interpretability ecosystem.

## pyvene (Stanford NLP)

### Architecture: Configuration-as-Data
- **IntervenableConfig**: dict-based specification (serializable, shareable via HuggingFace Hub) declaring what to intervene on
- **IntervenableModel**: hook-based decorator wrapping PyTorch models, executes interventions via Getter/Setter hook pairs
- **Key insight**: treating interventions as first-class serializable primitives, not imperative code. Decouples specification from execution.

### Intervention Types
1. **VanillaIntervention**: replace activation with alternative source (activation swapping)
2. **AdditionIntervention**: add noise/perturbation (corruption-based causal tracing)
3. **CollectIntervention**: pass-through gathering (no modification) — for supervised probes
4. **RotatedSpaceIntervention (DAS)**: trainable rotation matrices in subspaces — finds causally relevant low-rank subspaces
5. **Custom subclasses**: user-defined via Intervention base class

### Composition
- **Parallel**: multiple interventions on different components simultaneously
- **Sequential**: chain with state threading — later interventions see modified activations
- **Generative**: handle decoding-time interventions across sequence generation

### Model Agnosticism
Works across RNNs, ResNets, CNNs, Mamba, Transformers. Per-hook state variables solve stateful model limitations.

### Performance & Limitations
- Authors explicitly state: "designed to support complex intervention schemes at the cost of computational efficiency"
- FSDP support only recently initiated (v0.1.7) — multi-GPU intervention not yet mature
- No built-in batched intervention patterns or throughput benchmarks
- Memory overhead modest for small models (~0.14MB for Llama-2-7B schema)

### What to steal for rocket_surgeon
1. **Configuration-as-data**: serialize interventions as dicts/JSON for versioning, composition, LLM-driven synthesis
2. **Cross-architecture hook strategy**: per-hook state variables for non-Transformer support
3. **Trainable interventions**: RotatedSpaceIntervention shows how to learn intervention parameters
4. **Modularity**: clean separation of config (what) from model (how)

## inseq (Interpretability for Sequence Generation)

### Architecture
- **AttributionModel**: wraps any HuggingFace ForSeq2SeqLM or ForCausalLM model
- **attribute()** method: gradient/perturbation-based feature attribution
- **FeatureAttributionOutput**: structured result with sequence-level and per-step attributions

### Attribution Methods
**Gradient-based** (via Captum): Saliency, Input x Gradient, Integrated Gradients, DeepLift, Gradient SHAP, Sequential Integrated Gradients

**Perturbation-based**: Occlusion, LIME, Value Zeroing, Reagent

**Internals-based**: Attention weights, custom step functions (logits, entropy, contrast probability difference)

**Key innovation**: contrastive attribution via custom `attributed_fn` — "How does feature X contribute to prediction A rather than B?"

### Output Structure
- `sequence_attributions`: list of per-sequence results
- `step_attributions`: per-generation-step (with `include_steps=True`)
- Tensor shape: `(source_len, target_len[, hidden_size])` for layer-level granularity
- Customizable AggregatorPipeline, PairAggregator for contrastive analysis

### Limitations
1. HuggingFace-only (no JAX, no custom models)
2. No evaluation metrics (deliberate — planned ferret collaboration)
3. Sequence generation only (no classification)
4. No scalability guidance for 100B+ models

### What to steal for rocket_surgeon
1. **Per-step attribution structure**: maps naturally to step-through semantics
2. **Contrastive attribution**: "Why A not B?" analysis
3. **Structured output with metadata**: model name, method, execution time — good practice
4. **Multi-method comparisons**: run saliency vs integrated gradients vs perturbation for robustness

## CircuitsVis (TransformerLens Org)

### Architecture
- **React components** as source of truth (built with JS/Svelte)
- **Python wrapper** for Jupyter integration via CDN-hosted React bundles
- Pragmatic: React for interactive web UIs, Python wrapping for notebook access

### Visualization Types
- **Attention patterns**: grid layout (dest x source tokens), interactive hover/click, head-level granularity
- **Token coloring**: `colored_tokens(["My", "tokens"], [0.123, -0.226])` — maps values to color
- **Computational graphs**: circuits as DAGs, feature-to-feature attribution edges
- **Head/Neuron view**: QK vector contributions to attention computation

### Data Format
Simple, flat structures: arrays of strings (tokens), arrays of floats (values), optional metadata (position indices, layer/head IDs). No complex nesting required.

### Limitations
- Browser rendering: large attention matrices (2048x2048) may lag
- No virtual scrolling for long sequences
- No performance benchmarks published
- Tightly coupled to TransformerLens ecosystem

### What to steal for rocket_surgeon
1. **Flat data format**: avoid complex nesting, keep visualization data simple
2. **Component library approach**: modular building blocks (colored_tokens, attention_grid) scale better than monolithic dashboards
3. **React + Python dual-stack**: if building interactive viz, decouple UI iteration from core logic

## Anthropic Circuit Tracing (2025)

### Approach: Attribution Graphs
- Nodes = interpretable features (neurons, heads, concepts)
- Edges = causal interactions (feature A influences B's activation)
- Iterative pruning: remove nodes not on causal path to output

### Methodology
1. Feature identification (via SAEs or similar)
2. Edge pruning via attribution
3. Graph reduction (iteratively remove non-causal nodes)
4. Interactive exploration (Neuronpedia frontend)

### Key Innovation
Uses interpretable features (not raw neurons) to build human-readable causal graphs. Closer to neuroscience (lesion identified neurons) than raw circuit discovery.

### What to steal
1. **Feature-interpretable causal graphs**: easier to explain than raw circuits
2. **Pruning algorithm**: could adapt to surgical intervention workflow
3. **Separate compute backend from frontend**: clean separation

## Design Patterns for rocket_surgeon

### Intervention Specification as Data (from pyvene)
```json
{
  "name": "test_reasoning_head",
  "interventions": [
    {"layer": 10, "component": "attention_heads[2]", "type": "activation_patch", "source": "clean", "target": "corrupt"}
  ]
}
```
Enables version control, sharing, programmatic generation by LLMs.

### Step-Through as Iterator (from inseq)
Per-step attribution naturally maps to debugger step semantics. Each tick yields structured state + attribution data.

### Modular Visualization (from CircuitsVis)
Small composable components, not monolithic dashboards. Flat data formats for easy consumption.

### Avoid
1. Don't couple specification to execution (pyvene lesson)
2. Don't limit to one framework early (inseq's HF-only is pragmatic but limiting)
3. Don't ignore gradient computation overhead (2-3x memory for attribution)
4. Don't make visualization the bottleneck (keep it decoupled)
5. Don't underestimate serialization importance

## Research Gaps / Opportunities
1. No standard benchmarks comparing interpretability tools
2. Limited automated hypothesis generation (LLM-driven intervention synthesis)
3. Multi-GPU intervention underexplored (pyvene FSDP nascent)
4. No contrastive step-through debugging ("why does this input cause A while that causes B?")

## Sources

- github.com/stanfordnlp/pyvene, stanfordnlp.github.io/pyvene
- ACL 2024 NAACL demo: pyvene paper
- arxiv 2403.07809 (pyvene)
- github.com/inseq-team/inseq, inseq.org
- ACL 2023 demo: inseq paper
- arxiv 2302.13942 (inseq)
- github.com/TransformerLensOrg/CircuitsVis
- Anthropic circuit tracing announcement (2025)
- github.com/decoderesearch/circuit-tracer
