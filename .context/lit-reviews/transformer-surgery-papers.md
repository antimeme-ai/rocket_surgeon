---
topic: Academic literature on transformer interpretability, surgery, editing, and mechanistic understanding
status: draft
created: 2026-05-14
sources: ArXiv, transformer-circuits.pub, Anthropic, EleutherAI, Alignment Forum, LessWrong
---

# Transformer Surgery Papers: Lit Review

Academic foundations for building a transformer forward-pass debugger with surgical intervention.

## 1. Activation Patching & Causal Tracing

### ROME (Meng et al., 2022)
- Locating and Editing Factual Associations in GPT
- Uses causal intervention via activation patching to identify factual knowledge localized in specific MLP modules at middle layers during subject token processing
- Treats MLPs as learnable key-value stores — edits via rank-one matrix modifications
- **The foundational paper for model surgery**

### Methodology (Heimersheim)
- Denoising paradigm: run on clean input, cache activations; run on corrupted input, overwrite from clean; observe output changes
- Attribution patching: gradient-based approximation for efficiency

### For rocket_surgeon
Activation patching is the core surgical tool. Should be a first-class debugger operation: cache from run A, patch into run B, measure causal effect.

## 2. Model Editing

### MEMIT (Mass-Editing Memory In a Transformer)
- Scales ROME to thousands of edits
- Batches weight modifications across critical layers
- Suffers from "knowledge attenuation" with many simultaneous edits

### Knowledge Neurons
- Individual neurons responsible for specific facts

### Model Surgery (Parameter Editing)
- Direct manipulation of small parameter subsets via behavior probes
- Up to 90% toxicity reduction with minimal capability degradation (arxiv 2407.08770)

### Weight Arithmetic / Contrastive Weight Steering
- Post-training: edit weights by subtracting differences between two fine-tunes
- Often generalizes better than activation steering (arxiv 2511.05408)

## 3. Mechanistic Interpretability & Circuits (Anthropic)

### A Mathematical Framework for Transformer Circuits (2021)
- **Key insight**: attention heads have two independent computations:
  - QK circuits: compute attention patterns
  - OV circuits: compute output given attention
- Residual stream as communication bus — all components read/write to it
- Enables modular analysis: interventions at specific positions affect downstream predictably

### Induction Heads
- Specific circuit for in-context learning
- Pairs of attention heads: one looks for repeated sequences, second copies next token

### Zoom In: Introduction to Circuits
- From individual neurons to circuits — subgraphs of interpretable computational motifs
- Small, falsifiable circuits can be reverse-engineered

### Circuit Tracing (2025)
- Attribution graphs: DAGs of computation flow from input through features to output
- Nodes = interpretable features, edges = causal interactions
- Iterative pruning of non-causal paths

## 4. Superposition & Polysemanticity

### Toy Models of Superposition (2022)
- Networks pack multiple unrelated features into single neurons
- Phase diagram: when superposition occurs vs monosemanticity
- Links to adversarial examples
- May explain MoE effectiveness

### Implication
Individual neurons are polysemantic. SAEs address this by decomposing into interpretable features.

## 5. Sparse Autoencoders (SAEs) & Dictionary Learning

### Towards Monosemanticity (Anthropic, 2023)
- Sparse dictionary learning on activation vectors
- Produces sparse codes where each feature is interpretable

### Scaling Monosemanticity (May 2024)
- **Major breakthrough**: 34 million features extracted from Claude 3 Sonnet
- <300 active per token; 65% reconstruction of activation variance
- Abstract, multimodal, multilingual features discovered
- Safety-relevant features (deception, sycophancy, bias) identifiable and steerable

### SAE-Based Steering
- Steering Language Model Refusal (arxiv 2411.11296)
- Steering Knowledge Selection (arxiv 2410.15999)
- SAE-SSV: Supervised Steering in Sparse Representation Spaces (arxiv 2505.16188)

### For rocket_surgeon
SAEs solve polysemanticity. Support loading pre-trained SAEs, querying features, steering via coefficient manipulation. Enables surgically precise interventions on interpretable features rather than raw neurons.

## 6. Steering Vectors & Representation Engineering

### Activation Addition (Turner et al., 2023)
- Compute steering vectors from contrasting activation pairs ("Love" vs "Hate")
- Add during inference to steer output
- Strong evidence for feature linearity in activation space

### Representation Engineering (Zou et al.)
- Contrastive pairs to identify behavioral directions
- Scaling and perturbation of representations

### Feature Guided Activation Additions (2025)
- SAE-decoded features for computing steering vectors — more interpretable than raw activation steering

### Key Insight
Concepts (politeness, toxicity, honesty, verbosity) are linear directions in activation space. Experimentally validated. Enables post-hoc steering without training.

## 7. MoE Interpretability

### Polysemantic Experts, Monosemantic Paths (2024)
- Individual experts are polysemantic, but **routing patterns** are monosemantic
- Specific token+context combinations deterministically select experts for interpretable operations
- Suggests control-friendly architecture

### The Expert Strikes Back (2024)
- Expert-level analysis as first-class interpretability primitive
- Experts are architecturally monosemantic, causally validated, controllable at inference time

### Geometric Routing Enables Causal Expert Control (2024)
- Precisely control expert activation via geometric reasoning about routing

### For rocket_surgeon
MoE support is a major differentiator. Allow inspection of routing decisions, selective expert forcing, routing logit modification, expert output steering.

## 8. Knowledge Localization & Factual Recall

### Dissecting Recall of Factual Associations (Geva et al.)
Three-stage process:
1. Subject enrichment (early MLPs encode attributes)
2. Relation propagation (relation info flows to prediction)
3. Object extraction (attention queries enriched subject)

Knowledge localized to specific MLP modules at middle layers during subject token processing.

### Multilingual Knowledge Localization
- Subject enrichment is language-independent; object extraction is language-dependent
- Cross-lingual inconsistencies useful for isolating causal components

## 9. Attention Head Analysis

### Interpreting Transformers Through Attention Head Intervention (2026)
- 70-90% of heads removable without failure — massive redundancy
- Hierarchical: early heads compute primitives, later heads compose
- **Causal ablation differs from attention visualization** — high attention ≠ causal importance

### QK and OV Circuit Analysis
Independent computations that can be analyzed and manipulated separately.

## 10. Residual Stream & Information Flow

- All transformer components communicate via residual stream
- Each layer reads from and writes to it via linear projections
- Modular analysis: interventions at specific positions affect downstream predictably
- VISIT: Visualizing Semantic Information Flow (arxiv 2305.13417)

## 11. Gradient Checkpointing & Reverse Debugging

### Activation Checkpointing
- Save only checkpoint activations during forward, recompute intermediate during backward
- Reduces memory by O(sqrt(n)) at cost of ~20-30% training slowdown
- Optimal: checkpoint every sqrt(n)-th node

### For rocket_surgeon
- Must work with gradient-checkpointed models
- Provide on-demand recomputation of intermediate activations for inspection
- Reverse stepping = restore checkpoint + replay forward (aligns with rr/TTD model from debugger lit review)

## 12. Position-Specific Interventions

### Single-Position Intervention Fails (2025)
- Output templates are distributed across many positions
- Single-position interventions insufficient for ICL

### Positional Encoding Analysis
- RoPE induces "single-head deposit" patterns
- Content-position coupling varies by encoding type

### For rocket_surgeon
Support interventions at specific (layer, position, head) coordinates. Track token representations across positions. But be aware: single-position interventions have limits.

## Tooling Gaps This Project Fills

1. **Existing tools focus on post-hoc analysis** — real-time debugging during forward passes is underexplored
2. **Multi-GPU / distributed debugging** lacks standard tools
3. **MoE-specific interpretability** is nascent but growing
4. **Gradient checkpointing + interpretability** integration is limited
5. **Reversibility & transactions** for interventions are not standard
6. **LLM-native interfaces** for interpretability don't exist

## Key Design Implications

1. **Activation patching as core**: cache, patch, measure causal effects at (layer, position) granularity
2. **SAE integration**: load pre-trained SAEs, steer via coefficients, visualize features alongside raw activations
3. **Causal reasoning**: hypothesis testing — ablation, patching, steering, effect measurement
4. **Position & layer specificity**: (layer, position, head) coordinate system for all interventions
5. **MoE support**: routing viz, expert inspection, selective forcing
6. **Reversibility & transactions**: apply experiments that can be rolled back
7. **Circuit visualization**: graph view of computation flow
8. **Integration**: compatible with TransformerLens, HuggingFace, EleutherAI models

## Key Paper References

- Meng et al. 2022 — ROME (arxiv 2202.05262)
- Heimersheim — How to Use Activation Patching (arxiv 2404.15255)
- Anthropic — Mathematical Framework for Transformer Circuits (transformer-circuits.pub/2021)
- Anthropic — Toy Models of Superposition (transformer-circuits.pub/2022, arxiv 2209.10652)
- Anthropic — Scaling Monosemanticity (transformer-circuits.pub/2024)
- Turner et al. — Activation Addition (arxiv 2308.10248)
- Zou et al. — Representation Engineering (arxiv 2602.11169)
- Vig & Gehrmann — Causal Mediation Analysis (arxiv 2004.12265)
- MoE: arxiv 2604.17837, arxiv 2604.02178, arxiv 2604.14434
- Geva et al. — Dissecting Recall of Factual Associations (OpenReview)
- Attention heads: arxiv 2601.04398
- Circuit tracing: transformer-circuits.pub/2025
- Gradient checkpointing: arxiv 1904.10631
