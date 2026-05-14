# Interpretability Tooling Analysis

Analysis of 7 quarantine repos and 4 priority papers for rocket_surgeon's interpretability integration layer.

---

## 1. Per-Repo Findings

### 1.1 SAELens (`quarantine/SAELens/`)

**Architecture.** Two-tier class hierarchy: abstract `SAE` base (in `sae_lens/saes/sae.py`) defines `encode()`/`decode()` contract plus HookPoints for every internal activation; concrete implementations (`StandardSAE`, `TopKSAE`, `JumpReLUSAE`, `GatedSAE`, etc.) implement the actual forward logic. A registry pattern (`register_sae_class("standard", StandardSAE, StandardSAEConfig)`) maps architecture names to classes.

**Key abstractions:**
- `SAEConfig`: frozen dataclass with `d_in`, `d_sae`, `dtype`, `device`, `apply_b_dec_to_input`, `normalize_activations`, `reshape_activations`. Metadata (hook_name, model_name) lives in a separate `SAEMetadata` dict-like.
- `SAE` base class inherits `HookedRootModule`, giving every SAE six internal HookPoints: `hook_sae_input`, `hook_sae_acts_pre` (pre-activation), `hook_sae_acts_post` (post-activation = feature activations), `hook_sae_recons` (reconstruction), `hook_sae_output`, `hook_sae_error`.
- `StandardSAE.encode()`: `sae_in @ self.W_enc + self.b_enc` -> activation_fn -> feature_acts. `decode()`: `feature_acts @ self.W_dec + self.b_dec`.

**HookedSAETransformer** (in `sae_lens/analysis/hooked_sae_transformer.py`): extends TransformerLens's `HookedTransformer`. Uses `_SAEWrapper` to wrap an SAE and attach it at a hook point via `set_deep_attr()`. Critical pattern: **error term computation** -- recomputes clean SAE output without hooks, computes `original_output - sae_out_clean` as error term, adds it back so that interventions on individual features don't get masked by reconstruction error. Methods: `add_sae()`, `reset_saes()`, `saes()` context manager, `run_with_cache_with_saes()`.

**SAETransformerBridge** (in `sae_lens/analysis/sae_transformer_bridge.py`): Same pattern but for HuggingFace models (not HookedTransformer). Resolves hook aliases. Beta status.

**Loader ecosystem** (in `sae_lens/loading/pretrained_sae_loaders.py`): Multiple loaders for different providers -- `sae_lens`, `gemma_2`, `gemma_3`, `llama_scope`, `deepseek_r1`, `qwen_scope`, `sparsify`, `transcoders`. All return `(cfg_dict, state_dict, log_sparsity)`. Standardized config keys: `architecture`, `d_in`, `d_sae`, `hook_name`, `model_name`. Weight keys: `W_enc`, `W_dec`, `b_enc`, `b_dec`, `threshold` (for JumpReLU).

**SAE variants supported:** Standard (ReLU + L1), TopK, JumpReLU, BatchTopK, Gated, Matryoshka, Transcoders (input hook differs from output hook, e.g. MLP input -> MLP output).

**Gotchas for rocket_surgeon:**
- Tightly coupled to TransformerLens's HookPoint system; the bridge for HuggingFace is beta
- Error term computation is essential for meaningful feature-level intervention -- without it, removing a feature also removes the correlated reconstruction error
- SAE attachment uses `set_deep_attr()` which mutates the model object
- `normalize_activations` and `apply_b_dec_to_input` flags change the encode/decode pipeline subtly

### 1.2 Automatic Circuit Discovery (`quarantine/Automatic-Circuit-Discovery/`)

**Graph structure** (in `acdc/TLACDCCorrespondence.py`):
- `graph[name][TorchIndex] -> TLACDCInterpNode` -- the node registry
- `edges[child_name][child_index][parent_name][parent_index] -> Edge` -- 4-level nested dict
- `setup_from_model()` builds the full computational DAG from a HookedTransformer: residual stream nodes, MLP nodes (with mlp_in/mlp_out separation), attention head nodes (per-head via TorchIndex `[None, None, head_idx]`), Q/K/V input nodes

**Edge types** (in `acdc/TLACDCEdge.py`):
- `ADDITION` (0): parent contributes additively to child (residual stream)
- `DIRECT_COMPUTATION` (1): single parent computes child (e.g., hook_q_input -> hook_q)
- `PLACEHOLDER` (2): graph connectivity edges that are always included (multi-input nodes like hook_result from q/k/v)

**TorchIndex** (in `acdc/TLACDCEdge.py`): wraps a list of ints/Nones into both an indexable tuple (for tensor slicing) and a hashable tuple (for dict keys). E.g., `TorchIndex([None, None, 3])` represents `[:, :, 3]`.

**TLACDCExperiment** (in `acdc/TLACDCExperiment.py`): the main ACDC algorithm. Takes model, dataset, ref_dataset, threshold, metric. Uses `sender_hook`/`receiver_hook` for path patching -- sender hooks replace activations with corrupted values, receiver hooks capture the result. `step()` processes one node at a time in reverse topological order. Supports both zero ablation and corrupted-input ablation.

**Gotchas for rocket_surgeon:**
- Extremely tightly coupled to TransformerLens (references `model.cfg.n_layers`, `model.cfg.n_heads`, `model.cfg.attn_only` directly)
- The graph construction logic in `setup_from_model()` hardcodes the TransformerLens hook naming scheme (`blocks.{layer}.attn.hook_result`, `blocks.{layer}.hook_mlp_out`, etc.)
- No MoE support -- the graph structure assumes dense transformers
- The parent/child convention is **reversed** from typical usage: nodes closer to input tokens are *parents*

### 1.3 tuned-lens (`quarantine/tuned-lens/`)

**Core abstractions** (in `tuned_lens/nn/lenses.py`):
- `Lens` ABC: `transform_hidden(h, idx)` and `forward(h, idx)` -- transform then unembed
- `LogitLens`: identity transform (just unembed)
- `TunedLens`: `layer_translators` = `ModuleList` of `Linear(d_model, d_model)`, initialized to zero. `transform_hidden(h, idx)` returns `h + self[idx](h)` -- residual connection ensures it starts as identity. Forward: transform then unembed.

**Unembed** (in `tuned_lens/nn/unembed.py`): extracts `final_norm` + `unembedding` from model. Forward: `self.unembedding(self.final_norm(h))`. Has `invert()` method using L-BFGS/SGD to find hidden state that minimizes KL divergence from target logits.

**Key insight:** The tuned lens is dead simple architecturally -- one learned affine transform per layer, applied residually. Training uses KL divergence between tuned lens output and final layer logits. The value is in the trained weights, not the code.

**Gotchas for rocket_surgeon:**
- Requires pre-trained translator weights per model -- these need to be loaded from somewhere
- The unembedding extraction logic assumes specific model architecture naming (final_norm, lm_head)
- The `invert()` capability (finding hidden states that produce target logits) is interesting for surgery -- lets you compute "what hidden state would produce this output?"

### 1.4 steering-vectors (`quarantine/steering-vectors/`)

**SteeringVector** (in `steering_vectors/steering_vector.py`): dataclass with `layer_activations: dict[int, Tensor]` and `layer_type`. Core methods:
- `patch_activations()`: registers forward hooks that add `multiplier * activation` to model layer outputs. Supports `token_indices` for selective patching (only modify specific positions).
- `apply()`: context manager that wraps patch/remove.

**Training** (in `steering_vectors/train_steering_vector.py`):
- `train_steering_vector()`: takes model, tokenizer, contrastive pairs `(positive_str, negative_str)`. Extracts activations at `read_token_index` (default -1, i.e., last token). Aggregates via `mean_aggregator` (default): `mean(positive) - mean(negative)` per layer.

**Operators** (in `steering_vectors/steering_operators.py`):
- `addition_operator()`: returns steering vector unchanged (default, additive steering)
- `ablation_operator()`: projects activation onto steering direction and negates it (removes the component)
- `ablation_then_addition_operator()`: ablate then add (replace the component)

**Gotchas for rocket_surgeon:**
- Simple and clean API -- good model for rocket_surgeon's steering interface
- No per-head granularity -- operates on full layer outputs
- The contrastive training approach requires paired positive/negative examples
- `token_indices` parameter enables position-selective steering

### 1.5 repeng (`quarantine/repeng/`)

**ControlVector** (in `repeng/extract.py`): `directions: dict[int, np.ndarray]`. Training methods:
- `train()`: uses `read_representations()` with PCA on contrastive activation differences
- `train_with_sae()`: encodes hidden states through SAE before PCA, optionally decodes back
- `read_representations()`: extracts last-token hidden states, applies PCA. Two modes: `pca_diff` (PCA on positive-negative differences) and `pca_center` (PCA on centered pairs). Signs vectors so positive > negative projections.

**ControlModel** (in `repeng/control.py`): wraps `PreTrainedModel`, **replaces layers** with `ControlModule` wrappers. `set_control(vector, coeff)` applies per-layer. `ControlModule.forward()` adds control vector to output, supports: `normalize` (preserve activation norm after adding vector), custom operator, padding-aware masking.

**Key difference from steering-vectors:** repeng replaces model layers with wrapper modules; steering-vectors uses forward hooks. repeng's approach is more invasive but gives more control (normalization, padding awareness). Supports GGUF export for llama.cpp.

**Gotchas for rocket_surgeon:**
- `train_with_sae()` demonstrates SAE + steering vector composition
- The `normalize` option (preserving activation norm) is important for keeping model behavior stable during steering
- Layer replacement approach is harder to compose with other interventions than hooks

### 1.6 transformer-utils (`quarantine/transformer-utils/`)

**Logit lens** (in `src/transformer_utils/logit_lens/hooks.py`):
- `make_lens_hooks()`: registers forward hooks on layer names. Each hook captures output, applies decoder (`final_layernorm` + `lm_head`) to get logits.
- `make_decoder()`: composes layers with residual connections. Stores per-layer logits in `model._layer_logits`.
- Handles residual additions (tracks running residual stream, adds attn/mlp outputs)

**Key insight:** This is the original logit lens implementation, simpler than tuned-lens. The pattern is: intercept hidden state at each layer, project through final_norm + unembedding, get per-layer token predictions. No learned parameters.

**Gotchas for rocket_surgeon:**
- Stores results as a model attribute (`model._layer_logits`) -- side-effect based
- Assumes specific naming for model layers
- The residual tracking logic shows how to correctly attribute intermediate predictions

### 1.7 EleutherAI sae (`quarantine/sae/`)

**SparseCoder** (in `sparsify/sparse_coder.py`): TopK architecture with fused encoder. Key features:
- `encode()`: subtracts `b_dec` first (unless transcoding), then `fused_encoder` with top-k selection
- `decode()`: uses sparse `decoder_impl` (exploits sparsity for efficiency)
- Supports **transcoders** with `W_skip` for skip connections
- `load_many()`: loads per-hookpoint SAEs from HuggingFace

**Training** (in `sparsify/trainer.py`):
- Supports DDP and distributed modules across ranks
- Loss functions: FVU (local), CE (end-to-end), KL (end-to-end)
- Hookpoint-based activation capture via forward hooks
- K-decay schedule from `initial_k` to `final_k`
- Optimizers: Adam, Muon, Signum
- AuxK loss for dead feature revival
- Multi-TopK FVU regularization

**Key differences from SAELens:**
- More training-focused, less analysis-focused
- Sparse decoder implementation for efficiency
- Native DDP support for distributed training
- Transcoder support with skip connections
- No HookPoint integration -- uses raw PyTorch hooks

**Gotchas for rocket_surgeon:**
- The `load_many()` pattern (loading SAEs per hookpoint) is the right model for multi-layer SAE deployment
- Sparse decode is a performance consideration for real-time debugging
- DDP support relevant for multi-GPU rocket_surgeon deployments

---

## 2. SAE Integration Design

### 2.1 What rocket_surgeon needs from SAEs

SAEs decompose superposed activations into interpretable features. For a debugger, this means:
1. **Feature-level inspection**: at any tick, decompose the current activation into active features with their coefficients
2. **Feature-level surgery**: modify individual feature activations (amplify, suppress, zero, set to arbitrary value)
3. **Error-aware intervention**: when modifying features, preserve the reconstruction error so the model doesn't break (SAELens pattern)
4. **Multi-point attachment**: attach SAEs at residual stream, MLP output, attention output -- possibly different SAEs at different points

### 2.2 Architecture pattern to adopt

**SAE abstraction:**
```
trait SparsAutoencoder:
    encode(activation: Tensor) -> (feature_acts: Tensor, indices: Tensor)  # sparse
    decode(feature_acts: Tensor, indices: Tensor) -> Tensor
    config: SAEConfig  # d_in, d_sae, architecture name
```

**Loading:** Adopt SAELens's multi-loader pattern. The `(cfg_dict, state_dict)` return format is a clean interface. Support loading from: SAELens format, EleutherAI sparsify format, Gemma/Llama/DeepSeek provider formats.

**Error term computation (critical):**
```python
# SAELens pattern -- must replicate
original_activation = hook_output
sae_reconstruction = sae.decode(sae.encode(original_activation))
error_term = original_activation - sae_reconstruction
# After feature intervention:
modified_reconstruction = sae.decode(modified_features)
final_output = modified_reconstruction + error_term
```
Without the error term, feature interventions are corrupted by reconstruction error. This is the single most important pattern from SAELens.

**Attachment model:** Use rocket_surgeon's own hook system (not TransformerLens HookPoints). At each tick where an SAE is attached:
1. Capture the activation
2. Run encode() to get features
3. Present features to user/LLM (top-k active features with coefficients)
4. Allow modification
5. Run decode() on modified features + error term
6. Replace activation with result

### 2.3 Transcoder support

Transcoders (SAE where input hook != output hook, e.g., MLP input -> MLP output) need special handling:
- The encode point and decode point are different ticks
- Need to buffer the encoded features between ticks
- Both SAELens and EleutherAI's sparsify support these

### 2.4 Concrete weight layout

Standard SAE weights: `W_enc` (d_in, d_sae), `b_enc` (d_sae), `W_dec` (d_sae, d_in), `b_dec` (d_in).
TopK adds: no threshold, just top-k selection on pre-activation.
JumpReLU adds: `threshold` (d_sae) per-feature threshold.
Gated adds: `W_gate` (d_in, d_sae), `b_gate` (d_sae), `r_mag` (d_sae).

---

## 3. Logit Lens / Tuned Lens Patterns

### 3.1 Core operation

Both logit lens and tuned lens answer: "if the model stopped at layer L, what would it predict?"

**Logit lens:** `logits_L = unembed(final_norm(h_L))` -- identity transform, no parameters.
**Tuned lens:** `logits_L = unembed(final_norm(h_L + A_L @ h_L + b_L))` -- learned affine per layer, residual.

### 3.2 Integration into rocket_surgeon

At every tick that touches a residual stream position, rocket_surgeon should offer a "lens view":
1. Take current hidden state `h`
2. Apply tuned lens translator if available, otherwise identity (logit lens)
3. Apply final LayerNorm + unembedding
4. Return top-k token predictions with probabilities

**Implementation is trivial:** needs access to final_norm and unembedding weights (extract once at model load) plus optional tuned lens weights (one Linear per layer, ~2MB per layer for a 4096-dim model).

### 3.3 Inversion capability

Tuned-lens's `Unembed.invert()` finds a hidden state that produces target logits using L-BFGS. This enables **goal-directed surgery**: "I want this layer to predict token X -- what hidden state would do that?" rocket_surgeon should expose this as a surgery primitive.

### 3.4 Causal basis extraction (from Belrose2023)

Beyond per-layer predictions, the tuned lens enables **causal basis extraction** (CBE): finding the principal directions in hidden space that cause the largest changes in lens predictions. This is a more principled version of "which dimensions matter at this layer." rocket_surgeon could offer this as an analysis tool.

---

## 4. Steering / Intervention Mechanics

### 4.1 Taxonomy of interventions

From the repos and papers, interventions fall into a clean hierarchy:

| Level | What | Example |
|-------|------|---------|
| Activation replacement | Replace entire activation at a hook point | Activation patching (ACDC) |
| Additive steering | Add a vector to activations | Steering vectors, control vectors, ActAdd (Turner2023) |
| Feature-level modification | Modify individual SAE feature coefficients | SAELens feature intervention |
| Ablation | Zero out or project away a component | Zero ablation, directional ablation |
| Weight editing | Directly modify model weights | ROME (Meng2022) |

rocket_surgeon should support all of these through a unified intervention API.

### 4.2 Steering vector mechanics

**Training pattern** (consistent across steering-vectors and repeng):
1. Collect activations from contrastive pairs (positive/negative examples)
2. Compute difference vectors (positive_mean - negative_mean per layer)
3. Optionally apply PCA (repeng) for robustness
4. Result: one vector per layer

**Application pattern:**
1. Register hook at target layer(s)
2. In hook: `output = output + multiplier * steering_vector`
3. `multiplier` controls strength and direction (negative = opposite)

**Variations:**
- Position-selective: only steer at specific token positions (steering-vectors's `token_indices`)
- Norm-preserving: after adding vector, rescale to original norm (repeng's `normalize`)
- Ablation: project out the steering direction instead of adding
- Ablation-then-addition: remove existing component, add desired component

### 4.3 The activation patching family

From ACDC and general mech-interp practice:
- **Activation patching**: run model on clean input, save activations. Run on corrupted input. At one point, swap in the clean activation. Measure change in output. High change = that activation matters.
- **Path patching**: more fine-grained. For an edge (parent -> child), patch only the parent's contribution to the child, not the entire activation.
- **Zero ablation**: replace activation with zeros (simpler but less principled).
- **Mean ablation**: replace with dataset mean.

ACDC's `sender_hook`/`receiver_hook` pattern for path patching is elegant: sender hook saves the corrupted activation; receiver hook replaces only the component from that specific sender.

### 4.4 Concrete API implications for rocket_surgeon

```
# At a tick, the user/LLM should be able to:
surgeon.inspect(hook_name, index)           # see the activation
surgeon.replace(hook_name, index, tensor)   # full replacement
surgeon.add(hook_name, index, vector, mult) # additive steering
surgeon.ablate(hook_name, index, direction) # directional ablation
surgeon.set_feature(hook_name, index, feat_id, value)  # SAE feature
surgeon.patch_from(hook_name, index, source_run)  # activation patching
```

---

## 5. Circuit Discovery Exposure

### 5.1 What ACDC teaches about computational graphs

The ACDC codebase reveals a precise formalization of the transformer computational graph:

**Nodes:** Each node is `(hook_name, TorchIndex)` -- a specific slice of a hooked tensor. For attention: per-head slicing via `[:, :, head_idx]`. For MLP/residual: `[:]` (full tensor).

**Edges:** Three types modeling different computational relationships:
- ADDITION: residual stream composition (embedding + attn_0_out + mlp_0_out + attn_1_out + ...)
- DIRECT_COMPUTATION: single-input function (hook_q_input -> hook_q is just a matrix multiply)
- PLACEHOLDER: multi-input function where we include all parents by default (q + k + v -> hook_result)

**Graph construction** (`setup_from_model()`): builds the DAG by iterating layers in reverse, connecting:
- Residual stream nodes (each layer's hook_resid_post) to all downstream residual nodes
- MLP nodes (hook_mlp_out, hook_mlp_in) with ADDITION to residual, PLACEHOLDER internally
- Attention head nodes (hook_result per head) with ADDITION to residual
- Q/K/V nodes with DIRECT_COMPUTATION from inputs, PLACEHOLDER to hook_result

### 5.2 rocket_surgeon's computational graph

rocket_surgeon already models the forward pass as a sequence of ticks. The ACDC graph structure maps directly:

- Each tick corresponds to one or more nodes in the ACDC graph
- Edges between ticks represent data flow
- The edge type taxonomy (ADDITION vs DIRECT_COMPUTATION vs PLACEHOLDER) should be part of rocket_surgeon's graph metadata

**Critical addition for MoE:** ACDC has no concept of expert routing. rocket_surgeon needs:
- Router nodes that determine which experts activate
- Expert nodes with sparse connectivity (only active experts have live edges)
- Routing decision edges (which tokens go to which experts)

### 5.3 Exposing circuit discovery as a feature

Rather than reimplementing ACDC's full algorithm, rocket_surgeon should:
1. **Expose its computational graph in ACDC-compatible format** -- so users can run ACDC-like algorithms externally
2. **Support the activation patching primitive natively** -- the sender_hook/receiver_hook pattern
3. **Track edge effect sizes** -- when the user patches an edge, record the metric change

The ACDC algorithm itself is simple: iterate over edges, temporarily remove each, check if metric degrades beyond threshold, permanently remove if not. With rocket_surgeon's tick-level control, users/LLMs can implement this as a scripted session.

### 5.4 Per-paper insights on circuits

**Conmy2023 (ACDC paper):** The algorithm is threshold-sensitive -- too low and you keep noise edges, too high and you lose real ones. The paper recommends starting with a generous threshold and tightening. The metric choice matters enormously (KL divergence vs logit difference vs accuracy).

**Cunningham2023 (SAE paper):** Circuit discovery at the feature level (using SAE features as nodes instead of model components) produces more interpretable circuits with fewer edges. The IOI circuit found with SAE features matched known structure but was more precise. This is a key argument for SAE integration in rocket_surgeon.

**Zou2023 (RepE paper):** Linear Artificial Tomography (LAT) scans find that concepts are linearly represented in hidden states. This validates the linear probing / steering vector approach. Function vectors (how the model processes information) are also linearly represented, separate from content vectors (what information is stored).

**Belrose2023 (Tuned Lens paper):** The tuned lens reveals that different layers specialize -- early layers do syntax/structure, middle layers do semantics, late layers do output formatting. Stimulus-response alignment (correlation between input lens signal and output lens signal) can identify which layers are causally important for a given input.

---

## 6. Concrete API Implications

### 6.1 Core integration points

Based on the analysis, rocket_surgeon needs these integration surfaces:

**SAE subsystem:**
- `SaeRegistry`: load and manage SAEs per hook point. Support SAELens format, EleutherAI format.
- At each tick: optional SAE decomposition showing top-k features with coefficients
- Feature-level surgery: modify feature activations with error term preservation
- Transcoder support: buffer encoded features across ticks

**Lens subsystem:**
- `LensView`: at every residual stream tick, project through final_norm + unembed
- Optional tuned lens weights (one Linear per layer)
- Top-k token predictions with probabilities and entropy
- Inversion: "what hidden state produces these target logits?"

**Steering subsystem:**
- `SteeringVector`: per-layer direction vectors with multiplier
- Training: contrastive pair extraction + PCA (repeng pattern is more robust than simple mean-diff)
- Application: additive, ablation, ablation-then-addition, norm-preserving
- Position-selective steering (specific token positions only)

**Circuit graph:**
- `ComputationalGraph`: nodes = (hook_name, index) pairs, edges = typed (ADDITION/DIRECT_COMPUTATION/PLACEHOLDER/ROUTING for MoE)
- Built automatically from model architecture at load time
- Tracks edge presence and effect sizes
- Export format compatible with external circuit discovery tools

### 6.2 Protocol messages (JSON-RPC)

For the LLM-facing protocol:

```jsonc
// Inspect SAE features at current tick
{"method": "sae.features", "params": {"hook": "blocks.5.hook_resid_post", "top_k": 10}}
// -> {"features": [{"id": 4821, "activation": 3.72, "label": "..."}, ...], "error_norm": 0.15}

// Modify a feature
{"method": "sae.set_feature", "params": {"hook": "blocks.5.hook_resid_post", "feature_id": 4821, "value": 0.0}}

// Lens view
{"method": "lens.predict", "params": {"layer": 5, "position": 3, "top_k": 5}}
// -> {"predictions": [{"token": " the", "prob": 0.42}, ...], "entropy": 1.83}

// Apply steering
{"method": "steer.apply", "params": {"vector_name": "honesty", "multiplier": 1.5, "layers": [10, 11, 12]}}

// Get computational graph
{"method": "graph.edges", "params": {"filter": {"child": "blocks.5.attn.hook_result[:,:,3]"}}}
// -> edges with types, presence, effect sizes

// Activation patching
{"method": "patch.from_run", "params": {"source_run_id": "corrupted_01", "hook": "blocks.5.attn.hook_result", "index": [null, null, 3]}}
```

### 6.3 TUI views

For the human-facing TUI:

- **Feature panel**: when SAE is loaded, show active features as a sortable table (feature_id, activation, description)
- **Lens strip**: horizontal bar at bottom showing per-layer top-1 prediction, colored by confidence
- **Steering overlay**: active steering vectors shown as annotations on affected layers
- **Circuit view**: graph visualization (simplified) showing which edges are alive/dead, with effect sizes as edge thickness

### 6.4 Multi-GPU considerations

- SAE weights are small (typically 10-100MB) -- replicate on each rank, no sharding needed
- Lens computation needs final_norm + unembed weights -- extract and replicate at model load
- Steering vectors are per-layer vectors -- broadcast from rank 0
- Activation patching across ranks: need to gather activations from the rank that owns the relevant shard before patching. For tensor-parallel models, this means an all-gather on the activation before SAE decomposition.

### 6.5 What NOT to build

- Do not reimplement SAE training -- use SAELens or EleutherAI's trainer, load pretrained weights
- Do not reimplement ACDC's full algorithm -- expose the primitives (graph + patching), let users/scripts orchestrate
- Do not build a tuned lens trainer -- load pretrained weights from the tuned-lens repo
- Do not build a steering vector trainer into the debugger core -- provide an extraction utility that runs separately and saves vectors to load

### 6.6 Priority order

1. **SAE loading + feature inspection** -- highest value, enables feature-level understanding at any tick
2. **Logit lens** -- zero-cost (no trained weights needed), immediate insight into per-layer predictions
3. **Activation replacement/patching** -- core surgery primitive that enables everything else
4. **Steering vector application** -- load pre-computed vectors, apply during stepping
5. **Tuned lens** -- requires pre-trained weights but strictly better than logit lens
6. **SAE feature surgery with error term** -- more complex but enables precise interventions
7. **Circuit graph metadata** -- expose the computational graph structure for external tools
8. **MoE routing inspection** -- unique to rocket_surgeon, no existing tool does this well

---

## Appendix A: Papers Read

1. **Cunningham et al. 2023** -- "Sparse Autoencoders Find Highly Interpretable Features in Language Models." SAEs learn overcomplete dictionaries of interpretable features. Loss = reconstruction + alpha * L1(features). Features are more interpretable than neurons/PCA/ICA. Circuit discovery at feature level demonstrated on IOI task.

2. **Belrose et al. 2023** -- "Eliciting Latent Predictions from Transformers with the Tuned Lens." Logit lens is biased; tuned lens trains per-layer affine translators with KL distillation. Causal basis extraction finds principal features. Applications: secret elicitation, prompt injection detection.

3. **Conmy et al. 2023** -- "Towards Automated Circuit Discovery for Mechanistic Interpretability." ACDC automates circuit extraction by iteratively removing edges from computational DAG if removal doesn't degrade metric beyond threshold. Recovers known circuits (IOI, Greater-Than, Induction).

4. **Zou et al. 2023** -- "Representation Engineering: A Top-Down Approach to AI Transparency." LAT extracts concept/function directions via contrastive stimuli + PCA. Control via adding/subtracting representation vectors. Applications across honesty, emotion, fairness, knowledge, harmlessness.

**Not available on disk** (referenced from training knowledge):

5. **Turner et al. 2023** -- "Activation Addition: Steering Language Models Without Optimization." Adds contrastive activation differences (steering vectors) at forward-pass time. Key finding: single-layer addition at early-to-mid layers is sufficient. No optimization needed -- compute vector from one pair, apply to any input.

6. **Meng et al. 2022** -- "Locating and Editing Factual Associations in GPT." ROME uses causal tracing to locate factual knowledge in MLP layers, then edits weights via rank-one update to change stored facts. Key finding: factual associations concentrated in mid-layer MLPs. rocket_surgeon implication: weight editing is a valid surgery primitive, but activation-level intervention is simpler and reversible.

7. **Elhage et al. 2021** -- "A Mathematical Framework for Transformer Circuits." Foundational analysis of 1-layer and 2-layer attention-only transformers. Introduces QK/OV circuit decomposition, induction heads, virtual attention heads via composition. Key contribution: the residual stream as a shared communication channel between independent heads.

8. **Wang et al. 2023** -- "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small." IOI circuit: name movers, backup name movers, inhibition heads, S-inhibition heads, duplicate token heads, induction heads. Demonstrates that mechanistic circuits can be found and validated in real models on natural language tasks.

## Appendix B: Key File Paths

```
quarantine/SAELens/sae_lens/saes/sae.py                          # SAE base class + config
quarantine/SAELens/sae_lens/saes/standard_sae.py                 # StandardSAE encode/decode
quarantine/SAELens/sae_lens/analysis/hooked_sae_transformer.py   # HookedSAETransformer + error term
quarantine/SAELens/sae_lens/analysis/sae_transformer_bridge.py   # HuggingFace bridge
quarantine/SAELens/sae_lens/loading/pretrained_sae_loaders.py    # Multi-format loaders

quarantine/Automatic-Circuit-Discovery/acdc/TLACDCExperiment.py      # ACDC algorithm
quarantine/Automatic-Circuit-Discovery/acdc/TLACDCCorrespondence.py  # Graph structure
quarantine/Automatic-Circuit-Discovery/acdc/TLACDCEdge.py            # Edge types + TorchIndex

quarantine/tuned-lens/tuned_lens/nn/lenses.py    # LogitLens + TunedLens
quarantine/tuned-lens/tuned_lens/nn/unembed.py   # Unembed + inversion

quarantine/steering-vectors/steering_vectors/steering_vector.py       # SteeringVector
quarantine/steering-vectors/steering_vectors/train_steering_vector.py # Training
quarantine/steering-vectors/steering_vectors/steering_operators.py    # Operators

quarantine/repeng/repeng/control.py    # ControlModel + ControlModule
quarantine/repeng/repeng/extract.py    # ControlVector + PCA extraction

quarantine/transformer-utils/src/transformer_utils/logit_lens/hooks.py  # Original logit lens

quarantine/sae/sparsify/sparse_coder.py  # EleutherAI SparseCoder (TopK)
quarantine/sae/sparsify/trainer.py       # Distributed SAE training
```
