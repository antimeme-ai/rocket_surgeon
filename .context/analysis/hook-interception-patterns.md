# Hook Interception Patterns: Reference Implementation Analysis

Analysis of five reference repos and four papers informing rocket_surgeon's architecture.

**Repos analyzed:** nnsight, TransformerLens, pyvene, baukit, nnterp
**Papers analyzed:** Fiotto-Kaufman et al. 2024 (NNsight/NDIF), Wu et al. 2024 (pyvene), NNterp 2025, Ferrando et al. 2024 (Inseq)

---

## 1. Per-Repo Findings

### 1.1 nnsight

**Source:** `quarantine/nnsight/src/nnsight/intervention/`

#### Key Abstractions

- **`Envoy`** (`envoy.py:58`): Proxy wrapper around `torch.nn.Module`. Provides `.output`, `.input`, `.inputs` via `eproperty` descriptors. `__getattr__` falls through to wrapped module; `__setattr__` auto-wraps child modules. Each Envoy carries a `path` string (e.g. `"model.transformer.h.0"`) and a reference to the active `Interleaver`.

- **`eproperty`** (`interleaver.py:71`): Descriptor class. On `__get__`, runs a decorated stub (side effect: registers a one-shot PyTorch hook), then issues a blocking `request()` to the interleaver. On `__set__`, posts a `swap()`. Has `preprocess`/`postprocess`/`transform` chains for reshaping values between user and model representations. The stub body is a no-op; decorators stacked on it (from `hooks.py`) do the real work.

- **`Interleaver`** (`interleaver.py:412`): Coordinates model execution and interventions. Owns a list of `Mediator` objects. `wrap_module()` installs a thin forward wrapper + sentinel hook on each module. `handle()` broadcasts provider values to all mediators. Uses `__enter__`/`__exit__` to start/stop all mediator threads.

- **`Mediator`** (`interleaver.py:774`): One per `invoke()` call. Runs user intervention code in a dedicated worker thread. Communication via two single-item queues (`event_queue`, `response_queue`) using an `Events` enum: VALUE, SWAP, SKIP, END, EXCEPTION, BARRIER. The `handle()` method (`interleaver.py:1016`) is the central dispatch loop processing events until the mediator yields.

- **`SourceAccessor` / `FunctionCallWrapper`** (`source.py:1`, `source.py:95`): AST-based interception of individual call sites within a module's `forward()`. `FunctionCallWrapper` is an `ast.NodeTransformer` that wraps every function call with `wrap(fn, name=...)`. The `SourceAccessor` is cached on `module.__source_accessor__` and built lazily. `OperationAccessor` holds per-call-site hook lists (pre_hooks, post_hooks, fn_hooks, fn_replacement).

- **One-shot hooks** (`hooks.py:1-100`): Dynamic hook registration system. `add_ordered_hook()` inserts hooks in mediator-index order by sorting PyTorch's hook dict. `input_hook()`/`output_hook()` create one-shot hooks that self-remove after firing. `requires_output`/`requires_input` are decorators for `eproperty` stubs.

#### Patterns

- **Lazy hook registration**: Hooks only registered when user code accesses `.output`/`.input`, not upfront. This means zero overhead for modules nobody inspects. Key insight: `wrap_module()` installs a sentinel `register_forward_hook(lambda _, __, output: output)` to keep PyTorch in the hook dispatch path so dynamically-added hooks fire.

- **Thread-per-invoke**: Each `invoke()` spawns a worker thread. User code runs in the worker; the main thread runs the model. Communication is synchronous (blocking queues). The worker blocks on `request()` until the model's hook fires and delivers the value.

- **CUDA stream propagation**: `Mediator.start()` (`interleaver.py:967-970`) captures `torch.cuda.current_stream()` and sets it in the worker thread. Worker threads default to the NULL stream, but frameworks like vLLM use non-default streams. Without propagation, CUDA ops race.

- **Iteration tracking**: `iterate_requester()` (`interleaver.py:488`) appends `.i0`, `.i1`, etc. to requester strings. Dual-mode: explicit `mediator.iteration` (from `tracer.iter[i]` loops) or `mediator.iteration_tracker[requester]` (auto-incremented by persistent hooks).

#### Gotchas

- **Forward-pass ordering constraint**: Interventions MUST be written in forward-pass order within a single invoke. Accessing a layer's output then its input deadlocks because the one-shot output hook has already fired and self-removed. The `Mediator.OutOfOrderError` (`interleaver.py:816`) is raised when this is detected, but detection is best-effort.

- **Sentinel hook requirement**: Without the sentinel hook, PyTorch fast-paths when no hooks are registered, causing dynamically-added hooks to silently not fire. This is a subtle PyTorch behavior that bit nnsight.

- **`sys.settrace` capture**: The `Tracer` base class (`tracing/base.py`) uses `sys.settrace()` to intercept user code at trace entry, raises `ExitTracingException` to prevent normal execution, then compiles captured source. This is fragile across Python versions and debugger interactions.

#### Steal

- The `eproperty` descriptor pattern with `preprocess`/`postprocess`/`transform` chains. Clean separation between what the user sees and what the model needs.
- One-shot self-removing hooks with ordered insertion. Avoids permanent hook overhead.
- `SourceAccessor` approach to sub-module interception. AST rewriting per call site is powerful for fine-grained operation-level access.
- CUDA stream propagation to worker threads.

#### Avoid

- `sys.settrace`-based code capture. rocket_surgeon should use an explicit API rather than magic source capture.
- The complexity of the mediator threading model. Six event types, two queues per mediator, careful save/restore of `current` mediator on each `handle()` call. This works but is extremely hard to debug.
- The sentinel hook workaround. rocket_surgeon should register hooks before the forward pass begins, not dynamically mid-forward.

---

### 1.2 TransformerLens

**Source:** `quarantine/TransformerLens/transformer_lens/hook_points.py`

#### Key Abstractions

- **`HookPoint(nn.Module)`** (`hook_points.py:145`): Identity module (`forward` returns `x` unchanged). Inserted at strategic points in the model graph (between attention and MLP, at residual stream, etc.). Carries `fwd_hooks: list[LensHandle]` and `bwd_hooks: list[LensHandle]`, plus a `ctx` dict and `name` string.

- **`LensHandle`** (`hook_points.py:39`): Dataclass wrapping PyTorch's `RemovableHandle` with `is_permanent: bool` and `context_level: Optional[int]`. Context levels enable nested hook scoping.

- **`HookedRootModule(nn.Module)`** (`hook_points.py:379`): Base class for hooked models. `setup()` walks `named_modules()` to build `mod_dict` (all modules by name) and `hook_dict` (only HookPoints by name). Provides `run_with_hooks()`, `run_with_cache()`, `hooks()` context manager.

- **`_AliasedHookPoint`** (`hook_points.py:97`): Lightweight wrapper presenting a different `name` to hook functions. Used for backward compatibility when hook points are renamed.

- **`BaseTensorConversion`** (imported from `conversion_utils`): Used by `HookPoint.enable_reshape()` to convert/revert tensor shapes, allowing hooks to work with reshaped representations.

#### Patterns

- **Identity module insertion**: The fundamental pattern. Instead of hooking existing modules, insert new `nn.Module`s that are identity functions. Hooks on these modules intercept values without touching the model's real modules. This is clean but requires modifying the model architecture.

- **Context-level scoping**: `HookedRootModule.context_level` is an integer counter incremented/decremented by the `hooks()` context manager (`hook_points.py:587-625`). Each `add_hook()` call stamps the current context level onto the `LensHandle`. On exit, only hooks at the current level are removed. This enables nested hook contexts.

- **Global hook state with explicit cleanup**: As documented at `hook_points.py:387-396`, hooks are GLOBAL state. The `run_with_hooks()` pattern wraps execution in a context manager that removes hooks at exit. The codebase contains explicit warnings about this being the "main footgun."

- **Caching via hooks**: `add_caching_hooks()` (`hook_points.py:666`) creates `save_hook` closures that `detach().to(device)` tensors into a dict. Simple, effective. The `run_with_cache()` wrapper adds caching hooks, runs the model, removes hooks, returns the cache.

- **Hook function protocol**: `_HookFunctionProtocol` (`hook_points.py:84`) specifies `(tensor, *, hook: HookPoint) -> Union[Any, None]`. Returning `None` means no modification; returning a value replaces the activation.

#### Gotchas

- **Hooks are global state**: The biggest design issue. Adding a hook persists until explicitly removed. If you add a hook during debugging and forget to remove it, it affects all subsequent forward passes. The context-level system mitigates this but doesn't eliminate it.

- **Requires model modification**: HookPoints must be inserted into the model's `__init__` and `forward`. You can't hook an arbitrary HuggingFace model without rewriting it. This is why TransformerLens maintains its own model implementations.

- **Backward hooks are tricky**: `module_output` for backward hooks is a tuple of `(grad,)`. The code has special handling for this (`hook_points.py:198-200`). The `_ScaledGradientTensor` wrapper (`hook_points.py:56`) works around a PyTorch bug where multiplying gradient tensors element-wise in backward hooks gives incorrect sums.

- **Performance overhead**: Every HookPoint executes on every forward pass even with zero hooks, because `nn.Module.__call__` still runs. The `has_hooks()` check (`hook_points.py:285`) lets users skip computation, but the Module overhead remains.

#### Steal

- Context-level scoping for nested hook management. Simple integer counter, stamps each hook, removes by level on exit. Clean and effective.
- The `run_with_cache()` pattern. Users want this exact API. Provide a filter, get a dict of activations.
- The `_AliasedHookPoint` pattern for backward compatibility when renaming hook points.
- `BaseTensorConversion` for pluggable shape transforms at hook points.

#### Avoid

- Requiring model rewrites to insert HookPoints. rocket_surgeon must work with arbitrary models.
- Global hook state as the default. Use scoped hooks that auto-clean.
- The tight coupling between HookPoint and specific model implementations. TransformerLens maintains parallel implementations of GPT-2, LLaMA, etc. just for hookability.

---

### 1.3 pyvene

**Source:** `quarantine/pyvene/pyvene/models/`

#### Key Abstractions

- **`RepresentationConfig`** (`configuration_intervenable_model.py`): Named tuple with 14 fields (layer, component, unit, max_number_of_units, low_rank_dimension, intervention_type, intervention, subspace_partition, group_key, intervention_link_key, moe_key, source_representation, hidden_source_representation, latent_dim). Declarative specification of where to intervene.

- **`IntervenableConfig`** (`configuration_intervenable_model.py`): Extends HuggingFace `PretrainedConfig`. Holds a list of `RepresentationConfig`s plus metadata (model_type, sorted_keys, mode). Serializable to/from HuggingFace Hub.

- **`Intervention(nn.Module)`** (`interventions.py`): Base class. Hierarchy: `TrainableIntervention`, `ConstantSourceIntervention`, `SourcelessIntervention`, `BasisAgnosticIntervention`. Concrete types: `ZeroIntervention`, `CollectIntervention`, `SkipIntervention`. Interventions are nn.Modules, so they can be trained.

- **`BaseModel(nn.Module)`** (`intervenable_base.py:45`): Owns the model, interventions (`torch.nn.ModuleDict`), and hook state. `representations` dict maps keys to `RepresentationConfig`. `intervention_hooks` maps keys to module hooks. Uses `_key_getter_call_counter` / `_key_setter_call_counter` for generation-aware hook counting.

- **Intervention groups**: `_intervention_group` (`intervenable_base.py:207-235`) organizes interventions into ordered groups. Groups execute sequentially; no dependency between groups. Key ordering validated to be ascending.

#### Patterns

- **Intervention-as-data**: The core pattern. Interventions are declared via config objects, not written as code. This enables serialization to HuggingFace Hub, reproducibility, and systematic comparison. The `RepresentationConfig` specifies the what/where; the `Intervention` subclass specifies the how.

- **Abstract layer naming**: Config uses abstract names (e.g., "block_output", "mlp_activation") that are resolved to concrete module paths via `get_module_hook()`. This makes configs portable across model families.

- **Trainable interventions as nn.Modules**: Because `Intervention` extends `nn.Module`, trainable interventions (e.g., `BoundlessRotatedSpaceIntervention`) participate in standard PyTorch training loops. `get_trainable_parameters()` collects them alongside model params.

- **Getter/setter hooks with call counters**: Hooks track how many times they've been called. During generation, the same hook fires multiple times (once per token). The counter determines which call to intervene on.

- **Key collision handling**: `_key_collision_counter` (`intervenable_base.py:98`) appends `#0`, `#1`, etc. to keys when multiple interventions target the same location.

#### Gotchas

- **14-field RepresentationConfig**: Extremely complex. Most fields have reasonable defaults, but the surface area for misconfiguration is large. The `moe_key` field was added for MoE models but only works for specific architectures.

- **Generation hook counting is fragile**: The `_key_getter_call_counter` / `_key_setter_call_counter` mechanism assumes a specific calling pattern during generation. If the model's generate logic changes, counters get out of sync.

- **Tight coupling to HuggingFace**: `IntervenableConfig` extends `PretrainedConfig`. `get_internal_model_type()` and `get_dimension_by_component()` use HuggingFace-specific introspection. Supporting non-HF models requires significant work.

- **Silent error handling**: `intervenable_base.py:29-32` catches all exceptions from `import nnsight` with a bare `except:` and prints a message. This violates fail-fast. Similar patterns appear elsewhere in the codebase.

#### Steal

- **Intervention-as-data** for serialization and reproducibility. rocket_surgeon should have a config-based intervention description for LLM consumption.
- **Trainable interventions as nn.Modules**. If rocket_surgeon ever supports learned interventions, this is the right pattern.
- **Abstract component naming** that maps to concrete module paths. Decouples intervention specs from model internals.
- **Intervention groups** for ordered multi-intervention execution.

#### Avoid

- The 14-field config object. Start minimal, extend later.
- Bare `except:` blocks. Use specific exception types.
- The generation hook counting approach. Use a more robust mechanism (iteration tracking like nnsight, or explicit step indices).
- The `get_internal_model_type()` dispatcher pattern. It's a growing if/elif chain that doesn't scale.

---

### 1.4 baukit

**Source:** `quarantine/baukit/baukit/nethook.py`

#### Key Abstractions

- **`Trace`** (`nethook.py:19`): Context manager wrapping a single `register_forward_hook`. Constructor takes `module`, `layer`, `retain_output`, `retain_input`, `clone`, `detach`, `retain_grad`, `edit_output`, `stop`. On `__exit__`, removes the hook. On `StopForward`, swallows the exception.

- **`TraceDict(OrderedDict)`** (`nethook.py:111`): Multiple `Trace` instances keyed by layer name. `flag_last_unseen()` generator determines which layer is last (for `stop` propagation). Each kwarg can be a dict mapping layer names to per-layer values.

- **`StopForward(Exception)`** (`nethook.py:189`): Raised by hooks with `stop=True`. Caught by `Trace.__exit__` to allow early termination of forward passes.

- **Helper functions**: `get_module()` (`nethook.py:361`) resolves dotted names. `replace_module()` (`nethook.py:395`) uses `getattr`/`setattr` with parent resolution. `subsequence()` (`nethook.py:232`) creates sub-sequences from `nn.Sequential`. `invoke_with_optional_args()` (`nethook.py:406`) supports flexible function signatures.

#### Patterns

- **Minimal context-manager hooks**: The entire library is 472 lines. One hook per Trace, one dict of Traces per TraceDict. No threading, no queues, no descriptors. The `retain_hook` closure (`nethook.py:71-94`) does everything: edit, retain input/output, clone/detach, stop.

- **`invoke_with_optional_args()`**: Calls a function with only the arguments it accepts, matching by name first, then by position. This lets `edit_output` callbacks have flexible signatures (`fn(output)`, `fn(output, layer)`, `fn(output, layer, inputs)`).

- **`recursive_copy()`** (`nethook.py:205`): Handles tensors, dicts, lists, tuples. Respects `clone`, `detach`, `retain_grad` flags. Asserts on unknown types rather than silently passing through.

- **`StopForward` for early termination**: Clean pattern. Hook raises `StopForward`, model execution stops, context manager catches it. No need for special API.

#### Gotchas

- **No multi-GPU awareness**: Hooks capture tensors on whatever device they're on. No device management, no stream synchronization.
- **No generation support**: Single forward pass only. No iteration tracking.
- **No backward hook support**: Only `register_forward_hook`, no `register_full_backward_hook`.
- **`recursive_copy` asserts on unknown types**: If a module returns a custom dataclass, it crashes. But this is arguably correct (fail fast).

#### Steal

- **The entire design philosophy**: 472 lines, zero magic, explicit everything. baukit proves that hook interception doesn't need to be complex.
- **`invoke_with_optional_args()`**: Excellent for supporting callback evolution. rocket_surgeon's hook callbacks should use this pattern.
- **`StopForward` exception for early termination**: Simple, effective. No special return values, no flags.
- **`recursive_copy()` with flag-based behavior**: Clone, detach, retain_grad as composable options.
- **`TraceDict` with per-layer kwarg dicts**: `retain_output={"layer1": True, "layer2": False}` is ergonomic.

#### Avoid

- Nothing significant to avoid. baukit is intentionally minimal. The limitations (no multi-GPU, no generation, no backward) are scope choices, not design mistakes.

---

### 1.5 nnterp

**Source:** `quarantine/nnterp/nnterp/`

#### Key Abstractions

- **`StandardizedTransformer(LanguageModel)`** (`standardized_transformer.py`): Extends nnsight's `LanguageModel`. On init, calls `get_rename_dict()` and renames modules via nnsight's `Envoy` aliasing. Validates via `check_model_renaming()`. Provides standardized accessors: `layers_input[i]`, `layers_output[i]`, `attentions[i]`, `mlps[i]`, etc.

- **`RenameConfig`** (`rename_utils.py:47`): Dataclass with architecture-specific name mappings (attn_name, mlp_name, ln_final_name, lm_head_name, model_name, layers_name, attn_prob_source, ignore_mlp, ignore_attn, attn_head_config_key, hidden_size_config_key, vocab_size_config_key). Optional fields with sensible defaults.

- **`LayerAccessor`** (`rename_utils.py`): `__getitem__`/`__setitem__` for layer I/O with tuple detection. Handles the common case where a transformer layer returns `(hidden_states, ...)` and you want just `hidden_states`.

- **`AttnProbFunction`** (`rename_utils.py:31`): Abstract class for architecture-specific attention probability extraction. Implementations use nnsight's `.source` feature to reach into attention internals.

#### Patterns

- **Module renaming as architecture abstraction**: Instead of maintaining parallel model implementations (TransformerLens) or abstract component names (pyvene), nnterp renames existing modules to standard names. Works with any HuggingFace model without code changes.

- **Validation after renaming**: `check_model_renaming()` and `check_io()` verify that standardized accessors actually work. Catches renaming failures early rather than at intervention time.

- **Architecture-specific constants**: `ATTENTION_NAMES`, `LAYER_NAMES`, `LN_NAMES`, `MODEL_NAMES` are lists of known names per architecture. `get_rename_dict()` matches against these.

- **Attention probabilities via `.source`**: For architectures where attention probabilities aren't a standard output, nnterp uses nnsight's source tracing to reach into the attention module's forward pass and intercept the softmax output.

#### Gotchas

- **Name matching is fragile**: If a new model uses a non-standard name that's not in the constants lists, renaming silently fails. The validation catches this, but the error message may not be obvious.

- **Tuple detection heuristic**: `LayerAccessor.__getitem__` checks if the output is a tuple and takes `[0]`. This assumes `hidden_states` is always first. Some models return different tuple structures.

- **Tight coupling to nnsight**: `StandardizedTransformer` extends `LanguageModel`, uses `Envoy` aliasing. Not usable without nnsight.

#### Steal

- **The renaming approach over parallel implementations**: rocket_surgeon should map arbitrary model naming to a standard namespace. Much more scalable than maintaining separate model implementations.
- **Validation after standardization**: Always verify that the abstraction layer actually works.
- **Architecture constants as a knowledge base**: Maintain a registry of known architectures and their naming conventions. Update it as new models appear.

#### Avoid

- **Tuple detection heuristics**: Be explicit about what each layer returns. Assert on structure rather than guessing.
- **Silent renaming failures**: If a module can't be found, fail loudly.

---

## 2. Cross-Cutting Synthesis

### 2.1 Common Patterns

**All five repos use PyTorch forward hooks as the interception mechanism.** The differences are in how hooks are managed:

| Repo | Hook Registration | Hook Lifetime | Hook Ordering |
|------|------------------|---------------|---------------|
| nnsight | On-demand (lazy) | One-shot (self-removing) | Sorted by mediator index |
| TransformerLens | Explicit `add_hook()` | Until removed (global) | Insertion order, prepend option |
| pyvene | At init from config | Until model disposed | Config-defined order |
| baukit | In context manager `__init__` | Until `__exit__` | Single hook per module |
| nnterp | Delegates to nnsight | Delegates to nnsight | Delegates to nnsight |

**All repos that support it use context managers for hook lifecycle.** baukit (`Trace`/`TraceDict`), TransformerLens (`hooks()`), nnsight (`model.trace()`). This is the right pattern.

**Activation caching follows the same shape everywhere:** Hook captures tensor, optionally clones/detaches, stores in a dict keyed by module name. TransformerLens's `run_with_cache()` and baukit's `Trace(retain_output=True)` are nearly isomorphic.

**Module name resolution uses dotted paths everywhere.** baukit's `get_module()`, pyvene's `get_module_hook()`, nnsight's `Envoy.path`, TransformerLens's `mod_dict`. rocket_surgeon should use dotted paths as the canonical module identifier.

### 2.2 Where They Diverge

**Execution model:**
- baukit/TransformerLens: Synchronous. Register hooks, run forward pass, read results. Single thread.
- nnsight: Asynchronous interleaving. User code and model execution alternate in separate threads.
- pyvene: Synchronous but declarative. Config defines hooks; forward pass triggers them.

**Granularity:**
- baukit/TransformerLens/pyvene: Module-level only. Can only intercept at module boundaries.
- nnsight: Module-level AND operation-level (via `SourceAccessor` AST rewriting). Can intercept individual function calls within a forward method.

**Model compatibility:**
- baukit: Works with any `nn.Module`. Zero requirements on model structure.
- nnsight: Works with any `nn.Module` but `LanguageModel` adds HuggingFace-specific features.
- pyvene: Requires HuggingFace models or models with known structure.
- TransformerLens: Requires custom model implementations with HookPoints inserted.
- nnterp: Requires HuggingFace models that nnsight can load.

**Multi-GPU:**
- nnsight: CUDA stream propagation, device-aware. Paper benchmarks include multi-GPU.
- TransformerLens: Has `multi_gpu` utilities but they're rudimentary.
- pyvene: Acknowledges multi-GPU as a limitation.
- baukit: No multi-GPU support.

### 2.3 Implications for rocket_surgeon

1. **Use PyTorch forward hooks** as the base interception mechanism. Every library does. Don't reinvent this.

2. **Support both synchronous and asynchronous modes.** The synchronous mode (baukit-style) is simpler and covers most use cases. The asynchronous mode (nnsight-style) enables interactive stepping. rocket_surgeon's "one tick at a time" requirement maps to the asynchronous model, but the synchronous mode should exist for batch operations.

3. **Start with module-level granularity, design for operation-level.** nnsight's `SourceAccessor` shows that sub-module interception is valuable but complex. rocket_surgeon should plan for it but ship without it initially.

4. **Work with arbitrary models.** Follow baukit's lead: zero requirements on model structure. Use nnterp's renaming approach for convenience accessors, but never require it.

5. **Multi-GPU from day one.** CUDA stream propagation, device tracking, distributed tensor handling. This is non-negotiable given the requirements.

6. **Context-scoped hooks with auto-cleanup.** Combine baukit's context manager pattern with TransformerLens's context-level scoping. Hooks should never leak.

---

## 3. Specific Code Patterns Worth Adopting

### 3.1 baukit's `invoke_with_optional_args()` (nethook.py:406-471)

Calls a function with only the arguments it accepts. Matches by name first, then position. Allows hook callbacks to evolve without breaking existing code. rocket_surgeon's tick callbacks should use this pattern so users can write `def on_tick(activation):` or `def on_tick(activation, layer, step, metadata):` and both work.

### 3.2 nnsight's `eproperty` descriptor with transform chains (interleaver.py:71-269)

The `preprocess`/`postprocess`/`transform` chain cleanly separates user-facing representation from model-internal representation. Example: attention heads reshaped from `[B, S, H]` to `[B, n_heads, S, head_dim]` on read, reshaped back on write. rocket_surgeon should adopt this for exposing activations in user-friendly shapes while maintaining model compatibility.

### 3.3 TransformerLens's context-level hook scoping (hook_points.py:587-625)

```
context_level += 1  # on enter
# ... add hooks stamped with current level ...
reset_hooks(level=context_level)  # on exit
context_level -= 1
```

Simple, effective nested scoping. rocket_surgeon's breakpoint/inspection scopes should use this pattern.

### 3.4 baukit's `StopForward` exception (nethook.py:189-203)

Clean early termination without special API. The hook raises the exception; the context manager catches it. rocket_surgeon needs this for "run up to layer N and stop" workflows.

### 3.5 nnsight's ordered hook insertion (hooks.py:92-100)

`add_ordered_hook()` inserts hooks into PyTorch's hook dict at the correct position by sorting on a `mediator_idx` attribute. This guarantees deterministic hook execution order even when hooks are registered out of order. Essential for multi-intervention scenarios.

### 3.6 baukit's `recursive_copy()` (nethook.py:205-229)

Handles tensors, dicts, lists, tuples with composable `clone`/`detach`/`retain_grad` flags. Asserts on unknown types. rocket_surgeon needs this exact utility for activation capture.

### 3.7 nnsight's CUDA stream propagation (interleaver.py:967-970)

Worker threads must inherit the caller's CUDA stream. Without this, CUDA operations in worker threads race with main-thread operations. rocket_surgeon's multi-GPU tick execution must propagate streams.

### 3.8 nnterp's validation-after-standardization (rename_utils.py: `check_model_renaming()`)

After remapping module names, run validation to confirm the mapping works. Don't trust the mapping table; verify empirically. rocket_surgeon should validate its model introspection results.

---

## 4. Specific Mistakes/Limitations to Avoid

### 4.1 TransformerLens's global hook state (hook_points.py:387-396)

The code itself documents this as "the main footgun." Hooks persist until explicitly removed. If cleanup fails (exception, bug, interactive session), hooks accumulate and corrupt subsequent runs. rocket_surgeon MUST use scoped hooks that auto-clean. A hook that outlives its scope is a bug.

### 4.2 nnsight's `sys.settrace()` code capture (tracing/base.py)

Intercepting user code by patching the Python trace function is clever but fragile. It breaks with debuggers, coverage tools, and non-CPython implementations. rocket_surgeon should use explicit API calls, not magic. If the user wants to define an intervention, they call a function, not write code inside a `with` block that gets secretly AST-rewritten.

### 4.3 pyvene's 14-field RepresentationConfig

Over-designed from the start. Most users need layer, component, and intervention_type. The rest are edge cases that could be handled by subclassing or kwargs. rocket_surgeon's intervention config should start with 3-4 fields and grow.

### 4.4 pyvene's bare `except:` blocks (intervenable_base.py:29-32)

```python
try:
    import nnsight
except:
    print("nnsight is not detected...")
```

Catches KeyboardInterrupt, SystemExit, everything. Violates fail-fast. rocket_surgeon must use specific exception types.

### 4.5 TransformerLens's model reimplementation requirement

TransformerLens maintains complete reimplementations of GPT-2, LLaMA, Mistral, etc. with HookPoints inserted. This doesn't scale. Every new model requires a new implementation. rocket_surgeon must hook arbitrary models without modification.

### 4.6 nnsight's forward-pass ordering constraint

The requirement that interventions be written in forward-pass order within a single invoke is a fundamental limitation of the one-shot hook + thread model. If rocket_surgeon uses a step-through model where the user advances tick by tick, this constraint doesn't apply. But if it supports batch intervention definitions (for LLM consumption), it needs to handle arbitrary ordering.

### 4.7 pyvene's generation hook counting (intervenable_base.py:100-113)

`_key_getter_call_counter` / `_key_setter_call_counter` assume hooks fire in a specific pattern during generation. If the model's generate logic changes (different sampling strategy, speculative decoding, etc.), counters desync. rocket_surgeon should use explicit step/iteration identifiers rather than call counts.

### 4.8 nnsight's sentinel hook workaround (interleaver.py:603-605)

```python
module.register_forward_hook(lambda _, __, output: output)
```

This exists solely because PyTorch skips hook dispatch when no hooks are registered. If rocket_surgeon registers hooks before the forward pass (not mid-forward), this workaround isn't needed. But if lazy registration is adopted, this gotcha must be known.

### 4.9 Lack of MoE-specific interception

None of the five repos have first-class MoE support. pyvene has a `moe_key` field in `RepresentationConfig` but it's underspecified. nnsight can theoretically intercept expert routing via `SourceAccessor`, but there's no ergonomic API for it. rocket_surgeon's MoE support will need to be designed from scratch, but the existing AST-rewriting approach (nnsight's `SourceAccessor`) and abstract component naming (pyvene's `RepresentationConfig.component`) provide useful starting points.

### 4.10 No library handles tensor parallelism correctly

All five repos assume each module lives on a single device. With tensor parallelism, a single logical module is sharded across devices. Hook outputs are partial tensors. rocket_surgeon must be aware of sharding and either gather before presenting to the user or expose the sharded view explicitly.
