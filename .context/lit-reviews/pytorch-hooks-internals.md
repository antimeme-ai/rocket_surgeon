---
topic: PyTorch hook mechanisms, interception points, multi-GPU, MoE, torch.compile
status: draft
created: 2026-05-14
sources: PyTorch docs, forums, blog posts, TransformerLens docs
---

# PyTorch Hooks + Internals: Lit Review

Deep dive into every interception point available in PyTorch for building a forward-pass debugger.

## Core Hook API

### register_forward_hook
- Fires after forward() completes
- Signature: `hook(module, args, output) -> None or modified_output`
- Can modify output in-place or return new output

### register_forward_pre_hook
- Fires before forward() executes
- Receives module and input args
- Can modify inputs

### register_full_backward_hook
- Fires during backward
- Signature: `hook(module, grad_input, grad_output)`
- **Memory leak documented** with `create_graph=True` — BackwardHook objects don't auto-destruct
- Can modify gradients

### RemovableHandle
- All register_*_hook return RemovableHandle with .remove()
- Supports context manager protocol — critical for temporary inspection hooks
- Failing to remove causes memory leaks

## Module.__call__ Internals

When `model(input)` runs, `_call_impl` executes:
1. Pre-execution checks (hook existence)
2. Forward pre-hooks (FIFO order)
3. Actual forward() call
4. Forward post-hooks (FIFO order)
5. Optimization: if no hooks exist anywhere, skips to direct forward

Hook execution is synchronous and blocks the forward pass. This is our primary interception mechanism.

## Multi-GPU: Here Be Dragons

### DataParallel (deprecated but extant)
- Hooks fire multiple times (once per GPU replica)
- Not guaranteed coherent order across devices
- Module state updates in forward are LOST — module re-replicated per device

### DistributedDataParallel (DDP)
- **Critical**: forward/backward hooks registered on the module WON'T BE INVOKED unless registered inside forward() itself
- Pre-registered hooks are silently ignored
- Workaround: register hooks dynamically during forward
- All-reduce triggered by autograd hooks on parameter gradients

### FSDP
- Uses forward/backward hooks internally for all-gather/reshard
- Custom hooks CAN INTERFERE with FSDP's internal communication hooks
- FSDP2 has explicit prefetching to overlap all-gathers with computation

### Tensor Parallelism
- Patches hooks onto tensor operations via parallelize_module
- Injects communication before/after modules

### Pipeline Parallelism (PiPPy)
- Uses hooks to manage micro-batch execution and pipeline stage boundaries
- Hook firing order critical for synchronization

### DDP Communication Hooks
- register_comm_hook allows custom gradient sync
- Returns Future for async completion
- Must register before first backward pass

## torch.fx: Static Graph Alternative

### How it works
- symbolic_trace() captures module structure into torch.fx.Graph (doubly-linked list of Nodes)
- Proxy objects intercept operations via __torch_function__
- Result: GraphModule whose forward() runs the captured graph

### Limitations for debugger use
- **No dynamic control flow** — can't trace data-dependent branches/loops/conditionals
- Single static graph per trace
- Hooks registered during tracing are NOT recorded
- Struggles with custom tensor subclasses

### Tradeoffs vs hooks
- Hooks: runtime, dynamic, full support, less optimizable
- torch.fx: static, compilable, symbolic shapes, but loses dynamism

## torch.compile / Dynamo: The Elephant

### Critical issue
Forward hooks registered AFTER first compilation are **silently ignored**.

### Graph breaks
- Accessing module._forward_hooks causes graph breaks
- Backward hooks cause graph breaks AND delayed firing
- State dict hooks not yet supported

### Workarounds
1. Register ALL hooks before torch.compile()
2. skip_nnmodule_hook_guards=False — allows recompilation on hook changes (runtime penalty)
3. torch.compiler.allow_in_graph for known-safe hook functions
4. Compiled Autograd (under development) — future solution

**For rocket_surgeon**: we must assume hooks are registered pre-compilation and never modified at runtime. Or we operate on uncompiled models only.

## MoE-Specific Interception

### Components to hook
1. **Gating network output**: post-softmax logits, expert selection probabilities
2. **Token-to-expert assignment**: which tokens route to which experts, scores
3. **Per-expert forward**: register hooks on individual expert modules
4. **Capacity thresholding**: monitor token counts per expert, overflow handling
5. **Auxiliary loss**: load balancing loss computation

### Routing collapse prevention mechanisms
- Router z-loss: penalizes large gating logits
- Auxiliary load balancing loss: encourages uniform utilization
- Expert capacity limits: hard cap on tokens per expert

## Advanced Interception Mechanisms

### saved_tensors_hooks
- Control packing/unpacking of intermediate activations saved for backward
- Enables activation offloading to CPU, recomputation on backward
- Used by modern activation checkpointing

### Custom Autograd Functions
- torch.autograd.Function with forward()/backward() statics
- Full control over gradient computation
- Can be wrapped by hooks for inspection

### Tensor Subclassing + __torch_dispatch__
- Custom tensor subclasses override __torch_dispatch__ to intercept ALL operations
- Per-operation instrumentation without explicit hooks
- Used by torchao for transparent quant/dequant

### ATen Dispatcher (C++ level)
- Dispatcher selects kernel implementations based on dispatch keys
- Custom backends register kernels via TORCH_LIBRARY_IMPL
- Operates before Python hooks execute

## Known Gotchas

1. TransformerEncoder forward hooks don't fire if model.eval() AND torch.no_grad() both active
2. register_full_backward_hook + create_graph=True = memory leak
3. DDP silently ignores pre-registered hooks
4. torch.compile silently ignores post-compilation hooks
5. Hooks holding references prevent garbage collection
6. Long-running hooks block the forward pass (synchronous execution)

## Design Implications for rocket_surgeon

1. **Hook registration strategy**: all hooks before compilation/distributed wrappers. Context managers for temporary inspection.
2. **DDP/FSDP**: don't rely on module-level hooks. Register inside forward or use communication hooks.
3. **Step-through**: forward_pre_hook + forward_hook pairs capture layer I/O. For MoE, hook gating + per-expert forwards.
4. **Backward debugging**: register_full_backward_hook but watch memory. Consider saved_tensors_hooks for offloading.
5. **torch.compile compat**: design for pre-compilation hook registration. Or target uncompiled models.
6. **Consider __torch_dispatch__**: tensor subclassing gives per-op interception without module-level hooks. More granular but more complex.

## Sources

- PyTorch nn.Module docs (register_forward_hook, etc.)
- Stanford blog: Intermediate Activations — the forward hook (Nandita Bhaskhar)
- PyTorch forums: register_full_backward_hook memory leak discussion
- PyTorch Module.__call__ vs forward (Stephen Cow Chau)
- PyTorch autograd mechanics docs
- DDP documentation
- FSDP tutorial
- torch.fx paper (arxiv 2112.08429)
- torch.compile hooks interaction (PyTorch issue #117758)
- Compiled Autograd tutorial
- PyTorch MoE training blog
- Mixture of Experts survey (arxiv 2406.18219)
- TransformerLens docs
