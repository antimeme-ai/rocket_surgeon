---
id: BEAD-0002
title: Clone reference implementations to quarantine
status: done
priority: medium
created: 2026-05-14
completed: 2026-05-14
---

## Description

Clone and study reference implementations identified across all 13 lit reviews. These go in quarantine/ (gitignored).

## Resolution

47 repos cloned (shallow, --depth 1) totaling ~7.5GB across 8 categories. Two repos were private/nonexistent (anthropics/circuit-tracer, safety-research/open-source-model-interp). PyTorch and JAX cloned with --no-checkout to save space (git history only, checkout specific files on demand).

## Quarantine Catalog

### ML Interpretability & Surgery (primary references)
| Repo | What to study |
|------|---------------|
| nnsight | Envoy/proxy wrapping, execution-order tracing, multi-GPU via vLLM |
| TransformerLens | HookPoint abstraction, standardized naming, activation cache |
| pyvene | IntervenableConfig serialization, intervention-as-data pattern |
| CircuitsVis | React visualization components for attention, neurons |
| nnterp | Architecture-agnostic naming layer across HuggingFace models |
| transformer-debugger | OpenAI's closest prior art — UI patterns, token attribution |
| inseq | Feature attribution methods, gradient-based + perturbation-based |
| SAELens | SAE training pipeline, feature dashboards, dictionary learning |
| baukit | David Bau's toolkit — nethook, pbar, renormalize |
| rome | ROME/MEMIT — causal tracing, rank-one model editing |
| repeng | Representation engineering, steering vectors, control vectors |
| sae | EleutherAI SAE implementation, sparse coding |
| Automatic-Circuit-Discovery | ACDC — automated circuit finding in transformers |
| tuned-lens | Learned affine probes per layer, logit lens calibration |
| transformer-utils | nostalgebraist's logit lens implementation |
| steering-vectors | Steering vector extraction and application library |
| lit | Google PAIR's Learning Interpretability Tool |
| ecco | NLP model explanation via activations and attributions |

### Debugger & Protocol References
| Repo | What to study |
|------|---------------|
| rr | Record & replay architecture, checkpoint model, syscall interception |
| debug-adapter-protocol | DAP spec, message types, capability negotiation, state machine |
| mcp-spec | MCP protocol specification, resource/tool/prompt primitives |
| mcp-servers | MCP server implementations — patterns for tool exposure |

### TUI References
| Repo | What to study |
|------|---------------|
| ratatui | Immediate-mode rendering, widget system, layout engine |
| textual | Textual's CSS-based layout, widget tree (Python comparison) |
| taskwarrior-tui | Ratatui app architecture, keybindings, table widgets |
| trippy | Ratatui network diagnostic TUI — real-time data display |
| git-cliff | Ratatui config TUI — clean CLI patterns |

### GPU / CUDA Infrastructure
| Repo | What to study |
|------|---------------|
| cuda-samples | CUPTI examples, stream/event patterns, memory management |
| cuda-checkpoint | CRIU + CUDA state — checkpoint/restore implementation |
| nccl | Ring/tree algorithms, communicator internals, topology detection |
| open-gpu-kernel-modules | NVIDIA kernel driver — UVM, ioctl interface, memory management |
| DCGM | GPU health monitoring, Prometheus integration, telemetry |
| flash-attention | Fused attention kernel, memory-efficient tiling, SRAM management |
| triton | GPU kernel compiler, Python→PTX pipeline, tile-based programming |

### ML Frameworks (reference only)
| Repo | What to study |
|------|---------------|
| pytorch | (no-checkout) Dispatcher, autograd engine, c10, torch.compile |
| jax | (no-checkout) Jaxpr tracing, XLA lowering, transformations |
| tinygrad | Full stack in 10K LOC — UOp IR, scheduler, lazy eval, codegen |
| vllm | Multi-GPU inference serving, PagedAttention, continuous batching |

### Systems Observability / eBPF
| Repo | What to study |
|------|---------------|
| bcc | BPF tools, Python bindings, GPU tracing examples |
| bpftrace | One-liner tracing language, probe attachment patterns |
| libbpf-bootstrap | CO-RE scaffolding, BTF, portable eBPF programs |
| perfetto | Trace format, SQL analysis, production tracing architecture |

### Profilers
| Repo | What to study |
|------|---------------|
| py-spy | Out-of-process sampling, process_vm_readv, Rust profiler |
| scalene | Combined CPU+GPU+memory profiling, line-level attribution |

### Build Infrastructure
| Repo | What to study |
|------|---------------|
| pyo3 | Rust↔Python bindings, GIL handling, numpy/ndarray interop |
| safetensors | Fast tensor serialization, memory-mapped I/O, Rust + Python |
| ROCm | AMD GPU stack overview, HIP translation layer |

## Acceptance

- [x] All accessible repos cloned to quarantine/
- [x] Catalog with what to study in each
