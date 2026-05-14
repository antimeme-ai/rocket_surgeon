---
topic: Traditional debugger architecture and design patterns
status: draft
created: 2026-05-14
sources: GDB, LLDB, rr, Ghidra, TTD/UDB, DAP
---

# Traditional Debuggers + Reverse Engineering: Lit Review

What classic debugger design teaches us about building a neural network debugger.

## GDB

### Architecture
- **Target abstraction vector**: modular function-pointer system abstracting machine-specific details. Architecture definitions specify disassembly, stack walking, trap instructions. This is the extensibility pattern — write analyses once, apply to multiple targets.
- **Breakpoints**: replace instruction at address with trap (INT3/0xCC on x86). On SIGTRAP, save/restore original instruction. Binary patching for control flow interruption.
- **Ptrace**: OS-level API underneath everything. Catches exec syscalls, queries CPU registers, peeks/pokes memory.

### Machine Interface (MI)
Structured, line-based, machine-readable protocol for tool integration:
- Zero or more out-of-band records + single result record per interaction
- Sequence tokens for tracking request/response correlation
- **Variable objects**: complex types exposed as named objects in a tree format — inspect/modify any property of nested structures
- Supports multi-process, reverse debugging, conditional logging

**Key lesson**: MI proves structured output enables abstraction. IDEs consume MI without reimplementing GDB logic. This is exactly the pattern for LLM consumers.

### TUI
- Curses-based, separate synchronized windows (source, assembly, registers, command)
- Seamlessly toggles between curses and standard terminal
- **Key lesson**: UI layer is orthogonal to core debugger. Stepping/inspection engine should be independent of visualization.

### Pretty Printers
- Extensible via Python/Guile: (lookup function, printer) pairs
- Custom visualization of domain-specific types without modifying GDB
- **For us**: pattern for visualizing tensor shapes, dtypes, layer-specific semantics

## LLDB

### Architecture
- Built as reusable components leveraging LLVM libraries
- **Debugger is a library** (LLDB.framework) that the CLI links to — embedding in other systems is trivial
- Uses Clang's expression parser for robust C++ evaluation

### SB API (Scripting Bridge)
- Lightweight, stable C++ API: all classes follow SB<SomeName> naming, non-virtual single-inheritance for binary stability
- SBValue (thread-safe variable access), SBBreakpoint (configuration + sync), SBFrame (stack frame inspection)
- Full Python exposure via SWIG bindings
- Breakpoint scripting passes (frame, bp_loc) to user Python functions

**Key lesson**: explicit API stability via naming conventions and non-virtual classes. Design the public stepping/inspection API early and keep it stable.

## rr (Record and Replay)

### Core Innovation
Records all inputs to Linux processes from the kernel + nondeterministic CPU effects (rdtsc), then replays deterministically. Foundation for reverse debugging.

### Recording
- Records only what crosses process boundaries (syscalls, signals), not internal computation
- Recording slowdown ≤1.2x on complex workloads
- Trace = log of kernel inputs + nondeterministic CPU effects on disk

### Reverse Execution
- **Checkpoint-based backward stepping**: restore a previous checkpoint and execute forward to desired point
- Acts as gdbserver during replay — GDB's reverse-continue/step/next/finish all work
- Replay guarantees identical memory layout, addresses, register values, syscall results

**Key lesson for rocket_surgeon**: reverse execution does NOT require reversing operation semantics. It requires efficient checkpointing and forward replay. Don't reverse autograd — record the forward pass and replay backward to checkpoints.

## Ghidra

### P-Code IR
- Lifts assembly from all architectures into processor-independent intermediate representation
- Low P-Code (direct translation) and High P-Code (decompiler-transformed)
- Address spaces abstract all data uniformly (RAM, registers, stack, temporary)
- SLEIGH processor specification: architecture behavior is declarative, not hard-coded

**Key lesson**: cross-cutting concerns (breakpoints, dataflow, visualization) become feasible when semantics are abstracted to an IR. Suggests lifting tensor operations to a computation-neutral IR before analysis.

### Scripting
- Multiple paths: Jython, Ghidrathon (CPython+JVM), PyGhidra (JPype)
- Headless mode for batch processing
- **Key lesson**: multiple integration paths (GUI, headless, external Python) maximize adoption

### Visualization
- Synchronized windows: CFG, data flow, cross-references, decompiled code — all auto-synced
- **Key lesson**: coordinated multi-view visualization is critical. Stepping through one layer should auto-show activations, gradients, parameters in other views.

## Time-Travel Debugging (TTD, UDB)

### Recording Strategy
- Dynamic just-in-time instrumentation, record only non-deterministic inputs
- 99% of state can be reconstructed on demand
- Snapshots scattered at strategic points during execution
- Space-time tradeoff: fewer snapshots = smaller trace, more replay cost

**Key lesson**: laziness wins. Don't record everything — record only what you can't reconstruct. For neural nets: record random seeds and layer inputs, not all intermediate activations. Checkpoint strategically (every layer or every N ops).

## DAP (Debug Adapter Protocol)

### Protocol Design
- JSON-based, header/content structure similar to HTTP
- Message types: Request (command), Response (typed, correlated by request_seq), Event (unsolicited notifications)
- Machine-processable specification (JSON-schema)
- Async communication: debugger notifies client of state changes rather than polling

### Why DAP matters
1. Tool independence — works with VS Code, JetBrains, Vim/Neovim, custom tools
2. JSON responses parseable by any language (notebooks, CI/CD, programmatic analysis)
3. Async events enable reactive debugging
4. Building on DAP (or DAP-like) inherits ecosystem support

**This is likely the protocol model for rocket_surgeon's LLM interface.**

## Architectural Implications for rocket_surgeon

1. **Checkpoint + replay** (from rr/TTD): record forward pass, checkpoint every layer/N ops, reverse step = restore + replay forward
2. **Structured protocol** (from GDB-MI/DAP): JSON-based request/response/event protocol as the primary interface, TUI built on top
3. **Computation IR** (from Ghidra P-code): lift tensor operations to neutral representation for uniform analysis
4. **Target abstraction** (from GDB): abstract the forward pass computation so core logic works across PyTorch/JAX/etc
5. **Stable scripting API** (from LLDB SB): design clean public API early, expose via Python, keep stable
6. **Pretty printers** (from GDB): pluggable domain-specific visualization for tensor types
7. **Synchronized views** (from Ghidra): multi-pane TUI with auto-synced activation/gradient/parameter views

## Sources

### GDB
- Architecture of Open Source Applications v2 — GDB chapter
- How Does a C Debugger Work (blog.0x972.info)
- GDB/MI Specification (sourceware.org)

### LLDB
- LLDB SB API docs (lldb.llvm.org)
- LLDB Architecture Overview

### rr
- rr-project.org, github.com/rr-debugger/rr
- Engineering Record And Replay For Deployability (USENIX ATC '17)

### Ghidra
- ghidra-sre.org
- Guide to P-code Injection (Swarm/PT Security)

### TTD
- Microsoft TTD docs
- Undo/UDB resources

### DAP
- microsoft.github.io/debug-adapter-protocol
- DAP Specification
