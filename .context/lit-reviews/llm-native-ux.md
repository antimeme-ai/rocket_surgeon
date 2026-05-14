---
topic: LLM-native UX — model-first interface design, function calling, MCP, DAP/GDB-MI for LLMs, composable tool APIs, structured I/O
status: draft
created: 2026-05-14
sources: OpenAI/Anthropic/Google docs, MCP spec, DAP spec, LSP spec, Gorilla, ReAct, Toolformer, various research
---

# LLM-Native UX: Lit Review

How to design a tool that is instantly legible to a capable LLM with no system prompt scaffolding, skill files, or "AI learnings" necessary.

## Core Principle

**The tool's interface IS the documentation.** An LLM should be able to pick up rocket_surgeon's protocol, understand what it can do, and use it effectively from the schema alone. No training data. No special prompts. No wrappers.

## Function Calling Fundamentals

### How LLMs Use Tools
- Model receives JSON schema describing available functions
- Decides when and which to call based on conversation context
- Emits structured JSON matching the schema; runtime executes and returns result
- Model incorporates result and continues
- **Critical**: the schema IS the UX. Poor schemas = poor tool use regardless of model capability.

### What Makes a Good Tool Schema
1. **Descriptive names**: `step_forward` not `sf` or `advance_execution_pointer`
2. **Self-documenting parameters**: enum values that read as sentences, descriptions that explain WHY not just WHAT
3. **Constrained outputs**: always return same shape, never surprise the model with unexpected fields
4. **Actionable errors**: error responses include what went wrong AND what to do about it
5. **State in every response**: never require the model to remember state from previous calls

### Gorilla: LLM API Accuracy
- Benchmark: 1,645 API calls across TorchHub, TensorHub, HuggingFace
- Key finding: models hallucinate API calls — invent parameters, confuse similar APIs
- Retrieval-augmented generation reduces hallucination significantly
- **Implication**: fewer, well-documented tools >> many poorly-documented tools
- **For us**: 5-7 composable primitives with exhaustive schemas beat 50 specific commands

## Protocol Design Models

### DAP (Debug Adapter Protocol)
- JSON-RPC messages: Requests (client->adapter), Responses (adapter->client), Events (adapter->client, async)
- Stateful: initialize -> attach/launch -> running -> stopped (breakpoint/step) -> continue
- Standard capabilities negotiation at initialization
- **Key operations**: setBreakpoints, continue, next, stepIn, stepOut, evaluate, stackTrace, scopes, variables
- **For us**: DAP is the closest existing protocol to what we need. Extend, don't reinvent.

### GDB Machine Interface (MI)
- Line-oriented structured output: `^done,value="0x..."`
- Async records: `*stopped,reason="breakpoint-hit"`
- Every command gets a token for request-response correlation
- Designed for IDE consumption, not human reading
- **Lesson**: structured output format matters more than human readability for programmatic consumers

### LSP (Language Server Protocol)
- JSON-RPC 2.0 over stdio/TCP
- Capability-based: server declares what it supports, client adapts
- **Brilliant pattern**: `initialize` response lists capabilities → client knows exactly what's available
- Progressive disclosure: basic features always work, advanced features opt-in
- **For us**: capability declaration at init means LLM clients discover features without documentation

### MCP (Model Context Protocol)
- Anthropic's standard for LLM-tool integration
- Resources (read), Tools (execute), Prompts (templates)
- JSON-RPC 2.0 transport
- Tool schemas use JSON Schema — same format LLMs already understand
- **For us**: MCP server for rocket_surgeon = any MCP-capable LLM can use it natively

## Design Principles for LLM Consumers

### 1. State in Every Response
- **Every response includes full current state** — position in model, enabled probes, last intervention
- LLMs have no persistent memory between calls; treat each response as potentially the first thing they see
- Include "what just happened" AND "what can happen next" (affordances)
- Example: step response includes current layer, available operations, tensor summary

### 2. Composable Primitives Over Specific Commands
- **5-7 core operations** that compose to cover all use cases:
  1. `step` — move forward/backward by N ticks
  2. `inspect` — read any tensor/state at current position
  3. `intervene` — modify tensor/state at current position
  4. `probe` — attach/detach observation hooks
  5. `checkpoint` — save/restore named states
  6. `evaluate` — run expression against current state
  7. `status` — full state dump
- Composition: `step 1` + `inspect attention.weights` + `intervene head.3 zero` = surgical editing
- **Anti-pattern**: `step_forward_and_inspect_attention_at_layer_12` — too specific, doesn't compose

### 3. Strict Schemas, Predictable Shapes
- Every response is valid JSON with consistent top-level structure
- Required fields always present (never conditional on operation type)
- Use discriminated unions for variant types (type field determines shape)
- **Never** return unstructured text in a field meant for data
- Pagination for large results (tensor data) with explicit cursors

### 4. Actionable Errors
```json
{
  "error": {
    "code": "INVALID_POSITION",
    "message": "Cannot step backward past tick 0",
    "context": {"current_tick": 0, "requested": -1},
    "suggestions": ["step forward first", "load a checkpoint"]
  }
}
```
- Error code (for programmatic handling) + message (for understanding) + context (for debugging) + suggestions (for recovery)
- **Never** just "Error: invalid operation"

### 5. Self-Describing / Discoverable
- `status` returns: current position, available operations, model architecture summary, active probes
- Each operation's response includes `available_actions` listing valid next steps
- Enum values are human/LLM-readable: `"forward"` not `0`, `"attention_output"` not `"attn_out"`
- **No hidden state**: everything observable is inspectable

### 6. Idempotent Where Possible
- `inspect` is always idempotent (pure read)
- `checkpoint save "name"` is idempotent (overwrite is explicit)
- `intervene` is NOT idempotent (mutations) — but result includes before/after for verification
- Idempotency means retries are safe — critical for LLMs that may re-call on ambiguous results

### 7. No System Prompt Dependency
- The tool must be usable from its schema alone
- No "you are a debugger assistant" wrapper needed
- No skill files, harness configurations, or prompt engineering required
- **Test**: can a model use this tool correctly with ONLY the JSON schema and a task description?

## State Machine Design

### Debugger States
```
UNLOADED → LOADED → PAUSED ⇄ STEPPING → PAUSED
                              ↓
                          INSPECTING → PAUSED
                              ↓
                          INTERVENING → PAUSED
```
- Every response includes current state
- Valid transitions listed in response
- Invalid transitions return actionable error with valid alternatives

### State Visibility
- Current tick (position in forward pass)
- Current layer / component
- Active probes and their configurations
- Pending interventions
- Checkpoint list with tick positions
- Model metadata (architecture, parameter count, device mapping)

## LLM-Specific Affordances

### Context Window Management
- Tensor summaries by default (shape, dtype, statistics: min/max/mean/std/norm)
- Full tensor data only on explicit request with pagination
- Hierarchical drill-down: model → layer → component → head → tensor → slice
- **Never dump 768-dimensional vectors unsolicited**

### Streaming vs Batch
- Batch mode (default): complete response after operation finishes
- Streaming mode (opt-in): progress updates for long operations (checkpoint restore, multi-step)
- LLMs generally prefer batch (complete responses to reason about)
- Streaming useful for human TUI client

### Multi-Turn Patterns
- **Explore**: status → inspect layer 0 → inspect layer 1 → ... (systematic sweep)
- **Diagnose**: status → step to layer 8 → inspect attention → inspect MLP → compare
- **Intervene**: checkpoint save → intervene → step → inspect → (checkpoint restore if bad)
- **A/B test**: checkpoint save "baseline" → intervene → step to end → inspect output → checkpoint restore "baseline" → different intervention → compare
- These patterns emerge naturally from composable primitives — don't hardcode them

### Multimodal Considerations
- Vision-capable models can interpret attention heatmaps, activation visualizations
- Emit optional `visualization` field with base64-encoded images or SVG
- Text description always present alongside (accessibility + non-vision models)
- **Future**: generate circuit diagrams, attention pattern images, activation distribution plots

## Anti-Patterns to Avoid

### 1. Natural Language Interfaces
- "Tell me what's happening at layer 12" → ambiguous, model must guess at intent
- Structured command with parameters → precise, reproducible, composable
- NL is for the human TUI, not the machine protocol

### 2. Excessive Tool Count
- 50 tools = model spends context window understanding options instead of using them
- 5-7 composable tools = model learns the algebra quickly, invents novel combinations
- **Cognitive load applies to LLMs too**

### 3. Implicit State
- If the model must remember "I'm at layer 12" from a previous call, it WILL forget or hallucinate
- Include position in every response
- Include valid operations in every response

### 4. Unstructured Text in Structured Fields
- `{"result": "The attention weights show high activation in head 3..."}` — useless for programmatic consumption
- `{"result": {"head_3": {"max": 0.94, "entropy": 1.2}, ...}}` — LLM can reason about numbers

### 5. Documentation-Dependent Features
- If a feature requires reading docs to use correctly, the schema is wrong
- Parameter constraints should be in the schema (enums, ranges, patterns)
- Behavior should be predictable from the function name and parameter names alone

### 6. Asymmetric Interfaces
- If humans get a rich TUI and LLMs get a stripped-down API, the LLM experience suffers
- **Same protocol, different renderers**: TUI renders for human eyes, LLM reads JSON directly
- Feature parity is non-negotiable

## Reference Implementations Worth Studying

### GDB-MI
- Mature machine interface for debuggers
- Token-based request-response correlation
- Async event notifications
- Limitation: line-oriented text format, not JSON

### DAP
- Modern JSON-RPC debugger protocol
- IDE-agnostic (VS Code, Neovim, Emacs, JetBrains)
- Capability negotiation
- Rich type system for variables, scopes, stack frames
- **Closest to what we need** — extend for GPU/tensor domain

### LSP
- Brilliant capability discovery pattern
- Progressive disclosure
- Notification system for async events
- **Pattern to steal**: initialize handshake → capability declaration → feature-gated behavior

### MCP
- Native LLM integration protocol
- Tool schemas in JSON Schema
- Resource exposure for context injection
- **Pattern to steal**: resources for model state, tools for operations

### REPL Design
- Immediate feedback loop
- Tab completion (discoverable commands)
- History (reproducible sessions)
- **For LLMs**: "tab completion" = listing available_actions in every response

## Concrete Protocol Sketch

### Initialize
```json
// Request
{"method": "initialize", "params": {"client": "claude-3.5-sonnet", "capabilities": ["vision"]}}

// Response
{
  "capabilities": {
    "stepping": {"forward": true, "backward": true, "max_ticks": null},
    "inspection": {"tensors": true, "gradients": false, "activations": true},
    "intervention": {"tensor_modify": true, "head_ablation": true, "feature_steering": true},
    "probes": {"static": true, "dynamic": true, "sae_features": true},
    "checkpoints": {"max_saved": 32, "auto_checkpoint": true},
    "visualization": {"attention_maps": true, "activation_plots": true}
  },
  "model_info": {"name": "llama-3-8b", "layers": 32, "heads": 32, "hidden_dim": 4096, "type": "dense"},
  "state": "LOADED",
  "available_actions": ["step", "inspect", "probe", "checkpoint", "status"]
}
```

### Step + Inspect
```json
// Request
{"method": "step", "params": {"direction": "forward", "count": 1}}

// Response
{
  "state": "PAUSED",
  "position": {"tick": 1, "layer": 0, "component": "attention", "phase": "output"},
  "summary": {"output_norm": 12.4, "max_attention": 0.87, "entropy": 2.1},
  "active_probes": [],
  "available_actions": ["step", "inspect", "intervene", "probe", "checkpoint", "status"]
}
```

### Intervene
```json
// Request
{"method": "intervene", "params": {
  "target": "layer.0.attention.head.3",
  "operation": "scale",
  "value": 0.0
}}

// Response
{
  "state": "PAUSED",
  "intervention": {
    "target": "layer.0.attention.head.3",
    "operation": "scale",
    "before": {"norm": 3.2, "max": 0.94},
    "after": {"norm": 0.0, "max": 0.0}
  },
  "position": {"tick": 1, "layer": 0, "component": "attention", "phase": "output"},
  "available_actions": ["step", "inspect", "intervene", "probe", "checkpoint", "status"]
}
```

## Sources

- OpenAI function calling docs, Anthropic tool use docs, Google Gemini function calling docs
- Model Context Protocol specification (modelcontextprotocol.io)
- Debug Adapter Protocol specification (microsoft.github.io/debug-adapter-protocol)
- Language Server Protocol specification (microsoft.github.io/language-server-protocol)
- GDB Machine Interface docs (sourceware.org/gdb/onlinedocs/gdb/GDB_002fMI.html)
- Patil et al. (2023): Gorilla — Large Language Model Connected with Massive APIs
- Yao et al. (2023): ReAct — Synergizing Reasoning and Acting in Language Models
- Schick et al. (2023): Toolformer — Language Models Can Teach Themselves to Use Tools
- Norman (2013): Design of Everyday Things (affordances, signifiers)
- Various MCP server implementations (github)
- Anthropic Claude tool use best practices
