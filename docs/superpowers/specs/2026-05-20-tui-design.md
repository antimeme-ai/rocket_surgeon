# rocket_surgeon TUI Design Specification

Synthesized from sky-claude Volumes I–IV, the Addendum, and brainstorming with the
project lead. This spec is the canonical reference for the complete TUI vision. Phasing
is structural (dependency order), not calendar-bound.

**Protocol version target**: 0.3.0 (breaking changes from 0.2.0 identified in §6).

**Dependencies**: tokenizer crate (separate JSMNTL spec, referenced but not specified here).

---

## §1 Data Model

The TUI operates on eight canonical data structures. Three exist in the protocol today;
five are new (three core, one dependency, one derived). The data model is the spine — every view, every interaction, every LLM
verb is a projection of or action on these structures. The data model leads the
presentation by half a beat because it is less forgiving to change, but the presentation
is where the ambition lives and it receives the same rigor.

### 1.1 Existing structures (protocol v0.2.0)

**Tick position.** `TickPosition` with `tick_id`, `layer`, `component`, `phase`
(Prefill / Decode / PrefillChunked), `token_position`, `rank`, `direction`, `event`.
This is the cursor. Every view reacts to where it points.

**Tensor summary.** `TensorSummary` with shape, dtype, device, optional sharding info,
stats (mean, std, min, max, abs_max, sparsity, l2_norm), histogram, top-K entries. This
is what the Inspector consumes.

**Session state.** `SessionState` envelope on every response: status, position, active
probes, checkpoints, available actions. The LLM-ergonomic guarantee that any single
response is actionable without conversation history.

### 1.2 Three-clock event model (new)

The current `tick_id` is a monotonic counter, but the system has three incommensurable
clocks:

| Clock | Unit | Semantics |
|-------|------|-----------|
| `tick_token` | Token position in sequence | The fundamental time axis. Advances once per generated token (decode) or per position (prefill). |
| `tick_operator` | Within-token traversal index | Layer 0 component 0 through layer N component M. Resets each token. This is what the current `tick_id` actually counts. |
| `tick_wall` | Nanosecond wall time | For performance overlay, Perfetto correlation, heartbeat. |

The tick model gains a `TickClock` struct carrying all three. `TickPosition` exposes
which clock is authoritative for the current operation. Navigation can address any
clock: "go to token 41" vs. "step 3 operators forward" vs. "show me what happened at
wall-time T."

```
TickClock {
    token: u64,
    operator: u64,
    wall_ns: u64,
}
```

The existing `tick_id` becomes an alias for `clock.operator` for backward compatibility.

### 1.3 KV cache geometry (new)

A 4D logical tensor: `(layer, kv_group, position, slot ∈ {K, V}) → R^{d_head}`.

Not a tensor shipped over the wire — a schema the TUI and LLM clients understand for
addressing and projecting. Three natural 2D projections:

| Projection | Axes | Purpose |
|------------|------|---------|
| Ribbon | position × layer | The overview — what does the cache remember across the sequence? |
| Per-layer detail | position × head | What is each head attending to at this layer? |
| Per-position detail | head × layer | How does a single position's representation vary across layers and heads? |

**Cache state overlays**, stacked by priority when they overlap:

| Overlay | Glyph | Palette index | Priority |
|---------|-------|---------------|----------|
| Sink token | ★ | 220 | 5 |
| Heavy hitter | ◆ | 208 | 4 |
| Evicted | · | 240 | 3 |
| KIVI quantized | ≈ | 214 | 2 |
| Page boundary | │ | 244 | 1 |
| RadixAttention shared prefix | ⌐⌐+underline | 212 | 1 |

GQA is rendered physical-first (`kv_group`, not query heads) with a `:kv.view logical`
toggle to show the repeated-head view.

**K vs V intervention semantics differ**: ablating K redistributes softmax probability
(the query has nothing to attend to); ablating V retrieves a zero vector (the query
still attends, but gets nothing). These are different experiments and the protocol must
distinguish them.

### 1.4 Worldline branching graph (new)

A DAG where nodes are checkpoints (Cauchy slices of model state) and edges are forward
runs, optionally annotated with interventions.

**Branch lifecycle**: `live` (in VRAM) → `spilled` (CPU pinned / NVMe) → `dropped`
(resources released). Resource visibility is first-class — the user always knows what
is in VRAM and what each branch costs.

**Branching operations**:
- `branch.fork` — create a new branch from a checkpoint
- `branch.drop` — release a branch's resources (tier transition: live → dropped)
- `branch.compare` — compute divergence metrics between two branches

**Divergence metrics**: cosine similarity, max relative error, KL divergence on logit
distributions, per-layer norm delta. Returned as structured data, not raw tensors.

### 1.5 Token model (dependency)

Referenced from the tokenizer crate (separate spec). The TUI consumes a `Token` struct
per position:

```
Token {
    id: u32,
    char_start: u32,
    char_end: u32,
    text_repr_off: u32,
    text_repr_len: u16,
    flags: u16,
    char_width: u8,
    special_class: u8,
}
```

The TUI renders token blocks. It does not tokenize. The token axis is a view of this
data, not a tokenizer.

### 1.6 Structural observations (new)

The daemon computes cheap, model-agnostic structural metrics during stepping and
surfaces them on `tick.stopped` events as an optional `observations` array. These are
data facts, not MI opinions:

- Residual norm anomalies (layer norm > Nσ above cross-layer trend)
- Attention concentration (head allocates >X% to a single position)
- Logit lens deltas (prediction changes between adjacent layers)
- Sparsity shifts (activation sparsity jumps between components)
- Sink token detection (position receives disproportionate attention across heads/layers)

These observations drive the niche-construction loop: the system surfaces what is
structurally notable → the user (human or LLM) forms hypotheses → investigation
produces new data → the system surfaces new observations. The feedback loop starts
before the user asks their first question.

---

## §2 Dataflow

How data moves from the daemon through the TUI, and how intent flows back.

### 2.1 Event-driven reactive loop

The TUI is not a request-response client. It subscribes to the daemon's event stream
and maintains a local mirror of session state.

```
daemon events ──► event ingress ──► state reducer ──► diff engine ──► render
                                        ▲
user intent ──► input decoder ──► command dispatch ──┘
```

**Event ingress.** The TUI holds a persistent connection (Unix socket for local,
WebSocket + TLS 1.3 / ChaCha20-Poly1305 for remote). All events (`tick.stopped`,
`probe.fired`, `kv.update`, `branch.created`, etc.) flow in continuously. No polling.

**State reducer.** Elm-style: pure function `(State, Event) → State`. The state is the
single source of truth — current tick position (all three clocks), cached tensor
summaries, KV cache metadata, branch graph, active probes, resource usage. Every event
produces a new state. No mutation outside the reducer.

**Diff engine.** Compares previous and new state, determines which views are dirty.
Views declare their data dependencies; the diff engine uses those declarations to
minimize recomputation. A cursor move in the Tower dirties the Inspector but does not
dirty an unrelated KV ribbon in another split.

**Input decoder.** Translates raw input — keystrokes, MIDI CC/note messages, mouse —
into abstract `NavigationEvent` and `CommandEvent` values. The keyboard is one
controller. A MIDI mapping file is another. Both produce the same event types. The
decoder is a trait with swappable implementations.

### 2.2 The two directions

**Data → Information (render path).** Daemon emits `tick.stopped`. State reducer
updates position. Diff engine marks Tower and Inspector dirty. Tower queries state for
the new layer's components and cached stats. Inspector fetches the tensor summary for
the newly-focused component. Renderer draws both. Total budget: 16ms from event to
pixels.

**Intent → Action (command path).** User presses `j` (or hits a launchpad pad). Input
decoder emits `NavigationEvent::Down`. Command dispatch resolves this against the
current mode and focused view. If the new target needs data the local cache doesn't
have, dispatch fires an `inspect` request to the daemon. When the response arrives, it
enters event ingress and the render path takes over. Intent and data close the loop.

### 2.3 Subscription model

The TUI subscribes to what the current view configuration needs:

- Tower visible → `tick.stopped` (always on regardless)
- KV ribbon visible → `kv.update`, `kv.evicted`
- Worldline visible → `branch.created`, `branch.tier_changed`
- Probes active → `probe.fired`

When views open/close/switch, subscriptions update. The daemon only sends what the TUI
asked for. Event filtering parameters on the subscribe verb control granularity (layer
ranges, component patterns).

### 2.4 Prefetch and caching

The TUI aggressively caches tensor summaries and pre-fetches adjacent data. Cursor at
layer 12 → layers 11 and 13 already requested. Stepping through tokens → next token's
data pre-fetched during current frame's render. By the time you navigate somewhere, the
data is already local.

Cache eviction is LRU with a configurable memory budget. Tensor slice data (heavy
payloads) evicts first; summaries (lightweight) persist longer.

### 2.5 Context-reactive tiling

When state changes imply a different view arrangement, the tiling manager proposes a
layout transition:

- Cursor enters a layer with attention components → attention view auto-opens (or
  suggests opening, based on user preference)
- `branch.fork` invoked → Worldline view opens alongside current layout
- KV command issued → Ribbon view takes a split

These are opinionated defaults, user-overridable. Repeated overrides in the same context
are persisted as preferences (a context → layout map, not ML).

---

## §3 Interaction Design

The UI is not a viewport onto data. It is the transformation layer where data becomes
information and intent becomes action. Bidirectional. The views are not downstream of the
data model — they are the other half of the circuit.

When intent and information spark just right, you get a self-sustaining cascade where
intent and information niche-construct each other until intent dissolves into outcome.
The interaction design exists to create the conditions for that cascade.

### 3.1 Modal navigation

The TUI has modes, like vim. The active mode determines what input events do.

**Normal mode** — the default. Navigate between splits, move the cursor within the
focused view. `hjkl` or arrows in text views. In graphical views (Sugiyama DAG, KV
ribbon), navigation maps to the view's natural geometry. Each view defines its own
navigation semantics, but the vocabulary is consistent: directional, jump-to-boundary,
search-next.

**Command mode** — entered via `:`. The Bloomberg grammar lives here. `L 12 AT` to jump
to layer 12 attention. `:kv L12 H0:3` to open KV ribbon. `:t /defendant/` to find a
token. `:branch fork` to branch. Tab completion, history, structured command language.
Exit with `<CR>` (execute) or `<Esc>` (cancel).

**Inspect mode** — entered via `i` on a focused element. Deepens into the current
target. In Tower: inspect mode on a component opens/focuses the Inspector with that
tensor. In KV ribbon: inspect mode on a cell shows the full K and V vectors. The "zoom
in" gesture. `<Esc>` returns to Normal.

**Intervene mode** — entered via `I` (capital). The current target becomes editable.
Visual affordances show what is mutable and what the intervention will do. Confirmation
before applying. `<Esc>` cancels, `<CR>` confirms and fires the intervention verb.

### 3.2 The controller abstraction

```
InputSource (trait)
├── TerminalInput      — crossterm key/mouse events
├── MidiInput          — MIDI CC, note on/off, pad pressure
└── [future sources]   — haptic controllers, OSC, game controllers

InputSource → RawEvent → InputDecoder → NavigationEvent | CommandEvent | ModeEvent
```

`InputDecoder` holds a mapping table: `(mode, raw_event) → abstract_event`. The
keyboard mapping is the default. A MIDI mapping is loaded from a config file. Both
produce the same event types.

**Continuous controllers.** For MIDI faders, knobs, and pressure-sensitive pads:
`ContinuousAdjust { axis, value: f32 }`. A fader mapped to "layer" smoothly scrubs
through layers. A knob mapped to "token position" scrubs through the sequence. Views
respond to continuous values, not just discrete steps.

**Output to controllers.** When TUI state changes, the MIDI output port (or haptic
motor, or LED grid) reflects it back. Cursor layer maps to pad colors. Mode maps to pad
bank. The controller becomes a physical mirror of TUI state. The architecture assumes
haptic and audio output are coming and does not close those doors.

**Input tiers:**

| Tier | Source | Nature |
|------|--------|--------|
| 1 | Abstract navigation events | The native language (move, select, zoom, adjust, confirm, cancel) |
| 2 | Physical controller mapping | The envisioned ideal — spatial, tactile, LED/haptic feedback |
| 3 | QWERTY keybindings | The default controller — maps to the same abstract events |

The Bloomberg command bar sits outside this — it is for when you need to say something
the controller cannot express.

### 3.3 Transitions as information

Navigating is learning. The interaction model is tuned so that moving through the model
structure, watching views react, builds intuition.

- **Animate transitions.** When the cursor moves, views do not snap — they interpolate.
  You see residual norm grow as you descend layers. You see attention patterns shift as
  you move across positions. The transition itself carries information.
- **Adjacency is preloaded.** You never wait for data when moving one step. The
  experience of motion is continuous, not request-response.
- **Sound and haptics are not precluded.** If MIDI input is first-class, MIDI output and
  haptic feedback are natural extensions. Sonification of tensor statistics, haptic
  resistance when approaching dangerous interventions — these are future experiments the
  architecture supports.

### 3.4 The command grammar

Bloomberg-style, living in Command mode.

```ebnf
command     = context_sel? function arg* flag*
context_sel = layer_sel | token_sel | kv_sel | branch_sel
layer_sel   = "L" number
token_sel   = ":t" ( number | "/" regex "/" | anchor )
kv_sel      = ":kv" layer_sel head_range?
branch_sel  = ":branch" branch_id
function    = "AT" | "MLP" | "RESID" | "NORM" | "LENS" | "FORK" | "DROP" | "CMP" | ...
anchor      = "BOS" | "EOS" | "PAD" | "SINK" | "MAXATTN"
head_range  = "H" number ( ":" number )?
```

Every command maps to either a local state change (navigation) or a protocol verb
(action). The grammar is the text interface to the same abstract events that MIDI
produces — with more precision and composability.

---

## §4 Views

Each view is a bridge: data in, information out, intent back in. Not projections of
data — the interface between the user's intent and the data flow. Get it right and data
becomes information.

### 4.1 View framework

Every view implements a common contract:

| Method | Purpose |
|--------|---------|
| `data_deps` | Declares what state this view reads (tick position, tensor cache, KV metadata, branch graph). The diff engine uses this. |
| `dirty(prev, next)` | Given a state transition, is this view dirty? |
| `render(state, rect)` | Draw into the allocated rectangle. Views adapt to their size at render time. |
| `handle(event, state)` | Respond to navigation/command events when focused. Returns state mutations or protocol requests. |
| `nav_geometry` | Describes the view's navigable structure (grid, list, tree, graph) so the input decoder knows what directional movement means here. |

Views are the plugin boundary. Built-in views cover the sky-claude surfaces. Future
views (MoE routing, SAE activations, user-defined analyses) implement the same contract.

### 4.2 Tiling

The layout is a dynamic tiling window manager, not a fixed 4-panel grid. Four panels is
the maximum subdivision. The user controls splits:

- Full screen (1 view)
- Horizontal or vertical split (2 views)
- Quad split (4 views)

Context-reactive defaults are opinionated about what views belong together in common
workflows, but the user overrides freely. The tiling manager does not force a layout —
it proposes, and the user accepts or dismisses.

### 4.3 Built-in views

#### Tower — "Where am I in the model?"

The vertical slice: one token position, all layers. Residual stream flows downward.
Each layer expands to show components (attn.q, attn.k, attn.v, attn.out, mlp, resid).
Collapsed layers show a one-line summary with a residual norm sparkline. The cursor
moves through components — this is the primary navigation act that drives everything
else.

| Aspect | Detail |
|--------|--------|
| Data | Tick position, per-component tensor summaries (cached), residual stream norms per layer |
| Information | Where signal concentrates, where norms explode, which layers matter for this token |
| Intent | Select a component to inspect, jump to a layer, expand/collapse detail |

#### Inspector — "What is in this tensor?"

The deep dive on whatever the Tower cursor points at. Stats, histogram, top-K by
magnitude, attention pattern heatmap (for attention outputs), slice viewer for arbitrary
sub-tensors. Updates reactively when the Tower cursor moves.

| Aspect | Detail |
|--------|--------|
| Data | `TensorSummary` for focused component; optionally full tensor slice |
| Information | Distribution shape, outliers, sparsity structure, attention allocation |
| Intent | Slice into sub-dimensions, toggle between histogram/heatmap/raw, enter Intervene mode |

#### Distribution — "How is the signal evolving?"

Cross-layer aggregates. Residual stream norm growth across all layers. Logit lens
predictions per layer. Tuned lens if available. Per-layer statistics.

| Aspect | Detail |
|--------|--------|
| Data | Per-layer residual norms (all layers), logit lens / tuned lens outputs |
| Information | Signal flow health, where the model "makes up its mind," layer-level comparison |
| Intent | Jump to a layer of interest — navigating the distribution moves the Tower cursor |
| Protocol additions | `LogitLens` and `TunedLens` as `BuiltInView` variants |

#### Timeline — "Where am I in the sequence?"

Horizontal axis: token time. Sparklines of per-tick metrics (residual norm, attention
entropy, probe firings). Speculative decoding shown as an overlay (candidate tokens
dimmed until verified), not as branching.

| Aspect | Detail |
|--------|--------|
| Data | Per-token-position metrics (aggregated from tick events), probe firing markers, spec decode status |
| Information | Sequence-level patterns, where probes fire, which tokens are "interesting" |
| Intent | Jump to a token position — navigating horizontally updates the Tower |

#### Token Axis — "What text am I looking at?"

A strip, not a panel. Two rows: text representation above, token ID below. 5-level LOD
adapting to available width:

| Level | Content | Visible tokens |
|-------|---------|---------------|
| L0 | Full text_repr | ~8–16 |
| L1 | Truncated text_repr | ~24 |
| L2 | Abbreviated (3-char) | ~32 |
| L3 | First glyph + color | ~64 |
| L4 | Color-only column | ~200+ |

Whitespace, control characters, and byte-fallback tokens render with explicit escape
glyphs. Scrollable viewport tracks the token cursor.

| Aspect | Detail |
|--------|--------|
| Data | Token blocks (id, text_repr, char offsets, display width, special_class) |
| Information | Literal text, token boundaries, special token locations, tokenization artifacts |
| Intent | Select a token position, search by text/regex, jump to anchors (BOS, EOS, padding boundary) |
| Dependency | Tokenizer crate (separate spec) |

#### KV Ribbon — "What does the cache remember?"

Position × layer heatmap of KV cache state. Default metric: L2 norm of K vectors.
State overlays stacked by priority (see §1.3). GQA rendered physical-first.

| Aspect | Detail |
|--------|--------|
| Data | `kv.read` responses, `kv.update` / `kv.evicted` events |
| Information | What the model attends to, cache utilization, eviction patterns, sink behavior |
| Intent | Select a cache position to inspect K/V vectors, intervene on entries, toggle K/V display, switch physical/logical head view |
| Protocol additions | `kv.read`, `kv.intervene` verbs; `kv.update`, `kv.evicted` events |

#### Worldline — "What branches exist and how do they diverge?"

The branching DAG rendered via Sugiyama layout (C kernel). Nodes are checkpoints. Edges
are forward runs annotated with intervention summaries. Divergence sparklines on edges.
Resource tier badges on nodes.

| Aspect | Detail |
|--------|--------|
| Data | Branch graph, divergence metrics from `branch.compare`, resource tier per branch |
| Information | Experimental history, which interventions caused what divergence, resource pressure |
| Intent | Fork, drop, compare branches, restore a checkpoint |
| Protocol additions | `branch.fork`, `branch.drop`, `branch.compare`; `branch.created`, `branch.tier_changed` events |
| Rendering | Sugiyama (C kernel): cycle removal → layer assignment → crossing minimization → Brandes-Kopf. Reingold-Tilford for tree subgraphs. Kitty graphics primary; half-block fallback. |

#### Command Bar

Fixed chrome at the bottom, not a splittable panel. In Normal mode: context display
(current position, model, branch, resource usage). In Command mode: input line with
Bloomberg grammar, tab completion against component vocabulary, history. Inline error
display with recovery hints.

---

## §5 Rendering Architecture

### 5.1 Two rendering paths

**Text path — ratatui.** Tower, Inspector stats, Timeline sparklines, Token Axis text,
Command Bar. Standard terminal cells. ratatui owns layout, styling, differential update.

**Graphical path — librocket_viz (C).** KV ribbon heatmaps, Sugiyama DAG, attention
heatmaps, any view needing pixel-level control or computational geometry. C kernels
render to pixel buffers. Buffers are encoded for the terminal graphics protocol and
composited into ratatui's frame via image widgets.

The two paths coexist in the same frame. A split can have Tower (text) on the left and
KV ribbon (graphical) on the right.

### 5.2 librocket_viz — the C core

Pure C library, `extern "C"` interface, called from Rust via FFI. C is the right tool
for SIMD-heavy pixel manipulation and layout math.

**Kernel categories:**

| Category | Algorithms | SIMD |
|----------|-----------|------|
| Colormap | LUT scalar → palette color | SSE2, AVX2, NEON, scalar reference |
| Downsampling | M4 (min-max), LTTB (Largest-Triangle-Three-Buckets) | SSE2, AVX2, NEON, scalar reference |
| Sugiyama layout | Cycle removal (guarded DFS), layer assignment (natural / depth / longest-path / Coffman-Graham), crossing minimization (barycenter + median, 24 sweeps, early termination at <1% improvement), coordinate assignment (Brandes-Kopf with 2020 erratum Alg. 3b corrections) | N/A (graph, not vectorizable) |
| Reingold-Tilford | Tree layout for branching subgraphs | N/A |
| Encoding | Kitty Unicode-placeholder compositor, Sixel encoder | Vectorizable inner loops |

**Data structures**: CSR adjacency `rsviz_dag_t` for graph layout. Arena allocators for
node/edge metadata. Templated transformer layout: solve one ~30-node block, replicate
per layer.

**SIMD support matrix**: SSE2 and AVX2 on x86-64, NEON on ARM64, scalar reference
fallback always available. No `fast-math` — FP determinism is non-negotiable.

### 5.3 The 256-color semantic palette

A designed palette that carries meaning:

| Range | Scale | Purpose |
|-------|-------|---------|
| 16–127 | Viridis sequential | Magnitudes — norms, activations, probabilities. Low=cool, high=hot. |
| 128–207 | RdBu_r diverging | Signed quantities — logits, deltas, divergence. Zero=white, positive=blue, negative=red. |
| 208–215 | Okabe-Ito categorical | Discrete labels — head identity, expert assignment, branch ID. Colorblind-safe. |
| 216–255 | Reserved | UI chrome, overlays, state indicators. |

The palette is a shared visual language. Viridis in the Tower means the same as viridis
in the KV ribbon.

### 5.4 Degradation ladder

Sanctioned terminals (Kitty, Ghostty, WezTerm) get the full experience. Non-sanctioned
terminals are not refused but receive no warranty.

| Tier | Requirements | Rendering |
|------|-------------|-----------|
| 1 | Kitty graphics protocol + 24-bit color | Full fidelity — pixel buffers via Unicode placeholders |
| 2 | Sixel + 24-bit color | Sixel encoding, no Unicode placeholders |
| 3 | Half-block characters + 256 color | Text-only approximation of graphical views |
| 4 | ASCII + 16 color | Emergency fallback |

Capability detection at startup via terminal query sequences. The TUI reports which tier
it detected and what is missing.

### 5.5 Frame budget

16ms target (60fps).

| Stage | Budget | Work |
|-------|--------|------|
| Event ingress + state reduction | ~1ms | Deserialize events, run reducer |
| Diff engine + dirty determination | ~0.1ms | Compare states, flag dirty views |
| View render (text + graphical) | ~12ms | ratatui layout, C kernel pixel fills |
| Compositor (merge + encode) | ~2ms | Kitty/Sixel encoding, terminal write prep |
| Terminal write (differential) | ~1ms | Only changed cells/graphics |

C kernel performance: Sugiyama on a 30-node transformer block is fast. 80-layer model
with 20 branches uses incremental layout (recompute only the affected subgraph) and
caching (layout is stable until the graph changes).

Animations interpolate across frames. The state reducer emits intermediate states for
smooth transitions.

---

## §6 Protocol Backports

Everything the TUI requires that the current protocol (v0.2.0) does not support. This is
the sync point — the TUI spec and the protocol must agree before either moves forward.

### 6.1 New verbs

| Verb | Purpose | Driving section |
|------|---------|-----------------|
| `kv.read` | Read a slice of the KV cache (layer range, position range, head range) | §1.3, §4.3 KV Ribbon |
| `kv.intervene` | Modify KV cache entries (ablate, patch, scale K or V independently) | §1.3, §4.3 KV Ribbon |
| `branch.fork` | Create a new branch from a checkpoint | §1.4, §4.3 Worldline |
| `branch.drop` | Release a branch's resources | §1.4, §4.3 Worldline |
| `branch.compare` | Compute divergence metrics between two branches | §1.4, §4.3 Worldline |
| `view.focus` | Navigate the token axis programmatically (by_id, by_position, by_regex, by_anchor, by_range) | §4.3 Token Axis, §8 LLM ergonomics |
| `rocket/discover` | Probe-point discovery — takes partial/wildcard pattern, returns matching concrete points with metadata | §8 LLM ergonomics |
| `rocket/sweep` | Batch experiment automation — array of trial specs, structured results keyed by trial | §8 LLM ergonomics |
| `rocket/view.define` | Register a user-defined or LLM-defined analysis view composed from existing primitives | §8 LLM ergonomics |

### 6.2 New events

| Event | Purpose |
|-------|---------|
| `kv.update` | KV cache grew (new positions written) |
| `kv.evicted` | Positions evicted from cache |
| `branch.created` | New branch forked |
| `branch.tier_changed` | Branch moved between live / spilled / dropped |
| `spec.step` | Speculative decoding candidate token generated |
| `sweep.trial_complete` | One trial in a sweep finished (streamed intermediate results) |

### 6.3 New error codes

All errors follow the expressiveness contract (§7.1): every error tells you what
happened, why, and what to do about it. `details` carries structured machine-readable
context. `recovery_hint` carries human-readable guidance.

| Code | Name | Trigger | `details` must include |
|------|------|---------|----------------------|
| 32007 | `E_CROSS_REQUEST_KV` | KV read/intervene crosses a request boundary unsafely | `{ position, boundary_tick, current_tick }` |
| 32008 | `E_BRANCH_NOT_FOUND` | Referenced branch does not exist | `{ branch_id, available_branches: [...] }` |
| 32009 | `E_KV_EVICTED` | Referenced KV position was evicted between request and response | `{ position, evicted_at_tick, current_tick, nearest_checkpoint }` |
| 32010 | `E_BRANCH_MERGE_REFUSED` | Branch merge would violate causality constraints | `{ source_branch, target_branch, reason }` |
| 32011 | `E_VRAM_EXHAUSTED` | Operation would exceed VRAM budget | `{ used_mb, total_mb, headroom_mb, per_branch: [{id, size_mb, tier}], recommendation }` |

Existing errors also adopt this contract. `INVALID_TARGET` gains `details: { attempted,
nearest_matches: [...], valid_components_at_layer: [...] }`. `GPU_OOM` gains per-checkpoint
memory accounting.

### 6.4 BuiltInView additions

Current: `ResidualStreamNorm`, `AttentionPattern`, `HeadOutput`, `LogitLens`,
`RoutingDecision`, `RoutingEntropy`, `FeatureAttribution`, `SaeActivation`.

Add:

| View | Purpose |
|------|---------|
| `TunedLens` | Tuned lens per-layer readout |
| `KvCacheRibbon` | Position × layer cache overview |
| `KvCacheDetail` | Per-layer or per-position cache detail projection |
| `WorldlineDag` | Branch graph structure and divergence |

### 6.5 Three-clock tick model

`TickPosition` gains a `clock: TickClock` field. Existing `tick_id` becomes alias for
`clock.operator`. Events carry full clocks. Breaking change.

**Protocol version bumps to 0.3.0.**

### 6.6 AttachResponse extensions

`AttachResponse` must include:

| Field | Purpose |
|-------|---------|
| `component_vocabulary` | Concrete list of valid probe-point targets for the attached model, with tensor shapes |
| `module_tree` | Hierarchical model structure |
| `alias_table` | Bidirectional TransformerLens ↔ rocket_surgeon name mapping |
| `tick_map` | At each granularity level, ticks per layer and ordering |

### 6.7 Step extensions

`StepRequest` gains `run_to: Option<String>` — a probe-point target or the literal
`"completion"`. The daemon steps until the target is reached and returns the result.
The LLM says "run until layer 12 attn.out" or "finish the forward pass" without
counting ticks.

### 6.8 Activation patching refinements

`InterventionType::Ablate` gains a `mode` parameter:

| Mode | Semantics |
|------|-----------|
| `zero` | Replace with zero tensor (current default) |
| `mean` | Replace with dataset mean activation (computed server-side; reference dataset specified via `reference_run` checkpoint ID or inline `reference_tensor_id`) |
| `resample` | Replace with activation from a different input (requires a reference run) |

`InterventionType::Patch` supports cross-branch patching (activation from branch A
applied in branch B).

New intervention types:

| Type | Purpose |
|------|---------|
| `AttentionMask` | Modify attention weights directly (mask specific source-target pairs) |
| `EmbedSwap` | Swap embedding for a different token at a position |
| `EmbedNoise` | Add calibrated Gaussian noise to embeddings |

### 6.9 TransformerLens hook-name aliasing

The bidirectional alias table maps between TransformerLens naming conventions and
rocket_surgeon's internal component vocabulary:

```
blocks.0.attn.hook_q  ↔  model:*:0:attn.q:output
blocks.0.hook_resid_post  ↔  model:*:0:resid.post:output
```

Shipped in `AttachResponse` (§6.6). Derived from the model adapter at attach time.
Either vocabulary is valid in probe-point grammar and command bar input.

### 6.10 Response envelope compactness

Requests gain an `envelope` field:

| Value | Behavior |
|-------|----------|
| `full` (default) | Complete `SessionState` on every response |
| `position` | Status + tick position + tick clocks only |
| `none` | Data payload only, no envelope |

LLM clients negotiate compactness to manage context window pressure. Explicit
`rocket/status` call when full state is needed.

### 6.11 Subscribe filtering

`SubscribeRequest` gains a `filter` parameter:

```json
{
  "events": ["tick.stopped", "probe.fired"],
  "layers": [10, 11, 12],
  "components": ["attn.*"]
}
```

Reduces notification volume. Critical for LLM clients that cannot afford to process
500+ notifications per multi-step operation.

### 6.12 plan.md impact

Phase 4 (currently 13 tasks, dramatically underscoped) requires a full rewrite.

Additionally:
- Phase 3 (checkpoint/replay): `branch.fork` / `branch.drop` / `branch.compare` land
  here — branching is checkpoint infrastructure.
- Phase 0 schema: needs a revision plan for the v0.2.0 → v0.3.0 transition.
- KV cache verbs: cut across Phases 3–5 (KV inspection needs checkpoint engine for safe
  reads during forward pass).

The plan update is the next step after this spec is approved. This spec identifies what
must change; the plan says when and how.

---

## §7 System Contracts

### 7.1 Error expressiveness

Every error — protocol, rendering, input, internal — follows this contract:

- **What** happened: error code + machine-readable category.
- **Why** it happened: human-readable message with the values that caused the failure,
  not just the rule violated. "KV position 847 was evicted 3 ticks ago (at operator
  tick 12041) while your read was in flight" — not "KV position evicted."
- **What to do**: when actionable, `recovery_hint` (human-readable) and structured
  `details` (machine-readable) include a suggested recovery path.

The `ErrorEvent` struct gains `recovery_hint: Option<String>`. The `details` field
carries structured context sufficient for an LLM client to decide its next action
without human interpretation.

This applies retroactively to all existing error codes. Every error in the system must
be useful and expressive.

### 7.2 Dual interface

The TUI is one client. LLMs are another. The protocol serves both. Every view surface
has a structured protocol counterpart:

| TUI surface | Protocol equivalent |
|-------------|-------------------|
| Tower cursor position | `view.focus` with selector |
| Inspector display | `inspect` response |
| KV ribbon | `kv.read` response |
| Worldline graph | `branch.compare` response |
| Command bar input | Protocol verbs directly |

An LLM driving the protocol can do everything the TUI user can do, using the same verbs,
getting the same data. Neither interface is privileged. If a feature works in the TUI but
not via the protocol, that is a bug.

### 7.3 Resource visibility

VRAM is finite and the user must always know the score.

- Status bar always shows VRAM usage (used / total, percentage, visual bar).
- Branch list shows tier badges (live / spilled / dropped) and per-branch cost.
- Before any operation that allocates significantly (fork, checkpoint, second model
  attach), the system shows a cost preview: "This fork will consume ~2.1 GB. Current:
  18.2/24.0 GB. Proceed?"
- `E_VRAM_EXHAUSTED` fires before OOM, not after. The system tracks allocations and
  refuses operations that would exceed a configurable headroom threshold.

### 7.4 Session persistence

The daemon owns session state. The TUI's local cache is just a cache. Disconnect and
reconnect — the TUI rehydrates from the daemon's state and event history.

Multiple clients can connect to the same daemon session: one human in the TUI, one LLM
running automated experiments, both seeing the same model state. The subscription model
keeps them in sync.

---

## §8 LLM Ergonomics

LLMs are end users of this system with the same standing as human users. The protocol is
their TUI. This section is not an appendix — it is a co-equal design surface that
receives the same rigor as the rendering architecture.

### 8.1 Discovery

When an LLM attaches to a model, it receives everything it needs to plan an experiment
in a single response:

- **Component vocabulary**: the concrete list of valid probe-point targets for this
  model, with tensor shapes. Not the generic schema — the actual components present
  after loading this specific architecture.
- **Module tree**: hierarchical model structure for reasoning about parent-child
  component relationships.
- **Alias table**: bidirectional TransformerLens ↔ rocket_surgeon name mapping. An LLM
  trained on TransformerLens code uses the names it knows.
- **Tick map**: at each granularity level, ticks per layer and ordering. The LLM can
  calculate destinations without trial and error.

`rocket/discover` verb: takes a partial or wildcard probe point, returns all matching
concrete points with metadata. Tab completion for LLMs.

### 8.2 Addressing

Symbolic, not positional. The probe-point grammar is the LLM's coordinate system.
TransformerLens aliases mean the LLM uses the vocabulary it was trained on. `view.focus`
selectors let the LLM navigate the token axis natively.

When addressing fails, `INVALID_TARGET` includes nearest valid matches (edit-distance)
and the list of valid component names at the referenced layer. A dead end becomes a
correction.

### 8.3 Response ergonomics

**Self-contained by default, compactable by choice.** Every response carries
`SessionState` so any single response is actionable. The `envelope` field on requests
(`full` / `position` / `none`) lets the LLM negotiate compactness to manage context
window pressure.

**Errors are recovery instructions.** `details` is the LLM-actionable field:
- `E_VRAM_EXHAUSTED` → per-branch memory accounting + concrete drop recommendation
- `INVALID_TARGET` → nearest matches + valid components at referenced layer
- `E_KV_EVICTED` → eviction timing + nearest restorable checkpoint

### 8.4 Experiment automation

The primitive verbs (step, inspect, intervene, checkpoint, replay) are composable and
stay composable. Systematic experimentation gets automation support on top:

**`step` with `run_to`.** Name a destination or say "completion." No tick counting.

**`rocket/sweep`** for batch experiments:

```json
{
  "method": "rocket/sweep",
  "params": {
    "baseline_checkpoint": "ckpt-clean",
    "trials": [
      {
        "interventions": [{"type": "ablate", "target": "...:attn.o_proj:output", "params": {"mode": "zero"}}],
        "run_to": "completion",
        "collect": ["...:lm_head:output"]
      }
    ],
    "metric": {"type": "logit_diff", "tokens": [3681], "position": -1}
  }
}
```

Results keyed by trial index. `sweep.trial_complete` events stream intermediate
progress. The primitives are still there — sweep is explicit automation, not a black
box.

**Intervention IDs are optional.** Omit the ID, the daemon auto-generates and returns
it.

### 8.5 Hackable views

The view system is extensible by both interfaces. Humans set TUI preferences; LLMs
define analysis views. Same registration mechanism.

`rocket/view.define` accepts a view specification — a composition of existing
primitives, metrics, and analysis functions. The LLM names it and calls it repeatedly
via `rocket/view` like any built-in view. Over a session, the LLM accumulates a custom analytical toolkit tuned
to its investigation.

The experiment: do LLMs build useful custom views when given the capability? Build the
hook, watch what happens.

### 8.6 The LLM as operator

A human navigates by moving a cursor and watching views react. An LLM navigates by
issuing verbs and reading structured responses. Different modality, same
intent-information niche-construction loop.

- **One round trip per thought.** `view.focus` returns focused data in a single
  response. `rocket/sweep` returns all trial results in a single exchange (streamed via
  events).
- **Session transcripts as artifacts.** Every protocol exchange is structured JSON-RPC.
  The transcript is reproducible (replay the verbs), auditable (human-readable),
  composable (modify interventions and re-run).
- **Structural observations.** The `observations` array on `tick.stopped` events
  surfaces data-driven anomalies. The LLM reads "layer 9 residual norm is 3.2σ above
  trend" and decides whether to investigate. The system shapes intent without imposing
  interpretation.

### 8.7 The dual-interface invariant

Everything the TUI user can do, the LLM can do via protocol verbs. Every view surface
has a structured counterpart. Neither interface is privileged. If a feature works in the
TUI but not via the protocol, that is a bug. If a protocol verb has no TUI affordance,
that is also a bug.

---

## §9 Structural Phasing

How the TUI work decomposes, in dependency order. No calendar estimates.

### 9.1 Foundation (must exist before any view renders)

| Component | Description |
|-----------|-------------|
| Protocol backports | Three-clock tick model, `AttachResponse` extensions (component vocabulary, alias table, tick map), `envelope` compactness, `step.run_to`, `rocket/discover`. JSMNTL cycle: TCK → red → implement → green → review. |
| Input abstraction | `InputSource` trait, `NavigationEvent` / `CommandEvent` / `ModeEvent` types, terminal input decoder, mode state machine (Normal / Command / Inspect / Intervene). |
| State reducer | Elm-style `(State, Event) → State`, diff engine with dependency declarations, subscription manager. |
| Tiling manager | Dynamic split system (1 / 2 / 4 panels), context-reactive layout proposals, user override persistence. |
| Rendering scaffold | ratatui frame loop, 16ms budget, graphical widget placeholder for C integration, Kitty/Sixel capability detection and degradation ladder. |

### 9.2 Core views (first usable TUI)

| Component | Dependencies |
|-----------|-------------|
| Tower + Inspector | Foundation. The primary navigation pair — proves the reactive dataflow. |
| Token Axis | Foundation + tokenizer crate (separate spec). |
| Command Bar | Foundation. Bloomberg grammar parser, tab completion against component vocabulary. |
| Distribution | Foundation + `LogitLens` / `TunedLens` BuiltInView additions. |

### 9.3 Extended views (require checkpoint/branching infrastructure)

| Component | Dependencies |
|-----------|-------------|
| Timeline | Foundation + accumulated per-tick metrics. |
| KV Ribbon | Foundation + `kv.read` / `kv.intervene` verbs + KV cache events. |
| Worldline | Foundation + `branch.*` verbs + checkpoint infrastructure (plan Phase 3) + Sugiyama in librocket_viz. Heaviest dependency chain. |

### 9.4 librocket_viz (C core, parallel track)

Built independently of TUI integration:

| Kernel | Notes |
|--------|-------|
| Colormap | LUT-based, SIMD. Palette definition. |
| Downsampling | M4, LTTB. Required for any sparkline or time-series view. |
| Sugiyama | Four-phase graph layout. Required for Worldline. |
| Reingold-Tilford | Tree layout. Required for simple branching subgraphs. |
| Kitty/Sixel encoding | Pixel buffer → terminal graphics bytes. |

### 9.5 LLM surface (parallel track)

Built against the protocol without the TUI:

| Component | Notes |
|-----------|-------|
| `rocket/discover` | Probe-point discovery/completion. |
| `rocket/sweep` | Batch experiment automation. |
| `view.focus` | Token-axis navigation for LLMs. |
| `rocket/view.define` | LLM-hackable view registration. |
| Structural observations | Anomaly annotations on `tick.stopped` events. |

### 9.6 Protocol sync point

Before any TUI or LLM surface work begins: protocol backports must be designed,
specified in Gherkin, and types landed. This is the sync point — the TUI spec and the
protocol agree before implementation starts.

The protocol changes go through the full JSMNTL cycle:
1. Gherkin behavioral specs for all new verbs, events, and error codes
2. Red tests (failing against current daemon)
3. Implementation (types, messages, daemon handlers)
4. Green tests
5. Subagent code review, fix all findings
6. Repeat until clean
