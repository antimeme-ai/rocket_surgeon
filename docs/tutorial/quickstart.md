# Quickstart

Get rocket_surgeon running and step through a model's forward pass.

## Prerequisites

- Rust 1.88+
- Python 3.11+
- PyTorch 2.x

## Build

```bash
cargo xtask setup
```

This creates a virtualenv, installs Python dependencies, builds the Rust workspace, and compiles the PyO3 worker.

## Start the daemon

The daemon reads JSON-RPC 2.0 messages on stdin and writes responses on stdout. All messages use Content-Length framing.

```bash
./target/debug/rocket-surgeon
```

## Initialize a session

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "client_name": "my-client",
    "protocol_version": "0.3.0"
  }
}
```

The response includes the session state with a `session_id` and the server's capabilities.

## Attach a model

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "attach",
  "params": {
    "model_path": "gpt2",
    "model_family": "gpt2",
    "device": "cpu",
    "num_ranks": 1
  }
}
```

The daemon spawns an orchestrator and worker, loads the model via PyTorch, installs hooks on every module, and returns the model's architecture (layers, heads, hidden dim, component vocabulary).

## Step through the forward pass

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "rocket/step",
  "params": {
    "direction": "forward",
    "count": 5
  }
}
```

Each tick advances through one component in the model's execution order. The response reports where the debugger stopped (`stopped_at`).

## Inspect a tensor

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "rocket/inspect",
  "params": {
    "target": "gpt2:0:0:attn.q_proj:output"
  }
}
```

Returns tensor summary statistics: mean, std, min, max, shape, dtype, sparsity.

## Register an intervention

```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "method": "rocket/intervene",
  "params": {
    "action": "set",
    "recipe": {
      "id": "scale-layer0-attn",
      "type": "scale",
      "target": "gpt2:0:0:attn.o_proj:fwd",
      "params": {"factor": 0.5},
      "priority": 0
    }
  }
}
```

Interventions persist across steps. On each subsequent `rocket/step`, the intervention engine applies matching recipes at each hook barrier and reports which fired in `fired_interventions`.

Supported intervention types: `ablate`, `scale`, `add`, `patch`, `clamp`, `attention_mask`, `embed_swap`, `embed_noise`.

## Export a session bundle

```json
{
  "jsonrpc": "2.0",
  "id": 6,
  "method": "rocket/session.export",
  "params": {
    "path": "/tmp/my-session.tar.gz",
    "include_tensors": false
  }
}
```

Produces a tar.gz archive containing the session manifest, registered interventions, and protocol trace log.
