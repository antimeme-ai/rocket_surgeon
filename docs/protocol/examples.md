# Protocol Examples

Copy-pasteable JSON-RPC 2.0 request/response pairs for all core verbs. Send these to the daemon's stdin with Content-Length framing.

## initialize

**Request:**
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

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "state": {
      "session_id": "a1b2c3d4-...",
      "status": "initialized",
      "tick_id": null,
      "available_actions": ["attach"]
    },
    "data": {
      "capabilities": {
        "supported_models": ["llama", "gpt2", "mistral", "phi"],
        "max_ranks": 8
      }
    }
  }
}
```

## attach

**Request:**
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

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "state": {
      "session_id": "a1b2c3d4-...",
      "status": "stopped",
      "tick_id": 0,
      "available_actions": ["step", "inspect", "intervene", "detach"]
    },
    "data": {
      "model_id": "gpt2-cpu-0",
      "model_family": "gpt2",
      "num_layers": 12,
      "num_heads": 12,
      "hidden_dim": 768,
      "num_ranks": 1,
      "component_vocabulary": [
        {"canonical": "gpt2:0:0:attn.c_attn:output", "type_name": "Conv1D"},
        {"canonical": "gpt2:0:0:attn.c_proj:output", "type_name": "Conv1D"}
      ]
    }
  }
}
```

## rocket/step

**Request:**
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

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "state": {
      "session_id": "a1b2c3d4-...",
      "status": "stopped",
      "tick_id": 5
    },
    "data": {
      "ticks_executed": 5,
      "stopped_at": {
        "tick_id": 5,
        "layer": 1,
        "component": "attn.c_attn",
        "event": "output",
        "direction": "forward",
        "phase": "prefill",
        "rank": 0
      }
    }
  }
}
```

## rocket/inspect

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "rocket/inspect",
  "params": {
    "target": "gpt2:0:0:attn.c_attn:output"
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "result": {
    "state": {"status": "stopped"},
    "data": {
      "tensor_id": "abc123...",
      "shape": [1, 5, 2304],
      "dtype": "float32",
      "stats": {
        "mean": 0.0012,
        "std": 0.0834,
        "min": -0.312,
        "max": 0.298,
        "abs_max": 0.312,
        "l2_norm": 5.42,
        "sparsity": 0.0
      }
    }
  }
}
```

## rocket/intervene — set

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "method": "rocket/intervene",
  "params": {
    "action": "set",
    "recipe": {
      "id": "scale-attn-0",
      "type": "scale",
      "target": "gpt2:0:0:attn.c_proj:fwd",
      "params": {"factor": 0.5},
      "priority": 0
    }
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "result": {
    "state": {"status": "stopped"},
    "data": {
      "applied": true,
      "active_interventions": [
        {
          "id": "scale-attn-0",
          "type": "scale",
          "target": "gpt2:0:0:attn.c_proj:fwd",
          "params": {"factor": 0.5},
          "priority": 0
        }
      ]
    }
  }
}
```

## rocket/intervene — clear

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 6,
  "method": "rocket/intervene",
  "params": {
    "action": "clear",
    "intervention_id": "scale-attn-0"
  }
}
```

## rocket/intervene — list

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "method": "rocket/intervene",
  "params": {
    "action": "list"
  }
}
```

## rocket/session.export

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 8,
  "method": "rocket/session.export",
  "params": {
    "path": "/tmp/my-session.tar.gz",
    "include_tensors": false
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 8,
  "result": {
    "state": {"status": "stopped"},
    "data": {
      "path": "/tmp/my-session.tar.gz",
      "size_bytes": 4096,
      "artifact_count": 3
    }
  }
}
```

## Intervention Types

| Type | Params | Effect |
|------|--------|--------|
| `ablate` | `{"mode": "zero"}` (default) | Zeroes the tensor |
| `ablate` | `{"mode": "mean"}` | Replaces with running mean |
| `ablate` | `{"mode": "resample"}` | Replaces with random values matching distribution |
| `scale` | `{"factor": 0.5}` | Multiplies tensor by factor |
| `add` | `{"vector": [1.0, 0.0]}` | Adds vector to tensor |
| `patch` | `{"source_tensor_id": "abc..."}` | Replaces with a stored tensor |
| `clamp` | `{"min": -1.0, "max": 1.0}` | Clamps values to range |
| `attention_mask` | `{"source_positions": [0], "target_positions": [5], "mask_value": -10000.0}` | Masks attention connections |
| `embed_swap` | `{"position": 5, "new_token_id": 1234}` | Swaps token embedding |
| `embed_noise` | `{"position": 5, "std": 0.1, "seed": 42}` | Adds Gaussian noise to embedding |

## Target Format

Targets use the canonical probe point grammar:

```
family:rank:layer:component:event
```

Examples:
- `gpt2:0:11:attn.c_proj:fwd` — GPT-2 layer 11 attention output projection
- `llama:0:0:mlp:output` — Llama layer 0 MLP output
- `*:*:*:*:fwd` — All components (wildcard)
- `gpt2:0:9:attn.c_proj:fwd` — Specific layer for IOI name-mover head ablation
