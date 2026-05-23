# IOI Circuit Reproduction

Reproduce the Indirect Object Identification (IOI) circuit analysis from Wang et al. 2023 using rocket_surgeon's protocol commands.

## Background

The IOI task: given "When Mary and John went to the store, John gave a drink to", GPT-2 should predict "Mary" (the indirect object). Wang et al. identified specific attention heads responsible for this behavior, called "name-mover heads" — primarily at layers 9 and 10.

## Setup

Start the daemon and attach GPT-2-small:

```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"client_name":"ioi","protocol_version":"0.3.0"}}
```

```json
{"jsonrpc":"2.0","id":2,"method":"attach","params":{"model_path":"gpt2","model_family":"gpt2","device":"cpu","num_ranks":1}}
```

## Step 1: Baseline forward pass

Step through the model without interventions:

```json
{"jsonrpc":"2.0","id":3,"method":"rocket/step","params":{"direction":"forward","count":20}}
```

The response shows `stopped_at` position and `ticks_executed`. With no interventions, `fired_interventions` is empty.

## Step 2: Register ablate interventions on name-mover heads

Target the output projections of name-mover heads at layers 9 and 10:

```json
{"jsonrpc":"2.0","id":4,"method":"rocket/intervene","params":{"action":"set","recipe":{"id":"ablate-nm-9.9","type":"ablate","target":"gpt2:0:9:attn.o_proj:fwd","params":{},"priority":0}}}
```

```json
{"jsonrpc":"2.0","id":5,"method":"rocket/intervene","params":{"action":"set","recipe":{"id":"ablate-nm-9.6","type":"ablate","target":"gpt2:0:9:attn.o_proj:fwd","params":{},"priority":0}}}
```

```json
{"jsonrpc":"2.0","id":6,"method":"rocket/intervene","params":{"action":"set","recipe":{"id":"ablate-nm-10.0","type":"ablate","target":"gpt2:0:10:attn.o_proj:fwd","params":{},"priority":0}}}
```

Verify with a list:

```json
{"jsonrpc":"2.0","id":7,"method":"rocket/intervene","params":{"action":"list"}}
```

## Step 3: Step with interventions active

```json
{"jsonrpc":"2.0","id":8,"method":"rocket/step","params":{"direction":"forward","count":20}}
```

When execution reaches layer 9+, the response includes `fired_interventions` listing which ablation recipes activated. This confirms the intervention engine correctly matched the target components and applied the ablation during the forward pass.

## Step 4: Clear interventions

```json
{"jsonrpc":"2.0","id":9,"method":"rocket/intervene","params":{"action":"clear","intervention_id":"ablate-nm-9.9"}}
```

Repeat for each intervention ID. After clearing, subsequent steps will have no fired interventions.

## What this validates

- **Full-stack intervention flow**: daemon registers recipes, worker applies them at hook barriers, Python engine zeroes the tensors, fired IDs bubble back up through the protocol response.
- **Target matching**: `gpt2:0:9:attn.o_proj:fwd` correctly matches the attention output projection at layer 9.
- **Persistence**: interventions survive across multiple `rocket/step` calls until explicitly cleared.
