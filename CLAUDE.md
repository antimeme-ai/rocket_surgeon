# Project: rocket_surgeon

A proper debugger + in-situ surgery tool operating natively on multi-GPU forward passes. Step-through transformer internals (dense and MoE) one tick at a time, forward and backward, with full surgical intervention between steps.

## Repo structure

```
.context/          # LLM working memory (beads, decisions, lit-reviews, session-notes)
docs/
  adr/             # Architectural Decision Records (formal, load-bearing)
  specs/           # Design specs from brainstorming
  reference/       # Reference material
scripts/           # Operational tooling (hooks, CI helpers)
tck/               # Behavioral specs (Gherkin .feature files)
quarantine/        # Holding cell for reference repos — gitignored
```

`../papers/` — PDF library (outside repo, never in git)

## Development methodology: JSMNTL

Extreme rigor baseline. No shortcuts.

### Planning
- Deep literature review for everything built (papers, reference impls, community test suites)
- Written sub-plan even at task level
- Plans built from: papers, reference implementations, existing tools

### Development cycle
1. Written sub-plan
2. TCK red (Gherkin .feature specs first)
3. Get tests compiling/running (red)
4. Write implementation code
5. Run tests -> fix until green
6. Subagent code reviewer -> fix ALL findings
7. Repeat

### Infrastructure
- Pre-commit hooks: zero warnings policy
- ADRs for load-bearing architectural decisions
- Beads for ALL issue tracking (in .context/beads/)
- Git commits: frequent, atomic, descriptive

## Design principles

- Dual-interface: TUI for humans, structured protocol for LLMs
- LLM ergonomics are first-class — LLMs as end users is inalienable
- One "tick" at a time through the forward pass, forward and backward
- Full surgical intervention between ticks (modify activations, weights, routing, etc.)
- Must work on multi-GPU setups (DDP, FSDP, tensor/pipeline parallelism)
- Must handle both dense transformers and MoE architectures
- High-fidelity view into every layer, head, expert, activation
