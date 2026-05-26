## Git identity

This repo belongs to the antimeme-ai org. Commits must use the org identity:

```
git config user.name "antimemeai"
git config user.email "hiya@antimeme.ai"
```

History rewritten and repo-local config set — all commits use the org identity.

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

### Hard rules for git (LLM agents and contributors alike)

The requirement is that every change to `master` **has a PR**. That's the audit trail and the CI gate. **Whether and how the PR gets merged is decided by orchestration outside the scope of this Claude** — your job ends at "PR is open with green CI."

- **Never push directly to `master`.** Every change goes through a PR. The branch protection rule "Changes must be made through a pull request" is real, not advisory. Don't admin-bypass; use `gh auth login` to get write perms and open the PR.
- **Don't merge the PR yourself.** Open the PR, push commits, wait for CI. Merging is the orchestrator's call, not yours.
- **Never `--no-verify` on a push to `master`.** Feature branches are fine when local hooks are demonstrably broken (document the reason), but the PR's CI run is the real gate and must pass.
- **Admin overrides on branch protection are off-limits without explicit human confirmation.** Even if the user said "ship it," that's authorization to *open the PR* — not to bypass the flow.
- **If `gh pr create` fails for auth reasons, stop and ask the user to re-auth.** Don't improvise an alternative path that ends up at direct-push-to-master.
- **"gg2g" / "ship it" / "let's go" do NOT mean "skip the PR ceremony."** They mean "open the PR and get CI green." Merge happens elsewhere.

## Design principles

- Dual-interface: TUI for humans, structured protocol for LLMs
- LLM ergonomics are first-class — LLMs as end users is inalienable
- One "tick" at a time through the forward pass, forward and backward
- Full surgical intervention between ticks (modify activations, weights, routing, etc.)
- Must work on multi-GPU setups (DDP, FSDP, tensor/pipeline parallelism)
- Must handle both dense transformers and MoE architectures
- High-fidelity view into every layer, head, expert, activation
