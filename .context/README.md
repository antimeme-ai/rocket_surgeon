# .context/

LLM working memory for rocket_surgeon. Structured artifacts for cross-session continuity.

## Contents

- `beads/` — Issue tracking (open questions, todos, follow-ups, blockers)
- `decisions/` — Lightweight decision logs (formal ADRs go in docs/adr/)
- `lit-reviews/` — Literature review notes organized by topic
- `session-notes/` — Per-session working notes capturing state not obvious from code/git

## Conventions

- Files are markdown with YAML frontmatter for metadata
- Beads use: `BEAD-NNNN-short-slug.md`
- Decisions use: `YYYY-MM-DD-short-slug.md`
- Lit reviews use: `topic-name.md` (living documents, updated as papers are read)
- Session notes use: `YYYY-MM-DD-summary.md`
