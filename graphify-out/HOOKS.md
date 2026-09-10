# Graphify Git Hooks

This repo can auto-rebuild its knowledge graph (`graphify-out/`) on every
commit and branch switch using git's native `post-commit` / `post-checkout`
hooks.

## Setup (once per clone)

```
pip install graphifyy
graphify hook install
```

This writes `post-commit` and `post-checkout` hooks directly into
`.git/hooks/` and registers a merge driver for `graphify-out/graph.json`
(see `.gitattributes`). It does **not** touch `core.hooksPath` and does
**not** interfere with `pre-commit` or any other hook already installed in
`.git/hooks/` — git hooks of different names coexist fine in the same
directory.

## Notes

- `.git/hooks/` is not version-controlled by git, so each clone needs to run
  `graphify hook install` once. This is a deliberate limitation of git, not
  specific to graphify.
- Do **not** set `core.hooksPath` to a custom tracked directory as a way to
  "version" these hooks — doing so makes git ignore `.git/hooks/` entirely
  for *all* hook types, which will silently disable any other hook installed
  there (for example `pre-commit install`'s `pre-commit` hook). This repo
  previously did this and it broke local `pre-commit` enforcement; it has
  since been reverted.
- A merge driver for `graphify-out/graph.json` is registered by
  `graphify hook install` via `.gitattributes`, so merges of the graph file
  are handled automatically instead of producing conflicts.
