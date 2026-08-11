# Published-run artifacts

One directory per published run: the learner telemetry every published table reduces
(`learner.jsonl`, interior-window counter deltas), the resolved run configuration, the
run's stdout, and a `provenance.json` recording dates, commit pins, and the command.
`h2h/` holds every head-to-head log and the accumulated PGN.

Scope notes:

- Commit SHAs in `provenance.json` are **reconstructed** from run timestamps against
  branch history — these runs predate the run-manifest tooling (Arena-protocol matches
  onward append manifests automatically).
- Per-actor logs and replay buffers are omitted (bulk, not load-bearing for any
  published number). Deadline checkpoints (the head-to-head artifacts) are distributed
  as GitHub release assets rather than in-tree; until the release is cut they remain on
  the measurement instance's volume.
- The published tables and their interpretation live in the reinfors documentation
  (`docs/benchmarks/`); this directory is the raw side of those numbers.
