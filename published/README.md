# Published-run artifacts

One directory per campaign, each a **filtered mirror of `runs/`**: identical paths
(`<campaign>/<session>/<cell>/cycleN/…`), with model binaries and replay buffers
stripped — checkpoints are distributed as GitHub release assets. Everything else ships:
the start/completion manifests (command, environment, hashes, status, metrics), the
unified telemetry (`learner.jsonl`, `rows.jsonl`), logs, and PGNs. Because the paths
mirror `runs/`, any citation in the reinfors documentation resolves identically against
the public copy here and the complete private tree.

The published tables and their interpretation live in the reinfors documentation
(`docs/benchmarks/`); this directory is the raw side of those numbers.

## `pre-v1/`

Artifacts of the era before the campaign runner and manifest tooling: one directory per
published run (learner telemetry, resolved configuration, stdout, `provenance.json`),
plus `h2h/` with every head-to-head log and the accumulated PGN. These back the
currently published figures until the V1 refresh replaces them. Era-specific caveats:

- Commit SHAs in `provenance.json` are **reconstructed** from run timestamps against
  branch history — these runs predate the run-manifest tooling.
- Per-actor logs and replay buffers are omitted (bulk, not load-bearing for any
  published number). Deadline checkpoints (the head-to-head artifacts) are distributed
  as GitHub release assets rather than in-tree; until the release is cut they remain on
  the measurement instance's volume.
