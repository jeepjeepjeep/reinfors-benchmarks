# Published-run artifacts

One directory per campaign, each a **filtered mirror of `runs/`**: identical paths
(`<campaign>/<session>/<cell>/cycleN/…`), with model binaries and replay buffers
stripped — checkpoints are distributed as GitHub release assets. Everything else
ships: the start/completion manifests (command, environment, hashes, status,
metrics), the unified telemetry, raw logs, and PGNs. Because the paths mirror
`runs/`, any citation in the documentation resolves identically against this public
copy and the complete private tree, and every figure re-derives from the manifests
and telemetry here.

Campaign git tags were renamed after the V1 campaign completed (`v1`, `v1.0.1`…`v1.0.6`,
`v1.1`, `v1.2` → the same names prefixed `bench_`, at the same commits). Manifests record
the tag names as they were at run time; resolve them against the `bench_`-prefixed tags.
