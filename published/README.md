# Published-run artifacts

One directory per campaign, each a **filtered mirror of `runs/`**: identical paths
(`<campaign>/<session>/<cell>/cycleN/…`), with model binaries and replay buffers
stripped — checkpoints are distributed as GitHub release assets. Everything else
ships: the start/completion manifests (command, environment, hashes, status,
metrics), the unified telemetry, raw logs, and PGNs. Because the paths mirror
`runs/`, any citation in the documentation resolves identically against this public
copy and the complete private tree, and every figure re-derives from the manifests
and telemetry here.

Manifests record campaign tags without the `bench_` prefix (`v1`, `v1.0.4`, …); the
corresponding git tags are `bench_v1`, `bench_v1.0.4`, and so on.
