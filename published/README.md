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

Pre-campaign-era artifacts (the runs behind the figures published before V1) were
retired from tracking at commit `a1745b5` and are retained in the maintainers'
archive; `docs/history.md` remains their narrative record.
