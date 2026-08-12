# C — The comparison

*How do the stacks compare, each at its own best measured configuration — and was the
race fair?*

Three acts, in an order that is itself the argument: selection first, so neither side
races handicapped; a matched round whose fairness is verified by telemetry, not
assumed; then the head-to-head between the models that round produced.

## 1. Operating points

Both sides' topology grids run in `v1_grid` and are analysed in
[configuring the engines](configuring-the-engines.md), which selects each stack's best
measured configuration under the full round workload.
**Selected operating points: OpenSpiel TBD, reinfors TBD** — these are the topologies
encoded in `v1_training.json` (decision point: revise them there if the grids
disagree).

## 2. The matched round — `v1_training`

Three fresh 2-hour legs per side at the selected configurations, the wall-clock budget
enforced identically on both by [`train_leg.py`](../experiments/train_leg.py)'s
scheduled kill. Throughput figures reduce post-hoc from the archived telemetry
(`learner.jsonl` — native from the rf trainer, harness-sampled into the same schema for
the OpenSpiel binary).

**Fairness is measured, not assumed:** gradient-samples per state (target 3.0, both
sides) verifies matched training intensity, so scheduling differences cannot be
mistaken for intensity differences; telemetry fields that are *not* comparable across
stacks (the two cache-hit definitions differ) are flagged rather than compared. The
learning parameters themselves are matched constants
([`lib/protocol.py`](../experiments/lib/protocol.py)), never tuned.

Each rf leg publishes its final net as `model.pt`; every leg records its newest
checkpoint in its manifest.

### Results — V1 (pending)

> **Placeholder.** Figures land with the V1 campaign — every `TBD` fills from the
> manifests under `published/v1/`. Directional language reflects the pre-campaign
> measurements these tables replace.

Cells read as median over the three seeds (spread in parentheses):

| | OpenSpiel | reinfors |
|---|---|---|
| states collected (2h) | TBD | TBD |
| sustained states/s (interior window) | TBD | TBD |
| learn steps (1024-sample equivalents) | TBD | TBD |
| gradient-samples per state (target 3.0) | TBD | TBD |
| final cache hit rate¹ | TBD | TBD |

Every prior round showed a sustained-throughput edge to reinfors at matched wall-clock,
cadence and net architecture: **+TBD%** in V1.

¹ Not comparable across the columns: the two stacks define cache hits differently
(query structure and measurement window differ); the figures are reported per-stack,
never compared.

## 3. Head-to-head — `v1_h2h`

The cycle-k model pairs play 100 Arena-protocol games each: seeded uniform-random
openings, each played once per color, pair-level scoring; 64 simulations per move on
both sides; their side runs its own unmodified engine over the bridge
([`eval_h2h.py`](../experiments/eval_h2h.py)). The spec points both sides at their
training-leg directories (`rf-model` / `os-model`); the harness resolves each engine's
native model format (rf: `model.pt`; OpenSpiel: highest-numbered checkpoint, their
loader takes a directory + number) and records the resolved artifacts and hashes in the
match manifest. Every game is exported as PGN.

### Results — V1 (pending)

> **Placeholder.** Figures land with the V1 campaign — every `TBD` fills from the
> manifests under `published/v1/`. Directional language reflects the pre-campaign
> measurements these tables replace.

| | pooled (3 matches) |
|---|---|
| games (opening pairs) | 300 (150) |
| W / D / L (reinfors perspective) | TBD |
| score ± SE (paired) | TBD |
| implied Elo difference (95% CI) | TBD |

Per-match (one row per training seed pair):

| match | games | W / D / L | score ± SE |
|---|---|---|---|
| cycle 1 | 100 | TBD | TBD |
| cycle 2 | 100 | TBD | TBD |
| cycle 3 | 100 | TBD | TBD |

In prior campaigns the edge held in each match and pooled; interpretation uses the
pair-level standard error, and robustness across training draws is evidenced by the
per-seed replication, not by the pooled interval.

## Gates

`v1_smoke` (4-minute legs, both sides) and `v1_smoke_h2h` (a 2-game match against the
smoke checkpoints) run first after any fresh environment/build assembly, alongside the
binary-smoke pytest gate:

```bash
H2H_SMOKE_OS_PATH=<os smoke leg dir> H2H_SMOKE_OS_CKPT=<n> \
  .venv23/bin/python -m pytest experiments/tests/test_eval_h2h.py
```

Their outputs are gates, never evidence.
