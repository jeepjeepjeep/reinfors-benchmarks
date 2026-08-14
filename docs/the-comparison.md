# C — The comparison

*How do the stacks compare, each at its own best measured configuration — and was the
race fair?*

Three acts, in an order that is itself the argument: selection first, so neither side
races handicapped; a matched round whose fairness is verified by telemetry, not
assumed; then the head-to-head between the models that round produced.

The workload is chess, AlphaZero-style self-play: both stacks support it natively, and
its 4,672-move encoded action space exercises the wide-policy path where boundary costs
show — the path the claim is about. Where the resulting numbers differ, the causes
trace to structural design choices on each side
([design differences](design-differences.md)), measured under one shared
[methodology](methodology.md).

## 1. Operating points

Both sides' topology grids run in `v1_grid` (with the `v1_grid_ext` extension where
a sweep was still rising at its edge) and are analysed in
[configuring the engines](configuring-the-engines.md), which selects each stack's best
measured configuration under the full round workload.
**Selected operating points: OpenSpiel TBD, reinfors TBD** — these are the topologies
encoded in `v1_training.json` (decision point: revise them there if the grids
disagree). The reinfors configuration includes its
[compiled inference callback](configuring-the-engines.md#reinfors-throughput-levers) —
an rf-side implementation choice inside "each stack at its own best measured
configuration", exactly as OpenSpiel's all-C++ inference service is its own.

## 2. The matched round — `v1_training`

Three fresh 2-hour legs per side at the selected configurations, the wall-clock budget
enforced identically on both by [`train_leg.py`](../experiments/train_leg.py)'s
scheduled kill. Throughput figures reduce post-hoc from the archived telemetry
(`telemetry.jsonl` — written natively by the rf trainer; for the OpenSpiel binary the
harness derives it from their learner's own counters, which see every actor — their
per-actor logs are capped at 20 files and must never be counted).

**The matched knobs** ([`lib/protocol.py`](../experiments/lib/protocol.py)), exactly:
network architecture (layer-for-layer, verified by parameter count at startup); search
budget per move, including the convention that the root expansion counts against it;
exploration constants, Dirichlet noise, temperature schedule; replay-buffer size,
minibatch size, optimizer hyperparameters, with weight decay applied to the same
parameter-name set; and aligned loss definitions — masked policy cross-entropy over
legal actions, value MSE on outcomes in [−1, 1]. None of these is ever tuned.

**Fairness is measured, not assumed:** the two learners amortize training differently
(one minibatch per fixed state count vs periodic full-buffer sweeps), so matched
training intensity is verified from telemetry as gradient-samples per state (target
3.0, both sides) — scheduling differences cannot be mistaken for intensity differences.
Caches are **architecture, not a matched knob**: each stack runs its own cache design
at equal capacity, and cache-off is not a "clean" condition
([why](design-differences.md#consequence-2-one-inference-question-per-node-or-two)).
Telemetry fields that are *not* comparable across stacks (the two cache-hit
definitions differ) are flagged rather than compared. Loss curves are definitionally
aligned but each is measured on its own self-play distribution — they show per-system
learning progress, never head-to-head quality.

**Artifacts under the deadline:** the deadline is a kill, not a request — neither stack
gets to write a "final" checkpoint. The head-to-head loads each side's last *complete
periodic* checkpoint, with cadences configured so worst-case staleness is comparable
(~a minute on both sides); alias files written by the trainers themselves are never
trusted, since a kill can tear them mid-write. Each rf leg's `model.pt` satisfies the
rule: the harness copies the newest periodic checkpoint *after* the child is dead, and
records it, its source, and their hashes in the leg manifest.

### Results — V1 (pending)

> **Placeholder.** Figures land with the V1 campaign — every `TBD` fills from the
> manifests under `published/v1/`. Directional language reflects the pre-campaign
> measurements these tables replace.

Cells read as median over the three runs per side (spread in parentheses) — the
reinfors legs are seeded 1–3; their trainer exposes no seed surface, so its legs are
three fresh draws:

| | OpenSpiel | reinfors |
|---|---|---|
| states collected (2h) | TBD | TBD |
| sustained states/s (interior window) | TBD | TBD |
| learn steps (1024-sample equivalents) | TBD | TBD |
| gradient-samples per state (target 3.0) | TBD | TBD |
| final cache hit rate¹ | TBD | TBD |

The headline: the sustained-throughput difference at matched wall-clock, cadence and
net architecture — **TBD** in V1.

¹ Not comparable across the columns: the two stacks define cache hits differently
(query structure and measurement window differ); the figures are reported per-stack,
never compared.

## 3. Head-to-head — `v1_h2h`

The cycle-k model pairs play 100 Arena-protocol games each: seeded uniform-random
openings, each played once per color, pair-level scoring; 64 simulations per move on
both sides with their native chess solver disabled — search-plus-network against
search-plus-network, no solver assist; their side runs its own unmodified engine over
the bridge
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

Per-match (one row per training-run pair):

| match | games | W / D / L | score ± SE |
|---|---|---|---|
| cycle 1 | 100 | TBD | TBD |
| cycle 2 | 100 | TBD | TBD |
| cycle 3 | 100 | TBD | TBD |

Interpretation uses the pair-level standard error, and robustness across training
draws is evidenced by the per-run replication, not by the pooled interval.

## Gates

`v1_smoke` (4-minute legs, both sides) and `v1_smoke_h2h` (a 2-game match against the
smoke checkpoints) run first after any fresh environment/build assembly, alongside the
binary-smoke pytest gate:

```bash
H2H_SMOKE_OS_PATH=<os smoke leg dir> H2H_SMOKE_OS_CKPT=<n> \
  .venv23/bin/python -m pytest experiments/tests/test_eval_h2h.py
```

Their outputs are gates, never evidence.

## Scope of the claim

**What this comparison is — and is not.** OpenSpiel is a research library whose goals
are breadth and reference clarity; high throughput is not among its stated objectives.
reinfors narrows its scope to a modular game/search/training boundary and asks a
narrower question: **how much throughput does that modularity preserve against a mature
C++ implementation, on a workload both systems support well?** The claim under test is
not "reinfors is faster" — it is that throughput remains comparable while keeping the
pluggable seams that are reinfors' design goal, with every mismatch found treated as a
bug in the benchmark rather than a result.

The comparison is also bounded to its regime: one GPU with few CPU cores (4, SMT
off) — a common single-GPU cloud shape. reinfors' lockstep pooling decouples inference
batch size from CPU concurrency; OpenSpiel's actor fleet couples them, acquiring batch
by adding actors. What that coupling costs on this box is a measured question the
[unified sizing grid](configuring-the-engines.md) answers, not an assumption — and
their architecture is built to scale further with more cores and actors. Core-count
scaling is unmeasured in V1; nothing here ranks the libraries beyond this workload and
host.
