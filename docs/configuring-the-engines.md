# B — Configuring the engines

*How do we determine each stack's best configuration — and what does each reinfors
feature buy?*

Both stacks are sized by the same rule: sweep the topology under the **full round
workload** (learner, checkpoint writes and cache sharing the GPU) and select by
completed-game states/s — the rate that survives a hard deadline, not raw search
speed ([methodology](methodology.md)). reinfors additionally has opt-in throughput features, each priced here. None of
this touches learning: the learning parameters (lr, weight decay, buffer size, reuse)
are protocol-matched constants and are deliberately never tuned — see
[the comparison](the-comparison.md).

## Sizing reinfors — n_games × n_groups

The full factorial — n_games 32/64/128/256 × 1/2 groups, the `rf_*` cells of
`v1_grid` via [`measure_throughput.py`](../experiments/measure_throughput.py) —
measures the workload-level batch response and the grouping lever in one table. With
two groups each call carries ~n/2 rows, so every grouped config's **matched-rows
partner sits one diagonal away** (n64×1 ↔ n128×2, n128×1 ↔ n256×2): the grouping
advantage at equal call size is a direct table read, and the winner is bracketed by
measured neighbors on both sides.

Two supporting pieces:

- **The batch-response curve** — `engine_rate_vs_n_games` (`v1_internal`) via
  [`measure_inference.py`](../experiments/measure_inference.py): rows/s vs call size
  through reinfors' data-gen loop, cache off, no learner, per device. It isolates the
  mechanism (per-row rate peaks at the device's sweet spot and regresses past it),
  predicts the grid's ordering, and is the cheap transferable artifact — on different
  hardware, run this ~6-minute curve instead of the multi-hour grid. It also yields
  the engine-level CPU/CUDA crossover (from which n_games the GPU pays through this
  loop).
- **The grouping model** — with per-round search time `S` and inference time `I`,
  overlap lifts throughput by up to `(S + I) / max(S, I)` (equivalently
  `1 / max(p, 1 - p)` for inference share `p`, measured in the target condition's own
  telemetry), **minus** what the new call size pays on the batch curve. The
  matched-rows diagonal validates the prediction; the matched-games column (same n,
  ×2 groups) shows why splitting *without* doubling usually loses — half-size calls
  pay the curve's left flank.

## Sizing OpenSpiel — actors × inference batch

Their topology axis is independent actor threads feeding a central batcher, so actor
count is simultaneously their CPU concurrency and their inference batch size — batch
is *acquired with actors*, where reinfors acquires it by staging. The corrected
pre-campaign record (their learner's own counters; the earlier actor-log-based grid
undercounted by actors/20 and was retracted) shows their states/s **rising
monotonically** through the measured range — 144.9 / 177.4 / 199.6 at a16/a32/a64,
single ~25-minute legs — with the curve's turn, if any, beyond the old grid's edge.
The V1 grid — `os_*` cells of `v1_grid`, actors 32 through 256 — maps the curve in
two columns: **full-fill** (batch = actors) and **half-fill** (batch = actors/2),
the latter separating batch size from concurrency at every call size. The half-fill
64-row cell (a128:b64) runs 128 games at 64-row calls — the exact games-in-flight and
call size of reinfors' n128×2 operating point.

## reinfors throughput levers

| lever | measures | cells |
|---|---|---|
| f32 vs f64 callback outputs | rows/s per dtype arm, w128 and w256 | `f32_ab_f64` / `f32_ab_f32` (`v1_internal`) |
| inference-cache capacity | hit rate at 4k / 32k / 256k / 2M entries under the full workload | `rf_cache_*` (`v1_internal`) |

- **f32** — engine-mode A/B, identical except the callback output dtype. The gain
  grows as the net shrinks, because the boundary cost is a larger share of a smaller
  forward.
- **cache** — full-workload legs (`measure_throughput.py --cache N`). Hit rate depends
  on the game's transposition structure and *rises over training* as the net
  concentrates its own play — the trajectory reads from the matched round's telemetry.

## Results — V1 (pending)

> **Placeholder.** Figures land with the V1 campaign — every `TBD` fills from the
> manifests under `published/v1/`. Directional language reflects the pre-campaign
> measurements these tables replace.

**The unified sizing grid** — both stacks, full round workload, aligned on nominal
inference call size (with two groups, reinfors calls carry n/2 rows). Cells report
states/s; achieved batch and games-in-flight come from each cell's telemetry —
equal call size does *not* mean equal concurrency, and that difference is the design
comparison itself:

| call size | OpenSpiel (actors = size) | OpenSpiel half-fill (actors = 2×size) | rf ungrouped (n = size) | rf grouped (n = 2×size) |
|---|---|---|---|---|
| 32 | a32: TBD | a64:b32: TBD | n32×1: TBD | n64×2: TBD |
| 64 | a64: TBD | a128:b64: TBD | n64×1: TBD | n128×2: TBD |
| 128 | a128: TBD | a256:b128: TBD | n128×1: TBD | n256×2: TBD |
| 256 | a256: TBD | a512:b256: TBD | n256×1: TBD | n512×2: TBD |

The two "2×size" columns are structural mirrors: each runs twice the games of its
call size — OpenSpiel by capping the batch below the actor count, reinfors by
splitting into two overlapped groups — so every horizontal neighbor pair compares the
stacks at identical concurrency *and* identical call size (a128:b64 ↔ n128×2 being
the operating-point case).

Within the reinfors columns, the grouping lever reads on the diagonal (matched
rows-per-call: n64×1 → n128×2 realized ×TBD against a ×TBD predicted ceiling) and on
the verticals (matched games: half-size calls pay the batch curve). Rows/call and
inference share per rf cell accompany the published table.

**Batch-response curve** (isolated loop, per device; medians over 3 cycles):

| call size | CUDA rows/s | CPU rows/s |
|---|---|---|
| 32 | TBD | TBD |
| 64 | TBD | TBD |
| 128 | TBD | TBD |
| 256 | TBD | TBD |

The per-row rate has peaked at 64 with a measurable regression at 128 — the mechanism
pricing every column of the unified grid. Engine-level crossover (smallest call size
where CUDA clears CPU ×2 through this loop): TBD.

**Selected operating points: OpenSpiel TBD, reinfors TBD** — these become the encoded
topologies in `v1_training.json` ([the comparison](the-comparison.md)).

Pre-registered questions the grid answers: where does the OpenSpiel curve turn (its
corrected record still rises at a64); whether half-fill beats full-fill at matched
call size (concurrency helps completion but splits CPU further); whether a128:b64
matches n128×2 at identical concurrency and call size; and the per-row mechanism
comparison at each side's optimum (µs/row and inference-thread saturation, from the
same telemetry).

**f32 vs f64** (engine mode, call size 64):

| config (chess, CUDA) | f64 rows/s | f32 rows/s | gain |
|---|---|---|---|
| w256 d8 | TBD | TBD | +TBD% |
| w128 d8 | TBD | TBD | +TBD% |

**Inference-cache capacity** (chess self-play, early-training net):

| capacity | hit rate |
|---|---|
| 4,096 | TBD |
| 32,768 | TBD |
| 262,144 | TBD |
| 2M | TBD |

Hit rate has been monotone in capacity but flattens sharply — capacity is a
throughput/host-memory choice, not a cliff. At the 262,144-entry operating point it
reaches TBD% by two hours of training.

*Provenance: V1 campaign (tag TBD), g5.2xlarge (A10G); grids: `v1_grid`, 3 interleaved
cycles per cell; curve/f32: `v1_internal`, 3 cycles; capacity probes: 1 cycle each.
Medians with per-cycle spreads throughout.*
