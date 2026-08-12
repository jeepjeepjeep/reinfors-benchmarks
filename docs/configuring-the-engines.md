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

Their topology axis is independent actor threads feeding a central batcher: more
actors fill larger inference batches (rows/s rises) while decelerating every game's
progress — completed-game states/s can *fall* as rows/s improves, which is exactly why
selection is by states/s. The grid — `os_*` cells of `v1_grid`, actors 4/8/16/32/64 plus
the 64:32 **decoupling probe** (batch capped below the actor count, separating the
batch-size effect from the completion effect) — re-verifies their optimum explicitly
and brackets it from both sides: the previously unmeasured 4- and 8-actor cells probe
the left flank, where per-game progress is fastest but the batches starve the
inference service.

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

**reinfors factorial** (full workload; states/s is the selection metric):

| config | states/s | rows/call | infer share |
|---|---|---|---|
| n32 × 1 | TBD | TBD | TBD |
| n32 × 2 | TBD | TBD | TBD |
| n64 × 1 | TBD | TBD | TBD |
| n64 × 2 | TBD | TBD | TBD |
| n128 × 1 | TBD | TBD | TBD |
| n128 × 2 | TBD | TBD | TBD |
| n256 × 1 | TBD | TBD | TBD |
| n256 × 2 | TBD | TBD | TBD |

Previously n128×2 won: matched-rows (n64×1 → n128×2) realized close to the ceiling the
ungrouped inference share predicts, while matched-games (n64×1 → n64×2) gained
nothing — half-size calls pay the batch curve. V1 re-measures the factorial in full:
matched-rows realized ×TBD against a predicted ×TBD ceiling.

**Batch-response curve** (isolated loop, per device; medians over 3 cycles):

| call size | CUDA rows/s | CPU rows/s |
|---|---|---|
| 32 | TBD | TBD |
| 64 | TBD | TBD |
| 128 | TBD | TBD |

The per-row rate has peaked at 64 with a measurable regression at 128 — the mechanism
behind the factorial's ordering. Engine-level crossover (smallest call size where CUDA
clears CPU ×2 through this loop): TBD.

**OpenSpiel actor grid** (full workload):

| config | states/s | achieved batch |
|---|---|---|
| 4 actors | TBD | TBD |
| 8 actors | TBD | TBD |
| 16 actors | TBD | TBD |
| 32 actors | TBD | TBD |
| 64 actors | TBD | TBD |
| 64 actors, batch 32 | TBD | TBD |

Previously states/s fell monotonically with actor count on this 4-core box (16 actors
best measured; the decoupling probe showed batch size was not the cause), while their
rows/s *rose* — the canonical warning against sizing by rows-level reasoning. If the
"smallest actor count that still feeds the inference service" account is right, a4 and
a8 should fall *below* a16 — the optimum bracketed from the starved side, not just the
contended one.

**Selected operating points: OpenSpiel TBD, reinfors TBD** — these become the encoded
topologies in `v1_training.json` ([the comparison](the-comparison.md)). At each side's
optimum, the per-row mechanism comparison (from the same grid telemetry): OpenSpiel
TBD µs/row at its achieved batch with its inference thread TBD% saturated, against
reinfors' TBD µs/row at its call size — previously ~128 vs ~90 µs/row, the structural
gap [design differences](design-differences.md) traces.

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
