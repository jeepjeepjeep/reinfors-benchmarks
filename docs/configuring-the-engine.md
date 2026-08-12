# B — Configuring the engine

*Which throughput features should be on, and what does each buy at the real workload?*

Three opt-in reinfors features, each priced by measurement. None of them touches
learning: f32 widening is bit-identical, the cache clears on weight refresh, grouping
reschedules the same collection — these levers move throughput only. The actual
learning parameters (lr, weight decay, buffer size, reuse) are protocol-matched
constants and are deliberately never tuned — see
[the comparison](the-comparison.md).

| lever | measures | cells |
|---|---|---|
| f32 vs f64 callback outputs | rows/s per dtype arm, w128 and w256 | `f32_ab_f64` / `f32_ab_f32` (`v1_internal`) |
| inference-cache capacity | hit rate at 4k / 32k / 256k / 2M entries under the full workload | `rf_cache_*` (`v1_internal`) |
| grouped collection (`n_groups`) | states/s at matched rows-per-call vs matched game count | rf cells of `v1_grid`, matched-rows analysis |

**Per-lever notes:**

- **f32** — engine-mode A/B via
  [`measure_inference.py`](../experiments/measure_inference.py), identical except the
  callback output dtype. The gain grows as the net shrinks, because the boundary cost
  is a larger share of a smaller forward.
- **cache** — full-workload legs via
  [`measure_throughput.py`](../experiments/measure_throughput.py) `--cache N`. Hit rate
  depends on the game's transposition structure and *rises over training* as the net
  concentrates its own play — the trajectory reads from the matched round's telemetry.
- **grouping** — the decision comparison is matched rows-per-call (n64×1 vs n128×2),
  not matched game count: splitting your current games halves rows per call and pays
  on the [batch curve](sizing-the-compute.md). The transfer model is
  `(S + I) / max(S, I)` (search vs inference time per round) minus that batch term,
  with the inference share measured in the target condition's own telemetry.

**What it feeds:** the rf side's operating configuration for
[the comparison](the-comparison.md) — and the same decision procedure, applied to your
own workload's telemetry, for yours.

## Results — V1 (pending)

> **Placeholder.** Figures land with the V1 campaign — every `TBD` fills from the
> manifests under `published/v1/`. Directional language reflects the pre-campaign
> measurements these tables replace.

**f32 vs f64 callback outputs** (engine mode, batch 64; medians over 3 cycles):

| config (chess, CUDA) | f64 rows/s | f32 rows/s | gain |
|---|---|---|---|
| w256 d8 | TBD | TBD | +TBD% |
| w128 d8 | TBD | TBD | +TBD% |

The gain has grown as the net shrinks — the boundary cost is a larger share of a
smaller forward.

**Inference-cache capacity** (chess self-play, early-training net):

| capacity | hit rate |
|---|---|
| 4,096 | TBD |
| 32,768 | TBD |
| 262,144 | TBD |
| 2M | TBD |

Hit rate has been monotone in capacity but flattens sharply — capacity is a
throughput/host-memory choice, not a cliff. It also *rises over training* as the net
concentrates its own play: TBD% by two hours at the 262,144-entry operating point
(read from the [matched round](the-comparison.md)'s telemetry).

**Grouped collection** (rf cells of `v1_grid`, full round workload):

| config | states/s | rows/call | infer share |
|---|---|---|---|
| n64 × 1 group | TBD | TBD | TBD |
| n128 × 1 group | TBD | TBD | TBD |
| n128 × 2 groups | TBD | TBD | TBD |
| n64 × 2 groups | TBD | TBD | TBD |

In every prior grid the matched-rows comparison (n64×1 → n128×2) delivered close to the
ceiling the ungrouped inference share predicts, while the matched-games split (n64×1 →
n64×2) gained little — as the batch term predicts for half-sized calls. V1 re-measures
both: realized ×TBD against a predicted ×TBD ceiling.

*Provenance: V1 campaign (tag TBD), cells `f32_ab_*` / `rf_cache_*` (`v1_internal`) and
the rf grid (`v1_grid`), 3 cycles (capacity probes: 1), medians with spreads.*
