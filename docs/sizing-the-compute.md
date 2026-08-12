# A — Sizing the compute

*How do we choose batch sizes, group sizes, and device — independent of the training
workload?*

This family characterizes the platform: device × net response curves, measured on
isolated components (no learner, no self-play distribution, cache off). Because nothing
about the training workload enters, the curves transfer to any workload on the same net
and device — they are the calibration everything downstream cites.

| experiment | measures | cells |
|---|---|---|
| kernel rate vs batch | pure net forwards/s at batch 32/64/128, per device | `kernel_rate_vs_batch` (`v1_internal`) |
| engine rate vs n_games | realized rows/s through the data-gen loop at n_games 32/64/128, per device | `engine_rate_vs_n_games` (`v1_internal`) |
| CPU/CUDA crossover | the ratio between the device arms of both curves | analysis across the two cells above |

**Instrument:** [`experiments/measure_inference.py`](../experiments/measure_inference.py).
`--mode kernel` runs the net alone; `--mode engine` runs the self-play data-gen loop
with the net as callback. `--devices` is a sweep list — every point runs once per
listed device, and the crossover is the comparison between the arms. Three cycles per
cell; each run leaves `rows.jsonl` plus a finalized manifest.

**What it feeds:**

- the engine batch sweet spot → group sizing for
  [grouped collection](configuring-the-engine.md) and the batch term in its transfer
  model;
- the kernel ceiling → the bound every configuration in
  [the comparison](the-comparison.md) approaches;
- the crossover → whether the GPU is worth using at all for a given net size.

**Caveat carried into every downstream use:** these curves are strongly device- and
net-shape-dependent. They are measured at the benchmark net (w256 d8) and do not
transfer to other nets or devices — measure yours before sizing anything against them.

## Results — V1 (pending)

> **Placeholder.** Figures land with the V1 campaign — every `TBD` fills from the
> manifests under `published/v1/`. Directional language reflects the pre-campaign
> measurements these tables replace.

**Kernel rate vs batch** (pure forwards, w256 d8; medians over 3 cycles):

| batch | CUDA rows/s | CPU rows/s |
|---|---|---|
| 32 | TBD | TBD |
| 64 | TBD | TBD |
| 128 | TBD | TBD |

The A10G's sweet spot for this net has sat at batch 64, with a measurable per-row
*regression* at batch 128 — the term the [grouping lever](configuring-the-engine.md)
prices against.

**Engine rate vs n_games** (data-gen loop, cache off): same shape as the kernel curve,
lower absolute — the gap is the engine's per-row overhead.

| n_games | CUDA rows/s | CPU rows/s |
|---|---|---|
| 32 | TBD | TBD |
| 64 | TBD | TBD |
| 128 | TBD | TBD |

**Crossover:** CUDA has cleared CPU from small batches at this net size, while batch-1
GPU inference sits far *below* CPU — the regime pooled collection exists to escape.
The V1 verdict (smallest batch / n_games where the CUDA:CPU ratio clears 2.0): TBD.

*Provenance: V1 campaign (tag TBD), g5.2xlarge (A10G), cells `kernel_rate_vs_batch` /
`engine_rate_vs_n_games` in `v1_internal`, 3 cycles, medians with per-cycle spreads.*
