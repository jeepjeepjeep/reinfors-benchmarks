# A — Sizing the compute

*What can the device do with this net — independent of any training workload or
library?*

This family characterizes the platform: the response of the benchmark net's kernels to
batch size, per device, measured outside any engine. The claim is deliberately
stack-neutral: reinfors' callback and OpenSpiel's libtorch evaluator dispatch the
**same ATen kernels from the same pinned torch generation**, so the kernel ceiling and
batch sweet spot bound both sides of [the comparison](the-comparison.md).

| experiment | measures | cells |
|---|---|---|
| kernel rate vs batch | pure net forwards/s at batch 32/64/128, per device | `kernel_rate_vs_batch` (`v1_internal`) |
| CPU/CUDA crossover | the ratio between the device arms of the curve | analysis of the cell above |

**Instrument:** [`experiments/measure_inference.py`](../experiments/measure_inference.py)
`--mode kernel` — the net alone, no engine. `--devices` is a sweep list: every point
runs once per listed device, and the crossover is the comparison between the arms.
Three cycles; each run leaves `rows.jsonl` plus a finalized manifest. (The companion
`--mode engine` measurement — the same sweep through reinfors' data-gen loop — is
reinfors-specific and lives with [engine sizing](configuring-the-engines.md).)

**What it feeds:**

- the batch sweet spot → call sizing in
  [configuring the engines](configuring-the-engines.md) (engine batch, group size) and
  the batch term in the grouping model;
- the kernel ceiling → the bound every configuration in
  [the comparison](the-comparison.md) approaches, on either stack;
- the crossover → whether the GPU is worth using at all for a given net size.

**Caveat carried into every downstream use:** the curve is strongly device- and
net-shape-dependent. It is measured at the benchmark net (w256 d8) and does not
transfer to other nets or devices — measure yours before sizing anything against it.

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
*regression* at batch 128 — the term the
[grouping model](configuring-the-engines.md) prices against.

**Crossover:** CUDA has cleared CPU from small batches at this net size, while batch-1
GPU inference sits far *below* CPU — the regime pooled collection exists to escape.
The V1 verdict (smallest batch where the CUDA:CPU ratio clears 2.0): TBD.

*Provenance: V1 campaign (tag TBD), g5.2xlarge (A10G), cell `kernel_rate_vs_batch` in
`v1_internal`, 3 cycles, medians with per-cycle spreads
([methodology](methodology.md)).*
