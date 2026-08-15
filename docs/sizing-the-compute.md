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
| kernel rate vs batch | pure net forwards/s at batch 32-2,048, per device — the CUDA:CPU ratio of the two arms is the crossover curve | `kernel_rate_vs_batch` (`v1_curves`; 512-2,048 via `v1_curves_ext`) |

The cell pins the eager path (`callback: fast`) — the neutrality claim above holds
for eager ATen dispatch only. reinfors' operating configuration compiles its callback,
so its own batch response is the reinfors-specific companion cell
(`kernel_rate_vs_batch_compiled`) in [engine sizing](configuring-the-engines.md).

**Instrument:** [`experiments/measure_inference.py`](../experiments/measure_inference.py)
`--mode kernel` — the net alone, no engine. `--devices` is a sweep list: every point
runs once per listed device; the crossover is not a separate measurement but the
ratio between the arms.
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

## Results — V1 (provisional)

> The kernel-rate table is computed from the run telemetry (3 cycles, `v1_curves` +
> `v1_curves_ext`). It is marked provisional until the backing artifacts are committed to
> `published/v1/`, at which point the table is regenerated from those tracked artifacts and
> marked final.

**Kernel rate vs batch** (pure forwards, w256 d8; medians over 3 cycles):

| batch | CUDA rows/s | CPU rows/s | CUDA : CPU |
|---|---|---|---|
| 32 | 13,857 | 261 | 53 |
| 64 | 15,278 | 263 | 58 |
| 128 | 14,295 | 246 | 58 |
| 256 | 19,266 | 232 | 83 |
| 512 | 20,236 | 201 | 101 |
| 1024 | 20,655 | 203 | 102 |
| 2048 | 20,879 | 203 | 103 |

CUDA rises to a ~21k plateau (dip at 128); CPU declines past batch 64. The ratio widens
53× → 103×.

**Crossover:** CUDA clears CPU ×2 at every batch measured — the smallest (32) already runs
53× CPU. The batch-1 regime pooled collection exists to escape sits below this sweep; these
are all batched forwards.

*Provenance: V1 campaign (v1 wheel / v1.0.x benchmarks), g5.2xlarge (A10G), cell `kernel_rate_vs_batch` in
`v1_curves` (batch 32-256) and `v1_curves_ext` (512-2,048), 3 cycles, medians with
per-cycle spreads ([methodology](methodology.md)).*
