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

**Instrument:** [`benchmarks/measure_inference.py`](../benchmarks/measure_inference.py).
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
