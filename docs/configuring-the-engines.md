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
`v1_grid` via [`measure_throughput.py`](../experiments/measure_throughput.py),
extended through n512/n1024 × 1 and n1024/n2048 × 2 by `v1_grid_ext` after the base
sweep was still rising at its edge —
measures the workload-level batch response and the grouping lever in one table. With
two groups each call carries ~n/2 rows, so every grouped config's **matched-rows
partner sits one diagonal away** (n64×1 ↔ n128×2, n128×1 ↔ n256×2): the grouping
advantage at equal call size is a direct table read, and the winner is bracketed by
measured neighbors on both sides.

## Sizing OpenSpiel — actors × inference batch

Their topology axis is independent actor threads feeding a central batcher, so actor
count is simultaneously their CPU concurrency and their inference batch size — batch
is *acquired with actors*, where reinfors acquires it by staging. The V1 grid — `os_*` cells of `v1_grid`, actors 32 through 256, extended through
a1024 full-fill and a2048:b1024 half-fill by `v1_grid_ext` — maps the curve in
two columns: **full-fill** (batch = actors) and **half-fill** (batch = actors/2),
the latter separating batch size from concurrency at every call size. The half-fill
256-row cell (a512:b256) runs 512 games at 256-row calls — the exact games-in-flight and
call size of reinfors' n512×2 operating point.

## reinfors throughput levers

| lever | measures | cells |
|---|---|---|
| f32 vs f64 callback outputs | rows/s per dtype arm, w128 and w256 | `f32_ab_f64` / `f32_ab_f32` (`v1_levers`) |
| inference-cache capacity | hit rate at 4k / 32k / 256k entries under the full workload | `rf_cache_*` (`v1_levers`) |
| compiled inference callback | states/s at the operating point, compiled vs eager | `rf_n512_g2_compiled` / `rf_n512_g2_eager` (`v1_levers` — contemporaneous, cycle-interleaved, at the confirmed operating point; the grid's `rf_n512_g2` serves selection only) |
| compiled batch response | compiled-kernel rows/s at batch 32-2,048, CUDA — the batch term for the operating configuration's grouping model | `kernel_rate_vs_batch_compiled` (`v1_curves`; 512-2,048 via `v1_curves_ext`) |

- **f32** — engine-mode A/B, identical except the callback output dtype. The gain
  grows as the net shrinks, because the boundary cost is a larger share of a smaller
  forward.
- **cache** — full-workload legs (`measure_throughput.py --cache N`). Hit rate depends
  on the game's transposition structure and *rises over training* as the net
  concentrates its own play — the trajectory reads from the matched round's telemetry.
- **compiled** — `torch.compile` on the callback net in **default mode**: the gain is
  inductor's kernel generation (conv/batch-norm fusion on an evaluation-mode net) over
  natural, unpadded call shapes, and it is reinfors' operating configuration (the
  trainer default). Graph-capture modes (`reduce-overhead`/CUDA graphs, with padded
  static shapes) measured no faster than eager on this workload — the forward is
  GPU-bound and kernel-launch submission is already hidden inside each call, so graph
  replay has nothing to recover and its per-call bookkeeping costs the codegen gain.

## Results — V1 (provisional)

> The grid and batch-response numbers are computed from the run telemetry (3 interleaved
> cycles, event-aligned reduce). They are marked provisional until the backing manifests
> and telemetry are committed to `published/v1/`, at which point these tables are
> regenerated from those tracked artifacts and marked final. The lever tables
> (compiled/eager, f32, cache) await `v1_levers` and are marked `TBD`.

**The unified sizing grid** — both stacks, full round workload, aligned on nominal
inference call size (with two groups, reinfors calls carry n/2 rows). Cells report
states/s; achieved batch and games-in-flight come from each cell's telemetry —
equal call size does *not* mean equal concurrency, and that difference is the design
comparison itself. The reinfors cells run the operating configuration (compiled
callback); the eager baseline appears once, in the compiled-lever pair below:

| call size | OpenSpiel (actors = size) | OpenSpiel half-fill (actors = 2×size) | rf ungrouped (n = size) | rf grouped (n = 2×size) |
|---|---|---|---|---|
| 32 | a32: 174.9 | a64:b32: 197.3 | n32×1: 150.7 | n64×2: 186.6 |
| 64 | a64: 202.3 | a128:b64: 204.9 | n64×1: 173.2 | n128×2: 222.1 |
| 128 | a128: 197.1 | a256:b128: 195.2 | n128×1: 172.8 | n256×2: 227.4 |
| 256 | **a256: 236.2** | a512:b256: 205.1 | n256×1: 209.4 | **n512×2: 265.7** |
| 512 | a512: 218.4 | a1024:b512: 149.1 | n512×1: 205.5 | n1024×2: 235.8 |
| 1024 | a1024: 139.2 | a2048:b1024: 126.3 | n1024×1: 120.1 | n2048×2: 153.4 |

The two bold cells are each stack's overall best — the selected operating points. Both
stacks' maxima are interior to the sweep, with a lower neighbor on each side. rf leads
best-vs-best by 12.5%.

The two "2×size" columns are structural mirrors: each runs twice the games of its
call size — OpenSpiel by capping the batch below the actor count, reinfors by
splitting into two overlapped groups — so every horizontal neighbor pair compares the
stacks at identical concurrency *and* identical call size (a512:b256 ↔ n512×2 being
the operating-point case).

Within the reinfors columns, the grouping lever reads on the diagonal (matched
rows-per-call: n256×1 → n512×2 realized ×TBD against a ×TBD predicted ceiling) and on
the verticals (matched games: half-size calls pay the batch curve). The ceiling is
`1 / max(p, 1 − p)` for inference share `p` measured in the ungrouped cell's own
telemetry — overlap can hide at most the smaller of search and inference — minus what
the new call size pays on the batch-response curve. Rows/call and inference share per
rf cell accompany the published table.

**Batch-response curve** (compiled forward — the operating path — CUDA; medians over
3 cycles; cell `kernel_rate_vs_batch_compiled`, 512-2,048 via `v1_curves_ext` —
2,048 brackets the curve one octave past the largest grid call):

| call size | rows/s | µs/row |
|---|---|---|
| 32 | 14,602 | 68.5 |
| 64 | 17,562 | 56.9 |
| 128 | 17,164 | 58.3 |
| 256 | 24,507 | 40.8 |
| 512 | 19,666 | 50.8 |
| 1024 | 26,517 | 37.7 |
| 2048 | 26,980 | 37.1 |

The per-row rate keeps improving through 2,048 (shape-dependent, non-monotone), past the
256-row call where completed-game throughput turns over — so the kernel is not the binding
constraint at the operating point. Engine throughput vs n_games (cell
`engine_rate_vs_n_games`): CUDA 11.7k–17.6k rows/s across n32–1024; CPU ~320 (measured
n32–256). CUDA leads by >30× at every measured n, so the ×2 crossover is below the swept
range. The eager cross-device kernel curve is [section A's](sizing-the-compute.md).

**Selected operating points: OpenSpiel a256 full-fill (batch = actors), reinfors n512×2
compiled** — the grid maxima above; these become the encoded topologies in
`v1_training.json` ([the comparison](the-comparison.md)). OpenSpiel caps nothing: in the
half-fill column, extra actors beat full-fill only below call 256 (the small-call regime
where four cores still have slack), and lose at the optimum.

Pre-registered questions the grid answers: where each stack's curve turns over call
size; whether half-fill beats full-fill at matched
call size (concurrency helps completion but splits CPU further); whether a512:b256
matches n512×2 at identical concurrency and call size; and the per-row mechanism
comparison at each side's optimum (µs/row and inference-thread saturation, from the
same telemetry).

**Compiled vs eager callback** (full workload at the reinfors operating point,
n512×2, cache on; the compiled cell is the operating point itself):

| config (chess, CUDA) | eager states/s | compiled states/s | gain |
|---|---|---|---|
| n512×2, cache 262k | TBD | TBD | +TBD% |

**f32 vs f64** (engine mode, call size 256):

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

Capacity is a throughput/host-memory choice, not a cliff. The sweep tops out at 262,144
(the operating point) on this 30 GB host: a 2M-entry cache of chess policy+value rows
exhausts RAM during the first collect, so capacity above the host's memory is unmeasurable
here rather than a data point.

*Provenance: V1 campaign (tags v1 wheel / v1.0.x benchmarks), g5.2xlarge (A10G); grids:
`v1_grid` and its 512/1,024-call extension `v1_grid_ext`, 3 interleaved cycles per cell
(median states/s; cross-cycle spread ≤3.5% for rf cells, up to 16% for high-actor OS
cells); curves:
`v1_curves` with `v1_curves_ext` (kernel batch 512-2,048, engine n512/n1024), 3
cycles; compiled/eager pair, f32, and capacity
probes: `v1_levers` (run after the grid/curves checkpoint fixes the operating point and
call size). Medians with per-cycle spreads for all 3-cycle cells; the three cache
capacity probes are single runs and their figures are labeled as such.*
