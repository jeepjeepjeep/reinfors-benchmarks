# reinfors-benchmarks

Systems benchmarks comparing [reinfors](../reinfors) against peer RL frameworks. Kept out of the
reinfors repo so reinfors takes no benchmark-only dependencies (OpenSpiel, etc.).

## What is (and isn't) being measured

These are **systems benchmarks**: self-play data-generation throughput at a matched computational
shape — same game, same search budget (simulations/move), same net trunk, same device handling.
Metrics: **leaf evaluations/s**, **moves/s**, **% of wall spent in the net**.

They are **not** algorithm benchmarks. The frameworks implement different algorithms (reinfors:
UCT over per-action Q values + TreeStrap targets; OpenSpiel: AlphaZero-style MCTS with a
prior+value evaluator), so playing strength and sample efficiency are out of scope. See
`benchmarks/openspiel/common.py` for the full list of matched settings and accepted mismatches.

## Setup

Requires a local reinfors checkout as a sibling directory (until reinfors is published).

**Canonical env: `.venv23` (torch 2.3.0).** It matches the libtorch generation OpenSpiel's C++ AZ
pins (2.3.0) and measured ~1.29× faster than torch 2.13 on this small-net CPU workload — so all
current and future runs use it. `.venv` (torch 2.13) is kept only to reproduce the older
kernel-skew ablation rows below.

```bash
# 1. install reinfors (release build) straight into the canonical env
cd ../reinfors
VIRTUAL_ENV=../reinfors-benchmarks/.venv23 uvx maturin develop --release -m crates/reinfors-py/Cargo.toml
cd ../reinfors-benchmarks
```

## Run

```bash
uv run python benchmarks/openspiel/bench_reinfors.py       # reinfors: n_games=1 and 8, cpu and mps
uv run python benchmarks/openspiel/bench_openspiel_py.py   # OpenSpiel Python MCTSBot + torch
```

Benchmark hygiene: close other workloads, run each script a few times, prefer medians. All
numbers below are from an Apple-silicon Mac (release builds only).

## OpenSpiel C++ AlphaZero (the "all native" comparison)

OpenSpiel's flagship native path (C++ MCTS + libtorch net, batched async self-play actors) must be
built from source with `OPEN_SPIEL_BUILD_WITH_LIBTORCH=ON`; the pip wheel does not include it.
`scripts/setup_openspiel_cpp.sh` scaffolds that build into `open_spiel_cpp/` (gitignored).
Note: the libtorch AZ path is CPU/CUDA oriented upstream; on macOS it is effectively CPU-only,
which is itself a datapoint (their all-native path cannot use the Apple GPU; reinfors' callback
path can).

## Results

### Python paths — 2026-07-16, Apple silicon (macOS 26.5.2), torch 2.13, open_spiel 2.0

Medians of 3 runs. `bench_reinfors.py --records 2000`, `bench_openspiel_py.py --moves 400`;
64 sims/move, uct_c 1.4, shared trunk (see common.py).

| config | leaf evals/s | moves/s | % wall in net |
|---|---|---|---|
| reinfors + torch callback [cpu], n_games=1 | 7,049 | 132 | 99.2 |
| reinfors + torch callback [cpu], n_games=8 | **28,938** | 544 | 97.5 |
| reinfors + torch callback [mps], n_games=1 | 1,226 | 23 | 99.9 |
| reinfors + torch callback [mps], n_games=8 | 8,992 | 169 | 99.3 |
| open_spiel Python MCTSBot + torch [cpu] | 5,340 | 198 | 84.4 |
| open_spiel Python MCTSBot + torch [mps] | 839 | 31 | 97.1 |

Reading notes:
- **evals/s is the apples-to-apples metric.** moves/s is not compute-matched here: the OpenSpiel
  bot's evaluator cache carries across its (deterministic, hence repeating) games, cutting its
  evals/move to 26.9 vs reinfors' 53.2 — real AZ self-play adds root noise, so fresh positions
  would push it back toward one eval per new node.
- Sequential vs sequential (n_games=1 vs MCTSBot), both sides are torch-batch-1-bound and land
  within ~1.3x (cpu) / ~1.5x (mps) of each other; reinfors' win there is its near-zero search
  overhead (net is 99% of wall vs open_spiel's 84%).
- reinfors' native mode (n_games=8, pooled leaf evals) is **~5.4x** the OpenSpiel Python path on
  cpu. This — not the boundary — is the structural advantage: batching across parallel games.
- The tiny trunk + small batches keep the GPU (mps) uncompetitive on both sides; that is a
  property of the workload, not of either framework.

### C++ libtorch AlphaZero (all-native path) — 2026-07-16, same machine

`alpha_zero_torch_example` at the pinned commit (see scripts/setup_openspiel_cpp.sh), connect4,
64 sims/move, uct_c 1.4, `--nn_model resnet --nn_width 32 --nn_depth 1`, evaluators off; ~2-3 min
runs, learner idle (buffer filling), so this is pure self-play data generation. moves/s parsed
from the actor logs' per-game timestamps.

| config | moves/s | trunk-normalized* |
|---|---|---|
| open_spiel C++ AZ, 1 actor | 60 | ~98 |
| open_spiel C++ AZ, 8 actors, inference_batch_size 8 | 144 | ~235 |
| (reinfors + torch callback [cpu], n_games=1, from above) | 132 | 132 |
| (reinfors + torch callback [cpu], n_games=8, from above) | 544 | 544 |

*their resnet(32,1) torso is 810k MACs/row vs this benchmark's 497k trunk (1.63x) — no exactly
matching torso exists on both sides, so the normalized column scales their moves/s by 1.63 as if
net cost were the whole story (it is ~85-99% of wall on the python side, unmeasured in theirs).
Treat it as an upper bound on their adjusted throughput.

Reading notes:
- **moves/s at matched sims/move is the metric here** — the C++ AZ does not log its net-forward
  count (it also runs a 262k-entry inference cache, which only helps it).
- Even trunk-normalized, reinfors' callback path at n_games=8 is **~2.3x** their all-native
  batched path (8 actors); sequential vs sequential it is ~1.3-2.2x.
- Their AZ self-play adds dirichlet noise + temperature (algorithmic difference, negligible for
  throughput). Their 8-actor mode uses 8 threads + an async inference batcher; reinfors' engine
  is single-threaded, pooling requests into one torch call — fewer cores for more throughput.
- macOS caveat: their libtorch path is CPU-only (no MPS device exists in it); reinfors' callback
  can use any device torch supports. On CUDA hardware their batcher may fare better — this table
  is Apple-silicon CPU only.

### Sequential decomposition (1 game / 1 actor) — variables removed one at a time

The headline tables above conflate several variables. Removing them (identical net — an exact
torch replica of their resnet(32,1) incl. BatchNorm and both heads; kernels pinned to their
libtorch 2.3 generation; dirichlet noise off; their evaluator instrumented for
requests/cache-hits/forwards/forward-time):

| config (identical net) | moves/s | forwards/move | us/forward | engine us/move |
|---|---|---|---|---|
| reinfors, torch 2.13 (python) | 41.8 | 55.3 | 431 | 66 |
| reinfors, torch 2.3.0 (python) | 58.3 | 55.3 | 309 | 58 |
| open_spiel C++ AZ (libtorch 2.3), cache on | 61.9 | 30.5 (58% hit rate) | 526 | 133 |
| open_spiel C++ AZ (libtorch 2.3), cache off | 26.7 | 70.5 | 523 | ~130-500 (noise) |
| reinfors, near-zero net | 1187 | 56.9 | 14 | 31 |
| open_spiel C++ AZ, near-zero net (tiny mlp) | 103 | 71.9 | 133 | 164 |

**Conclusion: at truly matched settings the two frameworks are at sequential parity
(58.3 vs 61.9 moves/s)**, via two offsetting differences:

1. Their evaluator has a transposition/LRU cache (~2.3x fewer real forwards in connect4
   early self-play). reinfors has no eval cache and pays every leaf.
2. Their per-forward cost is ~1.6x higher: a fixed ~133 us/call on the libtorch C++ inference
   path (input assembly, dispatch, output unpacking — measured with a near-zero net) vs ~14 us
   for reinfors' whole numpy->pyo3->python-torch callback round trip. Kernel time proper is
   similar (same ATen generation).

Both engines are negligible sequentially (reinfors ~31-66 us/move, theirs ~130-160 us/move; 1-2%
of wall). The earlier headline gaps decompose as: sequential 2.2x = net-architecture choice
(their BN-heavy resnet is ~3x costlier at batch-1 than this benchmark's plain trunk) and NOT
engine efficiency; the kernel-version confound was real but ran AGAINST reinfors (torch 2.13 is
~40% slower than 2.3 at batch-1 on this net); the n_games=8 headline row remains to be redone
net-matched before the parallel comparison can be interpreted (async 1ms-deadline batcher vs
lockstep pooling).

### Parallel decomposition (8 games / 8 actors, identical net, matched kernels)

Same treatment as the sequential table: identical resnet(32,1) net both sides, ε=0, their
evaluator instrumented. reinfors rows at torch 2.3.0 (their libtorch generation) and 2.13.

| config (identical net) | moves/s | achieved batch | forwards/move | us/call (us/row) | scaling vs own seq |
|---|---|---|---|---|---|
| reinfors n_games=8, torch 2.3.0 | **355** | 8.00 | 6.9 | 400 (50) | 6.1x |
| reinfors n_games=8, torch 2.13 | 197 | 8.00 | 6.9 | 728 (91) | 4.7x |
| open_spiel 8 actors, batch 8 | 157 | 8.00 | 3.5 (61.6% cache) | 1741 (218) | 2.5x |
| open_spiel 8 actors, OMP_NUM_THREADS=1 | 200 | 8.00 | 3.6 | 1302 (163) | 3.2x |

What the parallel comparison actually verified (and falsified):

- **Batch formation is NOT the differentiator — both designs achieve full batch-8.** The
  async-batcher "convoy" hypothesis is dead at this scale: their 1ms-deadline batcher fills
  every batch when 8 actors feed one inference thread.
- **The differentiator is per-row wrapper cost.** Their ~133 us/call fixed cost (sequential
  table) is really ~133 us/ROW — input tensor build and per-element output unpacking scale
  linearly with batch rows: batch-8 costs 8x133 ~ 1064 us of plumbing + ~240 us of kernels
  (matches the measured 1302 with OMP pinned). Batching amortizes their kernels but not their
  marshaling, capping per-row improvement at 526->163 (3.2x). reinfors' seam marshals one flat
  buffer per call regardless of rows, so batching gets full amortization: 309->50 us/row (6.2x).
- **Intra-op thread oversubscription costs them another ~25%** (1741->1302 with OMP=1): free-
  running actors + libtorch's default intra-op pool compete for cores. reinfors' lockstep never
  oversubscribes (workers idle during the forward) — the flip side of forgoing overlap.
- Their shared cache rises slightly with 8 actors (61.6% vs 58.5% hit rate) — real but small;
  with dirichlet noise on (real training) it would be lower.
- **Kernel-version footnote applies here too**: on current torch 2.13 reinfors does 91 us/row
  (2.13's small-batch regression persists at batch-8) and lands at ~197 moves/s — tied with
  their best configuration. At matched kernel generations reinfors leads ~1.8x.

Fair reading: on a single CPU node with homogeneous rollouts, lockstep pooling + a thin
flat-buffer seam beats the async batcher by ~1.8x at matched kernels — for reasons that are
about marshaling design, not language or architecture-of-parallelism. The async design's real
advantages (search/inference overlap, heterogeneous/remote actors, GPU inference server) are
out of scope of this single-node benchmark and are expected to dominate in that topology.

## Summary of findings

The investigation arc: the headline tables above (kept for the record) suggested reinfors
handily beats both OpenSpiel paths. Removing confounds one at a time at 1 game/actor showed
those gaps were dominated by variables that have nothing to do with either framework's
engineering. What survives:

1. **At truly matched settings, the frameworks are at sequential parity** (58.3 vs 61.9
   moves/s). The supported claim is NOT "reinfors is faster than OpenSpiel"; it is: **the
   Python-callback pipeline carries no systems penalty versus an all-native C++ pipeline.**

2. **Per-forward fixed cost: 14 us (reinfors' numpy->pyo3->python-torch round trip) vs 133 us
   (their libtorch C++ inference wrapper)** — measured with a near-zero net. Counterintuitive
   but structural: at sub-millisecond kernel granularity the cost driver is copies, allocations,
   locks and per-element extraction, not language. Their generic threaded evaluator API
   (device loans, defensive copies, per-call mask construction, ActionsAndProbs unpacking,
   cache writes under a lock) does more work per call than reinfors' raw
   flat-buffer-in/values-out contract. Kernels proper are the same ATen either way.

3. **Their evaluator LRU cache is their one genuine edge** — but ~50% of its measured 58.5% hit
   rate refunds their own interface's Prior()/Evaluate() double-call per leaf (cache-off run:
   70.5 forwards/move = 2x unique leaves). The transferable part (within-search transpositions,
   cross-move reuse absent tree reuse, cross-game repeats) is worth ~10-20% in
   transposition-rich games like connect4, ~nothing in games whose states rarely repeat. It is
   sound in training because the learner clears the cache on every weight update
   (alpha_zero.cc: LoadCheckpoint then ClearCache each learn step). An optional infer-seam LRU
   in reinfors would capture the same; an AZ-style planner in reinfors should keep a
   one-forward-per-leaf contract (priors+value from one call), starting where their cached path
   ends without any cache.

4. **Both engines (game sim + tree ops) are negligible** sequentially: 1-2% of wall each.
   OpenSpiel's generic virtual-dispatch State API is NOT a bottleneck at this granularity;
   neither is reinfors' Rust core the source of its throughput.

5. **Algorithm comparability caveat**: PUCT-with-priors (their AZ) vs value-only UCT (reinfors'
   Mcts) makes end-to-end moves/s an algorithm-confounded metric; sample efficiency and
   strength-per-simulation are out of scope by design. The decomposition quantities
   (us/forward, fixed cost/call, engine us/move) are algorithm-independent; forwards/move is
   algorithm-dependent and reported separately with causes named.

6. **Version footnote**: python torch 2.13 is ~40% slower than 2.3 at batch-1 on the BN-heavy
   resnet (431 vs 309 us/forward) — an upstream small-batch regression worth knowing about
   independent of this benchmark.

7. **Parallel (now measured, matched)**: both designs form full batch-8 batches — batch
   formation is a tie, the convoy hypothesis was wrong. reinfors' ~1.8x parallel win at matched
   kernels comes from marshaling design: their inference wrapper costs ~133 us per ROW (scales
   with batch, capping amortization at 3.2x); reinfors' flat-buffer seam is per-CALL, giving
   full 6.2x amortization. Their intra-op oversubscription adds ~25% (reinfors' lockstep
   sidesteps it by idling workers during the forward). On current torch 2.13 the frameworks tie
   (~197 vs ~200 moves/s); the async design's genuine advantages (overlap, remote actors, GPU
   server topology) are out of this benchmark's scope.

This mirrors the companion finding (in the reinfors repo) that reinfors' own fully-fused Rust
training path buys ~0.5-3% over the Python callback: the boundary everyone designs to avoid —
Python in the loop — is cheap when calls are pooled; the costs that matter live in batch
formation, per-call plumbing, and redundant forwards.


## AlphaZero training comparison — quick round (2026-07-24, Apple silicon)

Like-for-like AZ training: connect4, matched search knobs (64 sims, c_puct 2.0, noise 0.25/0.3,
temperature 1.0 drop 10, cutoffs off), matched learner hyperparameters (buffer 65,536, batch 1,024,
reuse 3, lr 1e-4, wd 1e-4), same resnet(32,1) torso, 8 actors ↔ n_games=8, single 30-minute run per
stack (quick round: one seed — trends, not headlines).

### Throughput identity: states/s = net-rows/s ÷ rows-per-state (15-min ablation legs)

| | net rows/s | rows/state | states/s |
|---|---|---|---|
| reinfors, torch 2.13 | ~6,300 | 51.8 | 121 |
| reinfors, torch 2.3.0 | **8,088** | 51.8 | 156 |
| open_spiel, cache ON | 5,867 | **30.2** | 201 |
| open_spiel, cache OFF | 5,885 | 80.8 | 80 |

- **reinfors pushes 1.38× more net rows/s** at matched kernels (their row rate is identical cache
  on/off — they are row-bound; the callback seam + pooled batching is the faster system).
- **Their states/s lead is entirely rows-per-state**: cache-off exposes their Prior/Evaluate
  double-call (80.8 rows/state). Of the cache's effect, 80.8→40.4 refunds their own double-call
  (zero edge vs reinfors, which never pays it); **40.4→30.2 is genuine transposition reuse
  (1.34×)** — the transferable edge, matching the earlier sequential-benchmark estimate. The
  remaining 51.8-vs-40.4 (1.28×) is a search-shape difference (unique evaluations per move), not
  caching.
- **torch 2.13 → 2.3.0 is 1.29×** for reinfors (upstream small-batch regression, previously
  measured; pin accordingly).

### Strength (round-1 30-min checkpoints)

| metric | reinfors net | open_spiel net |
|---|---|---|
| vs open_spiel python MCTSBot referee | **0.70** | — |
| vs own-stack referee | 0.83 (py) | 0.52 (C++) |
| **head-to-head, 50 games, 64 sims both, solver off** | **0.85 (38W 9D 3L)** | 0.15 |

**The referee-free verdict: the reinfors-trained net beats the open_spiel-trained net 0.85 at
equal wall-clock budget** — despite the open_spiel run generating ~1.9× the states and gradient
steps. Learning-per-state strongly favors the reinfors pipeline (candidate causes: single-call π
targets, legal-set priors, no double-call dilution — not yet isolated).

Bench-archaeology notes (kept for honesty): the first h2h attempts scored 0.03/0.00 for reinfors —
(1) their `game_example` defaults `--solve=true`, silently arming their az bot with an exact
MCTS-Solver (net-vs-solver, not net-vs-net); (2) a terminal-value sign bug in OUR python search
(turn does not flip on terminal moves) made it flee winning moves — invisible in referee evals
where both sides shared the search, exposed only by the cross-implementation match, diagnosed via
an asymmetric-sims probe (4× sims failing to help ruled out a mere quality gap). Defaults and
shared assumptions are confounds until enumerated.


### Search-shape investigation + corrected attribution (2026-07-24, 4-min fresh-net legs)

The quick round left a "1.28× search-shape" residual. Splitting their evaluator's Prior vs
Evaluate counters resolved it: **the residual was an accounting artifact** (requests are E
evaluations + P interior-expansion priors, not 2 x per leaf — P only fires when a node is first
descended *through*, so frontier leaves never pay it). Measured anatomy, matched fresh-net
conditions:

| per state | open_spiel | reinfors |
|---|---|---|
| unique evaluations (E) | **60.2** | **52.6** (reinfors evaluates ~13% FEWER — more in-tree terminal hits, 11.4 vs 3.8 sims/move) |
| interior priors (P) | 18.9 (always cache-refunded; zero edge) | — (no double-dip by construction) |
| genuine cache reuse | 28.5 of E = **47%** (larger than the earlier ~23% estimate — same accounting error) | — |
| effective rows/state | 31.7 | 52.6 uncached -> **32.6 with the new reinfors infer cache** |

**With reinfors' infer cache (engine `infer_cache=` + `weights_updated()` contract, per-batch
weight-sync honesty) the ledger closes:**

| 4-min leg | rows/s | rows/state | states/s |
|---|---|---|---|
| reinfors, no cache, torch 2.13 | 6,681 | 52.6 | 127 |
| reinfors, cache, torch 2.13 | 5,545 | 32.6 | 170 (**1.34× — as priced**) |
| **reinfors, cache, torch 2.3** | 6,110 | 32.5 | **188** |
| open_spiel (cache on, libtorch 2.3) | 5,773 | 31.7 | 182 |

Corrected conclusion: OpenSpiel held **no structural search or systems advantage** — their edge
was one feature (the eval cache, reuse-rate ~47% at fresh-net conditions) plus our kernel-version
skew. With the cache implemented in reinfors core and kernels pinned, **reinfors leads states/s
outright** while keeping its 0.85 head-to-head strength win and tighter weight-sync cadence.
(Conditions: fresh nets, 4-min legs, one seed — the longer round should confirm at trained-net
reuse rates.)

## AlphaZero training comparison — 2h matched-cadence round (2026-07-23, Apple silicon)

The definitive round: 2h/side, one seed, cache ON both (262,144 entries), torch/libtorch 2.3.0,
CPU, matched net (resnet 32×1) + search (64 sims, c=2, ε=0.25/α=0.3, temp-drop 10) + learner
(replay 65,536, batch 1,024, reuse 3, lr 1e-4) knobs, and **matched refresh cadence**: one weight
refresh + cache clear per `replay_buffer_size/reuse = 21,845` collected states on both sides
(their learner's own outer-step pacing; ours via `--collect-size 21845`). Launcher:
`scripts/run_round_matched.sh`. reinfors ran the merged Evaluator build (engine-level cache,
`weights_updated` contract).

| | reinfors | open_spiel C++ AZ |
|---|---|---|
| states/s (2h avg) | **298** | 162 |
| total states | 2,142,558 | 1,158,506 |
| SGD steps (reuse≈3 both) | 6,220 | ~3,400 |
| cache hit rate | 71% | 66% (65.7% overall) |
| net rows/state | 15.2 | — |
| net share of wall | 97% | — |

**Head-to-head (final nets, 64 sims both, solver off, opening sampling): reinfors 43W 0D 7L —
score 0.86** (`results/h2h_120.txt`).

Reading:
- **The cadence-matching answered the quick round's open question.** Round 1 left "reinfors
  refreshes more often" as a candidate explanation for its 0.85 h2h win. With refresh + cache-clear
  cadence structurally equalized, the strength edge is unchanged (0.86) — the edge is not a
  staleness artifact.
- **Cache-clear cadence is a big throughput lever.** Clearing once per 21,845 states (vs per
  512-record stream batch in round 1) lifted reinfors' hit rate from ~47% to **71%** and cut
  rows/state to 15.2 — widening the throughput lead from 1.03× (round 1) to **1.85×**. Their side
  also slowed vs round 1 (162 vs 182 states/s), consistent with full-replay-buffer outer steps
  (~64 SGD passes) competing with actors for CPU over the longer run.
- Cadence was matched **per states** (their pacing rule), which at unequal throughput means
  unequal wall-clock intervals (~73s vs ~136s). This is the structural mirror of their pipeline,
  not wall-clock synchronization.
- Net verdict at matched algorithm + matched cadence + equal wall-clock: reinfors collects 1.85×
  the states, takes 1.8× the SGD steps, and its net wins the head-to-head 0.86. Combined with
  round 1 (fewer states, still 0.85), the equal-wall-clock strength advantage is robust in both
  data regimes.

## Chess encoder measurement — absolute vs mover-relative (2026-07-27, Apple silicon CPU)

Decision run for reinfors' default chess training encoder (`benchmarks/encoders/compare_chess_encoders.py`):
45 min/side AZ self-play, identical net/knobs/seed, `MinimalChess` (absolute) vs `RelativeChess`
(mover's frame, matching action view). Result: **no evidence for relative at this budget — absolute
mildly ahead on every axis.** Records 394k vs 380k (~4% — σ-transform cost); late-half cache hit
0.401 vs 0.352 (the predicted mirror-merging bonus NOT observed — outweighed, plausibly by
per-perspective splitting of tail/eval observations that absolute encoding shares bytewise);
policy loss 7.05→2.45 vs 7.16→2.54; value loss 0.0087 vs 0.0165; policy-head probe 20–20/40.
Caveats: one seed, small net, search-free probe. Decision: `MinimalChess` stays the default;
`RelativeChess` remains available. Revisit only with a longer multi-seed run if equivariance is
suspected to pay at scale.

## Phase 0 — GPU-advantage sweep (prerequisite for the AWS GPU round)

The GPU round (g5.2xlarge: 8 vCPU / 4 physical cores, 1x A10G) only means something at an
operating point where the GPU actually beats CPU on both stacks — otherwise the result is "you
shouldn't have used the GPU", not a framework comparison. Two levers set that point: net size
(width/depth of the shared resnet family) and batch rows per forward (`n_games` here; actors x
inference_batch_size there).

- `benchmarks/openspiel/phase0_gpu_sweep.py` — reinfors side + the stack-independent surface.
  `--mode net` sweeps pure forwards of the width/depth-parameterized resnet over batch x device
  (same ATen kernels both stacks); `--mode engine` sweeps end-to-end AZ self-play over
  n_games x device (realized rows/s, achieved batch, % wall in net). The verdict table prints,
  per net config, the smallest batch / n_games where CUDA clears `--gpu-threshold` (2x) over
  CPU. Pin cores and threads: `taskset -c 0-3 ... --torch-threads 4`.
- `scripts/phase0_openspiel_sweep.sh` — their side: short pinned legs of the C++ AZ across
  (nn_width, nn_depth) x actors x device, raw logs + best-effort moves/s summary. UNVALIDATED
  on CUDA until the instance exists; verify one leg's logs before trusting the parse.

Protocol on the box: one process at a time, pinned to the same physical-core set both stacks
(decide and record whether SMT siblings stay idle), medians over repeated legs, torch pinned to
the libtorch kernel generation of the OpenSpiel build. The head-to-head rounds (connect4
calibration, then chess) run at the operating point Phase 0 selects.

### Phase 0 results — 2026-07-31, g5.2xlarge (4 cores SMT-off, A10G), chess, 64 sims, cache OFF

reinfors: release build, torch 2.3.0+cu121, AZ engine legs (20s+, medians of 3 where noted).
OpenSpiel: MASTER 112b7770 (restored build glue — see docs/openspiel_upstream_notes.md; the
as-shipped pinned era measured 134 rows/s flat on CUDA from the since-fixed staging bug).
Raw legs: results/phase0*/ on the bench box (gitignored here).

Net rows/s, chess depth-8, CUDA (each column = batch: reinfors n_games / their actors):

| stack        | w    | 8     | 32     | 64     | 128    |
|--------------|------|-------|--------|--------|--------|
| reinfors     | 128  | ~4.8k | 10,035 | 16,202 | 22,209 |
| openspiel    | 128  | 4,714 | 14,604 | 22,245 | 29,054 |
| reinfors     | 256  | ~4.6k | 10,096 | 11,293 | 10,790 |
| openspiel    | 256  | 4,916 | 11,556 | 14,613 | 16,358 |

- GPU-advantage gate: PASS both stacks (reinfors ~9-30x, openspiel 20-44x over own CPU).
- At matched batch their fixed stack leads rows/s ~1.15-1.45x: the async actors overlap CPU
  tree work with GPU forwards, while reinfors' lockstep alternates the two — the topology
  advantage the CPU rounds predicted, now measured. Empirical motivation for reinfors'
  shelved collect_async double-buffer design.
- Their useful throughput (states/s) peaks at actors=64 (87.4 w128, 40.9 w256) and FALLS at
  128 actors while rows/s still climbs (thread thrash + redundant evals; note their
  "Collected" counter omits in-flight games, biasing states/s low at high actor counts).
- rows/state differs across stacks (~64 for reinfors at 64 sims; ~250 theirs, cache off) —
  rows/s is the systems metric only; learning throughput is settled by the matched-cadence
  round, cache ON.
- OPERATING POINT for the head-to-head rounds: **w256 d8, batch 64 both (reinfors
  n_games=64, openspiel actors=64), inference cache ON, CUDA both sides** — GPU genuinely
  engaged on both stacks, AZ-realistic net. Secondary/contested regime if hours allow:
  w128 d8. Before the rounds: repeat legs for medians + the learner-contention check
  (SGD shares the A10G with collection).
