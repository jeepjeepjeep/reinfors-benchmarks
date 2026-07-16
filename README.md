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

```bash
# 1. build the reinfors wheel (from the reinfors repo, using ITS venv's maturin — not uv;
#    `maturin build` writes only to the -o dir and never touches the source tree's .so)
cd ../reinfors
.venv/bin/maturin build --release --features nn-metal,nn-accelerate -o ../reinfors-benchmarks/wheels

# 2. create this repo's env and install everything
cd ../reinfors-benchmarks
uv sync
uv pip install wheels/reinfors-*.whl
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
