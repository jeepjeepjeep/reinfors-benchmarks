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

(to be filled in from runs; keep the raw command + machine + date next to each table)
