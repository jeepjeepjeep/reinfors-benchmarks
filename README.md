# reinfors-benchmarks

The companion benchmark harness for [reinfors](https://github.com/jeepjeepjeep/reinfors):
every measurement script, the campaign specs, and the raw artifacts behind every
published number. It lives outside the reinfors repo so reinfors takes no
benchmark-only dependencies. **Published results and their interpretation live in the
reinfors documentation**
([`docs/benchmarks/`](https://github.com/jeepjeepjeep/reinfors/tree/main/docs/benchmarks));
this repository is the executable and raw-data side of those pages.

## The three questions

Everything measured here answers one of three questions — **A characterizes the
platform, B prices the features, C races the stacks**:

- **[A — Sizing the compute](docs/sizing-the-compute.md)** — how do we choose batch
  sizes, group sizes, and device, independent of the training workload? Device × net
  response curves: kernel rate, engine rate, the CPU/CUDA crossover.
- **[B — Configuring the engine](docs/configuring-the-engine.md)** — which throughput
  features should be on, and what does each buy at the real workload? f32 outputs,
  inference-cache capacity, grouped collection — measured effects plus the model
  predicting where each transfers.
- **[C — The comparison](docs/the-comparison.md)** — how do the stacks compare, each
  at its own best measured configuration, and was the race fair? Operating-point
  selection grids → the matched 2-hour round with its fairness verified by telemetry →
  head-to-head.

A calibrates B (the batch curve prices grouping), A + B explain C's mechanisms, and C
is the headline. That is also the order the campaign runs them.

## Layout

| path | contents |
|---|---|
| `experiments/` | one script per experiment surface: `measure_throughput.py` (training throughput under the full round workload, both engines), `measure_inference.py` (kernel/engine curves, f32 A/B, device crossover), `train_leg.py` + `train_az_rf.py` (matched-cadence legs: harness + the rf workload), `eval_h2h.py` (Arena-protocol head-to-head), `runner.py` (the orchestrator) |
| `experiments/lib/` | shared, non-executable: `protocol.py` (matched constants), `run.py` (child runtime), `manifest.py` + `preflight.py` (evidence + freeze gate), `common.py` (the benchmark net) |
| `experiments/specs/` | the checked-in V1 campaign: every cell, repeat count and deadline of each experiment family, executed by `experiments/runner.py` |
| `experiments/tests/` | the test suite, one file per surface |
| `scripts/` | OpenSpiel source-build + patches (`setup_openspiel_cpp.sh`), telemetry panels (`plot_round.py`) |
| `runs/` | untracked, append-only: every campaign session's evidence (`<session>/<cell>/cycleN/`), box-synced verbatim after a campaign |
| `published/` | tracked artifacts behind every published number, one directory per campaign as a filtered mirror of `runs/` (same paths, model binaries stripped to the GitHub release); pre-campaign-era artifacts live in the maintainers' archive (and git history) |
| `docs/` | the three experiment families ([A](docs/sizing-the-compute.md), [B](docs/configuring-the-engine.md), [C](docs/the-comparison.md)) and the investigation log ([history](docs/history.md)) |
| `attic/` | superseded scripts (untracked, local-only; git history remains the record) |
| `archive/` | untracked, frozen raw record of the pre-V1 era: `pre-v1-box/` (box telemetry behind current published figures), `pre-v1-local/` (exploratory local runs) |

## Setup

Requires a local reinfors checkout as a sibling directory.

Two environments with distinct roles:

- **`.venv23` — the canonical measurement env.** Built from pinned requirements
  (`requirements-venv23.txt`; torch 2.3.0 is the load-bearing pin — the libtorch
  generation OpenSpiel's C++ AZ links against; never mix kernel generations across the
  stacks). Every published number runs here, and
  `experiments/lib/preflight.py` refuses measurement runs elsewhere.
- **`.venv` — the dev/test env** (`pyproject.toml` + `uv.lock`, current torch). For
  linting and the harness test suites only; never for measurement.

```bash
# canonical measurement env from pinned requirements
bash scripts/make_venv23.sh

# reinfors release wheel into it
cd ../reinfors
VIRTUAL_ENV=../reinfors-benchmarks/.venv23 uvx maturin develop --release -m crates/reinfors-py/Cargo.toml
cd ../reinfors-benchmarks

# OpenSpiel from source with CUDA libtorch (restores build glue, applies documented patches)
bash scripts/setup_openspiel_cpp.sh
```

## Running measurements

Publication runs go through the experiment runner — never by invoking the harnesses
directly. The runner enforces the freeze preflight (tagged clean builds on both repos,
pinned torch, SMT off, GPU visible), launches every subprocess itself with the exact
argv and environment captured, keeps run directories append-only, and finalizes a
completion manifest (status, exit code, output hashes) for every cell:

```bash
.venv23/bin/python experiments/runner.py experiments/specs/v1_grid.json --set tag=<frozen tag>
```

The specs under `experiments/specs/` are the reviewable experiment matrix: cells,
repeats, deadlines, pinned cores. Direct harness invocation remains available for
exploration, but nothing produced that way is publishable evidence.

The families are separate experiments, not a pipeline — results are analysed between
them, and later specs depend on decisions those analyses produce. Campaign order:
smoke gates first, then `v1_grid` (selects each side's configuration — a decision
point), `v1_training` (review telemetry before spending H2H hours), `v1_h2h`, with
`v1_internal` independent. Each family's details, decision points and gates are in its
[experiment doc](#the-three-questions); which runs fed which decisions is recorded per
campaign in [`docs/history.md`](docs/history.md).

## History

This repository accumulated its protocol the honest way — including two retracted result
classes (a broken-opponent era and a drain-inflated measurement era) that were diagnosed,
corrected, and kept on record. [`docs/history.md`](docs/history.md) preserves that log
intact; nothing in the published tables rests on a retracted number.
