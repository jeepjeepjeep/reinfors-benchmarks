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
platform, B configures the engines, C races the stacks**:

- **[A — Sizing the compute](docs/sizing-the-compute.md)** — what can the device do
  with this net, independent of any workload or library? The kernel batch curve and
  the CPU/CUDA crossover — the same ATen kernels bound both stacks.
- **[B — Configuring the engines](docs/configuring-the-engines.md)** — how is each
  stack's best configuration determined, and what does each reinfors feature buy?
  Topology grids for both engines (selected by completed-game states/s under the full
  workload), the batch-response curve and grouping model, f32 outputs, cache capacity.
- **[C — The comparison](docs/the-comparison.md)** — how do the stacks compare at the
  configurations B selected, and was the race fair? The matched 2-hour round with its
  fairness verified by telemetry, then head-to-head.

A calibrates B (the kernel curve prices call sizes), B fixes the configurations C races
at, and C is the headline. That is also the order the campaign runs them.

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
| `docs/` | the three experiment families ([A](docs/sizing-the-compute.md), [B](docs/configuring-the-engines.md), [C](docs/the-comparison.md)) and notes on the pinned OpenSpiel build |

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
[experiment doc](#the-three-questions); a campaign's session manifests record every
invocation, substitution and resume that produced its evidence.
