# reinfors-benchmarks

The companion benchmark harness for [reinfors](https://github.com/jeepjeepjeep/reinfors):
the OpenSpiel comparison machinery, measurement and round-orchestration scripts, the
head-to-head runner, and the raw artifacts of every published run. It lives outside the
reinfors repo so reinfors takes no benchmark-only dependencies.

**All published results and their interpretation live in the reinfors documentation**
([`docs/benchmarks/`](https://github.com/jeepjeepjeep/reinfors/tree/main/docs/benchmarks)):
environment, methodology, the comparison protocol, tuning, the matched round, and the
internal lever measurements. This repository is the executable and raw-data side of those
pages.

## Layout

| path | contents |
|---|---|
| `benchmarks/internal/` | the internal-family harness: CPU/parallel-scaling sweeps (`benchmark.py`), cross-framework connect4 tracks (`benchmark_vs.py`) |
| `benchmarks/openspiel/` | the trainer (`train_reinfors_az.py`), head-to-head runner (`eval_h2h_chess.py`, Arena protocol) and its tests, parity checks, sweep tooling, shared config (`common.py`) |
| `benchmarks/specs/` | the checked-in V1 campaign: every cell, repeat count and deadline of each experiment family, executed by `benchmarks/runner.py` |
| `benchmarks/grid/` | the topology-grid measurement: `measure_cell.py` (one harness, both engines, unified interior-window telemetry) and `protocol.py` (the matched constants, defined once) |
| `scripts/` | OpenSpiel source-build + patches (`setup_openspiel_cpp.sh`), round orchestration (`run_round_chess_gpu.sh`), campaign driver (`run_v1_campaign.sh`), telemetry panels (`plot_round.py`) |
| `published/` | per-run artifacts of every published number: learner telemetry, configs, logs, PGNs, provenance |
| `docs/history.md` | the full investigation log, including retained corrections and retractions |
| `attic/` | superseded scripts, kept for the historical record |

## Setup

Requires a local reinfors checkout as a sibling directory.

Two environments with distinct roles:

- **`.venv23` — the canonical measurement env.** Built from pinned requirements
  (`requirements-venv23.txt`; torch 2.3.0 is the load-bearing pin — the libtorch
  generation OpenSpiel's C++ AZ links against; never mix kernel generations across the
  stacks). Every published number runs here, and
  `benchmarks/openspiel/preflight.py` refuses measurement runs elsewhere.
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
# the whole frozen campaign, phase by phase (smoke gates first):
TAG=<frozen tag> bash scripts/run_v1_campaign.sh

# or one family:
.venv23/bin/python benchmarks/runner.py benchmarks/specs/v1_grid.json --set tag=<frozen tag>
```

The specs under `benchmarks/specs/` are the reviewable experiment matrix: cells,
repeats, deadlines, pinned cores. Direct shell-harness invocation remains available for
exploration, but nothing produced that way is publishable evidence.

Command templates and interpretation for every measurement are in the reinfors docs
([reproduction pages](https://github.com/jeepjeepjeep/reinfors/tree/main/docs/benchmarks));
on GPU instances, mind the per-boot checklist there (SMT off, release wheel, patched
binaries).

## History

This repository accumulated its protocol the honest way — including two retracted result
classes (a broken-opponent era and a drain-inflated measurement era) that were diagnosed,
corrected, and kept on record. [`docs/history.md`](docs/history.md) preserves that log
intact; nothing in the published tables rests on a retracted number.
