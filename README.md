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
| `benchmarks/internal/` | reinfors-only measurements: inference-path characterization: kernel/engine batch curves incl. the f32 A/B (`measure_inference.py`), CPU/parallel-scaling sweeps (`benchmark.py`), cross-framework connect4 tracks (`benchmark_vs.py`) |
| `benchmarks/h2h/` | the head-to-head strength evaluation: `eval_h2h.py` (Arena protocol, external OpenSpiel engine seat) and its mirror/lifecycle tests |
| `benchmarks/openspiel/` | the trainer (`train_reinfors_az.py`), parity checks, shared net config (`common.py`), manifest + preflight modules |
| `benchmarks/specs/` | the checked-in V1 campaign: every cell, repeat count and deadline of each experiment family, executed by `benchmarks/runner.py` |
| `benchmarks/harness/` | shared measurement runtime: `protocol.py` (the matched constants, defined once) and `run.py` (pinned launch, scheduled kill, crash detection, os-telemetry sampler) |
| `benchmarks/grid/` | the topology-grid measurement: `measure_throughput.py` (training throughput under the full round workload — one harness, both engines, unified interior-window telemetry) |
| `benchmarks/training/` | the matched-cadence training legs: `train.py` (identical wall-clock budget both engines; newest checkpoint recorded in the manifest) |
| `scripts/` | OpenSpiel source-build + patches (`setup_openspiel_cpp.sh`), telemetry panels (`plot_round.py`) |
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
.venv23/bin/python benchmarks/runner.py benchmarks/specs/v1_grid.json --set tag=<frozen tag>
```

The specs under `benchmarks/specs/` are the reviewable experiment matrix: cells,
repeats, deadlines, pinned cores. Direct harness invocation remains available for
exploration, but nothing produced that way is publishable evidence.

The families are separate experiments, not a pipeline — results are analysed between
them and later specs depend on decisions those analyses produce. The campaign order,
with its decision points:

1. **`v1_smoke` / `v1_smoke_h2h`** — cheap end-to-end gates (4-minute legs, a 2-game
   match) worth running after any fresh environment/build assembly. The binary-smoke
   pytest gate runs alongside:
   `H2H_SMOKE_OS_PATH=<os leg out> H2H_SMOKE_OS_CKPT=<n> .venv23/bin/python -m pytest benchmarks/h2h/test_h2h_mirror.py`
2. **`v1_grid`** — the topology grids, both engines. Its analysis *selects each side's
   best configuration*; `v1_training.json`'s topology args encode the currently best
   measured configs (os a16/b16, rf n128/g2) and must be revised here if the grid
   says otherwise.
3. **`v1_training`** — the matched 2h legs at the selected topologies. Review the
   telemetry and surviving checkpoints before spending H2H hours on them.
4. **`v1_h2h`** — strength evaluation of the cycle-k model pairs. The spec points
   both sides at their training-leg directories (`rf-model` / `os-model`, knowable at
   spec-authoring time because session directories are deterministic); the harness
   resolves each engine's native model format inside — `model.pt` for rf, the
   highest-numbered checkpoint for os (their loader takes a directory + number) —
   and records the resolved artifacts and hashes in the match manifest.
5. **`v1_internal`** — reinfors-only curves and probes; independent of the above.

Which runs fed which decisions is recorded per campaign in
[`docs/history.md`](docs/history.md), alongside the analysis between phases.

Command templates and interpretation for every measurement are in the reinfors docs
([reproduction pages](https://github.com/jeepjeepjeep/reinfors/tree/main/docs/benchmarks));
on GPU instances, mind the per-boot checklist there (SMT off, release wheel, patched
binaries).

## History

This repository accumulated its protocol the honest way — including two retracted result
classes (a broken-opponent era and a drain-inflated measurement era) that were diagnosed,
corrected, and kept on record. [`docs/history.md`](docs/history.md) preserves that log
intact; nothing in the published tables rests on a retracted number.
