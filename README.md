# reinfors-benchmarks

The benchmark suite for [reinfors](https://github.com/jeepjeepjeep/reinfors): every
measurement script, the campaign specs, the raw artifacts, **and the published results
and their interpretation** — this repository is the single home for all of it (the
[experiment docs](#the-three-questions) carry the figures). It lives outside the
reinfors repo so reinfors takes no benchmark-only dependencies; reinfors' README
carries one headline sentence and links here.

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
at, and C is the headline. That is also the order the campaign runs them. Two
cross-cutting pages govern every number: the [methodology](docs/methodology.md) (hard
kills, interior windows, states/s, the provenance rule) and the
[design differences](docs/design-differences.md) (both architectures, and why the
numbers differ).

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
| `docs/` | the three experiment families ([A](docs/sizing-the-compute.md), [B](docs/configuring-the-engines.md), [C](docs/the-comparison.md)), the shared [methodology](docs/methodology.md) and [design differences](docs/design-differences.md), and notes on the pinned OpenSpiel build |

## Setup

### Measurement host

Every published number comes from one machine, one measurement at a time:
AWS g5.2xlarge — 1× NVIDIA A10G (24 GB), 8 vCPU (4 physical cores, **SMT disabled**),
32 GiB, Ubuntu 22.04, gp3 storage (baseline 125 MB/s — relevant to checkpoint-write
costs). Isolation invariants, enforced by the harnesses and preflight rather than
assumed: SMT off (it silently resets on every instance stop/start), all benchmark
processes pinned to cores 0–3, `OMP_NUM_THREADS=1` for the OpenSpiel side (per its own
docs — the libtorch intra-op pool otherwise competes with its actor threads).

Per boot, before any session: re-disable SMT, pull both repos at the frozen tag,
rebuild the reinfors release wheel (with a fresh `REINFORS_BUILD_NONCE`), re-run
`setup_openspiel_cpp.sh` if any patch changed, and pass
`experiments/lib/preflight.py`.

### Environments

Requires a local reinfors checkout as a sibling directory.

Two environments with distinct roles:

- **`.venv23` — the canonical measurement env.** Built from pinned requirements
  (`requirements-venv23.txt`; torch 2.3.0 is the load-bearing pin — the libtorch
  generation OpenSpiel's C++ AZ links against; never mix kernel generations across the
  stacks — a 2.13-vs-2.3 skew was measured at 1.29–1.4× on this workload, large enough
  to dominate any real difference). Every published number runs here, and
  `experiments/lib/preflight.py` refuses measurement runs elsewhere.
- **`.venv` — the dev/test env** (`pyproject.toml` + `uv.lock`, current torch;
  includes pytest via the dev dependency group). For linting and the test suites only;
  never for measurement. Create it and run the tests with:

  ```bash
  uv sync
  cd ../reinfors && VIRTUAL_ENV=../reinfors-benchmarks/.venv \
    uvx maturin develop --release -m crates/reinfors-py/Cargo.toml && cd -
  .venv/bin/python -m pytest experiments/tests/
  ```

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
them, and later specs depend on decisions those analyses produce. Every spec file is
internally dependency-free (one session fixes all values at launch), so each
between-session analysis is a real update point. Campaign order: smoke gates first,
then `v1_grid` and `v1_curves` (together a decision point: the grid selects each
side's configuration, the curves supply the mechanism cross-check and call-size sweet
spot), then `v1_levers` (lever A/Bs at the confirmed operating point and call size),
`v1_training` (review telemetry before spending H2H hours), `v1_h2h`. Each family's details, decision points and gates are in its
[experiment doc](#the-three-questions); a campaign's session manifests record every
invocation, substitution and resume that produced its evidence.
