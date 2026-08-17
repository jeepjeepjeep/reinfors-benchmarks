# Reproducing the benchmarks

This page is for contributors reproducing or extending the campaign. Reading the
[results](the-comparison.md) does not require this setup.

## Requirements

- Ubuntu 22.04 on AWS g5.2xlarge for a like-for-like V1 run. Budget roughly 48 hours of
  sequential measurement for the full campaign.
- A sibling checkout of `reinfors`.
- `uv`, a Rust toolchain, CUDA and the build dependencies used by OpenSpiel.
- SMT disabled and benchmark processes pinned to physical cores 0–3.

The repository uses two environments:

| Environment | Purpose |
|---|---|
| `.venv23` | Canonical measurements. PyTorch 2.3 is pinned to OpenSpiel's libtorch generation. |
| `.venv` | Development, linting and tests. Never used for published measurements. |

Do not substitute the development PyTorch build for `.venv23`: the resulting kernel
skew was measured at 1.29–1.4× on this workload.

## Create the environments

```bash
uv sync

# Build reinfors into the development environment.
cd ../reinfors
VIRTUAL_ENV=../reinfors-benchmarks/.venv \
  uvx maturin develop --release -m crates/reinfors-py/Cargo.toml
cd ../reinfors-benchmarks

# Canonical measurement environment and reinfors wheel.
bash scripts/make_venv23.sh
cd ../reinfors
VIRTUAL_ENV=../reinfors-benchmarks/.venv23 \
  uvx maturin develop --release -m crates/reinfors-py/Cargo.toml
cd ../reinfors-benchmarks

# OpenSpiel with the recorded CUDA/libtorch build patches.
bash scripts/setup_openspiel_cpp.sh
```

Run the development tests with:

```bash
.venv/bin/python -m pytest experiments/tests/
```

Before each campaign boot, disable SMT again, check out both frozen tags, rebuild the
reinfors wheel with a fresh `REINFORS_BUILD_NONCE`, and rerun the OpenSpiel setup if its
patches changed. The runner's preflight rejects a mismatched environment.

## Run a campaign session

Publication runs always go through the runner:

```bash
.venv23/bin/python experiments/runner.py \
  experiments/specs/v1_grid.json --set tag=bench_v1.0.6
```

Campaign tags carry a `bench_` prefix; manifests record them without it, so a manifest's
`v1.0.6` corresponds to the git tag `bench_v1.0.6`.

The runner verifies the frozen environment, captures each command and environment,
keeps run directories append-only, and hashes required outputs. Direct harness runs are
for exploration only.

V1 session order:

1. `v1_smoke` and `v1_smoke_h2h`
2. `v1_grid`, `v1_grid_ext`, `v1_curves`, `v1_curves_ext`
3. `v1_levers` and `v1_levers_f32c`
4. `v1_training`
5. `v1_h2h`

Later sessions encode decisions from earlier results, so the campaign is intentionally
not one automatic pipeline.

Regenerate documentation figures after changing their source tables:

```bash
.venv/bin/python scripts/plot_docs.py
```

## H2H smoke gate

After rebuilding either stack, verify the bridge against smoke checkpoints:

```bash
H2H_SMOKE_OS_PATH=<os-smoke-leg> H2H_SMOKE_OS_CKPT=<number> \
  .venv23/bin/python -m pytest experiments/tests/test_eval_h2h.py
```

Smoke outputs are gates, never published evidence.

## Artifacts

`runs/<session>/<cell>/cycleN/` is the append-only private tree. Published campaigns
mirror those paths under `published/`, excluding large model and replay files. Every
published result must be reproducible from the tracked manifest and telemetry at that
path.

OpenSpiel-specific build decisions and patch history are documented in
[the upstream notes](openspiel_upstream_notes.md).
