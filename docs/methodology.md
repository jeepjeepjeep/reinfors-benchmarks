# Methodology

## Measurement host

Every V1 number comes from one AWS g5.2xlarge: NVIDIA A10G, four physical CPU cores
(SMT disabled), 32 GiB RAM, Ubuntu 22.04 and gp3 storage at its 125 MB/s baseline.
Experiments run sequentially, pinned to cores 0–3. Both stacks use the same pinned
PyTorch/libtorch generation; OpenSpiel runs with `OMP_NUM_THREADS=1`.

## Rates and deadlines

Timed legs end at a scheduled hard kill. Rates use counter deltas between samples inside
a pre-registered interior window; startup and shutdown work are excluded. Fewer than two
interior samples makes a cell fail rather than fall back to run totals.

**States/s**, not network rows/s, selects configurations. A row is one network input; a
state is one completed training record. AlphaZero value targets require the game outcome,
so unfinished games at the deadline produce no states. Rows/s remains diagnostic.

## Repeats and publication

- Sizing and comparison cells use three interleaved repeats unless marked as single-run.
- Tables report medians and include spread or uncertainty where relevant.
- Every published number names its hardware, run family and reduction method.
- Raw telemetry and finalized manifests must exist under `published/` before a result is
  marked final.

## Experiment map

| Result | Sessions | Repeats |
|---|---|---|
| Device batch curve | `v1_curves`, `v1_curves_ext` | Three-cycle median |
| Engine configuration | `v1_grid`, `v1_grid_ext` | Three-cycle median |
| Compiled callback and eager dtype | `v1_levers` | Three-cycle median |
| Compiled dtype | `v1_levers_f32c` | Three-cycle median |
| Cache capacity | `v1_levers` | Single run per capacity |
| Matched training | `v1_training` | Three independent legs per stack |
| Head-to-head | `v1_h2h` | Three training pairs, 100 games each |

The runner records commands, environment, output hashes and telemetry for every session;
head-to-head manifests also record checkpoint hashes and PGNs. Published artifacts mirror
the run paths under `published/v1/`.

## Fair comparison

The training round matches the network, search budget, learning objective, optimizer,
replay size, minibatch size and gradient-sample intensity. Each stack retains its native
batching, cache and training schedule. See [the comparison](the-comparison.md) for the
result and [`protocol.py`](../experiments/lib/protocol.py) for exact constants.

The runner enforces clean tagged builds, the pinned environment, CPU/GPU preflight,
append-only output directories and output hashes. See
[reproducing the benchmarks](reproducing.md) for commands.
