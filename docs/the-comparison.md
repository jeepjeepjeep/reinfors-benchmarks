# The comparison

V1 compares chess AlphaZero training at each stack's best measured configuration. The
network, search budget and learning protocol are matched; implementation architecture is
not artificially equalized.

## Selected configurations

| OpenSpiel | reinfors |
|---|---|
| 256 actors, batch 256 | 512 games, two groups, compiled callback |

These are the maxima of the [configuration sweep](configuring-the-engines.md), selected
before the training comparison was run.

## Matched two-hour training

> V1 result pending publication.

Each side runs three fresh two-hour legs. Values will be medians with repeat spread.

| Metric | OpenSpiel | reinfors |
|---|---:|---:|
| States collected | TBD | TBD |
| Sustained states/s | TBD | TBD |
| 1,024-sample-equivalent learning steps | TBD | TBD |
| Gradient samples per state (target 3.0) | TBD | TBD |

Cache hit rates are recorded but not compared: the two architectures define a lookup
and hit differently.

## Trained-agent head-to-head

> Final V1 experiment in progress.

Each training-cycle pair plays 100 games from seeded random openings, with every opening
played once per color. Both agents use 64 simulations per move and no native chess
solver.

| Metric | Pooled result |
|---|---:|
| Games (opening pairs) | 300 (150) |
| W / D / L, reinfors perspective | TBD |
| Paired score ± SE | TBD |
| Implied Elo difference, 95% CI | TBD |

| Training pair | Games | W / D / L | Score ± SE |
|---|---:|---:|---:|
| Cycle 1 | 100 | TBD | TBD |
| Cycle 2 | 100 | TBD | TBD |
| Cycle 3 | 100 | TBD | TBD |

The paired-opening standard error is the primary uncertainty estimate; the three rows
show robustness across independent training runs.

## Fairness controls

- Identical network architecture, verified by parameter count.
- Identical 64-simulation search budget, including the root-expansion convention.
- Matched exploration, replay size, minibatch size, optimizer settings and legal-action
  policy loss.
- Matched gradient-sample intensity, verified from telemetry rather than inferred from
  scheduler settings.
- Identical wall-clock deadlines and checkpoint-staleness bounds.

Each stack keeps its native batching, cache and training schedule. Those are the systems
being measured, not nuisance variables to remove. The exact constants live in
[`protocol.py`](../experiments/lib/protocol.py); the architectural differences are
summarized in [design differences](design-differences.md).

## Scope

The claim is not that reinfors is generally faster than OpenSpiel. V1 tests whether its
modular Rust/Python design retains comparable throughput on one shared workload and a
single-GPU, low-core-count host. It does not measure other games, networks, core counts
or distributed setups.
