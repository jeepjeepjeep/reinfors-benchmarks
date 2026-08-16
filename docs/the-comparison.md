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

Three fresh two-hour legs per side. Values are medians over the three legs, with the
per-leg range in brackets; rates use the standard event-aligned reduction over the
post-warmup interval [300 s, 7,200 s].

| Metric | OpenSpiel | reinfors |
|---|---:|---:|
| States collected | 1,845,946 [1,844,431–1,868,467] | 2,026,861 [2,007,938–2,028,903] |
| Sustained states/s | 263.9 [263.2–269.4] | **290.1** [288.5–291.4] |
| 1,024-sample-equivalent learning steps | 5,311 | 5,938 |
| Gradient samples per state (target 3.0) | 2.95 | 3.00 |

reinfors sustains **9.9% higher states/s** and collects **9.8% more training data** in
the same wall-clock budget. Gradient-sample intensity is matched, so the extra data is
the only systematic difference carried into the head-to-head.

## Trained-agent head-to-head

Each training-cycle pair plays 100 games from seeded random openings, with every opening
played once per color. Both agents use 64 simulations per move and no native chess
solver.

| Metric | Pooled result |
|---|---:|
| Games (opening pairs) | 300 (150) |
| W / D / L, reinfors perspective | **104 / 155 / 41** |
| Paired score ± SE | **0.605 ± 0.020** |
| Implied Elo difference, 95% CI | **+74** (+46 to +103) |

| Training pair | Games | W / D / L | Score ± SE |
|---|---:|---:|---:|
| Cycle 1 | 100 | 42 / 45 / 13 | 0.645 ± 0.037 |
| Cycle 2 | 100 | 28 / 55 / 17 | 0.555 ± 0.035 |
| Cycle 3 | 100 | 34 / 55 / 11 | 0.615 ± 0.031 |

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
