# C — The comparison

*How do the stacks compare, each at its own best measured configuration — and was the
race fair?*

Three acts, in an order that is itself the argument: selection first, so neither side
races handicapped; a matched round whose fairness is verified by telemetry, not
assumed; then the head-to-head between the models that round produced.

## 1. Operating-point selection — `v1_grid`

Both engines sweep their topology under the **full round workload** (learner,
checkpoint writes and cache sharing the GPU), measured over the pre-registered interior
window and selected by completed-game states/s — the rate that survives a hard
deadline, not raw search speed. OpenSpiel: actors 8/16/32/64 plus the 64:32 decoupling
probe. reinfors: n_games 64/128/256 × 1/2 groups. Three interleaved cycles per cell via
[`measure_throughput.py`](../experiments/measure_throughput.py).

**Decision point:** each side's winner becomes its configuration in
`v1_training.json` — revise the encoded topologies there if the grid disagrees.

## 2. The matched round — `v1_training`

Three fresh 2-hour legs per side at the selected configurations, the wall-clock budget
enforced identically on both by [`train_leg.py`](../experiments/train_leg.py)'s
scheduled kill. Throughput figures reduce post-hoc from the archived telemetry
(`learner.jsonl` — native from the rf trainer, harness-sampled into the same schema for
the OpenSpiel binary).

**Fairness is measured, not assumed:** gradient-samples per state (target 3.0, both
sides) verifies matched training intensity, so scheduling differences cannot be
mistaken for intensity differences; telemetry fields that are *not* comparable across
stacks (the two cache-hit definitions differ) are flagged rather than compared. The
learning parameters themselves are matched constants
([`lib/protocol.py`](../experiments/lib/protocol.py)), never tuned.

Each rf leg publishes its final net as `model.pt`; every leg records its newest
checkpoint in its manifest.

## 3. Head-to-head — `v1_h2h`

The cycle-k model pairs play 100 Arena-protocol games each: seeded uniform-random
openings, each played once per color, pair-level scoring; 64 simulations per move on
both sides; their side runs its own unmodified engine over the bridge
([`eval_h2h.py`](../experiments/eval_h2h.py)). The spec points both sides at their
training-leg directories (`rf-model` / `os-model`); the harness resolves each engine's
native model format (rf: `model.pt`; OpenSpiel: highest-numbered checkpoint, their
loader takes a directory + number) and records the resolved artifacts and hashes in the
match manifest. Every game is exported as PGN.

## Gates

`v1_smoke` (4-minute legs, both sides) and `v1_smoke_h2h` (a 2-game match against the
smoke checkpoints) run first after any fresh environment/build assembly, alongside the
binary-smoke pytest gate:

```bash
H2H_SMOKE_OS_PATH=<os smoke leg dir> H2H_SMOKE_OS_CKPT=<n> \
  .venv23/bin/python -m pytest experiments/tests/test_eval_h2h.py
```

Their outputs are gates, never evidence.
