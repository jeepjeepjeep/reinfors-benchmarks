# Methodology

The measurement discipline every family follows. Rules that only exist because two
frameworks are being compared live in [the comparison](the-comparison.md).

## Termination: hard kill, interior windows

Every timed leg ends in a SIGKILL at its scheduled deadline, on both stacks, and rates
reduce from counter deltas between timestamped samples strictly inside a pre-registered
interior window — never from run totals. A leg whose window holds fewer than two
samples **fails** (`no-interior-window`) rather than falling back to totals. The rule
has a scar behind it: an early sweep counted work completed while one stack drained
after its deadline, inflating that side by 18–31%. The warmup bound exists for the
same reason in the other direction — rf telemetry doesn't exist before the first learn
step (~160–224s), and everything before steady state biases the rate downward (the
window parameters and their rationale live in
[`lib/protocol.py`](../experiments/lib/protocol.py)).

## The training-relevant rate is states/s, not rows/s

A *row* is one position forwarded through the net: search effort. A *state* is one
training example delivered to the learner — and in AlphaZero-style training a position
only becomes an example when its game **finishes**, because the value target is the
realized outcome. Under a hard deadline, in-flight games count for nothing.
Configurations can trade the two against each other — more parallel games buy rows/s
while slowing every game's progress and growing the in-flight loss at the kill — so
topology selection and headline comparisons use completed-game states/s at matched
search budget; rows/s is recorded as diagnosis.

One rule binds every table downstream: **no published number appears without a
repeat-derived spread, a repeat-median label, or an explicit single-run label — and
every table states its provenance inline** (hardware, window, cells, run type).
