# Attic

Superseded tooling, kept for the historical record (`docs/history.md` explains the eras):

- `run_round_matched.sh` — the CPU-era connect4 matched round; superseded by
  `scripts/run_round_chess_gpu.sh` and the GPU protocol.
- `phase0_openspiel_sweep.sh` — the phase-0 OpenSpiel sweep driver; its numbers were
  drain-inflated (see history) and the corrected measurement path is
  `scripts/measure_states.sh`.
- `eval_h2h.py` — the connect4-era head-to-head bridge; superseded by
  `benchmarks/openspiel/eval_h2h_chess.py` (Arena protocol).
- `bench_openspiel_py.py`, `bench_reinfors.py`, `decompose_sequential.py`,
  `eval_os_referee.py` — CPU-era sequential/Python-path benchmarks and referees behind
  the retracted early conclusions; the current measurement path is
  `scripts/measure_states*.sh` under the hard-kill protocol.
