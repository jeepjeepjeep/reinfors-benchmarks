"""Matched-protocol constants shared by every measurement surface.

These are NOT knobs: the comparison is defined by both sides running the identical
net (width/depth), search budget (sims, c_puct), cache capacity, and learner recipe
(lr, weight decay, buffer, batch, reuse). Anything a spec is allowed to vary lives in
the measurement scripts' CLIs; everything here is fixed by the protocol and defined
exactly once.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_OS_EXAMPLES = REPO / "open_spiel_cpp" / "open_spiel" / "build" / "examples"
OS_TRAIN_BIN = _OS_EXAMPLES / "alpha_zero_torch_example"
OS_PLAY_BIN = _OS_EXAMPLES / "alpha_zero_torch_game_example"

# their binary is single-inference-thread by protocol; stray BLAS threading would
# poach the pinned cores from the actors
OS_CHILD_ENV = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

WIDTH = 256
DEPTH = 8
SIMS = 64
C_PUCT = 2.0
CACHE = (
    262144  # their default capacity, matched on both sides; hit rate is monotone in it
)

LR = 1e-4
WEIGHT_DECAY = 1e-4
BUFFER_SIZE = 65536
TRAIN_BATCH = 1024
REUSE = 3
COLLECT_SIZE = 21845
CHECKPOINT_EVERY = 60

# Interior-window defaults (pre-registered; see docs/benchmarks methodology).
# WARMUP: rf telemetry does not exist before the first learn step, measured at
# 160-224s across n64-n128 (first collect batch of COLLECT_SIZE records must
# complete first); 300 clears that with margin. Re-check before trusting it for
# topologies with a later first collect (n256/n512 are unmeasured).
# WINDOW: `states` advances in ~120s collect-round bursts, so the window must span
# many bursts; 900 covers ~7, keeping the one-partial-burst edge error small.
WARMUP_SECONDS = 300
WINDOW_SECONDS = 900


def rf_train_argv(
    out: Path,
    n_games: int,
    n_groups: int,
    cache: int,
    minutes: float,
    seed: int = 0,
    device: str = "cuda",
    infer: str = "fast",
    pad_rows_to: int = -1,
) -> list[str]:
    return [
        sys.executable,
        str(REPO / "experiments" / "train_az_rf.py"),
        "--minutes",
        str(minutes),
        "--device",
        device,
        "--game",
        "chess",
        "--out",
        str(out),
        "--seed",
        str(seed),
        "--n-games",
        str(n_games),
        "--n-groups",
        str(n_groups),
        "--sims",
        str(SIMS),
        "--c-puct",
        str(C_PUCT),
        "--width",
        str(WIDTH),
        "--depth",
        str(DEPTH),
        "--infer-cache",
        str(cache),
        "--collect-size",
        str(COLLECT_SIZE),
        "--checkpoint-every",
        str(CHECKPOINT_EVERY),
        "--infer",
        infer,
        "--pad-rows-to",
        str(pad_rows_to),
    ]


def os_train_argv(
    out: Path,
    actors: int,
    batch: int,
    cache: int,
    device: str = "/cuda:0",
) -> list[str]:
    return [
        str(OS_TRAIN_BIN),
        "--game=chess",
        f"--path={out}",
        f"--actors={actors}",
        "--evaluators=0",
        f"--devices={device}",
        f"--max_simulations={SIMS}",
        f"--uct_c={C_PUCT:g}",
        "--policy_alpha=0.3",
        "--policy_epsilon=0.25",
        "--temperature=1",
        "--temperature_drop=10",
        "--nn_model=resnet",
        f"--nn_width={WIDTH}",
        f"--nn_depth={DEPTH}",
        f"--inference_batch_size={batch}",
        "--inference_threads=1",
        f"--inference_cache={cache}",
        f"--replay_buffer_size={BUFFER_SIZE}",
        f"--replay_buffer_reuse={REUSE}",
        f"--train_batch_size={TRAIN_BATCH}",
        f"--learning_rate={LR:g}",
        f"--weight_decay={WEIGHT_DECAY:g}",
        "--checkpoint_freq=1",
        "--evaluation_window=100",
        "--eval_levels=7",
        "--cutoff_probability=0",
        "--cutoff_value=0.95",
        "--explicit_learning=false",
        "--max_steps=0",
    ]
