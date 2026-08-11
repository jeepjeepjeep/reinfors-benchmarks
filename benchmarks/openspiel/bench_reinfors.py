"""reinfors side: connect4 self-play data generation via Engine.collect with a torch callback net.

Runs n_games=1 (sequential, comparable with OpenSpiel's single-game Python bot) and n_games=8
(reinfors' native parallel mode). Search: UCT MCTS, SIMULATIONS per move. Reports leaf evals/s,
moves/s and % wall in the net (from the engine's infer telemetry).

Usage: uv run python benchmarks/openspiel/bench_reinfors.py [--records 4000] [--devices cpu,mps]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import reinfors as rf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SEED, SIMULATIONS, UCT_C, QNet, report, seed_all


def build_engine(n_games: int) -> rf.Engine:
    return rf.Engine(
        rf.games.Connect4(),
        rf.Reward(),  # terminal win/loss is the game's own outcome; no shaping
        rf.policies.Mcts(num_simulations=SIMULATIONS, uct_c=UCT_C),
        rf.learners.TreeStrap(gamma=1.0),
        n_games=n_games,
        seed=SEED,
    )


def make_infer(net: QNet, device: str):
    c = net.in_channels

    def infer(obs_batch: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(obs_batch)).reshape(-1, c, 6, 7).to(device)
            return net(x).cpu().double().numpy()

    return infer


def run(n_games: int, device: str, n_records: int) -> None:
    seed_all()
    net = QNet(in_channels=2).to(device).eval()
    engine = build_engine(n_games)
    infer = make_infer(net, device)
    engine.collect(min(500, n_records), infer)  # warmup
    t0 = time.perf_counter()
    batch = engine.collect(n_records, infer)
    wall = time.perf_counter() - t0
    tel = batch[len(batch) - 1]
    report(
        f"reinfors Mcts+callback torch[{device}] n_games={n_games}",
        wall,
        moves=int(tel["decisions"]),
        evals=int(tel["infer_rows"]),
        net_seconds=float(tel["infer_seconds"]),
        extra=f"  records/s {batch[0].shape[0] / wall:6.0f}",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=int, default=4000)
    ap.add_argument("--devices", type=str, default="cpu,mps")
    ap.add_argument("--n-games", type=str, default="1,8")
    args = ap.parse_args()

    print(f"reinfors {rf.__version__ if hasattr(rf, '__version__') else ''} build={rf._reinfors.core_build_profile()}"
          f"  sims/move={SIMULATIONS} uct_c={UCT_C}")
    for device in args.devices.split(","):
        for n_games in (int(n) for n in args.n_games.split(",")):
            run(n_games, device, args.records)


if __name__ == "__main__":
    main()
