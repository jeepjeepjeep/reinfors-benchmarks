"""OpenSpiel Python side: connect4 self-play with open_spiel's Python MCTSBot driving a torch
prior+value evaluator (AZ-style, batch-1 leaf evals — how the OpenSpiel Python path works).

Same search budget as the reinfors side (SIMULATIONS per move, same uct_c). Reports leaf evals/s,
moves/s and % wall in the net (measured inside the evaluator).

Usage: uv run python benchmarks/openspiel/bench_openspiel_py.py [--moves 200] [--devices cpu,mps]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyspiel
import torch
from open_spiel.python.algorithms import mcts

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    SEED,
    SIMULATIONS,
    UCT_C,
    PriorValueNet,
    report,
    seed_all,
)


class TorchEvaluator(mcts.Evaluator):
    """AZ-style evaluator over a torch net; counts calls + time for the report."""

    def __init__(self, game, net: PriorValueNet, device: str) -> None:
        self.net = net
        self.device = device
        self.shape = game.observation_tensor_shape()
        self.net_seconds = 0.0
        self.net_calls = 0
        self._cache: dict[str, tuple] = {}

    def _forward(self, state):
        # prior() and evaluate() hit the same leaf back-to-back; cache so that costs ONE forward,
        # mirroring open_spiel's own AlphaZeroEvaluator LRU cache.
        key = str(state)
        if key in self._cache:
            return self._cache[key]
        t0 = time.perf_counter()
        obs = torch.tensor(state.observation_tensor(), dtype=torch.float32)
        x = obs.reshape(1, *self.shape).to(self.device)
        with torch.no_grad():
            prior_logits, value = self.net(x)
        prior = torch.softmax(prior_logits, dim=-1)[0].cpu().numpy()
        v = float(value.item())
        self.net_seconds += time.perf_counter() - t0
        self.net_calls += 1
        if len(self._cache) > 60000:
            self._cache.clear()
        self._cache[key] = (prior, v)
        return prior, v

    def evaluate(self, state):
        _, v = self._forward(state)
        # value from the perspective of the current player -> per-player returns
        player = state.current_player()
        return np.array([v, -v]) if player == 0 else np.array([-v, v])

    def prior(self, state):
        p, _ = self._forward(state)
        legal = state.legal_actions()
        mass = sum(p[a] for a in legal) or 1.0
        return [(a, p[a] / mass) for a in legal]


def run(device: str, n_moves: int) -> None:
    seed_all()
    game = pyspiel.load_game("connect_four")
    net = PriorValueNet(in_channels=game.observation_tensor_shape()[0]).to(device).eval()
    evaluator = TorchEvaluator(game, net, device)
    bot = mcts.MCTSBot(
        game,
        uct_c=UCT_C,
        max_simulations=SIMULATIONS,
        evaluator=evaluator,
        solve=False,  # pure MCTS; alpha-beta solving would be a search-quality mismatch
        random_state=np.random.RandomState(SEED),
    )

    # warmup (JIT/lazy init), then reset counters
    state = game.new_initial_state()
    bot.step(state)
    evaluator.net_seconds = 0.0
    evaluator.net_calls = 0

    moves = 0
    t0 = time.perf_counter()
    state = game.new_initial_state()
    while moves < n_moves:
        if state.is_terminal():
            state = game.new_initial_state()
            continue
        action = bot.step(state)
        state.apply_action(action)
        moves += 1
    wall = time.perf_counter() - t0
    report(
        f"open_spiel py MCTSBot+torch[{device}]",
        wall,
        moves=moves,
        evals=evaluator.net_calls,
        net_seconds=evaluator.net_seconds,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--moves", type=int, default=200)
    ap.add_argument("--devices", type=str, default="cpu,mps")
    args = ap.parse_args()

    print(f"open_spiel {pyspiel.__version__ if hasattr(pyspiel, '__version__') else ''}"
          f"  sims/move={SIMULATIONS} uct_c={UCT_C}")
    for device in args.devices.split(","):
        run(device, args.moves)


if __name__ == "__main__":
    main()
