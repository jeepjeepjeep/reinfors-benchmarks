"""Strength eval of a reinfors-trained checkpoint using OPEN_SPIEL-authored machinery on both
sides: our net drives open_spiel's python `MCTSBot` (PUCT child selection, solver off) against a
vanilla `MCTSBot` + `RandomRolloutEvaluator` referee — removing the our-referee-vs-their-referee
implementation confound from the strength comparison (`eval_reinfors_az.py` remains the
self-contained variant).

Observation bridge: open_spiel's connect_four obs is (3, 6, 7) [player0, player1, empty], row 0 =
bottom — same orientation as reinfors' Connect4Planes; our net's 2 planes are (mover, opponent),
so the bridge is a per-state plane permutation by current player.

  uv run python benchmarks/openspiel/eval_os_referee.py results/rf30_cpu/ckpt_final.pt --games 30
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyspiel
import torch
from open_spiel.python.algorithms import mcts

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AZResnetReplica


class ReinforsNetEvaluator(mcts.Evaluator):
    """open_spiel Evaluator over a reinfors-trained two-headed net (priors + value, one forward)."""

    def __init__(self, net: AZResnetReplica) -> None:
        self.net = net

    def _forward(self, state: pyspiel.State) -> tuple[np.ndarray, float]:
        obs3 = np.array(state.observation_tensor(), dtype=np.float32).reshape(3, 6, 7)
        me = state.current_player()
        obs = np.stack([obs3[me], obs3[1 - me]])  # (mover, opponent) — our Connect4Planes view
        with torch.no_grad():
            logits, v = self.net.heads(torch.from_numpy(obs).unsqueeze(0))
        prior = torch.softmax(logits[0], dim=-1).numpy()
        return prior, float(v.item())

    def evaluate(self, state: pyspiel.State) -> np.ndarray:
        _, v = self._forward(state)
        me = state.current_player()
        return np.array([v, -v]) if me == 0 else np.array([-v, v])

    def prior(self, state: pyspiel.State) -> list[tuple[int, float]]:
        p, _ = self._forward(state)
        legal = state.legal_actions()
        mass = sum(p[a] for a in legal) or 1.0
        return [(a, p[a] / mass) for a in legal]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--uct-c", type=float, default=2.0)
    ap.add_argument("--rollout-count", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    net = AZResnetReplica(in_channels=2)
    net.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    net.eval()

    game = pyspiel.load_game("connect_four")
    rng = np.random.RandomState(args.seed)
    az_bot = mcts.MCTSBot(
        game,
        uct_c=args.uct_c,
        max_simulations=args.sims,
        evaluator=ReinforsNetEvaluator(net),
        solve=False,
        random_state=rng,
        child_selection_fn=mcts.SearchNode.puct_value,
    )
    referee = mcts.MCTSBot(
        game,
        uct_c=args.uct_c,
        max_simulations=args.sims,
        evaluator=mcts.RandomRolloutEvaluator(args.rollout_count, rng),
        solve=False,
        random_state=rng,
    )

    score = 0.0
    for g in range(args.games):
        state = game.new_initial_state()
        az_side = g % 2
        while not state.is_terminal():
            bot = az_bot if state.current_player() == az_side else referee
            state.apply_action(bot.step(state))
        r = state.returns()[az_side]
        score += 1.0 if r > 0 else (0.5 if r == 0 else 0.0)
    n = args.games
    print(
        f"az(reinfors ckpt, os MCTSBot/puct) vs os vanilla-mcts ({args.sims} sims, "
        f"{args.rollout_count} rollouts): score {score / n:.2f} over {n} games"
    )


if __name__ == "__main__":
    main()
