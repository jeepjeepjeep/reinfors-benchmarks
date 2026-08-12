"""Offline strength eval for reinfors-trained AZ checkpoints: PUCT(net) vs a vanilla-MCTS referee,
mirroring open_spiel's alpha_zero_torch_game_example (az vs mcts, --solve=false):

  referee   UCT (uct_c matched), `rollout_count` uniform-random rollouts per leaf, no solver
  az side   PUCT over the checkpointed AZResnetReplica — no root noise, acts by visit count
  game      pure-python connect4 mirroring reinfors' rules (all 7 columns always playable; a move
            into a full column is an immediate loss) and Connect4Planes obs (own/opp planes, row 0
            = bottom). NOTE the one referee-rule divergence from their side: open_spiel masks full
            columns as illegal. A trained net never plays one, so the practical effect is nil —
            documented, not hidden.

  uv run python benchmarks/openspiel/eval_reinfors_az.py results/rf_smoke/ckpt_final.pt --games 20
"""

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AZResnetReplica

ROWS, COLS = 6, 7


class C4:
    """Connect4, reinfors rules: cells[row*7+col], row 0 = bottom; full-column move = mover loses."""

    __slots__ = ("cells", "done", "turn", "winner")

    def __init__(self) -> None:
        self.cells = [0] * (ROWS * COLS)
        self.turn = 0  # mover: 0 or 1 (stored as 1 / 2 in cells)
        self.winner: int | None = None  # None while running; -1 for draw
        self.done = False

    def clone(self) -> "C4":
        c = C4.__new__(C4)
        c.cells = self.cells[:]
        c.turn = self.turn
        c.winner = self.winner
        c.done = self.done
        return c

    def play(self, col: int) -> None:
        me = self.turn + 1
        row = next((r for r in range(ROWS) if self.cells[r * COLS + col] == 0), None)
        if row is None:  # full column: immediate loss for the mover (reinfors rule)
            self.done, self.winner = True, 1 - self.turn
            return
        self.cells[row * COLS + col] = me
        if self._wins(row, col, me):
            self.done, self.winner = True, self.turn
        elif all(self.cells[(ROWS - 1) * COLS + c] != 0 for c in range(COLS)):
            self.done, self.winner = True, -1
        else:
            self.turn = 1 - self.turn

    def _wins(self, row: int, col: int, me: int) -> bool:
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            n = 1
            for s in (1, -1):
                r, c = row + s * dr, col + s * dc
                while (
                    0 <= r < ROWS and 0 <= c < COLS and self.cells[r * COLS + c] == me
                ):
                    n += 1
                    r, c = r + s * dr, c + s * dc
            if n >= 4:
                return True
        return False

    def obs(self) -> np.ndarray:
        """Connect4Planes: plane 0 = mover's pieces, plane 1 = opponent's."""
        mine = self.turn + 1
        out = np.zeros((2, ROWS, COLS), dtype=np.float32)
        for i, v in enumerate(self.cells):
            if v == mine:
                out[0, i // COLS, i % COLS] = 1.0
            elif v != 0:
                out[1, i // COLS, i % COLS] = 1.0
        return out


class Node:
    __slots__ = ("children", "prior", "state", "value_sum", "visits")

    def __init__(self, state: C4, prior: np.ndarray | None = None) -> None:
        self.state = state
        self.visits = [0] * COLS
        self.value_sum = [0.0] * COLS
        self.children: list[Node | None] = [None] * COLS
        self.prior = prior


def rollout(state: C4, rng: random.Random) -> float:
    """Uniform-random playout; value from the perspective of `state.turn`'s mover."""
    me = state.turn
    s = state.clone()
    while not s.done:
        s.play(rng.randrange(COLS))
    if s.winner == -1:
        return 0.0
    return 1.0 if s.winner == me else -1.0


def search(
    root_state: C4,
    sims: int,
    uct_c: float,
    rng: random.Random,
    net: AZResnetReplica | None,
    rollouts: int,
    return_visits: bool = False,
) -> int | list[int]:
    """One move's search. net=None -> vanilla UCT with `rollouts` random playouts per leaf
    (their RandomRolloutEvaluator); net set -> PUCT with net priors + value (their az bot)."""

    def evaluate(node: Node) -> float:
        if net is None:
            return sum(rollout(node.state, rng) for _ in range(rollouts)) / rollouts
        with torch.no_grad():
            logits, v = net.heads(torch.from_numpy(node.state.obs()).unsqueeze(0))
        node.prior = torch.softmax(logits[0], dim=-1).numpy()
        return float(v.item())

    root = Node(root_state.clone())
    if net is not None:
        evaluate(root)
    # matched budget: their MCTS spends simulation #1 expanding+evaluating the root (mcts.cc
    # MCTSearch), so with a net the separate root evaluate() above consumes one sim here too
    for _ in range(max(sims - 1, 0) if net is not None else sims):
        node, path = root, []
        # select
        while True:
            if node.state.done:
                # C4.play does NOT flip the turn on a terminal move, so at a terminal `turn` is the
                # mover who ended the game: winner == turn for a win, != turn for a full-column loss.
                w = node.state.winner
                value = 0.0 if w == -1 else (1.0 if w == node.state.turn else -1.0)
                break
            total = sum(node.visits)
            if net is None:
                scores = [
                    (
                        node.value_sum[a] / node.visits[a]
                        + uct_c * math.sqrt(math.log(max(total, 1)) / node.visits[a])
                    )
                    if node.visits[a] > 0
                    else math.inf
                    for a in range(COLS)
                ]
            else:
                st = math.sqrt(max(total, 1))
                scores = [
                    (node.value_sum[a] / node.visits[a] if node.visits[a] else 0.0)
                    + uct_c * node.prior[a] * st / (1 + node.visits[a])
                    for a in range(COLS)
                ]
            a = max(range(COLS), key=lambda i: scores[i])
            path.append((node, a))
            if node.children[a] is None:
                child_state = node.state.clone()
                child_state.play(a)
                child = Node(child_state)
                node.children[a] = child
                if child_state.done:
                    w = child_state.winner
                    value = 0.0 if w == -1 else (1.0 if w == child_state.turn else -1.0)
                else:
                    value = evaluate(child)
                node = child
                break
            node = node.children[a]
        # backprop (negamax: value is from the reached node's mover perspective)
        for parent, action in reversed(path):
            child = parent.children[action]
            v = value if child.state.turn == parent.state.turn else -value
            parent.value_sum[action] += v
            parent.visits[action] += 1
            value = v
    if return_visits:
        return root.visits
    return max(range(COLS), key=lambda a: root.visits[a])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--uct-c", type=float, default=2.0)
    ap.add_argument("--rollout-count", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    net = AZResnetReplica(in_channels=2)
    net.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    net.eval()
    rng = random.Random(args.seed)

    wins = draws = 0
    for g in range(args.games):
        state = C4()
        az_side = g % 2
        while not state.done:
            if state.turn == az_side:
                move = search(
                    state, args.sims, args.uct_c, rng, net, args.rollout_count
                )
            else:
                move = search(
                    state, args.sims, args.uct_c, rng, None, args.rollout_count
                )
            state.play(move)
        if state.winner == az_side:
            wins += 1
        elif state.winner == -1:
            draws += 1
    n = args.games
    print(
        f"az vs vanilla-mcts ({args.sims} sims, {args.rollout_count} rollouts): "
        f"{wins}W {draws}D {n - wins - draws}L  ->  score {(wins + 0.5 * draws) / n:.2f}"
    )


if __name__ == "__main__":
    main()
