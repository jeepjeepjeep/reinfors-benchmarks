"""Head-to-head: the reinfors-trained net vs the open_spiel-trained net, no referee.

Their side runs unmodified: `alpha_zero_torch_game_example` with its az bot on their checkpoint and
`human` as the other player. This bridge tracks the game locally, computes our moves with our
PUCT + checkpoint (the search from eval_reinfors_az), and feeds them to the HumanBot via stdin;
their moves are read from the stderr announcements (`Player N chose action: x3`). Sync is
confirmed by the echo announcement of every submitted move.

Diversity: with two deterministic players every game would be identical, so our side samples its
first `--opening-plies` moves from the visit distribution (seeded per game) and plays argmax
after; their az bot gets a fresh `--seed` per game.

Rules note: our search runs on reinfors rules (full column = legal loss) while their game masks
full columns as illegal; the bridge therefore restricts our chosen move to their legal set (a
trained net never wants a full column anyway).

  uv run python benchmarks/openspiel/eval_h2h.py results/rf30_cpu/ckpt_final.pt results/os30 \\
      --os-checkpoint 15 --games 50
"""

import argparse
import random
import re
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AZResnetReplica  # noqa: E402
from eval_reinfors_az import C4, COLS, search  # noqa: E402

BIN = Path(__file__).resolve().parents[2] / "open_spiel_cpp/open_spiel/build/examples/alpha_zero_torch_game_example"
CHOSE = re.compile(r"Player (\d) chose action: ([xo])(\d)")
RETURNS = re.compile(r"Returns: (-?[\d.]+), (-?[\d.]+)")


def our_move(state: C4, net: AZResnetReplica, sims: int, uct_c: float, ply: int,
             opening_plies: int, rng: random.Random) -> int:
    # PUCT over our checkpoint; sample the opening for game diversity, argmax after. Restrict to
    # their legal set (non-full columns).
    their_legal = [c for c in range(COLS) if state.cells[5 * COLS + c] == 0]
    if len(their_legal) == 1:
        return their_legal[0]
    visits = search(state, sims, uct_c, rng, net, rollouts=1, return_visits=True)
    assert isinstance(visits, list)
    masked = [float(visits[c]) if c in their_legal else 0.0 for c in range(COLS)]
    if ply >= opening_plies:
        return max(range(COLS), key=lambda c: masked[c])
    # opening diversity: sample proportional to visit counts (temperature 1) over their legal set
    total = sum(masked)
    if total <= 0:
        return rng.choice(their_legal)
    r = rng.random() * total
    for c in range(COLS):
        r -= masked[c]
        if masked[c] > 0 and r <= 0:
            return c
    return max(range(COLS), key=lambda c: masked[c])


def play_one(rf_ckpt_net: AZResnetReplica, os_path: str, os_ckpt: int, our_player: int,
             sims: int, uct_c: float, opening_plies: int, seed: int) -> float:
    """Returns our score for one game (1 win / 0.5 draw / 0 loss)."""
    p1, p2 = ("human", "az") if our_player == 0 else ("az", "human")
    cmd = [
        str(BIN), "--game", "connect_four",
        "--player1", p1, "--player2", p2,
        "--az_path", os_path, "--az_checkpoint", str(os_ckpt),
        "--max_simulations", str(sims), "--uct_c", str(uct_c),
        "--num_games", "1", "--quiet=false", "--seed", str(seed),
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, bufsize=1
    )
    assert proc.stdin is not None and proc.stderr is not None
    state = C4()
    rng = random.Random(seed * 7919 + our_player)
    ply = 0
    marks = "xo"
    submitted = False
    try:
        # If we open, submit before reading (their HumanBot blocks on stdin first).
        if state.turn == our_player:
            col = our_move(state, rf_ckpt_net, sims, uct_c, ply, opening_plies, rng)
            proc.stdin.write(f"{marks[our_player]}{col}\n")
            proc.stdin.flush()
            submitted = True
        for line in proc.stderr:
            m = CHOSE.search(line)
            if m:
                player, col = int(m.group(1)), int(m.group(3))
                state.play(col)
                ply += 1
                submitted = False
                if not state.done and state.turn == our_player:
                    col = our_move(state, rf_ckpt_net, sims, uct_c, ply, opening_plies, rng)
                    proc.stdin.write(f"{marks[our_player]}{col}\n")
                    proc.stdin.flush()
                    submitted = True
                continue
            r = RETURNS.search(line)
            if r:
                ours = float(r.group(1 + our_player))
                return 1.0 if ours > 0 else (0.5 if ours == 0 else 0.0)
        raise RuntimeError("game ended without a Returns line")
    finally:
        _ = submitted
        proc.stdin.close()
        proc.wait(timeout=30)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rf_checkpoint")
    ap.add_argument("os_path")
    ap.add_argument("--os-checkpoint", type=int, required=True)
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--uct-c", type=float, default=2.0)
    ap.add_argument("--opening-plies", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    net = AZResnetReplica(in_channels=2)
    net.load_state_dict(torch.load(args.rf_checkpoint, map_location="cpu"))
    net.eval()

    score = 0.0
    wins = draws = 0
    for g in range(args.games):
        s = play_one(
            net, args.os_path, args.os_checkpoint, our_player=g % 2,
            sims=args.sims, uct_c=args.uct_c, opening_plies=args.opening_plies,
            seed=args.seed + g,
        )
        score += s
        wins += s == 1.0
        draws += s == 0.5
        print(f"  game {g + 1:3d}  as P{g % 2}: {'W' if s == 1.0 else 'D' if s == 0.5 else 'L'}", flush=True)
    n = args.games
    print(
        f"head-to-head (reinfors net vs open_spiel net, {args.sims} sims both): "
        f"{wins}W {draws}D {n - wins - draws}L -> score {score / n:.2f}"
    )


if __name__ == "__main__":
    main()
