"""Chess head-to-head: the reinfors-trained net vs the open_spiel-trained net, no referee.

Same bridge shape as eval_h2h.py (connect4): their `alpha_zero_torch_game_example` runs
unmodified with its az bot on their checkpoint and `human` as the other player; we compute our
moves with a PUCT over rf.Env + our checkpoint and feed them to the HumanBot via stdin. All
state sync is single-source: every move (theirs AND the echo of ours) is applied from the
stderr announcements (`Player N chose action: <SAN>`).

Three state mirrors, each doing the one job it is authoritative for:
  rf.Env        our observation/legal-actions/terminal source (OpenSpielChess encoder, the
                trainer's exact composition) — steps by reinfors action ids
  python-chess  SAN <-> UCI conversion (their announcements are SAN; rf ids convert via
                rf.chess_uci_action / rf.chess_action_uci)
  pyspiel       THEIR action-id/string authority: announcements decode via string_to_action,
                and our moves are submitted as numeric ids from the same call — so their two
                dedicated castling ids (vs our in-grid king-slide encoding) need no hardcoded
                mapping. Wheel-vs-source skew is guarded by the echo desync assertion.

Diversity: fixed openings, each played TWICE with colors swapped (paired scoring cancels
opening imbalance to first order). Openings are seeded uniform-random legal lines of
`--opening-plies` plies, generated via the pyspiel mirror (their SAN rendering) and forced on
their side through game_example's positional initial_actions — so BOTH engines play pure
argmax from the exit; neither side is handicapped with exploration moves. Their az bot still
gets a fresh `--seed` per game (tie-breaks only). Scoring authority is their `Returns:` line.

Budget parity: their MCTS counts the root expansion+eval as simulation #1 (mcts.cc
MCTSearch), so our search runs the root eval + sims-1 traversals = sims net evals total.

Smoke test (untrained checkpoints, REQUIRED before the round — validates announcement format,
HumanBot numeric input, castling ids, draw handling):

  uv run python benchmarks/openspiel/eval_h2h_chess.py results/rf_smoke/ckpt_60s.pt \\
      results/os_smoke --os-checkpoint 0 --games 2 --sims 8 --device cuda
"""

import argparse
import json
import math
import random
import re
import subprocess
import sys
from pathlib import Path

import chess as pychess
import numpy as np
import pyspiel
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reinfors as rf  # noqa: E402
from common import SweepResnet  # noqa: E402

BIN = Path(__file__).resolve().parents[2] / "open_spiel_cpp/open_spiel/build/examples/alpha_zero_torch_game_example"
CHOSE = re.compile(r"Player (\d) chose action: (\S+)")
RETURNS = re.compile(r"Returns: (-?[\d.]+),? (-?[\d.]+)")
HEAD_ACTIONS = 4674


class Node:
    __slots__ = ("env", "mover", "legal", "prior", "visits", "value_sum", "children", "terminal_value")

    def __init__(self, env: "rf.Env", terminal_value: float | None = None) -> None:
        self.env = env
        self.terminal_value = terminal_value  # set on done envs, from the PARENT mover's perspective
        if terminal_value is None:
            self.mover = env.active_agents()[0]
            self.legal = env.legal_actions(self.mover)
            self.visits = [0] * len(self.legal)
            self.value_sum = [0.0] * len(self.legal)
            self.children: list[Node | None] = [None] * len(self.legal)
            self.prior: np.ndarray | None = None


def evaluate(node: Node, net: SweepResnet, device: str) -> float:
    """Net priors over the node's legal set + value from the node mover's perspective."""
    obs = node.env.observe(node.mover)
    with torch.inference_mode():
        logits, v = net.heads(torch.from_numpy(obs).unsqueeze(0).to(device))
    legal_logits = logits[0].cpu().numpy()[node.legal]
    e = np.exp(legal_logits - legal_logits.max())
    node.prior = e / e.sum()
    return float(v.item())


def search(root_env: "rf.Env", net: SweepResnet, sims: int, c_puct: float, device: str) -> list[int]:
    """PUCT with net priors/values, no root noise; returns per-legal-action visit counts.
    Mirrors eval_reinfors_az.search but drives rf.Env (fork per expansion, rewards as terminal
    values). Perspective handling is explicit per edge — value flips iff movers differ."""
    root = Node(root_env.fork())
    evaluate(root, net, device)
    # root eval counts against the budget (their MCTS spends simulation #1 expanding the root)
    for _ in range(max(sims - 1, 0)):
        node, path = root, []
        while True:
            if node.terminal_value is not None:
                value, value_mover = node.terminal_value, path[-1][0].mover
                break
            total = sum(node.visits)
            st = math.sqrt(max(total, 1))
            best = max(
                range(len(node.legal)),
                key=lambda i: (node.value_sum[i] / node.visits[i] if node.visits[i] else 0.0)
                + c_puct * node.prior[i] * st / (1 + node.visits[i]),
            )
            path.append((node, best))
            if node.children[best] is None:
                child_env = node.env.fork()
                child_env.step({node.mover: node.legal[best]})
                if child_env.done():
                    rewards = child_env.rewards
                    child = Node(child_env, terminal_value=float(rewards[node.mover]))
                    value, value_mover = child.terminal_value, node.mover
                else:
                    child = Node(child_env)
                    value = evaluate(child, net, device)
                    value_mover = child.mover
                node.children[best] = child
                break
            child = node.children[best]
            node = child
        for parent, action in reversed(path):
            v = value if value_mover == parent.mover else -value
            parent.value_sum[action] += v
            parent.visits[action] += 1
            value, value_mover = v, parent.mover
    return root.visits


def our_move(env: "rf.Env", net: SweepResnet, sims: int, c_puct: float, device: str) -> int:
    """Returns a reinfors action id — pure argmax by visits (diversity comes from the fixed
    openings, not from sampling; sampling here would handicap only our side)."""
    mover = env.active_agents()[0]
    legal = env.legal_actions(mover)
    if len(legal) == 1:
        return legal[0]
    visits = search(env, net, sims, c_puct, device)
    return legal[max(range(len(legal)), key=lambda i: visits[i])]


class Mirror:
    """The three lockstep states plus the SAN/UCI/id conversions between them."""

    def __init__(self) -> None:
        self.env = rf.Env(
            rf.games.Chess(encoder=rf.encoders.OpenSpielChess(), max_ticks=None),
            rf.Reward(win=1.0, loss=-1.0),
        )
        self.board = pychess.Board()
        self.os_state = pyspiel.load_game("chess").new_initial_state()

    def os_action_from_san(self, san: str) -> int:
        """Tolerant SAN -> their action id: exact string_to_action first, then a match over
        legal actions with check/mate suffixes stripped from both sides (python-chess and
        open_spiel disagree on +/# rendering; the wheel and their source build may too)."""
        try:
            return self.os_state.string_to_action(san)
        except (pyspiel.SpielError, RuntimeError, ValueError):
            want = san.rstrip("+#")
            cur = self.os_state.current_player()
            matches = [
                a for a in self.os_state.legal_actions()
                if self.os_state.action_to_string(cur, a).rstrip("+#") == want
            ]
            if len(matches) != 1:
                raise RuntimeError(f"SAN {san!r} matched {len(matches)} of their legal actions")
            return matches[0]

    def apply_san(self, san: str) -> None:
        """Advance all three mirrors by one announced move."""
        self.os_state.apply_action(self.os_action_from_san(san))
        move = self.board.parse_san(san)
        uci = move.uci()
        self.board.push(move)
        mover = self.env.active_agents()[0]
        rf_action = rf.chess_uci_action(uci, self.env.state()["fen"])
        self.env.step({mover: rf_action})

    def their_id(self, rf_action: int) -> int:
        """Our chosen rf action -> their action id (via UCI -> SAN -> their string parser)."""
        uci = rf.chess_action_uci(rf_action, self.env.state()["fen"])
        san = self.board.san(self.board.parse_uci(uci))
        return self.os_action_from_san(san)


def make_opening(plies: int, rng: random.Random) -> list[str]:
    """One uniformly-random legal opening line, SANs in THEIR rendering (generated on the
    pyspiel mirror so the strings exact-match the binary's GetAction). Uniform because
    generating with either net would bias the position distribution toward that net.
    Resamples the whole line if it ends the game early (fool's-mate-class accidents)."""
    while True:
        m = Mirror()
        sans: list[str] = []
        for _ in range(plies):
            acts = m.os_state.legal_actions()
            a = acts[rng.randrange(len(acts))]
            san = m.os_state.action_to_string(m.os_state.current_player(), a)
            sans.append(san)
            m.apply_san(san)
            if m.env.done() or m.os_state.is_terminal():
                break
        else:
            return sans


def play_one(net: SweepResnet, os_path: str, os_ckpt: int, our_player: int, sims: int,
             our_sims: int, uct_c: float, opening_sans: list[str], seed: int, device: str,
             verbose: bool) -> float:
    """Returns our score for one game (1 win / 0.5 draw / 0 loss)."""
    p1, p2 = ("human", "az") if our_player == 0 else ("az", "human")
    cmd = [
        str(BIN), "--game", "chess",
        "--player1", p1, "--player2", p2,
        "--az_path", os_path, "--az_checkpoint", str(os_ckpt),
        "--max_simulations", str(sims), "--uct_c", str(uct_c),
        # solver OFF: their game_example defaults --solve=true (MCTS-Solver bolted onto the az
        # bot) — with it the match is no longer net-vs-net.
        "--solve=false",
        "--num_games", "1", "--quiet=false", "--seed", str(seed),
        # positional args = forced initial actions (their game_example applies them before
        # play and announces them as "forced action" lines, which the loop below ignores)
        *opening_sans,
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, bufsize=1
    )
    assert proc.stdin is not None and proc.stderr is not None
    mirror = Mirror()
    for san in opening_sans:  # replay the forced opening into all three mirrors
        mirror.apply_san(san)
    ply = len(opening_sans)
    pending_id: int | None = None  # their id we submitted, awaiting its echo

    def submit() -> None:
        nonlocal pending_id
        action = our_move(mirror.env, net, our_sims, uct_c, device)
        pending_id = mirror.their_id(action)
        proc.stdin.write(f"{pending_id}\n")
        proc.stdin.flush()

    try:
        if mirror.env.active_agents()[0] == our_player:
            # we move first at the opening exit: their HumanBot blocks on stdin before any
            # announcement
            submit()
        for line in proc.stderr:
            m = CHOSE.search(line)
            if m:
                player, san = int(m.group(1)), m.group(2)
                if player == our_player:
                    echoed = mirror.os_action_from_san(san)
                    if echoed != pending_id:
                        raise RuntimeError(
                            f"desync: submitted their-id {pending_id} but echo announced "
                            f"{san!r} (their-id {echoed}) at ply {ply}"
                        )
                    pending_id = None
                mirror.apply_san(san)
                ply += 1
                if verbose:
                    print(f"    ply {ply:3d} P{player} {san}", flush=True)
                if not mirror.env.done() and mirror.env.active_agents()[0] == our_player:
                    submit()
                continue
            r = RETURNS.search(line)
            if r:
                ours = float(r.group(1 + our_player))
                return 1.0 if ours > 0 else (0.5 if ours == 0 else 0.0)
        raise RuntimeError("game ended without a Returns line")
    finally:
        proc.stdin.close()
        proc.wait(timeout=30)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rf_checkpoint")
    ap.add_argument("os_path")
    ap.add_argument("--os-checkpoint", type=int, required=True)
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--sims", type=int, default=64, help="their az bot's simulations")
    ap.add_argument("--our-sims", type=int, default=None, help="our side's simulations (default: same)")
    ap.add_argument("--uct-c", type=float, default=2.0)
    ap.add_argument("--opening-plies", type=int, default=6,
                    help="forced uniform-random opening length; each opening is played twice")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--width", type=int, default=None, help="default: the checkpoint dir's config.json")
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--verbose", action="store_true", help="print every ply")
    args = ap.parse_args()

    cfg_path = Path(args.rf_checkpoint).parent / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    width = args.width if args.width is not None else cfg.get("width", 32)
    depth = args.depth if args.depth is not None else cfg.get("depth", 1)

    space = rf.games.Chess(encoder=rf.encoders.OpenSpielChess()).observation_space()
    c, h, w = space.shape
    net = SweepResnet(c, h, w, HEAD_ACTIONS, width, depth).to(args.device)
    net.load_state_dict(torch.load(args.rf_checkpoint, map_location=args.device))
    net.eval()
    print(f"our net: SweepResnet w{width} d{depth}  params={sum(p.numel() for p in net.parameters()):,}")

    rng = random.Random(args.seed)
    openings = [make_opening(args.opening_plies, rng) for _ in range((args.games + 1) // 2)]

    score = 0.0
    wins = draws = 0
    for g in range(args.games):
        opening = openings[g // 2]
        s = play_one(
            net, args.os_path, args.os_checkpoint, our_player=g % 2,
            sims=args.sims, our_sims=args.our_sims or args.sims, uct_c=args.uct_c,
            opening_sans=opening, seed=args.seed + g, device=args.device,
            verbose=args.verbose,
        )
        score += s
        wins += s == 1.0
        draws += s == 0.5
        print(f"  game {g + 1:3d}  opening {g // 2 + 1:2d} as P{g % 2}: "
              f"{'W' if s == 1.0 else 'D' if s == 0.5 else 'L'}  [{' '.join(opening)}]", flush=True)
    n = args.games
    print(
        f"chess head-to-head (reinfors net vs open_spiel net, {args.sims} sims both): "
        f"{wins}W {draws}D {n - wins - draws}L -> score {score / n:.2f}"
    )


if __name__ == "__main__":
    main()
