"""Chess head-to-head: the reinfors-trained net vs the open_spiel-trained net, no referee.

PROTOCOL v1 (Arena): our side plays
through rf.Arena + PolicyHandle.choose (the real Rust search, leaves batched across all
concurrent games into one GPU call), and their unmodified `alpha_zero_torch_game_example`
runs as an Arena External seat, one process per game on bounded worker lanes.

Their process contract (unchanged from the original bridge): openings and any of our moves
made before their first turn are passed as positional forced actions at spawn — TheirBot
buffers every observed move and spawns lazily at its first act(), so their az bot plays
pure argmax from the exit. After spawn, our moves are submitted to their HumanBot via
stdin and every move (theirs AND the echo of ours) is verified against their stderr
announcements; the echo-desync assertion guards wheel-vs-source action-id skew for
INTERACTIVE moves. Forced positional openings are validated by the binary itself before
any echo exists — test_h2h_mirror.py's binary smoke covers those renderings. Their
`Returns:` line is cross-checked against the rf.Env outcome at game end.

Player-index conventions DIFFER between the stacks: open_spiel chess maps BLACK to
player 0 and WHITE to player 1 (chess.h ColorToPlayer); reinfors maps WHITE to agent 0.

Openings come from rf.starts.RandomStartingMoves (seeded uniform legal lines, each played
once per color — Arena's paired seat-swap), and scoring is Arena's pair-level payoff.
Every run gets a fresh --out directory holding games.pgn and a manifest.json with the
full lifecycle (checkpoints + hashes, sims, seeds, concurrency, environment; finalized
with the result).

Smoke test (untrained checkpoints, REQUIRED before any round — validates announcement
format, HumanBot numeric input, castling ids, draw handling, lazy spawn):

  uv run python benchmarks/eval_h2h.py \
      --rf-model runs/v1_smoke/rf_train_smoke/cycle1/training \
      --os-model runs/v1_smoke/os_train_smoke/cycle1/training \
      --games 2 --sims 8 --device cuda --az-device /cuda:0 --out /tmp/h2h_smoke
"""

import argparse
import hashlib
import itertools
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import chess as pychess
import chess.pgn as pychess_pgn
import pyspiel
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
import reinfors as rf
import manifest
import protocol
from common import SweepResnet

BIN = protocol.OS_PLAY_BIN
CHOSE = re.compile(r"Player (\d) chose action: (\S+)")
RETURNS = re.compile(r"Returns: (-?[\d.]+),? (-?[\d.]+)")
HEAD_ACTIONS = 4674

# Every spawned engine registers here; a main-thread finally kills whatever a failed run
# leaves behind (a lane blocked in stderr can never run close() for it).
_LIVE_PROCS: set[subprocess.Popen] = set()
_LIVE_LOCK = threading.Lock()


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _reap(proc: subprocess.Popen) -> bool:
    """Terminate, escalate to kill, and wait; True only on confirmed exit."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=5)
        return True
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=5)
        return True
    except subprocess.TimeoutExpired:
        return proc.poll() is not None


def kill_leftover_processes() -> None:
    with _LIVE_LOCK:
        procs = list(_LIVE_PROCS)
        _LIVE_PROCS.clear()
    for proc in procs:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            continue
    deadline = time.time() + 5
    for proc in procs:
        try:
            proc.wait(timeout=max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass


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
                a
                for a in self.os_state.legal_actions()
                if self.os_state.action_to_string(cur, a).rstrip("+#") == want
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"SAN {san!r} matched {len(matches)} of their legal actions"
                )
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


def write_pgn(
    board: pychess.Board,
    our_white: bool,
    score: float,
    game_no: int,
    opening_no: int,
    forced_plies: int,
    run_meta: dict[str, str],
    path: str,
) -> None:
    """Append the finished game (full move stack incl. the forced opening) as PGN. The default
    file accumulates runs, so every game carries the run metadata that distinguishes
    experiments (checkpoints, sims, seed) plus its own opening index and forced-plies count."""
    game = pychess_pgn.Game.from_board(board)
    game.headers["Event"] = "reinfors vs open_spiel h2h"
    game.headers["Round"] = str(game_no)
    game.headers["White"] = "reinfors" if our_white else "open_spiel"
    game.headers["Black"] = "open_spiel" if our_white else "reinfors"
    white_score = score if our_white else 1.0 - score
    game.headers["Result"] = {1.0: "1-0", 0.5: "1/2-1/2", 0.0: "0-1"}[white_score]
    game.headers["Opening"] = f"index {opening_no}, forced {forced_plies} plies"
    for k, v in run_meta.items():
        game.headers[k] = v
    with open(path, "a") as f:
        f.write(str(game) + "\n\n")


class TheirBot:
    """One game of their engine behind Arena's External seat.

    Lifecycle: on_action() calls arrive for every executed move in the game (the opening
    replay first). Before the process exists they are buffered as forced-action SANs; the
    process spawns at the first act() with that buffer as positional args. Afterwards our
    moves are submitted via stdin and their echo consumed; their own moves are parsed from
    their announcement in act() and applied to the mirror when Arena echoes them back.
    All blocking pipe I/O runs on the Arena worker lane, never the scheduler."""

    def __init__(
        self,
        cfg: argparse.Namespace,
        game_seed: int,
        done_counter: itertools.count,
        total: int,
    ) -> None:
        self.cfg = cfg
        self.game_seed = game_seed
        self.done_counter = done_counter
        self.total = total
        self.mirror = Mirror()
        self.forced_sans: list[str] = []
        self.proc: subprocess.Popen | None = None
        self.their_us: int | None = None  # their index: white=1, black=0
        self.expect_own: int | None = (
            None  # rf id of our returned move, awaiting Arena echo
        )
        self.returns_seen = False

    # conversions against the PRE-move mirror state -----------------------------

    def _san_of(self, rf_action: int) -> str:
        uci = rf.chess_action_uci(rf_action, self.mirror.env.state()["fen"])
        return self.mirror.board.san(self.mirror.board.parse_uci(uci))

    def _rf_of_san(self, san: str) -> int:
        move = self.mirror.board.parse_san(san)
        return rf.chess_uci_action(move.uci(), self.mirror.env.state()["fen"])

    # process -------------------------------------------------------------------

    def _spawn(self) -> None:
        mover = self.mirror.env.active_agents()[
            0
        ]  # rf: 0 = white — it is their turn now
        their_white = mover == 0
        self.their_us = 1 if their_white else 0
        p1, p2 = ("az", "human") if not their_white else ("human", "az")
        cmd = [
            str(BIN),
            "--game",
            "chess",
            "--player1",
            p1,
            "--player2",
            p2,
            "--az_path",
            self.cfg.os_path,
            "--az_checkpoint",
            str(self.cfg.os_checkpoint),
            "--max_simulations",
            str(self.cfg.sims),
            "--uct_c",
            str(self.cfg.uct_c),
            # solver OFF: their game_example defaults --solve=true (MCTS-Solver bolted onto
            # the az bot) — with it the match is no longer net-vs-net.
            "--solve=false",
            "--num_games",
            "1",
            "--quiet=false",
            "--seed",
            str(self.game_seed),
            "--az_device",
            self.cfg.az_device,
            *self.forced_sans,
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        with _LIVE_LOCK:
            _LIVE_PROCS.add(self.proc)

    def _read_announcement(self) -> tuple[int, str]:
        assert self.proc is not None and self.proc.stderr is not None
        for line in self.proc.stderr:
            m = CHOSE.search(line)
            if m:
                return int(m.group(1)), m.group(2)
            r = RETURNS.search(line)
            if r:
                raise RuntimeError(f"their process returned early: {line.strip()}")
        raise RuntimeError("their process closed stderr without an announcement")

    def _read_returns(self) -> None:
        assert self.proc is not None and self.proc.stderr is not None
        for line in self.proc.stderr:
            r = RETURNS.search(line)
            if r:
                theirs = float(r.group(1 + (self.their_us or 0)))
                env_theirs = self.mirror.env.rewards[0 if self.their_us == 1 else 1]
                if (theirs > 0) != (env_theirs > 0) or (theirs < 0) != (env_theirs < 0):
                    raise RuntimeError(
                        f"outcome desync: their Returns {theirs} vs rf.Env reward {env_theirs}"
                    )
                self.returns_seen = True
                return
        raise RuntimeError("game ended without a Returns line")

    # External contract ---------------------------------------------------------

    def act(self, view) -> int:
        if self.proc is None:
            self._spawn()
        player, san = self._read_announcement()
        if player != self.their_us:
            raise RuntimeError(
                f"expected their move, got announcement for player {player}"
            )
        rf_action = self._rf_of_san(san)
        self.expect_own = rf_action
        return rf_action

    def on_action(self, rf_action: int) -> None:
        if rf_action == self.expect_own:
            # Arena echoing their own move back: their process already played it
            self.expect_own = None
            self.mirror.apply_san(self._san_of(rf_action))
        elif self.proc is None:
            # positional forced actions must exactly match THEIR ActionToString rendering
            # (the binary rejects anything else; +/# suffixes differ between renderers)
            their_id = self.mirror.their_id(rf_action)
            their_san = self.mirror.os_state.action_to_string(
                self.mirror.os_state.current_player(), their_id
            )
            self.forced_sans.append(their_san)
            self.mirror.apply_san(their_san)
        else:
            san = self._san_of(rf_action)
            their_id = self.mirror.their_id(rf_action)
            assert self.proc.stdin is not None
            self.proc.stdin.write(f"{their_id}\n")
            self.proc.stdin.flush()
            player, echoed_san = self._read_announcement()
            echoed = self.mirror.os_action_from_san(echoed_san)
            if player == self.their_us or echoed != their_id:
                raise RuntimeError(
                    f"desync: submitted their-id {their_id} but echo announced "
                    f"{echoed_san!r} (their-id {echoed})"
                )
            self.mirror.apply_san(san)
        if self.mirror.env.done() and self.proc is not None:
            self._read_returns()

    def close(self) -> None:
        proc = self.proc
        if proc is not None:
            try:
                if self.mirror.env.done() and not self.returns_seen:
                    raise RuntimeError(
                        "game finished but their Returns line was never seen"
                    )
                if proc.stdin is not None:
                    proc.stdin.close()
                try:
                    code = proc.wait(timeout=30)
                except subprocess.TimeoutExpired as e:
                    raise RuntimeError("their process refused to exit; killed") from e
                if code != 0:
                    raise RuntimeError(f"their process exited with code {code}")
            except BaseException:
                # unregister only on confirmed exit; an unkillable process stays
                # registered so the top-level sweep keeps trying
                if _reap(proc):
                    with _LIVE_LOCK:
                        _LIVE_PROCS.discard(proc)
                raise
            with _LIVE_LOCK:
                _LIVE_PROCS.discard(proc)
        done = next(self.done_counter)
        print(f"  finished {done}/{self.total} games", flush=True)


def replay_for_pgn(actions: list[int]):
    """Rebuild the pychess move stack (and SAN list) from an Arena action trace."""
    env = rf.Env(
        rf.games.Chess(encoder=rf.encoders.OpenSpielChess(), max_ticks=None),
        rf.Reward(win=1.0, loss=-1.0),
    )
    board = pychess.Board()
    sans = []
    for action in actions:
        uci = rf.chess_action_uci(action, env.state()["fen"])
        move = board.parse_uci(uci)
        sans.append(board.san(move))
        board.push(move)
        env.step({env.active_agents()[0]: action})
    return board, sans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rf-model",
        required=True,
        help="training-leg dir (resolves model.pt inside) or a checkpoint file",
    )
    ap.add_argument(
        "--os-model",
        required=True,
        help="their training-leg dir (their model is a directory + checkpoint number)",
    )
    ap.add_argument(
        "--os-checkpoint",
        default="latest",
        help='their checkpoint number, or "latest" (default): highest-numbered '
        "checkpoint-N.pt in --os-path",
    )
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument(
        "--sims", type=int, default=protocol.SIMS, help="their az bot's simulations"
    )
    ap.add_argument(
        "--our-sims",
        type=int,
        default=None,
        help="our side's simulations (default: same)",
    )
    ap.add_argument("--uct-c", type=float, default=protocol.C_PUCT)
    ap.add_argument(
        "--opening-plies",
        type=int,
        default=6,
        help="forced uniform-random opening length; each opening is played twice",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="our side's torch device")
    ap.add_argument(
        "--az-device",
        default="/cpu:0",
        help='their side (patched flag, their notation: "/cuda:0"); on the box pass both cuda',
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="concurrent their-engine processes = concurrent games (our GPU batch size)",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="per-turn budget for their side, seconds",
    )
    ap.add_argument(
        "--width",
        type=int,
        default=None,
        help="default: the checkpoint dir's config.json",
    )
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--out", required=True, help="fresh match directory")
    args = ap.parse_args()
    if args.games % 2:
        ap.error("--games must be even: every opening is played once per color")
    if not hasattr(rf, "Arena"):
        ap.error("this protocol needs a reinfors build with rf.Arena (PRs #159/#160)")

    # each side's model reference resolves to its engine's native format here; the
    # resolved concrete artifact is what the manifest records
    rf_model = Path(args.rf_model)
    args.rf_checkpoint = str(rf_model / "model.pt" if rf_model.is_dir() else rf_model)
    args.os_path = args.os_model
    if args.os_checkpoint == "latest":
        numbered = [
            int(m.group(1))
            for p in Path(args.os_path).glob("checkpoint-*.pt")
            if (m := re.search(r"checkpoint-(-?\d+)\.pt$", p.name))
        ]
        if not numbered:
            sys.exit(f"no checkpoint-N.pt found in {args.os_path}")
        args.os_checkpoint = max(numbered)
    else:
        args.os_checkpoint = int(args.os_checkpoint)

    out = Path(args.out).resolve()
    if out.exists():
        sys.exit(f"refusing to overwrite {out} — pick a fresh --out")
    out.mkdir(parents=True)
    args.pgn = str(out / "games.pgn")

    cfg_path = Path(args.rf_checkpoint).parent / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    width = args.width if args.width is not None else cfg.get("width", 32)
    depth = args.depth if args.depth is not None else cfg.get("depth", 1)

    game = rf.games.Chess(encoder=rf.encoders.OpenSpielChess(), max_ticks=None)
    space = game.observation_space()
    c, h, w = space.shape
    net = SweepResnet(c, h, w, HEAD_ACTIONS, width, depth).to(args.device)
    net.load_state_dict(torch.load(args.rf_checkpoint, map_location=args.device))
    net.eval()
    print(
        f"our net: SweepResnet w{width} d{depth}  params={sum(p.numel() for p in net.parameters()):,}"
    )

    def infer(obs, n=None):
        with torch.inference_mode():
            logits, values = net.heads(torch.from_numpy(obs).to(args.device))
        return logits.cpu().numpy(), values.cpu().numpy()

    our_sims = args.our_sims or args.sims
    policy = rf.policies.AlphaZero(
        num_simulations=our_sims,
        c_puct=args.uct_c,
        temperature=0.0,
        noise=None,
    )
    game_seeds = itertools.count(args.seed)
    done_counter = itertools.count(1)

    def factory():
        return TheirBot(args, next(game_seeds), done_counter, args.games)

    arena = rf.Arena(
        game,
        rf.Reward(win=1.0, loss=-1.0),
        contestants=[
            (policy, infer, 1.0),
            rf.arena.External(factory, workers=args.workers, timeout=args.timeout),
        ],
        n_slots=args.workers,
        start=rf.starts.RandomStartingMoves(args.opening_plies),
        seed=args.seed,
    )
    manifest.write(
        out,
        command=[sys.executable, *sys.argv],
        run_kind="h2h",
        protocol="v1",
        rf_checkpoint=args.rf_checkpoint,
        rf_checkpoint_sha256=_sha256(Path(args.rf_checkpoint)),
        os_path=args.os_path,
        os_checkpoint=args.os_checkpoint,
        os_checkpoint_sha256=_sha256(
            Path(args.os_path) / f"checkpoint-{args.os_checkpoint}.pt"
        ),
        net={"width": width, "depth": depth},
        sims={"theirs": args.sims, "ours": our_sims},
        uct_c=args.uct_c,
        opening={"kind": "RandomStartingMoves", "plies": args.opening_plies},
        seed=args.seed,
        concurrency={
            "workers": args.workers,
            "n_slots": args.workers,
            "timeout": args.timeout,
        },
        devices={"ours": args.device, "theirs": args.az_device},
        external_cmd=str(BIN),
        external_cmd_sha256=_sha256(BIN),
        completed=False,
    )

    started = time.time()
    try:
        result = arena.play(args.games)
    finally:
        kill_leftover_processes()

    run_meta = {
        "RFCheckpoint": args.rf_checkpoint,
        "OSPath": args.os_path,
        "OSCheckpoint": str(args.os_checkpoint),
        "TheirSims": str(args.sims),
        "OurSims": str(our_sims),
        "MatchSeed": str(args.seed),
        "Protocol": "v1",
    }
    wins = draws = 0
    for g in result.games:
        score = g.payoffs[0] / 2 + 0.5
        wins += score == 1.0
        draws += score == 0.5
        board, sans = replay_for_pgn(g.actions)
        our_white = g.seats[0] == 0
        if args.pgn:
            write_pgn(
                board,
                our_white,
                score,
                g.game_id + 1,
                g.opening_id + 1,
                args.opening_plies,
                run_meta,
                args.pgn,
            )
        opening = " ".join(sans[: args.opening_plies])
        tag = "W" if score == 1.0 else "D" if score == 0.5 else "L"
        print(
            f"  game {g.game_id + 1:3d}  opening {g.opening_id + 1:2d} as "
            f"{'white' if our_white else 'black'}: {tag}  [{opening}]"
        )

    n = args.games
    mean, stderr = result.payoff(0)
    score = mean / 2 + 0.5
    sims_note = (
        f"{args.sims} sims both"
        if our_sims == args.sims
        else f"sims theirs={args.sims} ours={our_sims}"
    )
    print(
        f"chess head-to-head (reinfors net vs open_spiel net, v1 protocol, {sims_note}): "
        f"{wins}W {draws}D {n - wins - draws}L -> score {score:.3f} "
        f"(pair stderr {stderr / 2:.3f} over {n // 2} pairs)"
    )

    manifest.finalize(
        out,
        status="ok",
        wall_seconds=round(time.time() - started, 1),
        result={
            "games": n,
            "wins": wins,
            "draws": draws,
            "losses": n - wins - draws,
            "score": round(score, 4),
            "pair_stderr": round(stderr / 2, 4),
        },
        output_sha256={"games.pgn": _sha256(Path(args.pgn))},
    )


if __name__ == "__main__":
    main()
