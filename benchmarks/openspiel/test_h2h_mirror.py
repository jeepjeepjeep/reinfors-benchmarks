"""Mirror compatibility: THEIR SAN renderings must drive all three state mirrors.

Forced positional openings are only accepted by the C++ binary when they exactly match
its ActionToString rendering, so TheirBot stores pyspiel-rendered SANs. These tests prove
that rendering round-trips through the Mirror (rf.Env + python-chess + pyspiel) for the
move classes whose renderings diverge between libraries: castling, checks/mates, and
promotions — plus a seeded fuzz over random lines. The echo-desync assertion only guards
INTERACTIVE moves at runtime; forced positional openings are validated by the binary
before any echo exists, so the env-gated binary test below is the sole guard against
wheel-vs-source ActionToString skew in forced arguments.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_h2h_chess import Mirror


def apply_their_rendering(mirror: Mirror, loose_san: str) -> str:
    """Resolve a move by tolerant match, then apply it via THEIR exact rendering —
    the same path TheirBot uses for forced openings."""
    their_id = mirror.os_action_from_san(loose_san)
    their_san = mirror.os_state.action_to_string(
        mirror.os_state.current_player(), their_id
    )
    mirror.apply_san(their_san)
    return their_san


LINES = {
    "castling_both_sides": "e4 e5 Nf3 Nc6 Bc4 Bc5 O-O d6 d3 Bg4 Nc3 Qd7 h3 O-O-O",
    "checks_and_mate": "e4 e5 Qh5 Nc6 Bc4 Nf6 Qxf7#",
    "promotion_with_check": "g4 h5 gxh5 g6 hxg6 Nf6 g7 Rg8 gxf8=Q+",
    "underpromotion": "g4 h5 gxh5 g6 hxg6 Nf6 g7 Rg8 gxf8=N",
}


@pytest.mark.parametrize("name", sorted(LINES))
def test_their_renderings_drive_the_mirror(name: str) -> None:
    mirror = Mirror()
    for loose in LINES[name].split():
        their_san = apply_their_rendering(mirror, loose)
        assert their_san.rstrip("+#") == loose.rstrip("+#")
    assert mirror.env.ticks == len(LINES[name].split())


def test_fuzzed_lines_stay_in_lockstep() -> None:
    # uniform-random walks: every ply is applied via THEIR rendering; the three mirrors
    # must agree on legality and terminality throughout
    import random

    rng = random.Random(11)
    for _ in range(40):
        mirror = Mirror()
        for _ply in range(80):
            if mirror.env.done() or mirror.os_state.is_terminal():
                break
            actions = mirror.os_state.legal_actions()
            their_id = actions[rng.randrange(len(actions))]
            their_san = mirror.os_state.action_to_string(
                mirror.os_state.current_player(), their_id
            )
            mirror.apply_san(their_san)
            assert mirror.env.done() == mirror.os_state.is_terminal()
        assert len(mirror.board.move_stack) == mirror.env.ticks


def test_close_reaps_on_pre_wait_errors() -> None:
    # an error raised BEFORE wait() (e.g. missing Returns) must still kill the child and
    # only then unregister it — otherwise the top-level sweep cannot find the orphan
    import itertools
    import subprocess
    import sys

    from eval_h2h_chess import _LIVE_LOCK, _LIVE_PROCS, TheirBot

    bot = TheirBot(None, 0, itertools.count(1), 1)
    for san in ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7#"]:
        bot.mirror.apply_san(san)
    assert bot.mirror.env.done() and not bot.returns_seen
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    bot.proc = proc
    with _LIVE_LOCK:
        _LIVE_PROCS.add(proc)
    with pytest.raises(RuntimeError, match="Returns line was never seen"):
        bot.close()
    assert proc.poll() is not None, "child must be dead after the failed close"
    with _LIVE_LOCK:
        assert proc not in _LIVE_PROCS


@pytest.mark.skipif(
    not __import__("eval_h2h_chess").BIN.exists()
    or "H2H_SMOKE_OS_PATH" not in __import__("os").environ,
    reason="needs the openspiel binary and H2H_SMOKE_OS_PATH/H2H_SMOKE_OS_CKPT (an az "
    "checkpoint dir — the binary refuses to run without an az player); box only",
)
@pytest.mark.parametrize("name", sorted(LINES))
def test_forced_lines_accepted_by_the_binary(name: str) -> None:
    # the binary validates positional forced actions before any echo exists, so the
    # echo-desync assertion cannot catch wheel-vs-source SAN skew here
    import os
    import subprocess
    import time as _time

    from eval_h2h_chess import BIN

    mirror = Mirror()
    their_sans = [apply_their_rendering(mirror, loose) for loose in LINES[name].split()]
    # their --player1 is player 0 = BLACK; put az on the side to move at the opening
    # exit so its first announcement doubles as a forcing-complete marker
    white_to_move = len(their_sans) % 2 == 0
    p1, p2 = ("human", "az") if white_to_move else ("az", "human")
    proc = subprocess.Popen(
        [
            str(BIN),
            "--game",
            "chess",
            "--player1",
            p1,
            "--player2",
            p2,
            "--az_path",
            os.environ["H2H_SMOKE_OS_PATH"],
            "--az_checkpoint",
            os.environ.get("H2H_SMOKE_OS_CKPT", "0"),
            "--max_simulations",
            "2",
            "--az_device",
            "/cpu:0",
            "--solve=false",
            "--num_games",
            "1",
            "--quiet=false",
            *their_sans,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    # Success requires positive evidence that every forced action was consumed — a live
    # process may still be loading the checkpoint, which happens BEFORE forced-action
    # processing. Markers: one forced-action line per SAN, or play reaching an
    # announcement/Returns (both only happen after forcing completes).
    import re as _re
    import threading

    from eval_h2h_chess import CHOSE, RETURNS

    forced_marker = _re.compile(r"Player \d+ forced action:")
    lines: list[str] = []

    def _forcing_complete(snapshot: list[str]) -> bool:
        forced_seen = sum(1 for line in snapshot if forced_marker.search(line))
        played = any(CHOSE.search(line) or RETURNS.search(line) for line in snapshot)
        return forced_seen >= len(their_sans) or played

    def _reader() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            lines.append(line)

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()
    deadline = _time.monotonic() + 180  # checkpoint load may dominate
    try:
        while _time.monotonic() < deadline:
            snapshot = list(lines)
            if _forcing_complete(snapshot):
                return
            if proc.poll() is not None:
                # The process can exit after writing its final marker but before the
                # reader thread drains stderr. Consume the closed pipe before judging.
                reader.join(timeout=5)
                snapshot = list(lines)
                if _forcing_complete(snapshot):
                    return
                raise AssertionError(
                    f"binary rejected forced line {name} (exit {proc.returncode}): "
                    f"{''.join(snapshot)[-500:]}"
                )
            _time.sleep(0.5)
        raise AssertionError(
            f"no forced-action confirmation for {name} within the deadline: "
            f"{''.join(lines)[-500:]}"
        )
    finally:
        proc.kill()
        proc.wait(timeout=10)
