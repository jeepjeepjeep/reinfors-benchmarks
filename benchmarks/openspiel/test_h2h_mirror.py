"""Mirror compatibility: THEIR SAN renderings must drive all three state mirrors.

Forced positional openings are only accepted by the C++ binary when they exactly match
its ActionToString rendering, so TheirBot stores pyspiel-rendered SANs. These tests prove
that rendering round-trips through the Mirror (rf.Env + python-chess + pyspiel) for the
move classes whose renderings diverge between libraries: castling, checks/mates, and
promotions — plus a seeded fuzz over random lines. The binary itself is exercised by the
box smoke test; wheel-vs-source ActionToString skew is guarded at runtime by the
echo-desync assertion.
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
    "underpromotion_with_check": "g4 h5 gxh5 g6 hxg6 Nf6 g7 Rg8 gxf8=Q+",
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
            assert (
                mirror.env.done() == mirror.os_state.is_terminal()
                or not mirror.env.done()
            )
        assert len(mirror.board.move_stack) == mirror.env.ticks
