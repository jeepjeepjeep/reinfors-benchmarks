"""Observation parity: reinfors' OpenSpielChess encoder vs pyspiel's chess observation_tensor.

Random-walk lockstep: drive an rf.Env with random legal moves; at every position rebuild the
pyspiel state from the FEN reinfors exports and compare all planes float-for-float — EXCEPT
plane 13 (repetition count), which a FEN-built pyspiel state cannot know (no history). Plane 13
is covered by the two halves it factors into: reinfors' repetition_count is pinned by Rust tests
(the AZ-119 repetition suite), and pyspiel's normalization (rep-1)/2 is asserted here directly
by building a threefold sequence move-by-move on the pyspiel side.

    .venv23/bin/python benchmarks/openspiel/parity_chess_obs.py
"""

import numpy as np
import pyspiel
import reinfors as rf

GAME = pyspiel.load_game("chess")


def compare_walk(games: int = 30, max_plies: int = 80) -> int:
    rng = np.random.default_rng(0)
    checked = 0
    divergent: list[str] = []
    for g in range(games):
        env = rf.Env(rf.games.Chess(max_ticks=None, encoder=rf.encoders.OpenSpielChess()), seed=g)
        env.reset()
        for _ply in range(max_plies):
            if env.done():
                break
            fen = env.state()["fen"]
            os_state = GAME.new_initial_state(fen)
            if os_state.is_terminal():
                # Termination-rule divergence (they auto-draw positions we still play — e.g.
                # insufficient material). Counted, not compared: this is the item the
                # termination-matching task quantifies; obs parity is only defined on live states.
                divergent.append(fen)
                break
            ours = env.observe(env.active_agents()[0]).reshape(20, 64)
            theirs = np.asarray(os_state.observation_tensor(), dtype=np.float32).reshape(20, 64)
            for p in range(20):
                if p == 13:
                    continue  # repetition: unknowable from a bare FEN (see module docstring)
                assert np.array_equal(ours[p], theirs[p]), f"game {g} ply {_ply} plane {p}\n{ours[p]}\n{theirs[p]}"
            checked += 1
            mover = env.active_agents()[0]
            env.step({mover: int(rng.choice(env.legal_actions(mover)))})
    if divergent:
        print(f"termination divergence on {len(divergent)}/{games} walks (their auto-draw, we play on), e.g.:")
        print("  " + divergent[0])
    return checked


def check_their_repetition_normalization() -> None:
    s = GAME.new_initial_state()
    for san in ["Nf3", "Nf6", "Ng1", "Ng8"]:  # ONE cycle: rep=2 (a 2nd cycle threefold-draws the state)
        s.apply_action(s.string_to_action(san))
    rep_plane = np.asarray(s.observation_tensor()).reshape(20, 64)[13]
    assert np.allclose(rep_plane, 0.5), rep_plane[0]  # (2-1)/2
    print(f"their repetition plane after recurrences: {rep_plane[0]} — formula (rep-1)/2 confirmed")


DEAD_POSITION_TABLE = [
    # (fen, dead?) — FIDE's dead-position rule; reinfors' predicate is pinned to the same table
    # by its Rust tests (chess.rs::fide_dead_position_tests), so agreement here = cross parity.
    ("8/8/4k3/8/8/3K4/8/8 w - - 0 1", True),
    ("8/8/4k3/8/8/3KB3/8/8 w - - 0 1", True),
    ("8/8/4kn2/8/8/3K4/8/8 w - - 0 1", True),
    ("8/8/3bk3/8/8/3KB3/8/8 w - - 0 1", True),   # same-colored bishops
    ("8/8/2bk4/8/8/3KB3/8/8 w - - 0 1", False),  # opposite-colored bishops
    ("8/8/3nk3/8/8/2NK4/8/8 w - - 0 1", False),  # KN vs KN
    ("8/8/4k3/8/8/2NKN3/8/8 w - - 0 1", False),  # KNN vs K
]


def check_dead_position_parity() -> None:
    for fen, dead in DEAD_POSITION_TABLE:
        got = GAME.new_initial_state(fen).is_terminal()
        assert got == dead, f"pyspiel disagrees with the FIDE table at {fen}: {got}"
    print(f"dead-position parity: {len(DEAD_POSITION_TABLE)} material boundaries agree")


if __name__ == "__main__":
    n = compare_walk()
    check_their_repetition_normalization()
    check_dead_position_parity()
    print(f"PARITY OK: {n} positions, 19/20 planes exact + repetition formula confirmed")
