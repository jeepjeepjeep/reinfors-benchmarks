"""Backgammon movegen/encoding parity: walk reinfors and OpenSpiel in lockstep and compare at
every decision — the perft analogue for a game with no perft.

reinfors realizes chance inside its Env, so the walk synchronizes by *reading* each realized roll
from `env.state()["dice"]` and applying the matching chance outcome to the OpenSpiel state (the
roll tables share one ordering by construction). At every decision position we compare: the player
to move, the FULL legal-action set (identical 1352-id encodings), and the 200-dim observation
tensor for both players. At terminal: winner and margin (their full_scoring returns +-1/2/3).

    .venv/bin/python benchmarks/openspiel/parity_backgammon.py --games 50 --seed 0
"""

import argparse
import random

import pyspiel
import reinfors as rf

ROLLS = [
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (2, 3),
    (2, 4),
    (2, 5),
    (2, 6),
    (3, 4),
    (3, 5),
    (3, 6),
    (4, 5),
    (4, 6),
    (5, 6),
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5),
    (6, 6),
]


NON_DOUBLES = ROLLS[:15]


def os_roll_id(dice: tuple[int, int]) -> int:
    """The installed pyspiel wheel's in-game chance id for a roll: 36 uniform outcomes — each
    non-double under two ids (we pick the even one), doubles at 30..35. (The pinned source uses a
    21-outcome model instead; the two are probabilistically identical, ids differ.)"""
    if dice[0] == dice[1]:
        return 29 + dice[0]
    return 2 * NON_DOUBLES.index(dice)


def sync_chance(os_state: object, ours: dict) -> None:
    """Apply our realized roll (from the state dict) to their pending chance node."""
    dice = tuple(sorted(int(d) for d in ours["dice"]))
    os_state.apply_action(os_roll_id(dice))


def run_game(game: object, seed: int) -> tuple[int, int]:
    rng = random.Random(seed)
    env = rf.Env(rf.games.Backgammon(max_ticks=None), rf.Reward(win=1.0), seed=seed)
    os_state = game.new_initial_state()

    # Their opening chance node: 0-14 X starts with roll i, 15-29 O with roll i-15 — mirror the
    # opening our initial_state already realized.
    # The installed wheel's opening ids: 2 * non-double-roll-index + starter (even = X, odd = O).
    ours = env.state()
    dice = tuple(sorted(int(d) for d in ours["dice"]))
    opening = 2 * NON_DOUBLES.index(dice) + int(ours["to_move"])
    os_state.apply_action(opening)

    positions = 0
    plies = 0
    while not env.done():
        assert not os_state.is_terminal(), "OpenSpiel finished before reinfors"
        ours = env.state()
        mover = env.active_agents()[0]
        assert os_state.current_player() == mover, (
            f"mover mismatch: os={os_state.current_player()} rf={mover}\n{os_state}"
        )
        ours_legal = sorted(env.legal_actions(mover))
        theirs_legal = sorted(os_state.legal_actions())
        assert ours_legal == theirs_legal, (
            f"legal-set mismatch at ply {plies} (dice {ours['dice']}):\n"
            f"  ours-only:   {sorted(set(ours_legal) - set(theirs_legal))}\n"
            f"  theirs-only: {sorted(set(theirs_legal) - set(ours_legal))}\n{os_state}"
        )
        for player in (0, 1):
            mine = [round(float(x), 5) for x in env.observe(player).ravel()]
            theirs = [round(float(x), 5) for x in os_state.observation_tensor(player)]
            assert mine == theirs, (
                f"observation mismatch for player {player} at ply {plies}"
            )
        positions += 1

        action = rng.choice(ours_legal)
        env.step({mover: action})
        os_state.apply_action(action)
        plies += 1
        if not env.done() and os_state.is_chance_node():
            sync_chance(os_state, env.state())
        assert plies < 5000, "runaway game"

    assert os_state.is_terminal(), f"reinfors finished before OpenSpiel\n{os_state}"
    returns = os_state.returns()
    ours = env.state()
    rf_winner = 0 if ours["scores"][0] == 15 else 1
    os_winner = 0 if returns[0] > 0 else 1
    assert rf_winner == os_winner, "winner mismatch"
    assert abs(returns[os_winner]) in (1.0, 2.0, 3.0)
    return positions, int(abs(returns[os_winner]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    game = pyspiel.load_game("backgammon(scoring_type=full_scoring)")
    total_positions = 0
    margins = [0, 0, 0]
    for g in range(args.games):
        positions, margin = run_game(game, args.seed + g)
        total_positions += positions
        margins[margin - 1] += 1
        print(f"game {g:3d}: OK ({positions} positions, margin {margin})")
    print(
        f"PARITY OK: {args.games} games, {total_positions} positions compared "
        f"(margins 1/2/3: {margins[0]}/{margins[1]}/{margins[2]})"
    )


if __name__ == "__main__":
    main()
