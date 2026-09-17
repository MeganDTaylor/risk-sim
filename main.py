# -*- coding: utf-8 -*-
"""Entry point: run a single simulated game of classic Risk and print the result.

The actual rules/state/strategy code lives in risk_sim/ - see that package if
you want to plug in a different strategy (including a human-input one) or run
risk_sim/train.py to evolve ParameterizedStrategy via self-play.
"""
from risk_sim import BaselineStrategy, RandomStrategy, run_game


def main():
    strategies = [BaselineStrategy(), BaselineStrategy(), BaselineStrategy(), RandomStrategy()]
    winner, state = run_game(strategies, max_turns=300, verbose=True)
    if winner is None:
        print('No winner within the turn limit.')
    else:
        print(f'Player {winner} wins with {len(state.territories_owned_by(winner))} territories.')


if __name__ == '__main__':
    main()
