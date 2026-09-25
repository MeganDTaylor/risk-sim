# -*- coding: utf-8 -*-
"""Entry point: run a single simulated game of classic Risk and print the result.

The actual rules/state/strategy code lives in risk_sim/ - see that package if
you want to plug in a different strategy (including a human-input one), run
risk_sim/train.py to evolve ParameterizedStrategy via self-play, or run
risk_sim/train_neural.py to evolve NeuralStrategy instead (currently the
stronger of the two - see each module's saved win rate vs. BaselineStrategy).
"""
from risk_sim import (
    BaselineStrategy, NeuralStrategy, ParameterizedStrategy, RandomStrategy,
    load_best_genome, load_best_weights, run_game,
)


def _lead_strategy():
    """Prefers the trained neural strategy, falling back to the trained
    heuristic weights, then to plain BaselineStrategy defaults if neither has
    been trained yet."""
    best_genome = load_best_genome()
    if best_genome is not None:
        return NeuralStrategy.from_genome(best_genome)
    best_weights = load_best_weights()
    return ParameterizedStrategy(best_weights) if best_weights else BaselineStrategy()


def main():
    strategies = [_lead_strategy(), BaselineStrategy(), BaselineStrategy(), RandomStrategy()]
    winner, state = run_game(strategies, max_turns=300, verbose=True)
    if winner is None:
        print('No winner within the turn limit.')
    else:
        print(f'Player {winner} wins with {len(state.territories_owned_by(winner))} territories.')


if __name__ == '__main__':
    main()
