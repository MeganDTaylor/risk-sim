# -*- coding: utf-8 -*-
"""Self-play training scaffold for ParameterizedStrategy.

No generative AI here - it's a plain evolutionary algorithm. Each generation,
random weight vectors play full games of Risk against each other, the
winningest half survive, and the next generation is bred from them (crossover
+ mutation). Over enough generations the population's attack/fortify
thresholds should drift toward whatever actually wins games.

This is a starting scaffold, not a tuned trainer: population/generation counts
are small enough to run in a few seconds so you can iterate on the reward
signal (currently just "won the game") before spending real compute on it.
"""
import random
from typing import Dict, List, Tuple

from .engine import run_game
from .strategies import BaselineStrategy, ParameterizedStrategy

Weights = Dict[str, float]


def run_generation(
    population: List[Weights], group_size: int = 4, max_turns: int = 300
) -> List[int]:
    """Plays the population off in random groups of group_size, once each.
    Returns a parallel list of win counts (0 or 1 with one game per group)."""
    if len(population) % group_size != 0:
        raise ValueError('population_size must be a multiple of group_size')
    order = list(range(len(population)))
    random.shuffle(order)
    wins = [0] * len(population)
    for start in range(0, len(order), group_size):
        group = order[start:start + group_size]
        strategies = [ParameterizedStrategy(population[i]) for i in group]
        winner, _ = run_game(strategies, max_turns=max_turns)
        if winner is not None:
            wins[group[winner]] += 1
    return wins


def crossover(a: Weights, b: Weights) -> Weights:
    return {name: random.choice([a[name], b[name]]) for name in ParameterizedStrategy.PARAM_NAMES}


def mutate(weights: Weights, rate: float = 0.3) -> Weights:
    mutated = dict(weights)
    for name, (lo, hi) in ParameterizedStrategy.PARAM_BOUNDS.items():
        if random.random() < rate:
            span = hi - lo
            mutated[name] = min(hi, max(lo, weights[name] + random.gauss(0, span * 0.15)))
    return mutated


def train(
    population_size: int = 16,
    generations: int = 20,
    group_size: int = 4,
    max_turns: int = 300,
    verbose: bool = False,
) -> Tuple[Weights, List[Weights]]:
    population = [ParameterizedStrategy.random_weights() for _ in range(population_size)]
    survivor_count = population_size // 2

    for generation in range(generations):
        wins = run_generation(population, group_size, max_turns)
        ranked = sorted(range(population_size), key=lambda i: wins[i], reverse=True)
        survivors = [population[i] for i in ranked[:survivor_count]]

        if verbose:
            print(f'Generation {generation}: best={population[ranked[0]]} wins={wins[ranked[0]]}')

        children = []
        while len(children) < population_size - len(survivors):
            parent_a, parent_b = random.sample(survivors, 2)
            children.append(mutate(crossover(parent_a, parent_b)))
        population = survivors + children

    final_wins = run_generation(population, group_size, max_turns)
    best_index = max(range(len(population)), key=lambda i: final_wins[i])
    return population[best_index], population


def evaluate_against_baseline(
    weights: Weights, num_games: int = 20, group_size: int = 4, max_turns: int = 300
) -> float:
    """Win rate of one trained strategy seated against (group_size - 1) baseline
    opponents, reshuffling seats each game."""
    wins = 0
    for _ in range(num_games):
        strategies = [ParameterizedStrategy(weights)] + [BaselineStrategy() for _ in range(group_size - 1)]
        random.shuffle(strategies)
        trained_seat = next(i for i, s in enumerate(strategies) if isinstance(s, ParameterizedStrategy))
        winner, _ = run_game(strategies, max_turns=max_turns)
        if winner == trained_seat:
            wins += 1
    return wins / num_games


if __name__ == '__main__':
    best_weights, _ = train(verbose=True)
    win_rate = evaluate_against_baseline(best_weights)
    print(f'Best weights: {best_weights}')
    print(f'Win rate vs. 3 BaselineStrategy opponents: {win_rate:.0%}')
