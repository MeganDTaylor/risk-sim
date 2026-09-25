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
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .engine import run_game
from .strategies import BaselineStrategy, ParameterizedStrategy

Weights = Dict[str, float]

# Evolved weights are cheap to reuse and expensive to regenerate, so training
# results are persisted here by default rather than lost when the process exits.
DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / 'trained_weights.json'


def save_weights(
    best_weights: Weights, population: List[Weights], history: Optional[List[Dict]] = None,
    path: Path = DEFAULT_WEIGHTS_PATH,
) -> None:
    """Writes best_weights/population (overwritten each call) and appends
    history to whatever history is already on disk, so a chart built later
    can see progress across every resumed run, not just the latest one."""
    existing_history: List[Dict] = []
    if path.exists():
        existing_history = json.loads(path.read_text()).get('history', [])
    combined_history = existing_history + (history or [])
    path.write_text(json.dumps({
        'best_weights': best_weights, 'population': population, 'history': combined_history,
    }, indent=2))


def load_population(path: Path = DEFAULT_WEIGHTS_PATH) -> Optional[List[Weights]]:
    """Returns the saved population to resume training from, or None if no
    save file exists yet (e.g. first run)."""
    if not path.exists():
        return None
    return json.loads(path.read_text())['population']


def load_best_weights(path: Path = DEFAULT_WEIGHTS_PATH) -> Optional[Weights]:
    """Returns the best previously-trained weights, or None if none are saved."""
    if not path.exists():
        return None
    return json.loads(path.read_text())['best_weights']


def load_history(path: Path = DEFAULT_WEIGHTS_PATH) -> List[Dict]:
    """Returns per-generation stats logged across every training run so far
    (empty list if none saved yet) - intended for plotting training progress."""
    if not path.exists():
        return []
    return json.loads(path.read_text()).get('history', [])


def run_generation(
    population: List[Weights], group_size: int = 4, max_turns: int = 300, rounds: int = 4
) -> List[int]:
    """Plays the population off in random groups of group_size, `rounds` times
    over (each round reshuffles groupings), so a win/loss on one unlucky dice
    sequence doesn't by itself decide who reproduces. Returns a parallel list
    of win counts (0..rounds)."""
    if len(population) % group_size != 0:
        raise ValueError('population_size must be a multiple of group_size')
    wins = [0] * len(population)
    for _ in range(rounds):
        order = list(range(len(population)))
        random.shuffle(order)
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
    population_size: int = 32,
    generations: int = 40,
    group_size: int = 4,
    max_turns: int = 300,
    rounds: int = 8,
    verbose: bool = False,
    initial_population: Optional[List[Weights]] = None,
) -> Tuple[Weights, List[Weights], List[Dict]]:
    if initial_population is not None:
        if len(initial_population) != population_size:
            raise ValueError('initial_population must have population_size entries')
        population = initial_population
    else:
        population = [ParameterizedStrategy.random_weights() for _ in range(population_size)]
    survivor_count = population_size // 2
    history: List[Dict] = []

    for generation in range(generations):
        wins = run_generation(population, group_size, max_turns, rounds)
        ranked = sorted(range(population_size), key=lambda i: wins[i], reverse=True)
        survivors = [population[i] for i in ranked[:survivor_count]]

        history.append({
            'best_wins': wins[ranked[0]], 'mean_wins': sum(wins) / len(wins), 'rounds': rounds,
        })
        if verbose:
            print(f'Generation {generation}: best={population[ranked[0]]} wins={wins[ranked[0]]}/{rounds}')

        children = []
        while len(children) < population_size - len(survivors):
            parent_a, parent_b = random.sample(survivors, 2)
            children.append(mutate(crossover(parent_a, parent_b)))
        population = survivors + children

    final_wins = run_generation(population, group_size, max_turns, rounds)
    best_index = max(range(len(population)), key=lambda i: final_wins[i])
    history.append({
        'best_wins': final_wins[best_index], 'mean_wins': sum(final_wins) / len(final_wins),
        'rounds': rounds, 'final': True,
    })
    return population[best_index], population, history


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
    default_population_size = 32
    resume_population = load_population()
    if resume_population is not None and len(resume_population) != default_population_size:
        print(f'Saved population size ({len(resume_population)}) does not match '
              f'population_size ({default_population_size}); starting fresh instead.')
        resume_population = None
    elif resume_population is not None:
        print(f'Resuming from saved population ({DEFAULT_WEIGHTS_PATH})')
    best_weights, population, history = train(
        population_size=default_population_size, verbose=True, initial_population=resume_population
    )
    win_rate = evaluate_against_baseline(best_weights, num_games=40)
    print(f'Best weights: {best_weights}')
    print(f'Win rate vs. 3 BaselineStrategy opponents: {win_rate:.0%}')
    history[-1]['win_rate_vs_baseline'] = win_rate
    save_weights(best_weights, population, history)
    print(f'Saved to {DEFAULT_WEIGHTS_PATH}')
