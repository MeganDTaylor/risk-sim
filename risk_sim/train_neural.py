# -*- coding: utf-8 -*-
"""Self-play evolutionary trainer for NeuralStrategy - same idea as train.py
(random population, self-play games decide fitness, survivors breed) but
operating on flat numpy weight vectors instead of named scalar weights, since
a network's genome is too large to reason about term-by-term the way
ParameterizedStrategy's 7 knobs were.
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .engine import run_game
from .neural import NeuralStrategy
from .strategies import BaselineStrategy

Genome = np.ndarray

DEFAULT_GENOME_PATH = Path(__file__).resolve().parent.parent / 'trained_neural.json'

# Weights drift unboundedly under gaussian mutation with nothing else pulling
# them back in, so they're clipped here to keep tanh/matmul activations sane.
MUTATION_CLIP = 4.0


def save_genome(
    best_genome: Genome, population: List[Genome], history: Optional[List[Dict]] = None,
    path: Path = DEFAULT_GENOME_PATH,
) -> None:
    existing_history: List[Dict] = []
    if path.exists():
        existing_history = json.loads(path.read_text()).get('history', [])
    combined_history = existing_history + (history or [])
    path.write_text(json.dumps({
        'best_genome': best_genome.tolist(),
        'population': [g.tolist() for g in population],
        'history': combined_history,
    }, indent=2))


def load_population(path: Path = DEFAULT_GENOME_PATH) -> Optional[List[Genome]]:
    if not path.exists():
        return None
    return [np.array(g) for g in json.loads(path.read_text())['population']]


def load_best_genome(path: Path = DEFAULT_GENOME_PATH) -> Optional[Genome]:
    if not path.exists():
        return None
    return np.array(json.loads(path.read_text())['best_genome'])


def load_history(path: Path = DEFAULT_GENOME_PATH) -> List[Dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text()).get('history', [])


def run_generation(
    population: List[Genome], group_size: int = 4, max_turns: int = 300, rounds: int = 8
) -> List[int]:
    if len(population) % group_size != 0:
        raise ValueError('population_size must be a multiple of group_size')
    wins = [0] * len(population)
    for _ in range(rounds):
        order = list(range(len(population)))
        random.shuffle(order)
        for start in range(0, len(order), group_size):
            group = order[start:start + group_size]
            strategies = [NeuralStrategy.from_genome(population[i]) for i in group]
            winner, _ = run_game(strategies, max_turns=max_turns)
            if winner is not None:
                wins[group[winner]] += 1
    return wins


def crossover(a: Genome, b: Genome) -> Genome:
    mask = np.random.rand(len(a)) < 0.5
    return np.where(mask, a, b)


def mutate(genome: Genome, rate: float = 0.15, sigma: float = 0.4) -> Genome:
    mutated = genome.copy()
    mask = np.random.rand(len(genome)) < rate
    mutated[mask] += np.random.normal(0, sigma, size=int(mask.sum()))
    return np.clip(mutated, -MUTATION_CLIP, MUTATION_CLIP)


def train(
    population_size: int = 32,
    generations: int = 40,
    group_size: int = 4,
    max_turns: int = 300,
    rounds: int = 8,
    mutation_rate: float = 0.15,
    mutation_sigma: float = 0.4,
    verbose: bool = False,
    initial_population: Optional[List[Genome]] = None,
) -> Tuple[Genome, List[Genome], List[Dict]]:
    if initial_population is not None:
        if len(initial_population) != population_size:
            raise ValueError('initial_population must have population_size entries')
        population = initial_population
    else:
        population = [NeuralStrategy.random_genome() for _ in range(population_size)]
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
            print(f'Generation {generation}: best_wins={wins[ranked[0]]}/{rounds} '
                  f'mean={sum(wins) / len(wins):.2f}/{rounds}')

        children = []
        while len(children) < population_size - len(survivors):
            parent_a, parent_b = random.sample(survivors, 2)
            children.append(mutate(crossover(parent_a, parent_b), rate=mutation_rate, sigma=mutation_sigma))
        population = survivors + children

    final_wins = run_generation(population, group_size, max_turns, rounds)
    best_index = max(range(len(population)), key=lambda i: final_wins[i])
    history.append({
        'best_wins': final_wins[best_index], 'mean_wins': sum(final_wins) / len(final_wins),
        'rounds': rounds, 'final': True,
    })
    return population[best_index], population, history


def evaluate_against_baseline(
    genome: Genome, num_games: int = 20, group_size: int = 4, max_turns: int = 300
) -> float:
    wins = 0
    for _ in range(num_games):
        strategies = [NeuralStrategy.from_genome(genome)] + [BaselineStrategy() for _ in range(group_size - 1)]
        random.shuffle(strategies)
        trained_seat = next(i for i, s in enumerate(strategies) if isinstance(s, NeuralStrategy))
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
        print(f'Resuming from saved population ({DEFAULT_GENOME_PATH})')
    best_genome, population, history = train(
        population_size=default_population_size, verbose=True, initial_population=resume_population
    )
    win_rate = evaluate_against_baseline(best_genome, num_games=40)
    print(f'Win rate vs. 3 BaselineStrategy opponents: {win_rate:.0%}')
    history[-1]['win_rate_vs_baseline'] = win_rate
    save_genome(best_genome, population, history)
    print(f'Saved to {DEFAULT_GENOME_PATH}')
