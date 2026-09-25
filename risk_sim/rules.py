# -*- coding: utf-8 -*-
"""Pure rules logic: reinforcement math, card sets, combat resolution."""
import functools
import random
from itertools import combinations, product
from typing import Dict, List, Tuple

from . import board
from .state import Card, GameState, WILD

MIN_REINFORCEMENTS = 3
CARD_SET_SIZE = 3


def reinforcement_count(state: GameState, player_id: int) -> int:
    territory_count = len(state.territories_owned_by(player_id))
    base = max(MIN_REINFORCEMENTS, territory_count // 3)
    continent_bonus = sum(
        board.CONTINENT_BONUS[c] for c in state.continents_owned_by(player_id)
    )
    return base + continent_bonus


def card_set_bonus(trade_in_count: int) -> int:
    # Official escalating trade-in schedule: 4,6,8,10,12,15,20,25,30,... (+5 after the 6th).
    fixed = [4, 6, 8, 10, 12, 15]
    if trade_in_count < len(fixed):
        return fixed[trade_in_count]
    return 20 + (trade_in_count - len(fixed)) * 5


def is_valid_set(cards: Tuple[Card, ...]) -> bool:
    if len(cards) != CARD_SET_SIZE:
        return False
    types = [c.kind for c in cards if c.kind != WILD]
    distinct = set(types)
    # Valid iff it can form three-of-a-kind (<=1 distinct type) or one-of-each
    # (every non-wild card already a different type), with wilds filling gaps.
    return len(distinct) <= 1 or len(distinct) == len(types)


def find_valid_sets(cards: List[Card]) -> List[Tuple[Card, ...]]:
    return [combo for combo in combinations(cards, CARD_SET_SIZE) if is_valid_set(combo)]


def territory_bonus_armies(cards: Tuple[Card, ...], owned_territories: List[str]) -> List[str]:
    """Territories in the traded set the player occupies -> +2 armies each, placed there."""
    return [c.territory for c in cards if c.territory and c.territory in owned_territories]


def roll_dice(count: int) -> List[int]:
    return sorted((random.randint(1, 6) for _ in range(count)), reverse=True)


def max_attack_dice(attacker_armies: int) -> int:
    return max(0, min(3, attacker_armies - 1))


def max_defend_dice(defender_armies: int) -> int:
    return min(2, defender_armies)


def resolve_combat(attack_dice: int, defend_dice: int) -> Tuple[int, int]:
    """Returns (attacker_losses, defender_losses). Defender wins ties."""
    attacker_rolls = roll_dice(attack_dice)
    defender_rolls = roll_dice(defend_dice)
    comparisons = min(len(attacker_rolls), len(defender_rolls))
    attacker_losses = 0
    defender_losses = 0
    for i in range(comparisons):
        if attacker_rolls[i] > defender_rolls[i]:
            defender_losses += 1
        else:
            attacker_losses += 1
    return attacker_losses, defender_losses


def _combat_round_distribution(attack_dice: int, defend_dice: int) -> Dict[Tuple[int, int], float]:
    """Exact probability of each (attacker_losses, defender_losses) outcome for
    one clash at these dice counts, by enumerating every die-face combination."""
    outcomes: Dict[Tuple[int, int], int] = {}
    total = 0
    for attacker_rolls in product(range(1, 7), repeat=attack_dice):
        attacker_sorted = sorted(attacker_rolls, reverse=True)
        for defender_rolls in product(range(1, 7), repeat=defend_dice):
            defender_sorted = sorted(defender_rolls, reverse=True)
            comparisons = min(attack_dice, defend_dice)
            attacker_losses = sum(
                1 for i in range(comparisons) if attacker_sorted[i] <= defender_sorted[i]
            )
            key = (attacker_losses, comparisons - attacker_losses)
            outcomes[key] = outcomes.get(key, 0) + 1
            total += 1
    return {key: count / total for key, count in outcomes.items()}


_ROUND_DISTRIBUTIONS = {
    (attack_dice, defend_dice): _combat_round_distribution(attack_dice, defend_dice)
    for attack_dice in (1, 2, 3) for defend_dice in (1, 2)
}

# Win-probability recursion depth/cache size grows with army counts; beyond
# this many armies a side's fate is already all but decided, so clip inputs
# rather than let huge stacks blow up the memoization table.
_WIN_PROBABILITY_ARMY_CAP = 40


@functools.lru_cache(maxsize=None)
def _attacker_win_probability(attacker_armies: int, defender_armies: int) -> float:
    if defender_armies <= 0:
        return 1.0
    if attacker_armies <= 1:
        return 0.0
    attack_dice = min(3, attacker_armies - 1)
    defend_dice = min(2, defender_armies)
    return sum(
        probability * _attacker_win_probability(attacker_armies - a_loss, defender_armies - d_loss)
        for (a_loss, d_loss), probability in _ROUND_DISTRIBUTIONS[(attack_dice, defend_dice)].items()
    )


def attacker_win_probability(attacker_armies: int, defender_armies: int) -> float:
    """Probability the attacker eventually wipes out the defending territory if
    both sides fight to the end always committing max dice ('Risk odds'). Used
    as a static evaluation of an attack's favorability - a plain army ratio
    doesn't capture how non-linear dice-based attrition actually is (e.g. 3v2
    favors the attacker by far more than the 1.5x ratio suggests)."""
    return _attacker_win_probability(
        min(attacker_armies, _WIN_PROBABILITY_ARMY_CAP), min(defender_armies, _WIN_PROBABILITY_ARMY_CAP)
    )
