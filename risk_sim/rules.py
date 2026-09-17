# -*- coding: utf-8 -*-
"""Pure rules logic: reinforcement math, card sets, combat resolution."""
import random
from itertools import combinations
from typing import List, Tuple

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
