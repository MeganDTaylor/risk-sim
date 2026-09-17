# -*- coding: utf-8 -*-
"""Pluggable player strategies. Every decision point in the engine calls one
of these methods, so swapping a player's brain is just swapping the object
passed into GameState - human, heuristic, random, or a trained one."""
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from . import rules
from .state import Card, GameState


class Strategy(ABC):
    @abstractmethod
    def choose_initial_placement(self, state: GameState, player_id: int) -> str:
        """Return one owned territory to receive a single setup army."""

    @abstractmethod
    def choose_reinforcement_placement(
        self, state: GameState, player_id: int, army_count: int
    ) -> Dict[str, int]:
        """Return {territory: armies} summing to army_count, all owned."""

    @abstractmethod
    def choose_card_trade(
        self, state: GameState, player_id: int, mandatory: bool
    ) -> Optional[Tuple[Card, Card, Card]]:
        """Return a valid 3-card set to trade in, or None to decline (only
        allowed when mandatory is False)."""

    @abstractmethod
    def choose_attack(
        self, state: GameState, player_id: int
    ) -> Optional[Tuple[str, str, int]]:
        """Return (from_territory, to_territory, attack_dice) or None to stop
        attacking for the rest of this turn."""

    @abstractmethod
    def choose_defend_dice(self, state: GameState, player_id: int, max_dice: int) -> int:
        """Return how many dice (1..max_dice) to defend with."""

    @abstractmethod
    def choose_capture_move(
        self, state: GameState, player_id: int, from_territory: str,
        to_territory: str, min_move: int, max_move: int,
    ) -> int:
        """Armies to move into a just-conquered territory, in [min_move, max_move]."""

    @abstractmethod
    def choose_fortify(
        self, state: GameState, player_id: int
    ) -> Optional[Tuple[str, str, int]]:
        """Return (from_territory, to_territory, amount) or None to skip."""


def _border_territories(state: GameState, player_id: int) -> List[str]:
    owned = state.territories_owned_by(player_id)
    return [t for t in owned if state.enemy_neighbors(t, player_id)]


def _weakest(state: GameState, territories: List[str]) -> str:
    return min(territories, key=lambda t: state.armies(t))


class RandomStrategy(Strategy):
    """Picks uniformly among legal moves. Useful as a baseline opponent and
    for injecting diversity into a self-play population."""

    def choose_initial_placement(self, state, player_id):
        return random.choice(state.territories_owned_by(player_id))

    def choose_reinforcement_placement(self, state, player_id, army_count):
        owned = state.territories_owned_by(player_id)
        placement = {t: 0 for t in owned}
        for _ in range(army_count):
            placement[random.choice(owned)] += 1
        return {t: n for t, n in placement.items() if n > 0}

    def choose_card_trade(self, state, player_id, mandatory):
        valid_sets = rules.find_valid_sets(state.players[player_id].cards)
        if not valid_sets:
            return None
        if not mandatory and random.random() < 0.5:
            return None
        return random.choice(valid_sets)

    def choose_attack(self, state, player_id):
        options = []
        for t in state.territories_owned_by(player_id):
            if state.armies(t) < 2:
                continue
            for enemy in state.enemy_neighbors(t, player_id):
                options.append((t, enemy))
        if not options or random.random() < 0.3:
            return None
        frm, to = random.choice(options)
        dice = random.randint(1, rules.max_attack_dice(state.armies(frm)))
        return frm, to, dice

    def choose_defend_dice(self, state, player_id, max_dice):
        return random.randint(1, max_dice)

    def choose_capture_move(self, state, player_id, from_territory, to_territory, min_move, max_move):
        return random.randint(min_move, max_move)

    def choose_fortify(self, state, player_id):
        owned = state.territories_owned_by(player_id)
        movable = [t for t in owned if state.armies(t) > 1]
        if not movable or random.random() < 0.5:
            return None
        frm = random.choice(movable)
        friendly_neighbors = [n for n in state.neighbors(frm) if state.owner(n) == player_id]
        if not friendly_neighbors:
            return None
        to = random.choice(friendly_neighbors)
        amount = random.randint(1, state.armies(frm) - 1)
        return frm, to, amount


class BaselineStrategy(Strategy):
    """Simple greedy heuristic: reinforce the weakest border, attack whenever
    clearly favored, press captures hard, and shore up the front line."""

    def __init__(self, attack_ratio_threshold: float = 1.5, aggression: float = 0.8,
                 fortify_min_spare: int = 2):
        self.attack_ratio_threshold = attack_ratio_threshold
        self.aggression = aggression
        self.fortify_min_spare = fortify_min_spare

    def choose_initial_placement(self, state, player_id):
        borders = _border_territories(state, player_id)
        owned = borders or state.territories_owned_by(player_id)
        return _weakest(state, owned)

    def choose_reinforcement_placement(self, state, player_id, army_count):
        borders = _border_territories(state, player_id)
        target = _weakest(state, borders or state.territories_owned_by(player_id))
        return {target: army_count}

    def choose_card_trade(self, state, player_id, mandatory):
        valid_sets = rules.find_valid_sets(state.players[player_id].cards)
        if not valid_sets:
            return None
        owned = state.territories_owned_by(player_id)
        return max(
            valid_sets,
            key=lambda combo: len(rules.territory_bonus_armies(combo, owned)),
        )

    def _best_attack(self, state, player_id):
        best = None
        best_ratio = self.attack_ratio_threshold
        for t in state.territories_owned_by(player_id):
            armies = state.armies(t)
            if armies < 2:
                continue
            for enemy in state.enemy_neighbors(t, player_id):
                enemy_armies = state.armies(enemy)
                ratio = armies / enemy_armies
                if ratio > best_ratio:
                    best_ratio = ratio
                    best = (t, enemy)
        return best

    def choose_attack(self, state, player_id):
        best = self._best_attack(state, player_id)
        if best is None:
            return None
        frm, to = best
        dice = rules.max_attack_dice(state.armies(frm))
        return frm, to, dice

    def choose_defend_dice(self, state, player_id, max_dice):
        return max_dice

    def choose_capture_move(self, state, player_id, from_territory, to_territory, min_move, max_move):
        return max(min_move, round(max_move * self.aggression))

    def choose_fortify(self, state, player_id):
        owned = state.territories_owned_by(player_id)
        interiors = [t for t in owned if not state.enemy_neighbors(t, player_id) and state.armies(t) > 1]
        if not interiors:
            return None
        frm = max(interiors, key=lambda t: state.armies(t))
        border_neighbors = [
            n for n in state.neighbors(frm)
            if state.owner(n) == player_id and state.enemy_neighbors(n, player_id)
        ]
        if not border_neighbors:
            return None
        to = _weakest(state, border_neighbors)
        spare = state.armies(frm) - 1
        if spare < self.fortify_min_spare:
            return None
        return frm, to, spare


class ParameterizedStrategy(BaselineStrategy):
    """Same heuristic shape as BaselineStrategy but with its knobs exposed as
    a flat weight vector, so a training loop can search over them (e.g. the
    evolutionary self-play trainer in risk_sim.train)."""

    PARAM_NAMES = ('attack_ratio_threshold', 'aggression', 'fortify_min_spare')
    PARAM_BOUNDS = {
        'attack_ratio_threshold': (1.0, 3.0),
        'aggression': (0.3, 1.0),
        'fortify_min_spare': (1, 5),
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        weights = weights or {}
        super().__init__(
            attack_ratio_threshold=weights.get('attack_ratio_threshold', 1.5),
            aggression=weights.get('aggression', 0.8),
            fortify_min_spare=weights.get('fortify_min_spare', 2),
        )

    def as_weights(self) -> Dict[str, float]:
        return {name: getattr(self, name) for name in self.PARAM_NAMES}

    @classmethod
    def random_weights(cls) -> Dict[str, float]:
        return {
            name: random.uniform(*bounds)
            for name, bounds in cls.PARAM_BOUNDS.items()
        }
