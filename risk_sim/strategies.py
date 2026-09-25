# -*- coding: utf-8 -*-
"""Pluggable player strategies. Every decision point in the engine calls one
of these methods, so swapping a player's brain is just swapping the object
passed into GameState - human, heuristic, random, or a trained one."""
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from . import board, rules
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


def _threat_score(state: GameState, territory: str, player_id: int, exclude: Optional[str] = None) -> int:
    """Total enemy armies bordering territory (optionally ignoring one neighbor),
    used as a proxy for how exposed it is to a counter-attack."""
    return sum(
        state.armies(n) for n in state.neighbors(territory)
        if n != exclude and state.owner(n) != player_id
    )


def _completes_continent(state: GameState, player_id: int, territory: str) -> bool:
    """True if capturing territory would hand player_id every territory in its continent."""
    continent = board.TERRITORY_TO_CONTINENT[territory]
    return all(
        state.owner(t) == player_id for t in board.CONTINENTS[continent] if t != territory
    )


def _would_eliminate(state: GameState, defender_id: Optional[int]) -> bool:
    return defender_id is not None and len(state.territories_owned_by(defender_id)) == 1


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
    """Greedy heuristic that scores candidate moves on multiple weighted
    features (army ratio, continent completion, elimination, exposure, threat)
    rather than a single ratio, so evolution over the weights (see
    ParameterizedStrategy) can favor different tactical mixes - e.g. rushing
    a continent bonus or hunting a weak player's last territory - instead of
    just tuning how cautious/aggressive a single-feature heuristic is."""

    def __init__(self, attack_score_threshold: float = 1.5, aggression: float = 0.8,
                 fortify_min_spare: int = 2, continent_weight: float = 1.0,
                 elimination_weight: float = 1.5, safety_weight: float = 0.5,
                 reinforcement_weakness_weight: float = 2.0):
        self.attack_score_threshold = attack_score_threshold
        self.aggression = aggression
        self.fortify_min_spare = fortify_min_spare
        self.continent_weight = continent_weight
        self.elimination_weight = elimination_weight
        self.safety_weight = safety_weight
        self.reinforcement_weakness_weight = reinforcement_weakness_weight

    def choose_initial_placement(self, state, player_id):
        borders = _border_territories(state, player_id)
        owned = borders or state.territories_owned_by(player_id)
        return _weakest(state, owned)

    def _reinforcement_score(self, state: GameState, player_id: int, territory: str) -> float:
        threat = _threat_score(state, territory, player_id)
        weakness = self.reinforcement_weakness_weight / state.armies(territory)
        continent_incentive = self.continent_weight if any(
            _completes_continent(state, player_id, n)
            for n in state.enemy_neighbors(territory, player_id)
        ) else 0.0
        return threat + weakness + continent_incentive

    def choose_reinforcement_placement(self, state, player_id, army_count):
        borders = _border_territories(state, player_id) or state.territories_owned_by(player_id)
        scores = {t: max(self._reinforcement_score(state, player_id, t), 0.01) for t in borders}
        total = sum(scores.values())
        raw = {t: army_count * s / total for t, s in scores.items()}
        placement = {t: int(n) for t, n in raw.items()}
        remaining = army_count - sum(placement.values())
        by_remainder = sorted(borders, key=lambda t: raw[t] - int(raw[t]), reverse=True)
        for t in (by_remainder * (remaining // len(by_remainder) + 1))[:remaining]:
            placement[t] += 1
        return {t: n for t, n in placement.items() if n > 0}

    def choose_card_trade(self, state, player_id, mandatory):
        valid_sets = rules.find_valid_sets(state.players[player_id].cards)
        if not valid_sets:
            return None
        owned = state.territories_owned_by(player_id)
        return max(
            valid_sets,
            key=lambda combo: len(rules.territory_bonus_armies(combo, owned)),
        )

    def _attack_score(self, state: GameState, player_id: int, frm: str, to: str) -> float:
        score = state.armies(frm) / state.armies(to)
        if _completes_continent(state, player_id, to):
            score += self.continent_weight
        if _would_eliminate(state, state.owner(to)):
            score += self.elimination_weight
        score -= self.safety_weight * _threat_score(state, frm, player_id, exclude=to) / state.armies(frm)
        return score

    def _best_attack(self, state, player_id):
        best = None
        best_score = self.attack_score_threshold
        for t in state.territories_owned_by(player_id):
            if state.armies(t) < 2:
                continue
            for enemy in state.enemy_neighbors(t, player_id):
                score = self._attack_score(state, player_id, t, enemy)
                if score > best_score:
                    best_score = score
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
        to = max(border_neighbors, key=lambda t: _threat_score(state, t, player_id) - state.armies(t))
        spare = state.armies(frm) - 1
        if spare < self.fortify_min_spare:
            return None
        return frm, to, spare


class ParameterizedStrategy(BaselineStrategy):
    """Same heuristic shape as BaselineStrategy but with its knobs exposed as
    a flat weight vector, so a training loop can search over them (e.g. the
    evolutionary self-play trainer in risk_sim.train)."""

    PARAM_NAMES = (
        'attack_score_threshold', 'aggression', 'fortify_min_spare',
        'continent_weight', 'elimination_weight', 'safety_weight',
        'reinforcement_weakness_weight',
    )
    PARAM_BOUNDS = {
        'attack_score_threshold': (1.0, 3.0),
        'aggression': (0.3, 1.0),
        'fortify_min_spare': (1, 8),
        'continent_weight': (0.0, 3.0),
        'elimination_weight': (0.0, 6.0),
        'safety_weight': (0.0, 2.0),
        'reinforcement_weakness_weight': (0.0, 20.0),
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        weights = weights or {}
        defaults = BaselineStrategy()
        super().__init__(
            attack_score_threshold=weights.get('attack_score_threshold', defaults.attack_score_threshold),
            aggression=weights.get('aggression', defaults.aggression),
            fortify_min_spare=int(weights.get('fortify_min_spare', defaults.fortify_min_spare)),
            continent_weight=weights.get('continent_weight', defaults.continent_weight),
            elimination_weight=weights.get('elimination_weight', defaults.elimination_weight),
            safety_weight=weights.get('safety_weight', defaults.safety_weight),
            reinforcement_weakness_weight=weights.get(
                'reinforcement_weakness_weight', defaults.reinforcement_weakness_weight
            ),
        )

    def as_weights(self) -> Dict[str, float]:
        return {name: getattr(self, name) for name in self.PARAM_NAMES}

    @classmethod
    def random_weights(cls) -> Dict[str, float]:
        return {
            name: random.uniform(*bounds)
            for name, bounds in cls.PARAM_BOUNDS.items()
        }
