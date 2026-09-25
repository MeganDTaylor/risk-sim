# -*- coding: utf-8 -*-
"""Neural strategy: two tiny MLPs (attack scoring, defense/reinforcement
priority) whose weights are found by neuroevolution (risk_sim.train_neural)
rather than hand-tuned. Unlike ParameterizedStrategy - which searches a fixed
7-term formula - the network can learn its own combination of the underlying
signals, including interactions the formula never expressed.

Card trades, defend-dice count, capture-move sizing, and which owned
territory to fortify *from* stay on the same small heuristics BaselineStrategy
uses: there's little to learn there, so spending genome capacity on them would
just slow down evolution on the two decisions that actually shape a game.
"""
import math
from typing import Optional

import numpy as np

from . import board, rules
from .state import GameState
from .strategies import Strategy, _border_territories, _completes_continent, _threat_score


class TinyMLP:
    """A single-hidden-layer network (tanh hidden, linear output) with its
    weights stored as one flat vector, so a genome is just a concatenation of
    these vectors and evolution never needs to know the network's shape."""

    def __init__(self, in_dim: int, hidden_dim: int):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.w1 = np.zeros((hidden_dim, in_dim))
        self.b1 = np.zeros(hidden_dim)
        self.w2 = np.zeros(hidden_dim)
        self.b2 = 0.0

    @property
    def size(self) -> int:
        return self.hidden_dim * self.in_dim + self.hidden_dim + self.hidden_dim + 1

    def forward(self, x: np.ndarray) -> float:
        hidden = np.tanh(self.w1 @ x + self.b1)
        return float(self.w2 @ hidden + self.b2)

    def to_vector(self) -> np.ndarray:
        return np.concatenate([self.w1.ravel(), self.b1, self.w2, [self.b2]])

    @classmethod
    def from_vector(cls, vector: np.ndarray, in_dim: int, hidden_dim: int) -> 'TinyMLP':
        net = cls(in_dim, hidden_dim)
        i = 0
        n1 = hidden_dim * in_dim
        net.w1 = np.asarray(vector[i:i + n1]).reshape(hidden_dim, in_dim); i += n1
        net.b1 = np.asarray(vector[i:i + hidden_dim]); i += hidden_dim
        net.w2 = np.asarray(vector[i:i + hidden_dim]); i += hidden_dim
        net.b2 = float(vector[i])
        return net

    @classmethod
    def random(cls, in_dim: int, hidden_dim: int, scale: float = 0.6) -> 'TinyMLP':
        net = cls(in_dim, hidden_dim)
        net.w1 = np.random.randn(hidden_dim, in_dim) * scale / math.sqrt(in_dim)
        net.w2 = np.random.randn(hidden_dim) * scale / math.sqrt(hidden_dim)
        return net


def _attack_features(state: GameState, player_id: int, frm: str, to: str) -> np.ndarray:
    defender_id = state.owner(to)
    continent_bonus = board.CONTINENT_BONUS[board.TERRITORY_TO_CONTINENT[to]]
    return np.array([
        rules.attacker_win_probability(state.armies(frm), state.armies(to)),
        state.armies(frm) / 20.0,
        1.0 if _completes_continent(state, player_id, to) else 0.0,
        continent_bonus / 7.0,
        1.0 if defender_id is not None and len(state.territories_owned_by(defender_id)) == 1 else 0.0,
        _threat_score(state, frm, player_id, exclude=to) / max(1, state.armies(frm)),
        _threat_score(state, to, player_id) / max(1, state.armies(to)),
        min(state.turn_number / 60.0, 2.0),
        len(state.territories_owned_by(player_id)) / 42.0,
        (len(state.territories_owned_by(defender_id)) / 42.0) if defender_id is not None else 0.0,
    ])


def _defense_features(state: GameState, player_id: int, territory: str) -> np.ndarray:
    enemy_neighbors = state.enemy_neighbors(territory, player_id)
    strongest_enemy_stack = max((state.armies(n) for n in enemy_neighbors), default=0)
    continent_bonus = board.CONTINENT_BONUS[board.TERRITORY_TO_CONTINENT[territory]]
    fall_probability = (
        rules.attacker_win_probability(strongest_enemy_stack, state.armies(territory))
        if strongest_enemy_stack else 0.0
    )
    return np.array([
        math.log(_threat_score(state, territory, player_id) + 1),
        state.armies(territory) / 20.0,
        1.0 / state.armies(territory),
        1.0 if any(_completes_continent(state, player_id, n) for n in enemy_neighbors) else 0.0,
        continent_bonus / 7.0,
        fall_probability,
        len(enemy_neighbors) / 6.0,
        len(state.territories_owned_by(player_id)) / 42.0,
        min(state.turn_number / 60.0, 2.0),
    ])


ATTACK_FEATURES = 10
DEFENSE_FEATURES = 9
HIDDEN_DIM = 8


class NeuralStrategy(Strategy):
    def __init__(self, attack_net: Optional[TinyMLP] = None, defense_net: Optional[TinyMLP] = None):
        self.attack_net = attack_net or TinyMLP.random(ATTACK_FEATURES, HIDDEN_DIM)
        self.defense_net = defense_net or TinyMLP.random(DEFENSE_FEATURES, HIDDEN_DIM)

    @staticmethod
    def genome_size() -> int:
        return TinyMLP(ATTACK_FEATURES, HIDDEN_DIM).size + TinyMLP(DEFENSE_FEATURES, HIDDEN_DIM).size

    def to_genome(self) -> np.ndarray:
        return np.concatenate([self.attack_net.to_vector(), self.defense_net.to_vector()])

    @classmethod
    def from_genome(cls, vector: np.ndarray) -> 'NeuralStrategy':
        split = TinyMLP(ATTACK_FEATURES, HIDDEN_DIM).size
        attack_net = TinyMLP.from_vector(vector[:split], ATTACK_FEATURES, HIDDEN_DIM)
        defense_net = TinyMLP.from_vector(vector[split:], DEFENSE_FEATURES, HIDDEN_DIM)
        return cls(attack_net, defense_net)

    @classmethod
    def random_genome(cls) -> np.ndarray:
        return cls().to_genome()

    def _defense_score(self, state: GameState, player_id: int, territory: str) -> float:
        return self.defense_net.forward(_defense_features(state, player_id, territory))

    # ---- Strategy interface -------------------------------------------
    def choose_initial_placement(self, state, player_id):
        owned = state.territories_owned_by(player_id)
        return max(owned, key=lambda t: self._defense_score(state, player_id, t))

    def choose_reinforcement_placement(self, state, player_id, army_count):
        borders = _border_territories(state, player_id) or state.territories_owned_by(player_id)
        scores = {t: math.exp(self._defense_score(state, player_id, t)) for t in borders}
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
        return max(valid_sets, key=lambda combo: len(rules.territory_bonus_armies(combo, owned)))

    def _best_attack(self, state, player_id):
        best = None
        best_score = 0.0
        for t in state.territories_owned_by(player_id):
            if state.armies(t) < 2:
                continue
            for enemy in state.enemy_neighbors(t, player_id):
                score = self.attack_net.forward(_attack_features(state, player_id, t, enemy))
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
        return max(min_move, round(max_move * 0.8))

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
        to = max(border_neighbors, key=lambda t: self._defense_score(state, player_id, t))
        spare = state.armies(frm) - 1
        if spare < 2:
            return None
        return frm, to, spare
