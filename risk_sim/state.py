# -*- coding: utf-8 -*-
"""Mutable game state: territories, players, cards. No rules logic here."""
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import board

WILD = 'wild'
CARD_TYPES = ['infantry', 'cavalry', 'artillery']


@dataclass(frozen=True)
class Card:
    kind: str
    territory: Optional[str] = None


def build_deck() -> List[Card]:
    # 14 of each of the 3 territory-linked types (one card per territory,
    # cycling type) plus 2 wilds - matches the real 44-card Risk deck.
    cards = [
        Card(CARD_TYPES[i % 3], territory)
        for i, territory in enumerate(board.ALL_TERRITORIES)
    ]
    cards.append(Card(WILD))
    cards.append(Card(WILD))
    return cards


@dataclass
class Player:
    player_id: int
    strategy: object
    cards: List[Card] = field(default_factory=list)
    alive: bool = True


class GameState:
    def __init__(self, strategies: List[object]):
        self.num_players = len(strategies)
        if not 3 <= self.num_players <= 6:
            raise ValueError('Classic Risk supports 3-6 players')
        self.players: Dict[int, Player] = {
            i: Player(i, strategies[i]) for i in range(self.num_players)
        }
        self.territories: Dict[str, Dict] = {
            name: {'owner': None, 'armies': 0} for name in board.ALL_TERRITORIES
        }
        self.deck: List[Card] = build_deck()
        random.shuffle(self.deck)
        self.discard: List[Card] = []
        self.trade_in_count = 0
        self.turn_number = 0

    # --- ownership / board queries -------------------------------------
    def owner(self, territory: str) -> Optional[int]:
        return self.territories[territory]['owner']

    def armies(self, territory: str) -> int:
        return self.territories[territory]['armies']

    def neighbors(self, territory: str) -> List[str]:
        return board.TERRITORY_NEIGHBORS[territory]

    def is_adjacent(self, a: str, b: str) -> bool:
        return b in board.TERRITORY_NEIGHBORS[a]

    def territories_owned_by(self, player_id: int) -> List[str]:
        return [t for t, s in self.territories.items() if s['owner'] == player_id]

    def continents_owned_by(self, player_id: int) -> List[str]:
        return [
            continent
            for continent, territories in board.CONTINENTS.items()
            if all(self.territories[t]['owner'] == player_id for t in territories)
        ]

    def enemy_neighbors(self, territory: str, player_id: int) -> List[str]:
        return [n for n in self.neighbors(territory) if self.territories[n]['owner'] != player_id]

    def players_alive(self) -> List[int]:
        return [p.player_id for p in self.players.values() if p.alive]

    # --- cards -----------------------------------------------------------
    def draw_card(self, player_id: int) -> None:
        if not self.deck:
            if not self.discard:
                return
            self.deck, self.discard = self.discard, []
            random.shuffle(self.deck)
        card = self.deck.pop()
        self.players[player_id].cards.append(card)

    def discard_cards(self, player_id: int, cards: List[Card]) -> None:
        player = self.players[player_id]
        for card in cards:
            player.cards.remove(card)
            self.discard.append(card)
