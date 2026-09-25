# -*- coding: utf-8 -*-
"""Records a played-out game turn-by-turn for external playback (e.g. a web
UI), separate from the engine itself so run_game/run_turn stay free of any
recording overhead when nobody needs a replay."""
import copy
from typing import Dict, List, Optional

from . import board
from .engine import initial_setup, run_turn
from .state import GameState


def _snapshot(state: GameState, turn: int, acting_player: Optional[int]) -> Dict:
    return {
        'turn': turn,
        'acting_player': acting_player,
        'territories': {
            name: {'owner': t['owner'], 'armies': t['armies']}
            for name, t in state.territories.items()
        },
        'players_alive': state.players_alive(),
    }


def record_game(strategies: List[object], max_turns: int = 300) -> Dict:
    """Plays one full game like engine.run_game, but returns a full turn-by-turn
    history instead of just the final winner - each entry is the complete
    board state after one player's turn, suitable for scrubbing through in a
    replay viewer."""
    state = GameState(strategies)
    initial_setup(state)
    snapshots = [_snapshot(state, turn=0, acting_player=None)]

    round_robin = list(range(state.num_players))
    turns_taken = 0
    idx = 0
    while turns_taken < max_turns and len(state.players_alive()) > 1:
        player_id = round_robin[idx % state.num_players]
        idx += 1
        if state.players[player_id].alive:
            run_turn(state, player_id)
            turns_taken += 1
            snapshots.append(_snapshot(state, turn=turns_taken, acting_player=player_id))

    alive = state.players_alive()
    winner = alive[0] if len(alive) == 1 else None
    return {
        'territories': list(board.ALL_TERRITORIES),
        'neighbors': board.TERRITORY_NEIGHBORS,
        'continents': board.CONTINENTS,
        'num_players': state.num_players,
        'winner': winner,
        'turns_taken': turns_taken,
        'snapshots': snapshots,
    }
