# -*- coding: utf-8 -*-
"""Turn engine: wires state + rules + strategies into a full game loop.

Deliberate simplifications from the physical board game, called out because
they're the sort of thing that's easy to forget was a judgement call:
  - Fortifying moves armies between a single adjacent pair once per turn
    (the original 1959 rule), not along a chain of owned territories.
  - The 2-player "neutral army" variant isn't implemented; 3-6 human/AI
    players only.
  - Mandatory card trade-in uses one hand-size threshold (5) both at the
    start of a turn and immediately after absorbing an eliminated player's
    cards, rather than the two slightly different thresholds some editions
    use for those two cases.
"""
import random
from typing import List, Optional, Tuple

from . import board, rules
from .state import GameState


def _apply_placement(state: GameState, player_id: int, placement, expected_total: int) -> None:
    owned = set(state.territories_owned_by(player_id))
    total = sum(placement.values())
    if total != expected_total:
        raise ValueError(f'player {player_id} placed {total} armies, expected {expected_total}')
    for territory, count in placement.items():
        if territory not in owned:
            raise ValueError(f'player {player_id} cannot place armies on unowned {territory}')
        if count < 0:
            raise ValueError('cannot place a negative number of armies')
        state.territories[territory]['armies'] += count


def _apply_card_trade(state: GameState, player_id: int, combo) -> Tuple[int, List[str]]:
    bonus = rules.card_set_bonus(state.trade_in_count)
    state.trade_in_count += 1
    owned = state.territories_owned_by(player_id)
    territory_bonus = rules.territory_bonus_armies(combo, owned)
    state.discard_cards(player_id, list(combo))
    for territory in territory_bonus:
        state.territories[territory]['armies'] += 2
    return bonus, territory_bonus


def _force_trade_down(state: GameState, player_id: int) -> None:
    player = state.players[player_id]
    while len(player.cards) >= 5:
        valid_sets = rules.find_valid_sets(player.cards)
        if not valid_sets:
            break
        combo = player.strategy.choose_card_trade(state, player_id, mandatory=True)
        if combo not in valid_sets:
            combo = valid_sets[0]
        bonus, _ = _apply_card_trade(state, player_id, combo)
        placement = player.strategy.choose_reinforcement_placement(state, player_id, bonus)
        _apply_placement(state, player_id, placement, bonus)


def _check_elimination(state: GameState, defender_id: int, conqueror_id: int, verbose: bool) -> None:
    if state.territories_owned_by(defender_id):
        return
    state.players[defender_id].alive = False
    conqueror = state.players[conqueror_id]
    conqueror.cards.extend(state.players[defender_id].cards)
    state.players[defender_id].cards = []
    if verbose:
        print(f'  Player {defender_id} eliminated by player {conqueror_id}')
    _force_trade_down(state, conqueror_id)


def _resolve_attack(state: GameState, player_id: int, frm: str, to: str, dice: int, verbose: bool) -> bool:
    if state.owner(frm) != player_id:
        raise ValueError(f'player {player_id} does not own {frm}')
    defender_id = state.owner(to)
    if defender_id == player_id:
        raise ValueError(f'{to} is not an enemy territory')
    if not state.is_adjacent(frm, to):
        raise ValueError(f'{frm} and {to} are not adjacent')
    max_dice = rules.max_attack_dice(state.armies(frm))
    if not 1 <= dice <= max_dice:
        raise ValueError(f'illegal attack dice count {dice} from {frm} (max {max_dice})')

    max_def_dice = rules.max_defend_dice(state.armies(to))
    defend_dice = state.players[defender_id].strategy.choose_defend_dice(state, defender_id, max_def_dice)
    defend_dice = max(1, min(defend_dice, max_def_dice))

    attacker_losses, defender_losses = rules.resolve_combat(dice, defend_dice)
    state.territories[frm]['armies'] -= attacker_losses
    state.territories[to]['armies'] -= defender_losses

    conquered = state.armies(to) <= 0
    if conquered:
        state.territories[to]['owner'] = player_id
        state.territories[to]['armies'] = 0
        min_move = dice
        max_move = state.armies(frm) - 1
        move_amount = state.players[player_id].strategy.choose_capture_move(
            state, player_id, frm, to, min_move, max_move
        )
        move_amount = max(min_move, min(move_amount, max_move))
        state.territories[frm]['armies'] -= move_amount
        state.territories[to]['armies'] += move_amount
        if verbose:
            print(f'  Player {player_id} conquers {to} from player {defender_id}')
        _check_elimination(state, defender_id, player_id, verbose)
    return conquered


def _apply_fortify(state: GameState, player_id: int, move: Tuple[str, str, int]) -> None:
    frm, to, amount = move
    if state.owner(frm) != player_id or state.owner(to) != player_id:
        raise ValueError('fortify must move between two territories you own')
    if not state.is_adjacent(frm, to):
        raise ValueError('fortify must move between adjacent territories')
    if amount < 1 or state.armies(frm) - amount < 1:
        raise ValueError('fortify must leave at least 1 army behind')
    state.territories[frm]['armies'] -= amount
    state.territories[to]['armies'] += amount


def initial_setup(state: GameState) -> None:
    territories = list(board.ALL_TERRITORIES)
    random.shuffle(territories)
    order = list(range(state.num_players))
    for i, territory in enumerate(territories):
        owner = order[i % state.num_players]
        state.territories[territory]['owner'] = owner
        state.territories[territory]['armies'] = 1

    total_armies = board.STARTING_ARMIES[state.num_players]
    remaining = {p: total_armies - len(state.territories_owned_by(p)) for p in order}
    while any(count > 0 for count in remaining.values()):
        for p in order:
            if remaining[p] <= 0:
                continue
            territory = state.players[p].strategy.choose_initial_placement(state, p)
            if state.owner(territory) != p:
                territory = state.territories_owned_by(p)[0]
            state.territories[territory]['armies'] += 1
            remaining[p] -= 1


def run_turn(state: GameState, player_id: int, verbose: bool = False) -> None:
    player = state.players[player_id]
    if not player.alive:
        return
    state.turn_number += 1

    _force_trade_down(state, player_id)

    bonus_armies = 0
    valid_sets = rules.find_valid_sets(player.cards)
    if valid_sets:
        combo = player.strategy.choose_card_trade(state, player_id, mandatory=False)
        if combo is not None:
            if combo not in valid_sets:
                raise ValueError(f'player {player_id} offered an invalid card set')
            bonus_armies, _ = _apply_card_trade(state, player_id, combo)

    reinforcements = rules.reinforcement_count(state, player_id) + bonus_armies
    placement = player.strategy.choose_reinforcement_placement(state, player_id, reinforcements)
    _apply_placement(state, player_id, placement, reinforcements)

    conquered_any = False
    while True:
        if len(state.players_alive()) == 1:
            return
        move = player.strategy.choose_attack(state, player_id)
        if move is None:
            break
        frm, to, dice = move
        if _resolve_attack(state, player_id, frm, to, dice, verbose):
            conquered_any = True

    if conquered_any:
        state.draw_card(player_id)

    fortify_move = player.strategy.choose_fortify(state, player_id)
    if fortify_move is not None:
        _apply_fortify(state, player_id, fortify_move)


def run_game(strategies: List[object], max_turns: int = 500, verbose: bool = False) -> Tuple[Optional[int], GameState]:
    state = GameState(strategies)
    initial_setup(state)
    round_robin = list(range(state.num_players))
    turns_taken = 0
    idx = 0
    while turns_taken < max_turns and len(state.players_alive()) > 1:
        player_id = round_robin[idx % state.num_players]
        idx += 1
        if state.players[player_id].alive:
            run_turn(state, player_id, verbose)
            turns_taken += 1
    alive = state.players_alive()
    winner = alive[0] if len(alive) == 1 else None
    if verbose:
        print(f'Game over after {turns_taken} turns. Winner: {winner}')
    return winner, state
