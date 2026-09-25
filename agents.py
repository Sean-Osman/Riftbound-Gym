"""Scripted agents: fixed baselines to measure trained agents against.

GreedyAgent follows a few simple rules, using only the observation and the
legal actions like any other agent. It is a yardstick, not a good player: a
PPO agent should beat it clearly before its self-play results mean much.
"""

from __future__ import annotations

import random
from typing import Any

from env import _cost
from game import Action, ActionKind

# Spells that help the unit they target; every other targeted spell hurts it.
FRIENDLY_SPELLS = {"OGN-004", "OGN-104"}      # Cleave, Retreat
TIME_WARP = "OGN-122"


class GreedyAgent:
    """Develop the board, then attack where it wins:

    - keep the opening hand
    - play the most expensive card it can, without recycling runes if it can help it
    - aim harmful spells at the strongest enemy unit, helpful ones at its own
    - move units onto a battlefield when they'd win the fight (or it's empty)
    - in combat, kill as much enemy Might as possible
    - otherwise pass or end the turn
    """

    def __init__(self, seed: int | None = None, *, name: str = "Greedy"):
        self.rng = random.Random(seed)
        self.name = name

    def act(self, observation: dict[str, Any], legal: list[Action]) -> Action:
        info = _Board(observation)
        scored = [(self._score(a, info), self.rng.random(), i) for i, a in enumerate(legal)]
        return legal[max(scored)[2]]

    def _score(self, action: Action, board: _Board) -> float:
        kind = action.kind
        if kind is ActionKind.MULLIGAN:
            return 1.0 if not action.set_aside else 0.0
        if kind is ActionKind.CHOOSE:
            option = board.option(action.choice)
            return 0.0 if option.get("declines") else 1.0
        if kind is ActionKind.ASSIGN_DAMAGE:
            return sum(board.might.get(oid, 0) for oid, amount in action.damage
                       if amount >= board.toughness(oid))
        if kind is ActionKind.PLAY_CARD:
            return self._play(action, board)
        if kind is ActionKind.MOVE:
            return self._move(action, board)
        if kind is ActionKind.CONCEDE:
            return -100.0
        return 0.0                                      # PASS / END_TURN

    def _play(self, action: Action, board: _Board) -> float:
        card = board.cards.get(action.card_oid, {})
        energy, power = _cost(card.get("cost"))
        score = 5.0 + energy + power
        if card.get("card_id") == TIME_WARP:
            score += 5
        pay = action.payment
        if pay is not None:
            score -= 1.5 * len(pay.recycle)             # recycled runes leave the board
        if action.accelerate:
            score -= 1
        if action.targets:
            friendly = card.get("card_id") in FRIENDLY_SPELLS
            for oid in action.targets:
                mine = board.controller.get(oid) == board.me
                value = board.might.get(oid, 0)
                if mine != friendly:
                    return -10.0                        # hurting its own unit / helping theirs
                score += value
        if action.destination is not None:
            score += 1                                  # straight to a battlefield it holds
        return score

    def _move(self, action: Action, board: _Board) -> float:
        if action.destination is None:
            return -5.0                                 # retreating to base
        mine = sum(board.might.get(o, 0) for o in action.units)
        theirs = board.enemy_might(action.destination)
        if theirs == 0:
            held = board.battlefields[action.destination]["controller"] == board.me
            return -2.0 if held else 8.0 - 0.5 * len(action.units)
        if mine > theirs:
            return 9.0 + theirs - 0.5 * len(action.units)
        return -10.0


class _Board:
    """What GreedyAgent reads from an observation."""

    def __init__(self, obs: dict[str, Any]):
        self.obs = obs
        self.me = obs["viewer"]
        self.might = {int(k): v for k, v in obs["unit_might"].items()}
        self.battlefields = obs["battlefields"]
        self.cards: dict[int, dict[str, Any]] = {}
        self.controller: dict[int, int] = {}
        for p in obs["players"]:
            for zone in p["zones"].values():
                for view in zone["objects"]:
                    if not view.get("hidden"):
                        self.cards[view["oid"]] = view
                        self.controller[view["oid"]] = view["controller"]
        for bf in self.battlefields:
            for view in bf["units"]:
                self.cards[view["oid"]] = view
                self.controller[view["oid"]] = view["controller"]

    def enemy_might(self, index: int) -> int:
        return sum(self.might.get(v["oid"], 0) for v in self.battlefields[index]["units"]
                   if v["controller"] != self.me)

    def toughness(self, oid: int) -> int:
        view = self.cards.get(oid, {})
        return max(1, self.might.get(oid, 0) - view.get("damage", 0))

    def option(self, choice: int | None) -> dict[str, Any]:
        options = (self.obs["decision"] or {}).get("options", [])
        return options[choice] if choice is not None and choice < len(options) else {}
