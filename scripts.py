"""Card scripts: what each implemented card does.

The engine (game.py) looks cards up here by card_id. A spell is only playable
once it has an entry in SPELLS; units play without one but have no abilities
unless TRIGGERS lists some. Scripts only call Game's public helpers, so this
module doesn't import game.py.

Current coverage: every card in decks/kaisa.json.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from game import ChainItem, Game
    from zones import CardInstance


# ---------------------------------------------------------------------------
# Spells
# ---------------------------------------------------------------------------

# Target kinds (355.5 / 355.9). "unit" means a unit on the board (355.9.a.1).
UNIT = "unit"
UNIT_AT_BATTLEFIELD = "unit_at_battlefield"
FRIENDLY_UNIT = "friendly_unit"


@dataclass(frozen=True)
class SpellScript:
    resolve: Callable[[Game, ChainItem], None]
    targets: tuple[str, ...] = ()      # one entry per target chosen as it's played
    banish: bool = False               # "Banish this": goes to banishment, not the trash


def _deal(amount: int, *, draw: int = 0) -> Callable[[Game, ChainItem], None]:
    def resolve(game: Game, item: ChainItem) -> None:
        for unit in game.targets(item):
            game.deal(unit, amount)
        if draw:
            game.draw(item.controller, draw)
    return resolve


def _shrink(amount: int, *, draw: int = 0) -> Callable[[Game, ChainItem], None]:
    """Give a unit -X Might this turn, to a minimum of 1 Might."""
    def resolve(game: Game, item: ChainItem) -> None:
        for unit in game.targets(item):
            game.add_might(unit, -amount, floor=1)
        if draw:
            game.draw(item.controller, draw)
    return resolve


def _cleave(game: Game, item: ChainItem) -> None:
    for unit in game.targets(item):
        game.grant(unit, "Assault", 3)


def _retreat(game: Game, item: ChainItem) -> None:
    for unit in game.targets(item):
        owner = unit.owner
        game.return_to_hand(unit)
        game.channel(owner, 1, exhausted=True)


def _time_warp(game: Game, item: ChainItem) -> None:
    game.add_extra_turn(item.controller)


SPELLS: dict[str, SpellScript] = {
    "OGN-004": SpellScript(_cleave, (UNIT,)),                              # Cleave
    "OGN-009": SpellScript(_deal(3), (UNIT_AT_BATTLEFIELD,)),              # Hextech Ray
    "OGN-024": SpellScript(_deal(4, draw=1), (UNIT_AT_BATTLEFIELD,)),      # Void Seeker
    "OGN-029": SpellScript(_deal(3), (UNIT, UNIT)),                        # Falling Star: "do this twice"
    "OGN-093": SpellScript(_shrink(4), (UNIT,)),                           # Smoke Screen
    "OGN-095": SpellScript(_shrink(1, draw=1), (UNIT,)),                   # Stupefy
    "OGN-104": SpellScript(_retreat, (FRIENDLY_UNIT,)),                    # Retreat
    "OGN-122": SpellScript(_time_warp, banish=True),                       # Time Warp
}


# ---------------------------------------------------------------------------
# Triggered abilities (382-383)
# ---------------------------------------------------------------------------

# Events the engine emits, with the data it passes:
#   "played"     seat, obj         a card finished being played (359)
#   "conquer"    seat, bf          a player conquered a battlefield (469.1)
#   "hold"       seat, bf          a player held a battlefield (469.2)
#   "defend"     seat, bf          a player became the defender in a combat (464.2.c)
#   "dies"       obj               a unit is about to go to the trash (808.1.d)
#   "beginning"  seat              the start of a player's Beginning Phase (315.2.a)

Choice = tuple[str, Any]     # (label, value); a value of None declines a "you may" (383.3.a)


@dataclass(frozen=True)
class TriggerScript:
    event: str
    condition: Callable[[Game, CardInstance, dict], bool]
    resolve: Callable[[Game, ChainItem], None]
    text: str
    # "You may ..." abilities: the options offered as the ability is put on the chain
    choices: Callable[[Game, ChainItem], list[Choice]] | None = None


def _mine(game: Game, source: CardInstance, data: dict) -> bool:
    return data["seat"] == source.controller


def _darius(game: Game, item: ChainItem) -> None:
    if (me := game.source(item)) is not None:
        game.add_might(me, 2)
        me.exhausted = False


def _ravenbloom(game: Game, item: ChainItem) -> None:
    if (me := game.source(item)) is not None:
        game.add_might(me, 1)


def _watcher(game: Game, item: ChainItem) -> None:
    for unit in game.units():
        if unit.controller != item.controller:
            game.add_might(unit, -3, floor=1)


def _draw_one(game: Game, item: ChainItem) -> None:
    game.draw(item.controller)


def _arena(game: Game, item: ChainItem) -> None:
    game.gain_points(item.data["seat"], 1)


def _peak_choices(game: Game, item: ChainItem) -> list[Choice]:
    return [("Channel 1 rune exhausted", True), ("Don't channel", None)]


def _peak(game: Game, item: ChainItem) -> None:
    game.channel(item.controller, 1, exhausted=True)


def _row_choices(game: Game, item: ChainItem) -> list[Choice]:
    bf = item.data["bf"]
    units = [u for u in bf.units if u.controller == item.controller]
    return [(f"Move {u.card.name} to base", (u, u.oid)) for u in units] + [("Don't move", None)]


def _row(game: Game, item: ChainItem) -> None:
    unit, oid = item.data["choice"]
    if unit.oid == oid and unit.zone is item.data["bf"].units:
        game.move_unit(unit, None)


TRIGGERS: dict[str, tuple[TriggerScript, ...]] = {
    "OGN-027": (TriggerScript(                                             # Darius, Trifarian
        "played", lambda g, s, d: _mine(g, s, d) and g.cards_played[d["seat"]] == 2,
        _darius, "+2 Might this turn and ready"),),
    "OGN-103": (TriggerScript(                                             # Ravenbloom Student
        "played", lambda g, s, d: _mine(g, s, d) and d["obj"].card.is_spell,
        _ravenbloom, "+1 Might this turn"),),
    "OGN-116": (TriggerScript(                                             # Thousand-Tailed Watcher
        "played", lambda g, s, d: d["obj"] is s,
        _watcher, "enemy units get -3 Might this turn"),),
    "OGN-039": (TriggerScript(                                             # Kai'Sa, Survivor
        "conquer", lambda g, s, d: _mine(g, s, d) and s.zone is d["bf"].units,
        _draw_one, "draw 1"),),
    "OGN-096": (TriggerScript(                                             # Watchful Sentry
        "dies", lambda g, s, d: d["obj"] is s,
        _draw_one, "Deathknell: draw 1"),),
    "OGN-290": (TriggerScript(                                             # The Arena's Greatest
        "beginning", lambda g, s, d: g.beginning_phases[d["seat"]] == 1,
        _arena, "first Beginning Phase: gain 1 point"),),
    "OGN-288": (TriggerScript(                                             # Startipped Peak
        "hold", lambda g, s, d: d["bf"].card is s,
        _peak, "may channel 1 rune exhausted", _peak_choices),),
    "OGN-285": (TriggerScript(                                             # Reaver's Row
        "defend", lambda g, s, d: d["bf"].card is s,
        _row, "may move a friendly unit here to base", _row_choices),),
}


# ---------------------------------------------------------------------------
# Other card rules
# ---------------------------------------------------------------------------

# Legends with "[E]: [Reaction] — Add [A]" that can help pay costs. The value is
# whether that Power can only be used to play spells.
LEGEND_POWER: dict[str, bool] = {
    "OGN-247": True,       # Kai'Sa, Daughter of the Void
}

# [Legion] — I cost [X] less (812): Energy discount once you've played another card this turn.
LEGION_DISCOUNT: dict[str, int] = {
    "OGN-012": 2,          # Noxus Hopeful
}
