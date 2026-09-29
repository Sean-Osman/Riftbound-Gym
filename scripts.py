"""Card scripts: what each implemented card does.

The engine (game.py) looks cards up here by card_id. A spell is only playable
once it has an entry in SPELLS; units and gear play without one but have no
abilities unless TRIGGERS / ACTIVATED list some. Scripts only call Game's public
helpers, so this module doesn't import game.py.

Current coverage: every card in the Main Board and Rune Deck of decks/kaisa.json,
decks/annie.json, decks/master_yi.json and decks/miss_fortune.json.

Choices a script needs while it resolves (look at the top 3 and pick one, which
card to discard, how much to pay) are asked through `ask`: it returns an Ask
until every answer it needs is in `item.data`, and the engine turns each Ask
into a Decision for that player. It is called again after every answer, so it
must only depend on the game state and `item.data`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from cards import Cost, Pip

if TYPE_CHECKING:
    from game import ChainItem, Game
    from zones import CardInstance


Choice = tuple[str, Any]     # (label, value); a value of None declines a "you may" (383.3.a)


@dataclass(frozen=True)
class Ask:
    """A choice made while a spell or ability resolves (355.17). The answer goes in
    item.data[key]. `private` hides the options from other players (the cards a
    player looks at in their own deck or hand)."""
    seat: int
    key: str
    options: list[Choice]
    private: bool = False


# ---------------------------------------------------------------------------
# Spells
# ---------------------------------------------------------------------------

# Target kinds (355.5 / 355.9). "unit" means a unit on the board (355.9.a.1) and
# "spell" a spell on the chain (355.9.a.2).
UNIT = "unit"
UNIT_AT_BATTLEFIELD = "unit_at_battlefield"
SMALL_UNIT_AT_BATTLEFIELD = "small_unit_at_battlefield"     # "with 3 Might or less"
FRIENDLY_UNIT = "friendly_unit"
FRIENDLY_UNIT_AT_BATTLEFIELD = "friendly_unit_at_battlefield"
ENEMY_UNIT = "enemy_unit"
SPELL = "spell"
CHEAP_SPELL = "cheap_spell"            # "costs no more than [4] and no more than [A]"
BATTLEFIELD = "battlefield"

UNIT_KINDS = frozenset({UNIT, UNIT_AT_BATTLEFIELD, SMALL_UNIT_AT_BATTLEFIELD, FRIENDLY_UNIT,
                        FRIENDLY_UNIT_AT_BATTLEFIELD, ENEMY_UNIT})
SPELL_KINDS = frozenset({SPELL, CHEAP_SPELL})
AT_BATTLEFIELD_KINDS = frozenset({UNIT_AT_BATTLEFIELD, SMALL_UNIT_AT_BATTLEFIELD,
                                  FRIENDLY_UNIT_AT_BATTLEFIELD})


@dataclass(frozen=True)
class SpellScript:
    resolve: Callable[[Game, ChainItem], None]
    targets: tuple[str, ...] = ()      # one entry per target chosen as it's played
    banish: bool = False               # "Banish this": goes to banishment, not the trash
    min_targets: int | None = None     # "up to N" (355.13): fewer targets allowed, at least this many
    distinct: bool = False             # the targets must be different objects
    move: bool = False                 # moves its target: Action.destination is where to (355.4)
    ask: Callable[[Game, ChainItem], Ask | None] | None = None


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


def _pump(amount: int, *, draw: int = 0) -> Callable[[Game, ChainItem], None]:
    """Give a unit +X Might this turn."""
    def resolve(game: Game, item: ChainItem) -> None:
        for unit in game.targets(item):
            game.add_might(unit, amount)
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


def _bounce(game: Game, item: ChainItem) -> None:
    """Return a unit to its owner's hand."""
    for unit in game.targets(item):
        game.return_to_hand(unit)


def _to_base(game: Game, item: ChainItem) -> None:
    """Move units to base (Fight or Flight, Flash)."""
    units = game.targets(item)
    if units:
        game.move_units(item.controller, units, None)


def _move_target(ready: bool) -> Callable[[Game, ChainItem], None]:
    """Move a unit to the location chosen as it was played (Charm, Ride the Wind)."""
    def resolve(game: Game, item: ChainItem) -> None:
        for unit in game.targets(item):
            game.move_units(item.controller, [unit], item.data["destination"])
            if ready:
                game.ready(unit)
    return resolve


def _counter(game: Game, item: ChainItem) -> None:
    for spell in game.targets(item):
        game.counter(spell)


def _stun(game: Game, item: ChainItem) -> None:
    for unit in game.targets(item):
        game.stun(unit)


def _challenge(game: Game, item: ChainItem) -> None:
    """Both units deal damage equal to their Might to each other, at once. If either
    is no longer a legal target, neither deals damage (359.3.e.12)."""
    kinds = [kind for _, _, kind in item.targets]
    units = game.targets(item)
    if len(units) == 2 and kinds == [FRIENDLY_UNIT, ENEMY_UNIT]:
        mine, theirs = units
        a, b = max(0, game.might(mine)), max(0, game.might(theirs))
        game.deal(theirs, a)
        game.deal(mine, b)


def _channel_or_draw(n: int) -> Callable[[Game, ChainItem], None]:
    """Channel n runes exhausted. If you couldn't channel n, draw 1."""
    def resolve(game: Game, item: ChainItem) -> None:
        if game.channel(item.controller, n, exhausted=True) < n:
            game.draw(item.controller)
    return resolve


def _find_your_center(game: Game, item: ChainItem) -> None:
    game.draw(item.controller)
    game.channel(item.controller, 1, exhausted=True)


def _find_your_center_discount(game: Game, seat: int) -> int:
    """If an opponent's score is within 3 points of the Victory Score, [2] less."""
    close = any(game.points[s] >= game.victory_score - 3 for s in game.opponents(seat))
    return 2 if close else 0


def _stacked_deck_ask(game: Game, item: ChainItem) -> Ask | None:
    if "card" in item.data:
        return None
    top = game.deck_top(item.controller, 3)                 # 431.1.c: no Burn Out for looking
    if not top:
        return None
    options, seen = [], set()
    for obj in top:
        if obj.card.card_id not in seen:
            seen.add(obj.card.card_id)
            options.append((f"Put {obj.card.name} into your hand", (obj, obj.oid)))
    return Ask(item.controller, "card", options, private=True)


def _stacked_deck(game: Game, item: ChainItem) -> None:
    top = game.deck_top(item.controller, 3)
    choice = item.data.get("card")
    rest = list(top)
    if choice is not None:
        obj, oid = choice
        if obj.oid == oid and obj in top:
            game.put_in_hand(obj)
            rest.remove(obj)
    game.recycle(rest)


def _sabotage_ask(game: Game, item: ChainItem) -> Ask | None:
    if "card" in item.data:
        return None
    opponent = game.opponents(item.controller)[0]        # "Choose an opponent": the only one in a Duel
    cards = [o for o in game.hand(opponent) if not o.card.is_unit]
    game.reveal_hand(opponent)
    return Ask(item.controller, "card", _card_options(cards, "Recycle")) if cards else None


def _sabotage(game: Game, item: ChainItem) -> None:
    choice = item.data.get("card")
    if choice is not None and choice[0].oid == choice[1]:
        game.recycle([choice[0]])


def _acceptable_losses_ask(game: Game, item: ChainItem) -> Ask | None:
    for seat in game.turn_order():                        # each player chooses their own gear
        key = f"gear{seat}"
        if key not in item.data:
            gear = [o for o in game.permanents(seat) if o.card.is_gear]
            return Ask(seat, key, _card_options(gear, "Kill"))
    return None


def _acceptable_losses(game: Game, item: ChainItem) -> None:
    for seat in game.turn_order():
        choice = item.data.get(f"gear{seat}")
        if choice is not None and choice[0].oid == choice[1]:
            game.kill(choice[0])


def _invert_timelines(game: Game, item: ChainItem) -> None:
    for seat in game.turn_order():
        for obj in list(game.hand(seat)):
            game.discard(obj)
    for seat in game.turn_order():
        game.draw(seat, 4)


def _bullet_time_ask(game: Game, item: ChainItem) -> Ask | None:
    """Pay any amount of [A] (a game action on resolution, not a cost: 205)."""
    if "amount" in item.data or not game.targets(item):
        return None
    most = min(game.power_available(item.controller), 8)
    options = [(f"Pay {n} Power: deal {n} to each enemy unit there", ("amount", n)) for n in range(most + 1)]
    return Ask(item.controller, "amount", options)


def _bullet_time(game: Game, item: ChainItem) -> None:
    targets = game.targets(item)
    amount = (item.data.get("amount") or ("amount", 0))[1]
    if not targets or amount <= 0:
        return
    bf = game.battlefield_index(targets[0])
    paid = game.pay_power(item.controller, amount)
    for unit in game.units_at(bf):
        if unit.controller != item.controller:
            game.deal(unit, paid)


def _card_options(objs: list[CardInstance], verb: str) -> list[Choice]:
    """One option per distinct card: copies are interchangeable."""
    options, seen = [], set()
    for obj in objs:
        if obj.card.card_id not in seen:
            seen.add(obj.card.card_id)
            options.append((f"{verb} {obj.card.name}", (obj, obj.oid)))
    return options


SPELLS: dict[str, SpellScript] = {
    # Kai'Sa
    "OGN-004": SpellScript(_cleave, (UNIT,)),                              # Cleave
    "OGN-009": SpellScript(_deal(3), (UNIT_AT_BATTLEFIELD,)),              # Hextech Ray
    "OGN-024": SpellScript(_deal(4, draw=1), (UNIT_AT_BATTLEFIELD,)),      # Void Seeker
    "OGN-029": SpellScript(_deal(3), (UNIT, UNIT)),                        # Falling Star: "do this twice"
    "OGN-093": SpellScript(_shrink(4), (UNIT,)),                           # Smoke Screen
    "OGN-095": SpellScript(_shrink(1, draw=1), (UNIT,)),                   # Stupefy
    "OGN-104": SpellScript(_retreat, (FRIENDLY_UNIT,)),                    # Retreat
    "OGN-122": SpellScript(_time_warp, banish=True),                       # Time Warp
    # Annie
    "OGN-168": SpellScript(_to_base, (UNIT_AT_BATTLEFIELD,)),              # Fight or Flight
    "OGN-169": SpellScript(_bounce, (SMALL_UNIT_AT_BATTLEFIELD,)),         # Gust
    "OGN-172": SpellScript(_bounce, (UNIT_AT_BATTLEFIELD,)),               # Rebuke
    "OGN-173": SpellScript(_move_target(ready=True), (FRIENDLY_UNIT,), move=True),   # Ride the Wind
    "OGN-183": SpellScript(_stacked_deck, ask=_stacked_deck_ask),          # Stacked Deck
    "OGS-011": SpellScript(_to_base, (FRIENDLY_UNIT_AT_BATTLEFIELD, FRIENDLY_UNIT_AT_BATTLEFIELD),
                           min_targets=1, distinct=True),                  # Flash: "up to 2"
    # Master Yi
    "OGN-043": SpellScript(_move_target(ready=False), (ENEMY_UNIT,), move=True),     # Charm
    "OGN-045": SpellScript(_counter, (CHEAP_SPELL,)),                      # Defy
    "OGN-047": SpellScript(_find_your_center),                             # Find Your Center
    "OGN-050": SpellScript(_stun, (UNIT,)),                                # Rune Prison
    "OGN-058": SpellScript(_pump(2, draw=1), (UNIT,)),                     # Discipline
    "OGN-064": SpellScript(_counter, (SPELL,)),                            # Wind Wall
    "OGN-128": SpellScript(_challenge, (FRIENDLY_UNIT, ENEMY_UNIT)),       # Challenge
    "OGN-134": SpellScript(_channel_or_draw(1)),                           # Mobilize
    "OGN-138": SpellScript(_channel_or_draw(2)),                           # Catalyst of Aeons
    "OGN-156": SpellScript(_sabotage, ask=_sabotage_ask),                  # Sabotage
    # Miss Fortune
    "OGN-154": SpellScript(_pump(7), (UNIT,)),                             # Primal Strength
    "OGN-179": SpellScript(_acceptable_losses, ask=_acceptable_losses_ask),   # Acceptable Losses
    "OGN-201": SpellScript(_invert_timelines),                             # Invert Timelines
    "OGN-268": SpellScript(_bullet_time, (BATTLEFIELD,), ask=_bullet_time_ask),   # Bullet Time
}

# Energy discounts that depend on the game state (356.4)
ENERGY_DISCOUNT: dict[str, Callable[[Game, int], int]] = {
    "OGN-047": _find_your_center_discount,     # Find Your Center
}


# ---------------------------------------------------------------------------
# Triggered abilities (382-383)
# ---------------------------------------------------------------------------

# Events the engine emits, with the data it passes:
#   "played"     seat, obj         a card finished being played (359)
#   "chosen"     seat, targets     a spell was finalized choosing these objects (383.4.b)
#   "move"       seat, units, dest, first   units moved (420); `seat` is responsible (411),
#                                  `first` holds the oids moving for the first time this turn
#   "conquer"    seat, bf          a player conquered a battlefield (469.1)
#   "hold"       seat, bf          a player held a battlefield (469.2)
#   "defend"     seat, bf          a player became the defender in a combat (464.2.c)
#   "dies"       obj               a unit is about to go to the trash (808.1.d)
#   "killed"     obj               a permanent is about to be killed (428)
#   "leaves"     obj               a permanent is about to leave the board
#   "discard"    obj               a card was discarded (422.1.b); obj is in the trash
#   "beginning"  seat              the start of a player's Beginning Phase (315.2.a)
#   "end_turn"   seat              the Ending Step of a player's turn (317.1)


@dataclass(frozen=True)
class TriggerScript:
    event: str
    condition: Callable[[Game, CardInstance, dict], bool]
    resolve: Callable[[Game, ChainItem], None]
    text: str
    # "You may ..." abilities and targets: the options offered as the ability is put
    # on the chain. An ability with no options can't be put on the chain (355.8).
    choices: Callable[[Game, ChainItem], list[Choice]] | None = None
    ask: Callable[[Game, ChainItem], Ask | None] | None = None
    # "The first time ... each turn" (383.1): once per turn per source and controller
    once_per_turn: bool = False


def _mine(game: Game, source: CardInstance, data: dict) -> bool:
    return data["seat"] == source.controller


def _played_me(game: Game, source: CardInstance, data: dict) -> bool:
    return data["obj"] is source


def _moved_me(game: Game, source: CardInstance, data: dict) -> bool:
    return any(obj is source for obj in data["units"])


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
    units = [u for u in bf.units if u.controller == item.controller and u.card.is_unit]
    return [(f"Move {u.card.name} to base", (u, u.oid)) for u in units] + [("Don't move", None)]


def _row(game: Game, item: ChainItem) -> None:
    unit, oid = item.data["choice"]
    if unit.oid == oid and unit.zone is item.data["bf"].units:
        game.move_units(item.controller, [unit], None)


def _annie_legend(game: Game, item: ChainItem) -> None:
    """Ready up to 2 runes. Which ones doesn't matter: Energy has no domain and
    exhausted runes can still be recycled."""
    for rune in [r for r in game.runes(item.controller) if r.exhausted][:2]:
        game.ready(rune)


def _annie_stubborn_choices(game: Game, item: ChainItem) -> list[Choice]:
    spells = [o for o in game.trash(item.controller) if o.card.is_spell]
    return _card_options(spells, "Return")


def _return_from_trash(game: Game, item: ChainItem) -> None:
    obj, oid = item.data["choice"]
    if obj.oid == oid and obj in game.trash(item.controller):
        game.put_in_hand(obj)


def _discard_ask(game: Game, item: ChainItem) -> Ask | None:
    """Discard 1 (the discarding player chooses, 422.1.a)."""
    if "discard" in item.data:
        return None
    hand = game.hand(item.controller)
    return Ask(item.controller, "discard", _card_options(hand, "Discard"), private=True) if hand else None


def _discard_then_draw(game: Game, item: ChainItem) -> None:
    choice = item.data.get("discard")
    if choice is not None and choice[0].oid == choice[1]:
        game.discard(choice[0])
    game.draw(item.controller)


def _faefolk(game: Game, item: ChainItem) -> None:
    game.channel(item.controller, 2, exhausted=True)
    game.draw(item.controller)


def _treasure_trove(game: Game, item: ChainItem) -> None:
    game.draw(item.controller)
    game.channel(item.controller, 1, exhausted=True)


def _any_unit_choices(verb: str) -> Callable[[Game, ChainItem], list[Choice]]:
    def choices(game: Game, item: ChainItem) -> list[Choice]:
        return [(f"{verb} {game.describe_unit(u)}", (u, u.oid)) for u in game.distinct_units(game.units())]
    return choices


def _whiteflame(game: Game, item: ChainItem) -> None:
    unit, oid = item.data["choice"]
    if unit.oid == oid and unit.zone is not None and unit.zone.is_board:
        game.add_might(unit, 8)


def _volibear_moves(game: Game, source: CardInstance, data: dict) -> bool:
    """When an opponent moves to a battlefield other than mine."""
    dest = data["dest"]
    return (data["seat"] != source.controller and dest is not None
            and all(u.controller == data["seat"] for u in data["units"])
            and game.battlefield_index(source) != dest)


def _mf_choices(game: Game, item: ChainItem) -> list[Choice]:
    """Ready something else that's exhausted: a friendly unit, gear, rune or legend.
    Runes of one domain are interchangeable, and so are identical units."""
    me = item.source
    seat = item.controller
    tired = [o for o in game.permanents(seat) if o.exhausted and o is not me]
    options: list[Choice] = []
    for unit in game.distinct_units([o for o in tired if o.card.is_unit]):
        options.append((f"Ready {game.describe_unit(unit)}", (unit, unit.oid)))
    for gear in _card_options([o for o in tired if o.card.is_gear], "Ready"):
        options.append(gear)
    legend = game.legend(seat)
    if legend is not None and legend.exhausted:
        options.append((f"Ready {legend.card.name}", (legend, legend.oid)))
    for rune in _card_options([r for r in game.runes(seat) if r.exhausted], "Ready"):
        options.append(rune)
    return options + [("Don't ready anything", None)] if options else []


def _ready_choice(game: Game, item: ChainItem) -> None:
    obj, oid = item.data["choice"]
    if obj.oid == oid and obj.zone is not None and obj.zone.is_board:
        game.ready(obj)


def _mindsplitter_ask(game: Game, item: ChainItem) -> Ask | None:
    if "card" in item.data:
        return None
    opponent = game.opponents(item.controller)[0]
    hand = game.hand(opponent)
    game.reveal_hand(opponent)
    return Ask(item.controller, "card", _card_options(hand, "Discard")) if hand else None


def _mindsplitter(game: Game, item: ChainItem) -> None:
    choice = item.data.get("card")
    if choice is not None and choice[0].oid == choice[1]:
        game.discard(choice[0])


def _soulgorger_choices(game: Game, item: ChainItem) -> list[Choice]:
    """You may play a unit from your trash, ignoring its Energy cost: offered only
    if its Power cost can be paid."""
    seat = item.controller
    units = [o for o in game.trash(seat) if o.card.is_unit
             and game.can_pay(seat, o.card, Cost(0, o.card.cost.power))]
    options = _card_options(units, "Play")
    return options + [("Don't play a unit", None)] if options else []


def _play_location_ask(key: str) -> Callable[[Game, ChainItem], Ask | None]:
    """Where a unit played by this ability enters (355.2)."""
    def ask(game: Game, item: ChainItem) -> Ask | None:
        if "location" in item.data:
            return None
        unit = _unit_to_play(game, item, key)
        if unit is None:
            return None
        options = [(f"Play {unit.card.name} to {game.location_name(dest)}", ("location", dest))
                   for dest in game.unit_destinations(item.controller, unit.card)]
        return Ask(item.controller, "location", options)
    return ask


def _unit_to_play(game: Game, item: ChainItem, key: str) -> CardInstance | None:
    if key == "aurora":
        return game.reveal_until_unit(item.controller)
    choice = item.data.get(key)
    if choice is None or choice[0].oid != choice[1] or choice[0] not in game.trash(item.controller):
        return None
    return choice[0]


def _soulgorger(game: Game, item: ChainItem) -> None:
    unit = _unit_to_play(game, item, "choice")
    if unit is None:
        return
    seat = item.controller
    if game.pay_first_option(seat, unit.card, Cost(0, unit.card.cost.power)):
        game.play_unit(seat, unit, item.data["location"][1])


def _aurora(game: Game, item: ChainItem) -> None:
    """Reveal from the top until a unit, banish it and play it ignoring its cost;
    recycle the rest. With no unit in the deck, everything revealed is recycled."""
    seat = item.controller
    revealed = game.reveal_from_top_until_unit(seat)
    unit = revealed[-1] if revealed and revealed[-1].card.is_unit else None
    game.recycle([o for o in revealed if o is not unit])
    if unit is not None:
        game.banish(unit)
        game.play_unit(seat, unit, item.data["location"][1])


def _obelisk(game: Game, item: ChainItem) -> None:
    game.channel(item.data["seat"], 1)


def _dreaming_tree_chosen(game: Game, source: CardInstance, data: dict) -> bool:
    """A player chose a friendly unit here with a spell."""
    bf = game.battlefield_index(source)
    return any(t.card.is_unit and t.controller == data["seat"] and game.battlefield_index(t) == bf
               for t in data["targets"])


def _here(game: Game, source: CardInstance, data: dict) -> bool:
    return data["bf"].card is source


TRIGGERS: dict[str, tuple[TriggerScript, ...]] = {
    # Kai'Sa
    "OGN-027": (TriggerScript(                                             # Darius, Trifarian
        "played", lambda g, s, d: _mine(g, s, d) and g.cards_played[d["seat"]] == 2,
        _darius, "+2 Might this turn and ready"),),
    "OGN-103": (TriggerScript(                                             # Ravenbloom Student
        "played", lambda g, s, d: _mine(g, s, d) and d["obj"].card.is_spell,
        _ravenbloom, "+1 Might this turn"),),
    "OGN-116": (TriggerScript(                                             # Thousand-Tailed Watcher
        "played", _played_me, _watcher, "enemy units get -3 Might this turn"),),
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
        "hold", _here, _peak, "may channel 1 rune exhausted", _peak_choices),),
    "OGN-285": (TriggerScript(                                             # Reaver's Row
        "defend", _here, _row, "may move a friendly unit here to base", _row_choices),),
    # Annie
    "OGS-017": (TriggerScript(                                             # Annie, Dark Child
        "end_turn", _mine, _annie_legend, "ready up to 2 runes"),),
    "OGS-010": (TriggerScript(                                             # Annie, Stubborn
        "played", _played_me, _return_from_trash, "return a spell from the trash to hand",
        _annie_stubborn_choices),),
    "OGN-182": (                                                           # Scrapheap
        TriggerScript("played", _played_me, _draw_one, "played: draw 1"),
        TriggerScript("discard", lambda g, s, d: d["obj"] is s, _draw_one, "discarded: draw 1"),
        TriggerScript("killed", lambda g, s, d: d["obj"] is s, _draw_one, "killed: draw 1"),
    ),
    "OGN-185": (TriggerScript(                                             # Traveling Merchant
        "move", _moved_me, _discard_then_draw, "discard 1, then draw 1", ask=_discard_ask),),
    "OGN-298": (TriggerScript(                                             # Zaun Warrens
        "conquer", _here, _discard_then_draw, "discard 1, then draw 1", ask=_discard_ask),),
    "OGN-292": (TriggerScript(                                             # The Dreaming Tree
        "chosen", _dreaming_tree_chosen, _draw_one, "first friendly unit chosen here this turn: draw 1",
        once_per_turn=True),),
    # Master Yi
    "OGN-075": (TriggerScript(                                             # Tasty Faefolk
        "dies", lambda g, s, d: d["obj"] is s, _faefolk,
        "Deathknell: channel 2 runes exhausted and draw 1"),),
    "OGN-082": (TriggerScript(                                             # Whiteflame Protector
        "played", _played_me, _whiteflame, "give a unit +8 Might this turn", _any_unit_choices("+8 Might to")),),
    "OGN-158": (TriggerScript(                                             # Volibear, Imposing
        "move", _volibear_moves, _draw_one, "an opponent moved to another battlefield: draw 1"),),
    "OGN-160": (TriggerScript(                                             # Dazzling Aurora
        "end_turn", _mine, _aurora, "reveal until a unit and play it", ask=_play_location_ask("aurora")),),
    "OGN-284": (TriggerScript(                                             # Obelisk of Power
        "beginning", lambda g, s, d: g.beginning_phases[d["seat"]] == 1,
        _obelisk, "first Beginning Phase: channel 1 rune"),),
    # Miss Fortune
    "OGN-162": (TriggerScript(                                             # Miss Fortune, Captain
        "move", _moved_me, _ready_choice, "first move this turn: may ready something else",
        _mf_choices, once_per_turn=True),),
    "OGN-186": (TriggerScript(                                             # Treasure Trove
        "leaves", lambda g, s, d: d["obj"] is s, _treasure_trove,
        "left the board: draw 1 and channel 1 rune exhausted"),),
    "OGN-192": (TriggerScript(                                             # Mindsplitter
        "played", _played_me, _mindsplitter, "the opponent discards a card of your choice",
        ask=_mindsplitter_ask),),
    "OGN-196": (TriggerScript(                                             # Soulgorger
        "played", _played_me, _soulgorger, "may play a unit from your trash",
        _soulgorger_choices, ask=_play_location_ask("choice")),),
}


# ---------------------------------------------------------------------------
# Activated abilities (376-381)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AbilityScript:
    resolve: Callable[[Game, ChainItem], None]
    text: str
    cost: Cost = Cost()                  # Energy / Power, paid like a card's
    exhaust: bool = False                # [E]: exhaust the source (414.5)
    recycle_trash: int = 0               # "Recycle N from your trash" (416.3)
    targets: tuple[str, ...] = ()
    reaction: bool = False               # 813.1.c.2: usable in Closed States on any turn


def _vi(game: Game, item: ChainItem) -> None:
    if (me := game.source(item)) is not None:
        game.add_might(me, 1)


def _give_ganking(game: Game, item: ChainItem) -> None:
    for unit in game.targets(item):
        game.grant(unit, "Ganking", 1)


def _trove_kill(game: Game, item: ChainItem) -> None:
    if (me := game.source(item)) is not None:
        game.kill(me)


ACTIVATED: dict[str, tuple[AbilityScript, ...]] = {
    "OGN-036": (AbilityScript(_vi, "+1 Might this turn", recycle_trash=1),),         # Vi, Destructive
    # Miss Fortune, Bounty Hunter. Ganking only changes Standard Moves (810.1.c), so
    # only friendly units at battlefields are offered.
    "OGN-267": (AbilityScript(_give_ganking, "give a unit Ganking this turn", exhaust=True,
                              targets=(FRIENDLY_UNIT_AT_BATTLEFIELD,)),),
    "OGN-186": (AbilityScript(_trove_kill, "kill this", cost=Cost(0, (Pip.CHAOS,)), exhaust=True),),  # Treasure Trove
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

# "I enter ready" (369.3)
ENTERS_READY: frozenset[str] = frozenset({
    "OGS-009",             # Master Yi, Honed
})

# Extra places a unit may be played to (355.2.b): "open" battlefields (unoccupied and
# uncontrolled, 170.11.c) or "occupied enemy" ones.
PLAY_LOCATIONS: dict[str, str] = {
    "OGN-176": "open",               # Sneaky Deckhand
    "OGN-161": "occupied_enemy",     # Deadbloom Predator
}

# Legends: "While a friendly unit defends alone, it gets +X Might" (740.2.a)
DEFEND_ALONE_BONUS: dict[str, int] = {
    "OGS-019": 2,          # Master Yi, Wuju Bladesman
}

# Gear with "If a friendly unit would die, kill this instead. Heal that unit,
# exhaust it, and recall it." (367)
DEATH_REPLACEMENT: frozenset[str] = frozenset({
    "OGN-077",             # Zhonya's Hourglass
})

# Battlefields: "Increase the points needed to win the game by 1."
VICTORY_BONUS: dict[str, int] = {
    "OGN-276": 1,          # Aspirant's Climb
}

# Battlefields: "Units can't move from here to base."
NO_RETREAT: frozenset[str] = frozenset({
    "OGN-295",             # Vilemaw's Lair
})
