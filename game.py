"""Game engine: decks, setup, the turn, paying costs, the chain, combat and agents.

Implemented: setup with the mulligan (110-118), the phases of the turn with the
Ending Step and extra turns (314-317, 734-738), runes and rune pools (160-168),
paying costs with Accelerate, Legion and Deflect (356-357, 805, 809, 812),
playing cards, triggered and activated abilities on the chain with priority
(327-340, 349-359, 376-383), choices made on resolution (355.17), targets
(355.5-355.10, 359.3.e), "this turn" effects, Standard Moves with Ganking (144,
810), moves and recalls (445-458), hiding and playing from Hidden (421, 811),
battlefield control, showdowns and focus (190, 341-348), combat with Assault,
Shield, Tank and stuns (423, 459-466, 807, 814, 815), countering (425), gear,
discarding and revealing (422, 424), cleanups (323), Deathknell (808),
replacement effects on death (367), scoring (467-471) and Burn Out (431).

What each card does lives in scripts.py. Every card in the decklists in decks/
is scripted; any other spell is unplayable and any other unit has no abilities.

The decision loop is `acting_player` / `legal_actions` / `step` / `observation`.
Everything that needs no decision runs inside `step`, and when a player's only
option is to pass, end their turn, or make a forced choice, the engine does it
for them. Game state is plain data (no generators or callbacks; chain items
point at scripts by key), so games deep-copy and pickle.

    python3 game.py            # headless: random agent vs random agent
"""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from cards import HIDE_COST, CardDef, Cost, Domain, Keyword, Pip, deck_errors, load_card_pool
from scripts import (
    ACTIVATED, AT_BATTLEFIELD_KINDS, BATTLEFIELD, CHEAP_SPELL, DEATH_REPLACEMENT, DEFEND_ALONE_BONUS,
    ENEMY_UNIT, ENERGY_DISCOUNT, ENTERS_READY, FRIENDLY_UNIT, FRIENDLY_UNIT_AT_BATTLEFIELD,
    LEGEND_POWER, LEGION_DISCOUNT, NO_RETREAT, PLAY_LOCATIONS, SMALL_UNIT_AT_BATTLEFIELD, SPELL_KINDS,
    SPELLS, TRIGGERS, UNIT_AT_BATTLEFIELD, UNIT_KINDS, VICTORY_BONUS, AbilityScript, Ask, SpellScript,
    TriggerScript,
)
from zones import Battlefield, CardInstance, PlayerZones, Power, Zone, ZoneKind

ROOT = Path(__file__).parent
VICTORY_SCORE = 8          # 485.3 (battlefields like Aspirant's Climb can raise it: Game.victory_score)
OPENING_HAND = 4           # 116
MULLIGAN_MAX = 2           # 117.1
CHANNEL_PER_TURN = 2       # 315.3.b
SECOND_PLAYER_EXTRA = 1    # 485.7: extra rune on the second player's first Channel Phase


# ---------------------------------------------------------------------------
# Decks
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Deck:
    """103: a player's full deck. `main` excludes the Chosen Champion."""
    name: str
    legend: CardDef
    champion: CardDef
    main: tuple[CardDef, ...]
    runes: tuple[CardDef, ...]
    battlefields: tuple[CardDef, ...]

    def errors(self) -> list[str]:
        return deck_errors(self.legend, self.champion, self.main, self.runes, self.battlefields)

    @classmethod
    def load(cls, path: str | Path, pool: dict[str, CardDef]) -> Deck:
        """Load an exported decklist:

            {"metadata": {"name", "author"},
             "deck": {"Main Board": [{"id": "OGN-039", "count": 1}, ...],
                      "Rune Deck": [{"id": "OGN-007", "count": 6}, ...],
                      "Side Board": [...]}}

        The Rune Deck (103.3) holds the runes and nothing else. Main Board holds
        everything else and is sorted by card type (103): the legend, the
        battlefields, and the main deck. One copy of the champion unit whose
        champion tag matches the legend is set aside as the Chosen Champion
        (103.2.a); any further copies stay in the main deck. The Side Board is
        ignored - sideboarding isn't a Core Rules deck zone.
        """
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
        sections = spec["deck"] if "deck" in spec else spec

        def expand(name: str) -> list[CardDef]:
            return [pool[e["id"]] for e in sections.get(name, []) for _ in range(e.get("count", 1))]

        cards = expand("Main Board")
        runes = expand("Rune Deck")
        if misplaced := sorted({c.name for c in cards if c.is_rune}):
            raise ValueError(f"runes belong in the Rune Deck, not the Main Board: {misplaced}")
        if misplaced := sorted({c.name for c in runes if not c.is_rune}):
            raise ValueError(f"the Rune Deck can only hold runes: {misplaced}")

        legends = [c for c in cards if c.is_legend]
        if len(legends) != 1:
            raise ValueError(f"decklist needs exactly 1 legend, found {[c.name for c in legends]}")
        legend = legends[0]
        champions = [c for c in cards if c.is_champion_unit and c.champion_tag == legend.champion_tag]
        if not champions:
            raise ValueError(f"decklist has no champion unit with the tag {legend.champion_tag!r}")
        champion = champions[0]

        main, battlefields = [], []
        champion_taken = False
        for card in cards:
            if card is legend:
                continue
            if card.is_battlefield:
                battlefields.append(card)
            elif card is champion and not champion_taken:
                champion_taken = True          # the Chosen Champion, not a main deck slot
            else:
                main.append(card)
        return cls(
            name=spec.get("metadata", {}).get("name", Path(path).stem),
            legend=legend,
            champion=champion,
            main=tuple(main),
            runes=tuple(runes),
            battlefields=tuple(battlefields),
        )


# ---------------------------------------------------------------------------
# Actions and agents
# ---------------------------------------------------------------------------

class ActionKind(Enum):
    MULLIGAN = "mulligan"      # 117
    PLAY_CARD = "play_card"    # 349
    MOVE = "move"              # 144: Standard Move
    ASSIGN_DAMAGE = "assign_damage"   # 465.2.c: combat damage assignment
    CHOOSE = "choose"          # a trigger's choice (383.3.a), which showdown starts (323.12), or a choice on resolution
    HIDE = "hide"              # 421: put a card with Hidden facedown at a battlefield
    ACTIVATE = "activate"      # 376: an activated ability
    PASS = "pass"              # 338.1.b / 347.2: pass priority on a chain, or focus in a showdown
    END_TURN = "end_turn"      # 316.9
    CONCEDE = "concede"        # 650


@dataclass(frozen=True)
class Payment:
    """How a card's cost gets paid (357). Each rune in `exhaust` adds [1] and each
    rune in `recycle` adds one Power of its domain (164.2); `legend` uses the
    legend's Add ability (LEGEND_POWER). A rune can be in both."""
    exhaust: tuple[int, ...] = ()
    recycle: tuple[int, ...] = ()
    legend: bool = False

    def to_json(self) -> dict[str, Any]:
        return {"exhaust": list(self.exhaust), "recycle": list(self.recycle), "legend": self.legend}


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    label: str
    card_oid: int | None = None
    payment: Payment | None = None
    set_aside: tuple[int, ...] = ()     # MULLIGAN: the cards to set aside
    units: tuple[int, ...] = ()         # MOVE: the units moving together (144.3)
    # PLAY_CARD / MOVE / HIDE: battlefield index, None for base. For a spell that
    # moves its target (Charm), where the target moves to (355.4).
    destination: int | None = None
    damage: tuple[tuple[int, int], ...] = ()   # ASSIGN_DAMAGE: (unit oid, amount) pairs
    targets: tuple[int, ...] = ()       # PLAY_CARD / ACTIVATE: chosen targets, in the script's order (355.5)
    accelerate: bool = False            # PLAY_CARD: pay Accelerate (805)
    choice: int | None = None           # CHOOSE: index into the offered options; ACTIVATE: which ability

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind.value, "label": self.label}
        if self.card_oid is not None:
            result["card_oid"] = self.card_oid
        if self.payment is not None:
            result["payment"] = self.payment.to_json()
        if self.kind is ActionKind.MULLIGAN:
            result["set_aside"] = list(self.set_aside)
        if self.kind is ActionKind.MOVE:
            result["units"] = list(self.units)
        if self.kind in (ActionKind.MOVE, ActionKind.PLAY_CARD, ActionKind.HIDE):
            result["destination"] = self.destination
        if self.kind in (ActionKind.PLAY_CARD, ActionKind.ACTIVATE):
            result["targets"] = list(self.targets)
        if self.kind is ActionKind.PLAY_CARD:
            result["accelerate"] = self.accelerate
        if self.kind is ActionKind.ASSIGN_DAMAGE:
            result["damage"] = [list(pair) for pair in self.damage]
        if self.kind in (ActionKind.CHOOSE, ActionKind.ACTIVATE):
            result["choice"] = self.choice
        return result


class Agent(Protocol):
    name: str

    def act(self, observation: dict[str, Any], legal: list[Action]) -> Action:
        """Pick one of `legal`. `observation` only holds what this seat may know."""
        ...


class RandomAgent:
    def __init__(self, seed: int | None = None, *, may_concede: bool = False, name: str = "Random"):
        self.rng = random.Random(seed)
        self.may_concede = may_concede
        self.name = name

    def act(self, observation: dict[str, Any], legal: list[Action]) -> Action:
        options = [a for a in legal if self.may_concede or a.kind is not ActionKind.CONCEDE]
        return self.rng.choice(options or legal)


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------

class Phase(Enum):
    PLAYING = "playing"
    GAME_OVER = "game_over"


class TurnPhase(Enum):
    MULLIGAN = "mulligan"      # 117, before the first turn
    AWAKEN = "awaken"          # 315.1
    BEGINNING = "beginning"    # 315.2.a: Beginning Step
    SCORING = "scoring"        # 315.2.b: Scoring Step
    CHANNEL = "channel"        # 315.3
    DRAW = "draw"              # 315.4
    MAIN = "main"              # 316
    ENDING = "ending"          # 317


START_OF_TURN = (TurnPhase.AWAKEN, TurnPhase.BEGINNING, TurnPhase.SCORING,
                 TurnPhase.CHANNEL, TurnPhase.DRAW)


@dataclass(eq=False)
class ChainItem:
    """329: a card being played, a triggered ability or an activated ability, on
    the chain or waiting to be put there. Abilities point at their script by key
    so the state stays plain data."""
    controller: int
    obj: CardInstance | None = None                 # the card, for spells
    trigger: tuple[str, int] | None = None          # (card_id, index) into scripts.TRIGGERS
    ability: tuple[str, int] | None = None          # (card_id, index) into scripts.ACTIVATED
    source: CardInstance | None = None              # the ability's source
    source_oid: int | None = None
    targets: list[tuple[CardInstance, int, str]] = field(default_factory=list)   # (object, oid, kind)
    data: dict[str, Any] = field(default_factory=dict)
    label: str = ""


@dataclass(eq=False)
class Decision:
    """A choice made outside of priority."""
    seat: int
    kind: str                           # "trigger", "showdown" or "resolve"
    options: list[tuple[str, Any]]      # (label, value)
    item: ChainItem | None = None
    key: str = "choice"                 # "resolve": where the answer goes in item.data
    private: bool = False               # other players only learn how many options there are


class Game:
    def __init__(self, decks: list[Deck], names: list[str] | None = None, *, seed: int | None = None,
                 allow_concede: bool = False, cache_actions: bool = False):
        """`allow_concede` offers Concede (650) as an action. It is off by default so
        RL agents never see it: conceding would end early training games at random
        and skew matchup win rates. The browser sim turns it on for humans.

        `cache_actions` reuses the legal actions between `step` calls instead of
        enumerating them again (about 4x fewer enumerations per decision). Only use
        it when nothing changes the game except `step`, as in RL rollouts: tests
        that edit the state by hand would see stale actions."""
        if len(decks) != 2:
            raise ValueError("only 1v1 Duel (485) is supported")
        n = len(decks)
        self.rng = random.Random(seed)
        self.seed = seed
        self.allow_concede = allow_concede
        self.cache_actions = cache_actions
        self._legal_cache: tuple[int, list[Action]] | None = None
        self.names = names or [f"Player {i + 1}" for i in range(n)]
        self.decks = decks
        self.players = [PlayerZones(i) for i in range(n)]
        self.points = [0] * n
        self.battlefields: list[Battlefield] = []
        self.set_aside: list[CardInstance] = []           # unused battlefields (485.5)
        self.chain = Zone(ZoneKind.CHAIN, None)           # holds the cards on the chain
        self.chain_items: list[ChainItem] = []            # everything on the chain, oldest first
        self.chain_by_trigger = False                     # the chain was started by a trigger (346.1)
        self.pending: list[ChainItem] = []                # triggered abilities not yet on the chain
        self.decision: Decision | None = None
        self.turn = 0
        self.turn_player = 0
        self.regular_player = 0                           # whose regular turn was last (737)
        self.extra_turns: list[int] = []                  # additional turns queued (735)
        self.turn_phase = TurnPhase.MULLIGAN
        self.first_player = 0
        self.phase = Phase.PLAYING
        self.winner: int | None = None
        self.priority: int | None = None                  # 312
        self.focus: int | None = None                     # 313
        self.showdown: int | None = None                  # index of the battlefield with a showdown
        self.showdown_passes = 0                          # consecutive focus passes (347.2.a)
        self.attacker: int | None = None                  # set while the showdown is a combat (464.2.c)
        self.assigning: list[int] = []                    # seats still to assign combat damage
        self.assignments: dict[int, tuple[tuple[int, int], ...]] = {}
        self.passes = 0                                   # consecutive passes on the chain (339.1)
        self.mulligan_queue: list[int] = []
        self.channel_phases = [0] * n                     # for 485.7
        self.beginning_phases = [0] * n                   # for "first Beginning Phase" effects
        self.cards_played = [0] * n                       # this turn, for Legion and "second card"
        self.scored: set[tuple[int, int]] = set()         # (seat, battlefield oid) scored this turn (470)
        self.moved: set[int] = set()                      # oids of units that moved this turn
        self.once: set[tuple[int, int, int]] = set()      # (source oid, controller, index): "first time each turn"
        self.log: list[str] = []
        self._setup()

    # --- setup (110-118) ------------------------------------------------------

    def _setup(self) -> None:
        for seat, (deck, zones) in enumerate(zip(self.decks, self.players)):
            new = lambda card: CardInstance(card, owner=seat, controller=seat)   # noqa: E731
            self._put(new(deck.legend), zones.legend)                            # 111
            self._put(new(deck.champion), zones.champion)                        # 112
            for card in deck.main:
                self._put(new(card), zones.main_deck)
            for card in deck.runes:
                self._put(new(card), zones.rune_deck)
            # 113 / 485.5: each player randomly picks one of their battlefields
            fields = [new(card) for card in deck.battlefields]
            chosen = self.rng.choice(fields)
            self.battlefields.append(Battlefield.create(chosen))
            self.set_aside += [f for f in fields if f is not chosen]
            zones.main_deck.shuffle(self.rng)                                    # 114
            zones.rune_deck.shuffle(self.rng)
            self._log(f"{self.names[seat]} enters with {deck.name} ({deck.legend.name}); "
                      f"battlefield: {chosen.card.name}")
        self.first_player = self.turn_player = self.rng.randrange(len(self.players))   # 115
        self._log(f"{self.names[self.first_player]} goes first")
        for seat in range(len(self.players)):                                   # 116
            self.draw(seat, OPENING_HAND)
        second = self._opponent(self.first_player)
        self.mulligan_queue = [self.first_player, second]                       # 117: in turn order

    # --- zone movement --------------------------------------------------------

    def _put(self, obj: CardInstance, dest: Zone, *, bottom: bool = False) -> None:
        if obj.zone is not None:
            obj.zone.objects.remove(obj)
        if dest.capacity is not None and len(dest) >= dest.capacity:
            raise ValueError(f"{dest} is full (107.3.b)")
        if bottom:
            dest.objects.insert(0, obj)
        else:
            dest.objects.append(obj)
        obj.zone = dest

    def move(self, obj: CardInstance, dest: Zone, *, bottom: bool = False) -> None:
        """Change an object's zone, applying 056 and 124."""
        if not dest.is_board and dest.owner is not None and dest.owner != obj.owner:
            dest = self.players[obj.owner].by_kind(dest.kind)                   # 056.2
        on_board = obj.zone is not None and obj.zone.kind in (ZoneKind.BASE, ZoneKind.BATTLEFIELD)
        if on_board and not dest.is_board and obj.card.is_permanent:
            self._emit("leaves", obj=obj)                                       # "when this leaves the board"
        leaving_or_entering_non_board = obj.zone is None or not obj.zone.is_board or not dest.is_board
        self._put(obj, dest, bottom=bottom)
        if leaving_or_entering_non_board:
            obj.become_new_object()                                             # 124

    def draw(self, seat: int, n: int = 1) -> None:
        """413. Drawing from an empty Main Deck Burns Out first (413.4)."""
        zones = self.players[seat]
        for _ in range(n):
            if not zones.main_deck.objects:
                self._burn_out(seat)
                if not zones.main_deck.objects:
                    continue        # 431.3: the trash was empty too; the next draw burns out again
            self.move(zones.main_deck.objects[-1], zones.hand)

    def _burn_out(self, seat: int) -> None:
        """431.2: recycle the trash into the Main Deck, then an opponent gains 1 point."""
        zones = self.players[seat]
        self._recycle_to_main_deck(list(zones.trash))
        opponent = self._opponent(seat)                   # 431.2.c: the only opponent in a Duel
        self.points[opponent] += 1
        self._log(f"{self.names[seat]} burns out; {self.names[opponent]} gains 1 point")

    def _recycle_to_main_deck(self, objs: list[CardInstance]) -> None:
        """416.5: several cards recycled at once go to the bottom in random order."""
        objs = list(objs)
        self.rng.shuffle(objs)
        for obj in objs:
            self.move(obj, self.players[obj.owner].main_deck, bottom=True)

    def channel(self, seat: int, n: int, *, exhausted: bool = False) -> int:
        """430: put the top n runes of the Rune Deck into the base. Returns how many
        were channeled (430.3: as many as possible)."""
        zones = self.players[seat]
        runes = zones.rune_deck.top(n)
        for rune in runes:
            self.move(rune, zones.base)
            rune.exhausted = exhausted                                          # 430.2.a
        return len(runes)

    # --- helpers for card scripts ---------------------------------------------

    @property
    def victory_score(self) -> int:
        """485.3, raised by battlefields like Aspirant's Climb (each copy counts)."""
        return VICTORY_SCORE + sum(VICTORY_BONUS.get(bf.card.card.card_id, 0) for bf in self.battlefields)

    def units(self) -> list[CardInstance]:
        """Every unit on the board."""
        objs = [o for p in self.players for o in p.base if o.card.is_unit]
        return objs + [o for bf in self.battlefields for o in bf.units if o.card.is_unit]

    def units_at(self, index: int) -> list[CardInstance]:
        return [o for o in self.battlefields[index].units if o.card.is_unit]

    def permanents(self, seat: int) -> list[CardInstance]:
        """Units and gear `seat` controls on the board (not facedown cards)."""
        return [o for o in self._on_board(seat) if o.card.is_permanent]

    def runes(self, seat: int) -> list[CardInstance]:
        return self.players[seat].runes()

    def legend(self, seat: int) -> CardInstance | None:
        objs = self.players[seat].legend.objects
        return objs[0] if objs else None

    def hand(self, seat: int) -> list[CardInstance]:
        return list(self.players[seat].hand)

    def trash(self, seat: int) -> list[CardInstance]:
        return list(self.players[seat].trash)

    def deck_top(self, seat: int, n: int) -> list[CardInstance]:
        """The top n cards of the Main Deck, top first (fewer if the deck is smaller)."""
        return self.players[seat].main_deck.top(n)

    def opponents(self, seat: int) -> list[int]:
        return [s for s in range(len(self.players)) if s != seat]

    def turn_order(self) -> list[int]:
        return [self.turn_player, *(s for s in range(len(self.players)) if s != self.turn_player)]

    def battlefield_index(self, obj: CardInstance) -> int | None:
        """The battlefield an object is at (or is), None if it isn't at one."""
        for i, bf in enumerate(self.battlefields):
            if obj.zone is bf.units or obj.zone is bf.facedown or obj is bf.card:
                return i
        return None

    def might(self, obj: CardInstance) -> int:
        """A unit's current Might: printed Might, buffs, "this turn" changes, Assault
        while it's an attacker (807), Shield while it's a defender (814), and
        legend passives like Master Yi's."""
        value = (obj.card.might or 0) + obj.buffs
        for amount, minimum in obj.might_mods:
            new = value + amount
            if minimum is not None:
                new = max(new, min(value, minimum))     # "to a minimum of N" never raises Might
            value = new
        if self._is_attacker(obj):
            value += obj.card.keyword_value(Keyword.ASSAULT) + obj.granted.get("Assault", 0)
        elif self._is_defender(obj):
            value += obj.card.keyword_value(Keyword.SHIELD) + obj.granted.get("Shield", 0)
            legend = self.legend(obj.controller)
            if legend is not None and legend.card.card_id in DEFEND_ALONE_BONUS \
                    and not any(o is not obj and o.controller == obj.controller for o in obj.zone):
                value += DEFEND_ALONE_BONUS[legend.card.card_id]                # 740.2.a: alone
        return value

    def add_might(self, obj: CardInstance, amount: int, *, floor: int | None = None) -> None:
        """Give a unit +/-X Might this turn (optionally "to a minimum of `floor`")."""
        obj.might_mods.append((amount, floor))

    def grant(self, obj: CardInstance, keyword: str, value: int) -> None:
        """Give a unit a keyword this turn; valued ones add up (807.2, 809.2, 814.2)."""
        obj.granted[keyword] = obj.granted.get(keyword, 0) + value

    def has_ganking(self, obj: CardInstance) -> bool:
        return obj.card.has_keyword(Keyword.GANKING) or "Ganking" in obj.granted

    def deal(self, obj: CardInstance, amount: int) -> None:
        """417: mark damage. Death happens in the next cleanup."""
        if amount > 0:
            obj.damage += amount

    def stun(self, obj: CardInstance) -> None:
        """423: a stunned unit deals no combat damage this turn."""
        obj.stunned = True

    def ready(self, obj: CardInstance) -> None:
        obj.exhausted = False                                                   # 415

    def return_to_hand(self, obj: CardInstance) -> None:
        self.move(obj, self.players[obj.owner].hand)

    def put_in_hand(self, obj: CardInstance) -> None:
        """A card from the deck or trash goes to its owner's hand (not a draw)."""
        self.move(obj, self.players[obj.owner].hand)

    def recycle(self, objs: list[CardInstance]) -> None:
        """416: cards go to the bottom of their owner's Main Deck, in random order if
        several (416.5)."""
        self._recycle_to_main_deck([o for o in objs if o.zone is not None])

    def banish(self, obj: CardInstance) -> None:
        self.move(obj, self.players[obj.owner].banishment)                      # 427

    def discard(self, obj: CardInstance) -> None:
        """422: from hand to trash; "when discarded" abilities trigger afterwards."""
        self.move(obj, self.players[obj.owner].trash)
        self._log(f"{self.names[obj.owner]} discards {obj.card.name}")
        self._emit("discard", extra=[obj], obj=obj)

    def reveal_hand(self, seat: int) -> None:
        """424: the whole hand becomes public information for the moment."""
        names = ", ".join(o.card.name for o in self.players[seat].hand) or "nothing"
        line = f"{self.names[seat]} reveals their hand: {names}"
        if not self.log or self.log[-1] != line:
            self._log(line)

    def reveal_until_unit(self, seat: int) -> CardInstance | None:
        """The first unit from the top of the Main Deck, without moving anything."""
        return next((o for o in reversed(self.players[seat].main_deck.objects) if o.card.is_unit), None)

    def reveal_from_top_until_unit(self, seat: int) -> list[CardInstance]:
        """Reveal cards from the top until a unit (424, 431.1.c: no Burn Out). Returns
        them top first; the last one is the unit unless the deck ran out."""
        revealed = []
        for obj in reversed(self.players[seat].main_deck.objects):
            revealed.append(obj)
            if obj.card.is_unit:
                break
        if revealed:
            self._log(f"{self.names[seat]} reveals " + ", ".join(o.card.name for o in revealed))
        return revealed

    def kill(self, obj: CardInstance) -> None:
        """428.1.a.1: a kill instruction. Units can have their death replaced."""
        if obj.zone is None or obj.zone.kind not in (ZoneKind.BASE, ZoneKind.BATTLEFIELD):
            return
        if obj.card.is_unit:
            self._kill_units([obj])
        else:
            self._emit("killed", obj=obj)
            self._log(f"{obj.card.name} is killed")
            self.move(obj, self.players[obj.owner].trash)

    def counter(self, obj: CardInstance) -> None:
        """425: a countered spell does nothing and goes to its owner's trash."""
        item = next((i for i in self.chain_items if i.obj is obj), None)
        if item is None:
            return
        self.chain_items.remove(item)
        self._log(f"{obj.card.name} is countered")
        self.move(obj, self.players[obj.owner].trash)

    def move_unit(self, obj: CardInstance, dest: int | None) -> bool:
        """A move caused by an effect (449): no exhaust cost, unlike a Standard Move.
        Returns False if the move was impossible (359.3.e.6)."""
        origin = self.battlefield_index(obj)
        if obj.zone is None or obj.zone.kind not in (ZoneKind.BASE, ZoneKind.BATTLEFIELD) or origin == dest:
            return False
        if dest is None:
            if self.battlefields[origin].card.card.card_id in NO_RETREAT:
                return False                                                    # Vilemaw's Lair
            self.move(obj, self.players[obj.controller].base)
            return True
        bf = self.battlefields[dest]
        if not bf.contested and bf.controller != obj.controller:               # 450
            bf.contested, bf.contested_by = True, obj.controller
        self.move(obj, bf.units)
        return True

    def move_units(self, seat: int, objs: list[CardInstance], dest: int | None) -> list[CardInstance]:
        """Move units as one game action `seat` is responsible for (411), then
        trigger "when ... moves" abilities once. Returns the units that moved."""
        moved = [o for o in objs if self.move_unit(o, dest)]
        if moved:
            first = {o.oid for o in moved if o.oid not in self.moved}
            self.moved |= {o.oid for o in moved}
            self._emit("move", seat=seat, units=moved, dest=dest, first=first)
        return moved

    def unit_destinations(self, seat: int, card: CardDef) -> list[int | None]:
        """355.2: where `seat` may play a unit: their base, battlefields they control,
        plus any the card allows (open or occupied enemy battlefields)."""
        extra = PLAY_LOCATIONS.get(card.card_id)
        dests: list[int | None] = [None]
        for i, bf in enumerate(self.battlefields):
            occupied = any(o.card.is_unit for o in bf.units)
            if bf.controller == seat \
                    or (extra == "open" and bf.controller is None and not occupied) \
                    or (extra == "occupied_enemy" and bf.controller not in (None, seat) and occupied):
                dests.append(i)
        return dests

    def play_unit(self, seat: int, obj: CardInstance, dest: int | None) -> None:
        """Play a unit as part of an effect (419.3), its costs already handled."""
        zone = self.players[seat].base if dest is None else self.battlefields[dest].units
        self.move(obj, zone)
        obj.controller = seat
        obj.exhausted = obj.card.card_id not in ENTERS_READY                    # 359.2.c
        self.cards_played[seat] += 1
        self._log(f"{self.names[seat]} plays {obj.card.name} to {self.location_name(dest)}")
        self._emit("played", seat=seat, obj=obj)

    def location_name(self, dest: int | None) -> str:
        return "base" if dest is None else self._bf_name(self.battlefields[dest])

    def gain_points(self, seat: int, n: int) -> None:
        self.points[seat] += n
        self._log(f"{self.names[seat]} gains {n} point(s): {self.points[seat]} points")

    def add_extra_turn(self, seat: int) -> None:
        self.extra_turns.append(seat)                                           # 735
        self._log(f"{self.names[seat]} will take an extra turn")

    def power_available(self, seat: int) -> int:
        """How much Power `seat` could pay right now: their pool plus every rune."""
        return len(self.players[seat].rune_pool.power) + len(self.runes(seat))

    def pay_power(self, seat: int, n: int) -> int:
        """Pay up to n Power of any domain: from the pool first, then by recycling
        runes, exhausted ones first (164.2.b). Returns how much was paid."""
        pool = self.players[seat].rune_pool
        paid = 0
        while paid < n and pool.power:
            pool.power.pop()
            paid += 1
        runes = sorted(self.runes(seat), key=lambda r: (not r.exhausted, r.oid))
        for rune in runes[:n - paid]:
            self.move(rune, self.players[seat].rune_deck, bottom=True)          # 416.1.b
            paid += 1
        return paid

    def can_pay(self, seat: int, card: CardDef, cost: Cost) -> bool:
        return bool(self.payment_options(seat, card, cost))

    def pay_first_option(self, seat: int, card: CardDef, cost: Cost) -> bool:
        """Pay a cost during an effect, the way that keeps the most runes ready."""
        options = self.payment_options(seat, card, cost)
        if not options:
            return False
        self._pay(seat, card, cost, options[0])
        return True

    def distinct_units(self, units: list[CardInstance]) -> list[CardInstance]:
        """One unit per group of interchangeable units (same card, place and state)."""
        seen: dict[tuple, CardInstance] = {}
        for obj in sorted(units, key=lambda o: o.oid):
            seen.setdefault(self._state_key(obj), obj)
        return list(seen.values())

    def describe_unit(self, obj: CardInstance) -> str:
        return self._describe_unit(obj)

    def targets(self, item: ChainItem) -> list[CardInstance]:
        """The item's targets that are still legal (359.3.e); a unit chosen twice
        appears twice."""
        at = item.data.get("hidden_at")
        return [obj for obj, oid, kind in item.targets
                if obj.oid == oid and self._valid_target(obj, kind, item.controller, at=at)]

    def source(self, item: ChainItem) -> CardInstance | None:
        """An ability's source, if it is still the same object on the board."""
        src = item.source
        if src is None or src.oid != item.source_oid or src.zone is None or not src.zone.is_board:
            return None
        return src

    # --- decision loop --------------------------------------------------------

    @property
    def is_over(self) -> bool:
        return self.phase is Phase.GAME_OVER

    @property
    def acting_player(self) -> int | None:
        """The seat that must decide next, or None if the game is over."""
        if self.is_over:
            return None
        if self.turn_phase is TurnPhase.MULLIGAN:
            return self.mulligan_queue[0]
        if self.decision is not None:
            return self.decision.seat
        if self.assigning:
            return self.assigning[0]
        return self.priority

    def legal_actions(self, seat: int | None = None) -> list[Action]:
        seat = self.acting_player if seat is None else seat
        if self.is_over or seat is None or seat != self.acting_player:
            return []
        if self._legal_cache is not None and self._legal_cache[0] == seat:
            return list(self._legal_cache[1])
        if self.turn_phase is TurnPhase.MULLIGAN:
            actions = self._mulligan_options(seat)
        elif self.decision is not None:
            actions = [Action(ActionKind.CHOOSE, label, choice=i)
                       for i, (label, _) in enumerate(self.decision.options)]
        elif self.assigning:
            actions = self._assignment_options(seat)
        else:
            actions = self._play_options(seat) + self._ability_options(seat)
            if self.chain_items:
                actions.append(Action(ActionKind.PASS, "Pass"))
            elif self.showdown is not None:
                actions.append(Action(ActionKind.PASS, "Pass focus"))
            else:
                actions += self._hide_options(seat) + self._move_options(seat)
                actions.append(Action(ActionKind.END_TURN, "End turn"))
        if self.allow_concede:
            actions.append(Action(ActionKind.CONCEDE, "Concede"))
        if self.cache_actions:
            self._legal_cache = (seat, list(actions))
        return actions

    def step(self, action: Action) -> None:
        seat = self.acting_player
        if action not in self.legal_actions(seat):
            raise ValueError(f"illegal action {action} for seat {seat}")
        self._apply(seat, action)
        self._settle()
        self._advance()

    def _apply(self, seat: int, action: Action) -> None:
        self._legal_cache = None
        if action.kind is ActionKind.CONCEDE:
            self._log(f"{self.names[seat]} concedes")
            self._end(winner=self._opponent(seat))
        elif action.kind is ActionKind.MULLIGAN:
            self._mulligan(seat, action.set_aside)
        elif action.kind is ActionKind.CHOOSE:
            self._choose(action.choice)
        elif action.kind is ActionKind.PLAY_CARD:
            self._play_card(seat, action)
        elif action.kind is ActionKind.ACTIVATE:
            self._activate(seat, action)
        elif action.kind is ActionKind.HIDE:
            self._hide(seat, action)
        elif action.kind is ActionKind.MOVE:
            self._standard_move(seat, action)
        elif action.kind is ActionKind.ASSIGN_DAMAGE:
            self._assign_damage(seat, action.damage)
        elif action.kind is ActionKind.PASS:
            if self.chain_items:
                self._pass_priority(seat)
            else:
                self._pass_focus(seat)
        elif action.kind is ActionKind.END_TURN:
            self._end_turn()

    def _settle(self) -> None:
        """Everything that happens between decisions: cleanups (319), putting
        triggered abilities on the chain, and the start-of-turn phases."""
        while not self.is_over:
            self._cleanup()
            if self.is_over or self.decision is not None or self.assigning:
                return
            if self.pending:
                self._flush_triggers()
                continue
            if not self.chain_items and self.turn_phase in START_OF_TURN:       # 335
                self._next_start_step()
                continue
            if not self.chain_items and self.turn_phase is TurnPhase.ENDING and self.showdown is None:
                self._expire_turn()                                             # 317.2
                continue
            return

    def _advance(self) -> None:
        """Take every action that isn't a real choice: when passing, ending the turn
        or a forced choice is all a player can do (conceding aside), do it."""
        automatic = (ActionKind.PASS, ActionKind.END_TURN, ActionKind.ASSIGN_DAMAGE, ActionKind.CHOOSE)
        while not self.is_over:
            seat = self.acting_player
            options = [a for a in self.legal_actions(seat) if a.kind is not ActionKind.CONCEDE]
            if len(options) != 1 or options[0].kind not in automatic:
                return
            self._apply(seat, options[0])
            self._settle()

    # --- triggered abilities (382-383) ----------------------------------------

    def _emit(self, event: str, *, extra: list[CardInstance] = (), **data: Any) -> None:
        """Queue every triggered ability whose condition `event` meets. Sources are
        the objects on the board and the battlefields, plus `extra` (a discarded
        card). The turn player's abilities go on the chain first (383.3.d.1)."""
        sources = [o for seat in self._turn_order() for o in self._on_board(seat)]
        sources += [bf.card for bf in self.battlefields] + list(extra)
        found = []
        for source in sources:
            for i, script in enumerate(TRIGGERS.get(source.card.card_id, ())):
                if script.event != event or not script.condition(self, source, data):
                    continue
                # 190.6.c: a battlefield's ability belongs to the player it names
                controller = data["seat"] if source.card.is_battlefield else source.controller
                if script.once_per_turn:                                        # 383.1: "the first time"
                    key = (source.oid, controller, i)
                    if key in self.once:
                        continue
                    self.once.add(key)
                found.append(ChainItem(controller, trigger=(source.card.card_id, i), source=source,
                                       source_oid=source.oid, data=dict(data),
                                       label=f"{source.card.name}: {script.text}"))
        order = self._turn_order()
        found.sort(key=lambda item: order.index(item.controller))
        self.pending += found

    def _turn_order(self) -> list[int]:
        return [self.turn_player, *(s for s in range(len(self.players)) if s != self.turn_player)]

    def _flush_triggers(self) -> None:
        """Put pending triggered abilities on the chain (383.3), asking "you may"
        abilities whether to go ahead and choosing targets first (383.3.a). An
        ability with nothing to choose can't be put on the chain (355.8)."""
        while self.pending:
            item = self.pending[0]
            script = _trigger_script(item)
            if script.choices is not None and "choice" not in item.data:
                options = script.choices(self, item)
                if not options:
                    self.pending.pop(0)
                    continue
                if len(options) > 1:
                    self.decision = Decision(item.controller, "trigger", options, item)
                    return
                item.data["choice"] = options[0][1]
            self.pending.pop(0)
            if script.choices is not None and item.data["choice"] is None:
                self._log(f"{self.names[item.controller]} declines {item.label}")
                continue                                                        # 383.3.a.2
            if not self.chain_items:
                self.chain_by_trigger = True
            self.chain_items.append(item)
            self._log(f"Triggers: {item.label}")
            self.priority = item.controller                                     # 337.4
            self.passes = 0

    def _choose(self, index: int) -> None:
        decision, self.decision = self.decision, None
        label, value = decision.options[index]
        if decision.kind == "trigger":
            decision.item.data["choice"] = value
        elif decision.kind == "resolve":
            decision.item.data[decision.key] = value
            self._resolve_top()                                                 # carry on resolving
        else:
            self._begin_showdown(value)

    # --- mulligan (117) -------------------------------------------------------

    def _mulligan_options(self, seat: int) -> list[Action]:
        """Every distinct set of up to 2 cards to set aside. Copies of the same card
        are interchangeable, so each set of card names is offered once."""
        hand = self.players[seat].hand.objects
        actions, seen = [], set()
        for n in range(MULLIGAN_MAX + 1):
            for combo in itertools.combinations(hand, n):
                key = tuple(sorted(o.card.card_id for o in combo))
                if key in seen:
                    continue
                seen.add(key)
                label = "Keep hand" if not combo else "Mulligan " + ", ".join(o.card.name for o in combo)
                actions.append(Action(ActionKind.MULLIGAN, label, set_aside=tuple(o.oid for o in combo)))
        return actions

    def _mulligan(self, seat: int, set_aside: tuple[int, ...]) -> None:
        zones = self.players[seat]
        objs = [o for o in zones.hand if o.oid in set_aside]
        for obj in objs:                                                        # 117.1
            zones.hand.objects.remove(obj)
            obj.zone = None
        self.draw(seat, len(objs))                                              # 117.2
        self._recycle_to_main_deck(objs)                                        # 117.3
        self._log(f"{self.names[seat]} mulligans {len(objs)}")
        self.mulligan_queue.pop(0)
        if not self.mulligan_queue:
            self.regular_player = self.first_player
            self._start_turn(self.first_player)                                 # 118

    # --- the turn (314-317) ---------------------------------------------------

    def _start_turn(self, seat: int) -> None:
        self.turn += 1
        self.turn_player = seat
        self.scored.clear()
        self.moved.clear()
        self.once.clear()
        self.cards_played = [0] * len(self.players)
        self._log(f"Turn {self.turn}: {self.names[seat]}")
        self.turn_phase = TurnPhase.AWAKEN                                      # 315.1
        self.priority = None
        for obj in self._on_board(seat):
            obj.exhausted = False
        # _settle runs the rest of the start of turn, pausing for any chain

    def _next_start_step(self) -> None:
        seat = self.turn_player
        if self.turn_phase is TurnPhase.AWAKEN:
            self.turn_phase = TurnPhase.BEGINNING                               # 315.2.a
            self.beginning_phases[seat] += 1
            self._emit("beginning", seat=seat)
        elif self.turn_phase is TurnPhase.BEGINNING:
            self.turn_phase = TurnPhase.SCORING                                 # 315.2.b: Hold
            for bf in self.battlefields:
                if bf.controller == seat:
                    self._score(seat, bf, conquer=False)
        elif self.turn_phase is TurnPhase.SCORING:
            self.turn_phase = TurnPhase.CHANNEL                                 # 315.3
            n = CHANNEL_PER_TURN
            if seat != self.first_player and self.channel_phases[seat] == 0:
                n += SECOND_PLAYER_EXTRA                                        # 485.7
            self.channel_phases[seat] += 1
            self.channel(seat, n)
        elif self.turn_phase is TurnPhase.CHANNEL:
            self.turn_phase = TurnPhase.DRAW                                    # 315.4
            self.draw(seat)
        elif self.turn_phase is TurnPhase.DRAW:
            self.turn_phase = TurnPhase.MAIN                                    # 316
            for p in self.players:                                              # 316.3
                p.rune_pool.empty()
            self.priority = seat                                                # 312.2.a
            self.passes = 0

    def _end_turn(self) -> None:
        """317.1: the Ending Step. "At the end of your turn" abilities trigger now;
        once their chain is done, _settle runs the Expiration Step."""
        seat = self.turn_player
        self._log(f"{self.names[seat]} ends turn {self.turn}")
        self.turn_phase = TurnPhase.ENDING                                      # 317
        self.priority = None
        self._emit("end_turn", seat=seat)

    def _expire_turn(self) -> None:
        """317.2: the Expiration Step, then the next turn."""
        for p in self.players:
            for obj in self._on_board(p.seat):
                if obj.card.is_unit:
                    obj.damage = 0                                              # 317.2.b: heal all units
                    obj.stunned = False                                         # 423.1.a.2
                obj.might_mods, obj.granted = [], {}                            # 317.2.c: "this turn" expires
            p.rune_pool.empty()                                                 # 317.2.d
        if self.extra_turns:                                                    # 737
            nxt = self.extra_turns.pop(0)
            self._log(f"{self.names[nxt]} takes an extra turn")
        else:
            nxt = self.regular_player = self._opponent(self.regular_player)
        self._start_turn(nxt)

    def _on_board(self, seat: int) -> list[CardInstance]:
        """Every object `seat` controls in their base, their legend zone, and at
        battlefields (not facedown cards)."""
        zones = self.players[seat]
        objs = [*zones.base, *zones.legend]
        for bf in self.battlefields:
            objs += [o for o in bf.units if o.controller == seat]
        return objs

    # --- movement (144, 445-453) and showdowns (341-348) -----------------------

    def _move_options(self, seat: int) -> list[Action]:
        """Standard Moves (144): ready units go from base to a battlefield, or from
        battlefields back to base, several at once if they share a destination
        (144.3.b: origins may differ). Units with Ganking may also go from one
        battlefield to another (144.4.c). Units that are interchangeable (same card,
        place and state) are grouped, so each distinct set of movers is offered
        once per destination."""
        if not (seat == self.turn_player and self.turn_phase is TurnPhase.MAIN):   # 144.1
            return []
        zones = self.players[seat]
        in_base = [o for o in zones.base if o.card.is_unit and not o.exhausted]
        at_bfs = {i: [o for o in bf.units if o.controller == seat and o.card.is_unit and not o.exhausted]
                  for i, bf in enumerate(self.battlefields)}
        actions = []
        for i, bf in enumerate(self.battlefields):                              # 144.4.a
            gankers = [o for j, units in at_bfs.items() if j != i for o in units if self.has_ganking(o)]
            actions += self._move_groups(in_base + gankers, i, self._bf_name(bf))
        retreating = [o for i, units in at_bfs.items() for o in units           # 144.4.b
                      if self.battlefields[i].card.card.card_id not in NO_RETREAT]   # Vilemaw's Lair
        actions += self._move_groups(retreating, None, "base")
        return actions

    def _move_groups(self, units: list[CardInstance], dest: int | None, dest_name: str) -> list[Action]:
        groups: dict[tuple, list[CardInstance]] = {}
        for obj in sorted(units, key=lambda o: o.oid):
            groups.setdefault(self._state_key(obj), []).append(obj)
        members = list(groups.values())
        actions = []
        for counts in itertools.product(*(range(len(m) + 1) for m in members)):
            chosen = [obj for m, n in zip(members, counts) for obj in m[:n]]
            if not chosen:
                continue
            names = ", ".join(f"{n}x {m[0].card.name}" if n > 1 else m[0].card.name
                              for m, n in zip(members, counts) if n)
            actions.append(Action(ActionKind.MOVE, f"Move {names} to {dest_name}",
                                  units=tuple(o.oid for o in chosen), destination=dest))
        return actions

    def _standard_move(self, seat: int, action: Action) -> None:
        zones = self.players[seat]
        movable = [*zones.base, *(o for bf in self.battlefields for o in bf.units)]
        units = [o for o in movable if o.oid in action.units]
        for obj in units:
            obj.exhausted = True                                                # 144.2 / 144.3.c
        names = ", ".join(o.card.name for o in units)
        self._log(f"{self.names[seat]} moves {names} to {self.location_name(action.destination)}")
        self.move_units(seat, units, action.destination)

    def _begin_showdown(self, index: int) -> None:
        bf = self.battlefields[index]
        self.showdown = index
        self.focus = self.priority = bf.contested_by                            # 345 / 313.2
        self.showdown_passes = 0
        if len({o.controller for o in bf.units if o.card.is_unit}) > 1:
            self._begin_combat()
        else:
            self._log(f"Showdown at {self._bf_name(bf)}")

    def _begin_combat(self) -> None:
        """464.2: the player who contested the battlefield attacks and has focus."""
        bf = self.battlefields[self.showdown]
        self.attacker = bf.contested_by                                         # 464.2.c.1
        self._log(f"Combat at {self._bf_name(bf)}: {self.names[self.attacker]} attacks")
        self._emit("defend", seat=self._opponent(self.attacker), bf=bf)        # 464.2.e

    def _pass_focus(self, seat: int) -> None:
        self.showdown_passes += 1
        if self.showdown_passes >= len(self.players):                           # 347.2.a
            self._end_showdown()
        else:
            self.focus = self.priority = self._opponent(seat)                   # 347.2.b

    def _end_showdown(self) -> None:
        """348.2: a Non-Combat Showdown closes. If one player's units remain,
        they establish control, which is a Conquer if not yet scored this turn."""
        if self.attacker is not None:
            return self._combat_damage_step()                                   # 348.1
        bf = self.battlefields[self.showdown]
        self.showdown = self.focus = None
        self.showdown_passes = 0
        present = {o.controller for o in bf.units if o.card.is_unit}
        if len(present) == 1:
            [seat] = present
            if bf.controller != seat:                                           # 348.2.a
                bf.controller = seat
                bf.contested, bf.contested_by = False, None
                self._log(f"{self.names[seat]} takes control of {self._bf_name(bf)}")
                self._score(seat, bf, conquer=True)                             # 348.2.a.1
        self.priority = self.turn_player if self.turn_phase is TurnPhase.MAIN else None

    # --- combat (459-466) ------------------------------------------------------

    def _is_attacker(self, obj: CardInstance) -> bool:
        if self.attacker is None or self.showdown is None:
            return False
        return obj.controller == self.attacker and obj.zone is self.battlefields[self.showdown].units

    def _is_defender(self, obj: CardInstance) -> bool:
        if self.attacker is None or self.showdown is None:
            return False
        return obj.controller != self.attacker and obj.zone is self.battlefields[self.showdown].units

    def _combat_damage_step(self) -> None:
        """465: if both sides still have units here, each assigns its total Might
        as damage, attacker first; then it is all dealt at once."""
        bf = self.battlefields[self.showdown]
        sides = {o.controller for o in bf.units if o.card.is_unit}
        self.priority = None
        if len(sides) == 2:
            self.assigning = [self.attacker, self._opponent(self.attacker)]     # 465.2.c
            self.assignments = {}
        else:
            self._combat_resolution()

    def _assignment_options(self, seat: int) -> list[Action]:
        """465.2.c.3-4: lethal damage goes to one unit at a time, so what a player
        really chooses is which enemy units die. Offer every set of kills that
        uses the damage fully (no other enemy unit could also be killed with what
        is left). Units with Tank must get lethal damage before any unit without
        it (815.1.b). Stunned units add no damage (423.1.b). Leftover damage lands
        on a survivor (a Tank first), who heals right after."""
        bf = self.battlefields[self.showdown]
        total = sum(max(0, self.might(o)) for o in bf.units                     # 465.2.a-b
                    if o.controller == seat and o.card.is_unit and not o.stunned)
        targets = sorted((o for o in bf.units if o.controller != seat and o.card.is_unit), key=lambda o: o.oid)
        lethal = {o.oid: max(1, self.might(o) - o.damage) for o in targets}     # 142.4.b
        tanks = {o.oid for o in targets if o.card.has_keyword(Keyword.TANK) or "Tank" in o.granted}
        groups: dict[tuple, list[CardInstance]] = {}
        for obj in targets:
            groups.setdefault(self._state_key(obj), []).append(obj)
        members = list(groups.values())

        def allowed(killed: list[CardInstance]) -> bool:
            ids = {o.oid for o in killed}
            return ids <= tanks or tanks <= ids

        actions = []
        for counts in itertools.product(*(range(len(m) + 1) for m in members)):
            killed = [o for m, n in zip(members, counts) for o in m[:n]]
            cost = sum(lethal[o.oid] for o in killed)
            rest = [o for o in targets if o not in killed]
            if cost > total or not allowed(killed) \
                    or any(cost + lethal[o.oid] <= total and allowed(killed + [o]) for o in rest):
                continue
            damage = {o.oid: lethal[o.oid] for o in killed}
            leftover = total - cost
            if leftover:
                rest.sort(key=lambda o: o.oid not in tanks)
                spill = rest[0] if rest else killed[0]
                damage[spill.oid] = damage.get(spill.oid, 0) + leftover
            names = ", ".join(o.card.name for o in killed)
            label = f"Kill {names}" if killed else "Kill nothing"
            actions.append(Action(ActionKind.ASSIGN_DAMAGE, f"{label} ({total} damage)",
                                  damage=tuple(sorted(damage.items()))))
        return actions

    def _assign_damage(self, seat: int, damage: tuple[tuple[int, int], ...]) -> None:
        self.assignments[seat] = damage
        self.assigning.pop(0)
        if self.assigning:
            return
        bf = self.battlefields[self.showdown]
        units = {o.oid: o for o in bf.units}
        for assigned in self.assignments.values():                              # 465.2.d: all at once
            for oid, amount in assigned:
                self.deal(units[oid], amount)
        self.assignments = {}
        self._combat_resolution()

    def _combat_resolution(self) -> None:
        """466: combat cleanup (kill, heal, recall attackers if defenders remain),
        then the survivor takes the battlefield."""
        bf = self.battlefields[self.showdown]
        attacker = self.attacker
        self._kill_lethal()                                                     # 323.4-323.5
        for obj in self.units():                                                # 466.1.a.1: heal all units
            obj.damage = 0
        if any(o.controller != attacker for o in self.units_at(self.showdown)):                     # 466.1.a.2: recall attackers
            for obj in [o for o in self.units_at(self.showdown) if o.controller == attacker]:
                self._put(obj, self.players[obj.controller].base)               # 455: not a move
        present = {o.controller for o in bf.units if o.card.is_unit}
        self.showdown = self.focus = self.attacker = None
        self.showdown_passes = 0
        bf.contested, bf.contested_by = False, None                             # 466.5.a
        if len(present) == 1:                                                   # 466.5
            [winner] = present
            self._log(f"{self.names[winner]} wins the combat at {self._bf_name(bf)}")
            if bf.controller != winner:
                bf.controller = winner
                self._score(winner, bf, conquer=True)                           # 466.5.d
        elif not present:
            bf.controller = None                                                # 466.5.b
            self._log(f"No units are left at {self._bf_name(bf)}")
        self.priority = self.turn_player if self.turn_phase is TurnPhase.MAIN else None

    def _kill_lethal(self) -> None:
        """323.4-323.5: units with lethal damage die and go to their owner's trash."""
        self._kill_units([o for o in self.units() if o.damage > 0 and o.damage >= self.might(o)])  # 142.4.b

    def _kill_units(self, dying: list[CardInstance]) -> None:
        """Units die together. Death replacements apply first (367): each Zhonya's
        Hourglass saves one friendly unit, the one with the most Might (373 lets
        the controller pick; this picks for them), and is killed instead. Then
        Deathknell triggers while the rest are still on the board (808.1.d.2)."""
        saved = []
        for seat in self.turn_order():                                          # 373.1
            savers = [o for o in self.permanents(seat) if o.card.card_id in DEATH_REPLACEMENT]
            mine = sorted((o for o in dying if o.controller == seat), key=lambda o: (-self.might(o), o.oid))
            for unit, saver in zip(mine, savers):
                self._log(f"{saver.card.name} is killed instead of {unit.card.name}")
                self.kill(saver)
                unit.damage = 0                                                 # heal, exhaust and recall it
                unit.exhausted = True
                self._put(unit, self.players[unit.controller].base)             # 455: a recall, not a move
                saved.append(unit)
        for obj in dying:
            if obj not in saved:
                self._emit("dies", obj=obj)
                self._emit("killed", obj=obj)
                self._log(f"{obj.card.name} dies")
                self.move(obj, self.players[obj.owner].trash)

    # --- scoring and winning --------------------------------------------------

    def _score(self, seat: int, bf: Battlefield, *, conquer: bool) -> None:
        """469-471: Hold or Conquer a battlefield, at most once per battlefield per turn."""
        key = (seat, bf.card.oid)
        if key in self.scored:                                                  # 470
            return
        self.scored.add(key)
        how = "conquers" if conquer else "holds"
        scored_all = all((seat, b.card.oid) in self.scored for b in self.battlefields)
        if conquer and self.points[seat] >= self.victory_score - 1 and not scored_all:
            self._log(f"{self.names[seat]} {how} {self._bf_name(bf)} but draws instead of the final point")
            self.draw(seat)                                                     # 471.1.b.1
        else:
            self.points[seat] += 1
            self._log(f"{self.names[seat]} {how} {self._bf_name(bf)}: {self.points[seat]} points")
        self._emit("conquer" if conquer else "hold", seat=seat, bf=bf)          # 471.2

    def _cleanup(self) -> bool:
        """323. Returns True if the game ended."""
        if self.is_over:
            return True
        best = max(self.points)
        leaders = [seat for seat, p in enumerate(self.points) if p == best]
        if best >= self.victory_score and len(leaders) == 1:                    # 323.1 / 194.2
            self._end(leaders[0])
            return True

        if not self.assigning:
            self._kill_lethal()                                                 # 323.4-323.5
        for bf in self.battlefields:                                            # 323.7
            for obj in [o for o in bf.units if not o.card.is_unit]:
                self._put(obj, self.players[obj.controller].base)               # 457.1: recall gear
            for obj in [o for o in bf.facedown if o.controller != bf.controller]:
                self._log(f"{self.names[obj.controller]}'s hidden {obj.card.name} is removed")
                self.move(obj, self.players[obj.owner].trash)                   # 107.3.d / 421.4

        open_state = not self.chain_items
        if (self.showdown is not None and self.attacker is None and open_state
                and len({o.controller for o in self.units_at(self.showdown)}) > 1):
            self._begin_combat()                                                # 323.14 / 460.1
        for i, bf in enumerate(self.battlefields):
            present = {o.controller for o in bf.units if o.card.is_unit}
            busy = self.showdown == i
            if bf.controller is not None and bf.controller not in present and open_state and not busy:
                self._log(f"{self.names[bf.controller]} loses control of {self._bf_name(bf)}")
                bf.controller = None                                            # 323.6 / 190.4.c
            if bf.contested and bf.contested_by not in present and not busy:
                bf.contested, bf.contested_by = False, None                     # 323.11
            others = present - {bf.controller}
            if not bf.contested and len(others) == 1 and not busy:
                bf.contested, bf.contested_by = True, others.pop()              # 323.11.a

        if (self.showdown is None and open_state and self.turn_phase in (TurnPhase.MAIN, TurnPhase.ENDING)
                and not self.pending and self.decision is None and not self.assigning):
            staged = [i for i, bf in enumerate(self.battlefields)               # 323.12-323.13
                      if bf.contested and any(o.controller == bf.contested_by and o.card.is_unit for o in bf.units)]
            if len(staged) == 1:
                self._begin_showdown(staged[0])
            elif staged:                                                        # 461.1: the turn player picks
                options = [(f"Start at {self._bf_name(self.battlefields[i])}", i) for i in staged]
                self.decision = Decision(self.turn_player, "showdown", options)
        return False

    def _end(self, winner: int) -> None:
        self.winner = winner
        self.phase = Phase.GAME_OVER
        self.priority = None
        self._log(f"{self.names[winner]} wins")

    # --- playing cards (349-359) ----------------------------------------------

    def _playable_now(self, seat: int, card: CardDef, *, hidden: bool = False) -> bool:
        """Card has rules support, and 308-310 timing allows it. A card played from
        Hidden has Reaction (811.6)."""
        if card.is_spell and card.card_id not in SPELLS:
            return False
        if not (card.is_permanent or card.is_spell):
            return False
        if self.turn_phase in START_OF_TURN or self.turn_phase is TurnPhase.MULLIGAN:
            return False
        reaction = hidden or card.has_keyword(Keyword.REACTION)
        if self.chain_items:                                                    # Closed: 309.1.a
            return reaction
        if self.showdown is not None:                                           # Showdown Open: 308.1.a
            return card.has_keyword(Keyword.ACTION) or reaction
        if reaction:
            return True
        # Neutral Open: the turn player, in their Main Phase (310.1.a)
        return seat == self.turn_player and self.turn_phase is TurnPhase.MAIN

    def total_cost(self, seat: int, card: CardDef, *, accelerate: bool = False, deflect: int = 0,
                   hidden: bool = False) -> Cost:
        """356: the base cost with discounts and additional costs applied. A card
        played from Hidden ignores its base cost (811.1.b, 356.1.b)."""
        energy = 0 if hidden else card.cost.energy
        pips = [] if hidden else list(card.cost.power)
        if card.card_id in LEGION_DISCOUNT and self.cards_played[seat] >= 1:   # 812
            energy -= LEGION_DISCOUNT[card.card_id]
        if card.card_id in ENERGY_DISCOUNT:                                     # 356.4
            energy -= ENERGY_DISCOUNT[card.card_id](self, seat)
        if accelerate:                                                          # 805.1.a: [1][C]
            energy += 1
            pips.append(Pip.CARD)
        pips += [Pip.ANY] * deflect                                             # 809.1.c
        return Cost(max(0, energy), tuple(pips))                                # 356.6

    def _deflect(self, seat: int, targets: list[CardInstance]) -> int:
        """809: +[A] for each time an opponent's Deflect unit is chosen."""
        return sum(t.card.keyword_value(Keyword.DEFLECT) + t.granted.get("Deflect", 0)
                   for t in targets if t.controller != seat and t.card.is_unit)

    def _playable_objects(self, seat: int) -> list[tuple[CardInstance, int | None]]:
        """Cards `seat` could play: hand, Chosen Champion (108.3.d), and cards they
        hid on an earlier turn (811.1.b), with the battlefield they're hidden at."""
        zones = self.players[seat]
        objs: list[tuple[CardInstance, int | None]] = [(o, None) for o in [*zones.hand, *zones.champion]]
        for i, bf in enumerate(self.battlefields):
            objs += [(o, i) for o in bf.facedown
                     if o.controller == seat and o.hidden_turn is not None and o.hidden_turn < self.turn]
        return objs

    def _play_options(self, seat: int) -> list[Action]:
        actions, seen = [], set()
        for obj, hidden_at in self._playable_objects(seat):
            card = obj.card
            hidden = hidden_at is not None
            key = (obj.zone.kind, card.card_id, hidden_at)     # copies in one zone are interchangeable
            if key in seen or not self._playable_now(seat, card, hidden=hidden):
                continue
            seen.add(key)
            script = SPELLS.get(card.card_id) if card.is_spell else None
            target_sets: list[tuple[CardInstance, ...]] = [()]
            if script is not None:
                target_sets = self._target_options(seat, script, at=hidden_at)
            accelerate = [False]
            if card.is_unit and card.has_keyword(Keyword.ACCELERATE) and not hidden:
                accelerate.append(True)                                         # 805.2
            for targets in target_sets:
                if card.is_permanent:
                    destinations = [hidden_at] if hidden else \
                        (self.unit_destinations(seat, card) if card.is_unit else [None])   # 811.1.d.1 / 355.2
                elif script is not None and script.move:
                    destinations = self._move_destinations(targets[0])          # 355.4
                else:
                    destinations = [None]
                deflect = self._deflect(seat, list(targets))
                for accel in accelerate:
                    cost = self.total_cost(seat, card, accelerate=accel, deflect=deflect, hidden=hidden)
                    for payment in self.payment_options(seat, card, cost):
                        for dest in destinations:
                            label = f"Play {card.name}"
                            if hidden:
                                label += f" from hidden at {self.location_name(hidden_at)}"
                            if targets:
                                label += " on " + " and ".join(self._describe_target(t) for t in targets)
                            if dest is not None or (script is not None and script.move):
                                if not hidden:
                                    label += f" to {self.location_name(dest)}"
                            if accel:
                                label += ", accelerated"
                            if payment != Payment():
                                label += f" ({self._describe(seat, payment)})"
                            actions.append(Action(ActionKind.PLAY_CARD, label, obj.oid, payment,
                                                  destination=dest, targets=tuple(t.oid for t in targets),
                                                  accelerate=accel))
        return actions

    def _move_destinations(self, unit: CardInstance) -> list[int | None]:
        """Where an effect can move a unit: anywhere but where it is (355.4.a).
        Moves that would be ignored (out of Vilemaw's Lair to base) aren't offered."""
        origin = self.battlefield_index(unit)
        dests: list[int | None] = []
        if origin is not None and self.battlefields[origin].card.card.card_id not in NO_RETREAT:
            dests.append(None)
        dests += [i for i in range(len(self.battlefields)) if i != origin]
        return dests

    def _target_options(self, seat: int, script: SpellScript | AbilityScript,
                        at: int | None = None) -> list[tuple[CardInstance, ...]]:
        """355.5-355.9: every distinct way to choose the targets. Interchangeable
        units (same card, place and state) count once, and when every choice has
        the same kind their order doesn't matter. A spell with a target and no
        legal choice can't be played (355.8). `at` restricts choices to the
        battlefield a hidden card is played from (811.1.d.2)."""
        kinds = script.targets
        if not kinds:
            return [()]
        candidates = self._target_candidates()
        slots = [[o for o in candidates if self._valid_target(o, k, seat, at=at)] for k in kinds]
        least = len(kinds) if getattr(script, "min_targets", None) is None else script.min_targets
        distinct = getattr(script, "distinct", False)
        unordered = len(set(kinds)) == 1
        result, seen = [], set()
        for n in range(len(kinds), least - 1, -1):
            for combo in itertools.product(*slots[:n]):
                if not combo or (distinct and len({o.oid for o in combo}) < len(combo)):
                    continue
                if unordered:
                    counts: dict[int, int] = {}
                    for obj in combo:
                        counts[obj.oid] = counts.get(obj.oid, 0) + 1
                    per_group: dict[tuple, list[int]] = {}
                    for obj in {o.oid: o for o in combo}.values():
                        per_group.setdefault(self._target_key(obj), []).append(counts[obj.oid])
                    key = (n, tuple(sorted((k, tuple(sorted(v))) for k, v in per_group.items())))
                    if distinct and any(len(v) > 1 for v in per_group.values()):
                        # two interchangeable units: which two doesn't matter
                        key = (n, tuple(sorted((k, len(v)) for k, v in per_group.items())))
                else:
                    key = (n, tuple(self._target_key(o) for o in combo))
                if key not in seen:
                    seen.add(key)
                    result.append(combo)
        return result

    def _target_candidates(self) -> list[CardInstance]:
        """Units on the board, spells on the chain and battlefields, by oid."""
        spells = [i.obj for i in self.chain_items if i.obj is not None]
        objs = self.units() + spells + [bf.card for bf in self.battlefields]
        return sorted(objs, key=lambda o: o.oid)

    def _target_key(self, obj: CardInstance) -> tuple:
        return self._state_key(obj) if obj.card.is_unit else ("object", obj.oid)

    def _valid_target(self, obj: CardInstance, kind: str, seat: int, *, at: int | None = None) -> bool:
        if kind in SPELL_KINDS:                                                 # 355.9.a.2
            if obj.zone is not self.chain or not obj.card.is_spell:
                return False
            if kind == CHEAP_SPELL:                                             # 206: printed cost
                return obj.card.cost.energy <= 4 and obj.card.cost.power_amount <= 1
            return True
        if kind == BATTLEFIELD:
            index = self.battlefield_index(obj)
            return obj.card.is_battlefield and index is not None and (at is None or index == at)
        if kind not in UNIT_KINDS:
            return False
        if not obj.card.is_unit or obj.zone is None or obj.zone.kind not in (ZoneKind.BASE, ZoneKind.BATTLEFIELD):
            return False
        where = self.battlefield_index(obj)
        if at is not None and where != at:                                      # 811.1.d.2
            return False
        if kind in AT_BATTLEFIELD_KINDS and where is None:
            return False
        if kind == SMALL_UNIT_AT_BATTLEFIELD:
            return self.might(obj) <= 3
        if kind in (FRIENDLY_UNIT, FRIENDLY_UNIT_AT_BATTLEFIELD):
            return obj.controller == seat
        if kind == ENEMY_UNIT:
            return obj.controller != seat
        return True

    def _find_playable(self, seat: int, oid: int) -> tuple[CardInstance, int | None]:
        return next((o, at) for o, at in self._playable_objects(seat) if o.oid == oid)

    def _play_card(self, seat: int, action: Action) -> None:
        obj, hidden_at = self._find_playable(seat, action.card_oid)
        hidden = hidden_at is not None
        card = obj.card
        candidates = {o.oid: o for o in self._target_candidates()}
        targets = [candidates[oid] for oid in action.targets]
        script = SPELLS.get(card.card_id) if card.is_spell else None
        kinds = script.targets if script is not None else ()
        cost = self.total_cost(seat, card, accelerate=action.accelerate, deflect=self._deflect(seat, targets),
                               hidden=hidden)
        self.move(obj, self.chain)                                              # 354
        item = ChainItem(seat, obj=obj, targets=[(t, t.oid, k) for t, k in zip(targets, kinds)],
                         label=card.name)
        if hidden:
            item.data["hidden_at"] = hidden_at
        if script is not None and script.move:
            item.data["destination"] = action.destination
        if not self.chain_items:
            self.chain_by_trigger = False
        self.chain_items.append(item)
        self._pay(seat, card, cost, action.payment or Payment())                # 357
        self.cards_played[seat] += 1
        on = f" on {' and '.join(self._describe_target(t) for t in targets)}" if targets else ""
        accelerated = ", accelerated" if action.accelerate else ""
        from_hidden = " from hidden" if hidden else ""
        to = f" to {self.location_name(action.destination)}" \
            if card.is_unit and action.destination is not None else ""
        self._log(f"{self.names[seat]} plays {card.name}{from_hidden}{on}{to}{accelerated}")
        if card.is_permanent:                                                   # 337.2: resolves at once
            self.chain_items.remove(item)
            dest = self.players[seat].base if action.destination is None \
                else self.battlefields[action.destination].units
            self.move(obj, dest)                                                # 355.2
            obj.controller = seat
            obj.exhausted = card.is_unit and not action.accelerate \
                and card.card_id not in ENTERS_READY                            # 359.2.c-d / 805.6 / 369.3
            self._emit("played", seat=seat, obj=obj)
            self._after_resolve()
        else:
            self.priority = seat                                                # 337.4
            self.passes = 0
            self._emit("played", seat=seat, obj=obj)
            if targets:
                self._emit("chosen", seat=seat, targets=targets)                # 383.4.b

    # --- hiding (421, 811) ------------------------------------------------------

    def _hide_options(self, seat: int) -> list[Action]:
        """421.2: a Discretionary Action on your own turn in a Neutral Open State:
        pay [A] to put a card with Hidden facedown at a battlefield you control
        whose facedown zone is empty (107.3.b)."""
        if not (seat == self.turn_player and self.turn_phase is TurnPhase.MAIN):
            return []
        zones = self.players[seat]
        spots = [i for i, bf in enumerate(self.battlefields) if bf.controller == seat and not bf.facedown.objects]
        actions, seen = [], set()
        for obj in [*zones.hand, *zones.champion]:
            card = obj.card
            if not card.has_keyword(Keyword.HIDDEN) or (obj.zone.kind, card.card_id) in seen:
                continue
            seen.add((obj.zone.kind, card.card_id))
            for payment in self.payment_options(seat, card, HIDE_COST, spell=False):
                for i in spots:
                    actions.append(Action(ActionKind.HIDE, f"Hide {card.name} at {self.location_name(i)} "
                                          f"({self._describe(seat, payment)})", obj.oid, payment, destination=i))
        return actions

    def _hide(self, seat: int, action: Action) -> None:
        zones = self.players[seat]
        obj = next(o for o in [*zones.hand, *zones.champion] if o.oid == action.card_oid)
        self._pay(seat, obj.card, HIDE_COST, action.payment or Payment(), spell=False)
        self.move(obj, self.battlefields[action.destination].facedown)
        obj.controller = seat                                                   # 191.1
        obj.facedown = True
        obj.hidden_turn = self.turn
        self._log(f"{self.names[seat]} hides a card at {self.location_name(action.destination)}")

    # --- activated abilities (376-381) -----------------------------------------

    def _ability_options(self, seat: int) -> list[Action]:
        """381: on your own turn in an Open State, unless the ability has Reaction."""
        if self.turn_phase is not TurnPhase.MAIN:
            return []
        actions = []
        for source in self._on_board(seat):
            for index, ability in enumerate(ACTIVATED.get(source.card.card_id, ())):
                if not ability.reaction and (seat != self.turn_player or self.chain_items):
                    continue
                if ability.exhaust and source.exhausted:
                    continue
                if len(self.players[seat].trash) < ability.recycle_trash:       # 416.3
                    continue
                target_sets = self._target_options(seat, ability) if ability.targets else [()]
                for targets in target_sets:
                    cost = self.total_cost_ability(ability, self._deflect(seat, list(targets)))
                    for payment in self.payment_options(seat, source.card, cost, spell=False):
                        label = f"{source.card.name}: {ability.text}"
                        if targets:
                            label += " on " + " and ".join(self._describe_target(t) for t in targets)
                        if payment != Payment():
                            label += f" ({self._describe(seat, payment)})"
                        actions.append(Action(ActionKind.ACTIVATE, label, source.oid, payment,
                                              targets=tuple(t.oid for t in targets), choice=index))
        return actions

    @staticmethod
    def total_cost_ability(ability: AbilityScript, deflect: int) -> Cost:
        return Cost(ability.cost.energy, ability.cost.power + (Pip.ANY,) * deflect)

    def _activate(self, seat: int, action: Action) -> None:
        source = next(o for o in self._on_board(seat) if o.oid == action.card_oid)
        ability = ACTIVATED[source.card.card_id][action.choice]
        candidates = {o.oid: o for o in self._target_candidates()}
        targets = [candidates[oid] for oid in action.targets]
        if ability.exhaust:
            source.exhausted = True                                             # 414.5
        if ability.recycle_trash:
            # Which cards get recycled is picked for the player: non-spells first,
            # since spells in the trash can be returned by other cards.
            trash = sorted(self.players[seat].trash, key=lambda o: (o.card.is_spell, o.oid))
            self.recycle(trash[:ability.recycle_trash])
        cost = self.total_cost_ability(ability, self._deflect(seat, targets))
        self._pay(seat, source.card, cost, action.payment or Payment(), spell=False)
        item = ChainItem(seat, ability=(source.card.card_id, action.choice), source=source,
                         source_oid=source.oid, targets=[(t, t.oid, k) for t, k in zip(targets, ability.targets)],
                         label=f"{source.card.name}: {ability.text}")
        if not self.chain_items:
            self.chain_by_trigger = False
        self.chain_items.append(item)                                           # 377.3
        on = f" on {' and '.join(self._describe_target(t) for t in targets)}" if targets else ""
        self._log(f"{self.names[seat]} uses {source.card.name}{on}")
        self.priority = seat
        self.passes = 0

    # --- the chain --------------------------------------------------------------

    def _pass_priority(self, seat: int) -> None:
        self.passes += 1
        if self.passes >= len(self.players):                                    # 339.1
            self._resolve_top()
        else:
            self.priority = self._opponent(seat)                                # 339.2

    def _item_script(self, item: ChainItem) -> SpellScript | TriggerScript | AbilityScript:
        if item.obj is not None:
            return SPELLS[item.obj.card.card_id]
        if item.ability is not None:
            card_id, index = item.ability
            return ACTIVATED[card_id][index]
        return _trigger_script(item)

    def _resolve_top(self) -> None:
        """340.1: the newest item resolves. Choices it needs are asked first (355.17);
        the item stays on the chain until they're made. A spell then goes to its
        owner's trash."""
        item = self.chain_items[-1]
        script = self._item_script(item)
        if not item.data.get("_resolving"):
            item.data["_resolving"] = True
            self._log(f"{item.label} resolves")
        ask_fn = getattr(script, "ask", None)
        while ask_fn is not None:
            ask: Ask | None = ask_fn(self, item)
            if ask is None:
                break
            if len(ask.options) <= 1:
                item.data[ask.key] = ask.options[0][1] if ask.options else None
                continue
            self.decision = Decision(ask.seat, "resolve", ask.options, item, key=ask.key, private=ask.private)
            self.priority = None
            return
        self.chain_items.pop()
        script.resolve(self, item)
        if item.obj is not None:
            owner = self.players[item.obj.owner]
            if item.obj.zone is self.chain:
                self.move(item.obj, owner.banishment if script.banish else owner.trash)   # 359.3.d / 427
        self._after_resolve()

    def _after_resolve(self) -> None:
        self.passes = 0
        if self.chain_items:
            self.priority = self.chain_items[-1].controller                     # 340.4
            return
        by_trigger, self.chain_by_trigger = self.chain_by_trigger, False
        if self.showdown is not None:
            if not by_trigger:                                                  # 340.2.a / 346.1
                self.focus = self._opponent(self.focus)
                self.showdown_passes = 0
            self.priority = self.focus
        elif self.turn_phase is TurnPhase.MAIN:
            self.priority = self.turn_player                                    # 335
        else:
            self.priority = None                                                # the start or end of turn carries on

    # --- paying costs (356-357) -----------------------------------------------

    def payment_options(self, seat: int, card: CardDef, cost: Cost | None = None, *,
                        spell: bool | None = None) -> list[Payment]:
        """Every sensible way `seat` can pay `cost` (by default the card's own) right now.

        Energy has no domain and an exhausted rune can still be recycled, so what
        a payment leaves behind comes down to how many runes stay ready, which
        domains were recycled, and whether the legend was used. Options are offered
        once per such outcome. Anything already in the pool is spent first, nothing
        is added beyond the cost, and options that leave fewer ready runes than
        another option with the same recycles are dropped. `spell` says whether
        spells-only Power may be used (by default: whether the card is a spell).
        """
        cost = card.cost if cost is None else cost
        spell = card.is_spell if spell is None else spell
        zones = self.players[seat]
        pool = zones.rune_pool
        pips = [p.payable_with(card.domains) for p in cost.power]
        usable = [u for u in pool.power if spell or not u.spells_only]
        energy_needed = max(0, cost.energy - pool.energy)
        power_needed = len(pips) - len(_match(pips, usable))

        ready: dict[str, list[CardInstance]] = {}
        tired: dict[str, list[CardInstance]] = {}
        for rune in sorted(zones.runes(), key=lambda r: r.oid):
            (tired if rune.exhausted else ready).setdefault(_rune_domain(rune), []).append(rune)
        domains = sorted(set(ready) | set(tired))

        legend = zones.legend.objects[0] if zones.legend.objects else None
        legend_uses = [False]
        if (power_needed and legend is not None and not legend.exhausted
                and legend.card.card_id in LEGEND_POWER
                and (spell or not LEGEND_POWER[legend.card.card_id])):
            legend_uses.append(True)

        candidates: dict[tuple, Payment] = {}
        for exhaust in _splits(energy_needed, [len(ready.get(d, [])) for d in domains]):
            for use_legend in legend_uses:
                recycle_total = power_needed - use_legend
                limits = [len(ready.get(d, [])) + len(tired.get(d, [])) for d in domains]
                for recycle in _splits(recycle_total, limits):
                    units = usable + [Power(d) for d, k in zip(domains, recycle) for _ in range(k)]
                    if use_legend:
                        units.append(Power("A", LEGEND_POWER[legend.card.card_id]))
                    if len(_match(pips, units)) < len(pips):
                        continue
                    payment, ready_after = _build_payment(domains, ready, tired, exhaust, recycle, use_legend)
                    candidates.setdefault((tuple(recycle), use_legend, sum(ready_after)), payment)

        # drop options another option beats: same recycles, legend use no worse,
        # and at least as many ready runes left
        keys = list(candidates)
        best = []
        for key in keys:
            recycle, use_legend, ready_after = key
            dominated = any(
                other != key and other[0] == recycle and other[1] <= use_legend and other[2] >= ready_after
                for other in keys)
            if not dominated:
                best.append(candidates[key])
        return best

    def _pay(self, seat: int, card: CardDef, cost: Cost, payment: Payment, *, spell: bool | None = None) -> None:
        spell = card.is_spell if spell is None else spell
        zones = self.players[seat]
        pool = zones.rune_pool
        runes = {r.oid: r for r in zones.runes()}
        for oid in payment.exhaust:                                             # 164.2.a
            rune = runes[oid]
            if rune.exhausted:
                raise ValueError(f"{rune.card.name} is already exhausted")
            rune.exhausted = True
            pool.energy += 1
        for oid in payment.recycle:                                             # 164.2.b
            rune = runes[oid]
            pool.power.append(Power(_rune_domain(rune)))
            self.move(rune, zones.rune_deck, bottom=True)                       # 416.1.b
        if payment.legend:
            legend = zones.legend.objects[0]
            legend.exhausted = True
            pool.power.append(Power("A", LEGEND_POWER[legend.card.card_id]))

        if pool.energy < cost.energy:
            raise ValueError(f"not enough Energy to play {card.name}")
        pips = [p.payable_with(card.domains) for p in cost.power]
        usable = [u for u in pool.power if spell or not u.spells_only]
        matched = _match(pips, usable)
        if len(matched) < len(pips):
            raise ValueError(f"not enough Power to play {card.name}")
        pool.energy -= cost.energy
        for unit in matched:
            pool.power.remove(unit)

    def _describe(self, seat: int, payment: Payment) -> str:
        runes = {r.oid: r for r in self.players[seat].runes()}

        def domains(oids: tuple[int, ...]) -> str:
            counts: dict[str, int] = {}
            for oid in oids:
                name = Domain(_rune_domain(runes[oid])).name.title()
                counts[name] = counts.get(name, 0) + 1
            return " + ".join(f"{n} {name}" for name, n in counts.items())

        parts = []
        if payment.exhaust:
            parts.append("exhaust " + domains(payment.exhaust))
        if payment.recycle:
            parts.append("recycle " + domains(payment.recycle))
        if payment.legend:
            parts.append("use " + self.players[seat].legend.objects[0].card.name)
        return "; ".join(parts)

    # --- helpers --------------------------------------------------------------

    def _state_key(self, obj: CardInstance) -> tuple:
        """Everything that tells two units apart for the rules, besides identity."""
        where = ("base", obj.controller) if obj.zone is None or obj.zone.kind is ZoneKind.BASE else \
            ("bf", next(i for i, bf in enumerate(self.battlefields) if bf.units is obj.zone))
        return (where, obj.controller, obj.card.card_id, obj.exhausted, obj.damage, obj.buffs,
                tuple(obj.might_mods), tuple(sorted(obj.granted.items())))

    def _describe_unit(self, obj: CardInstance) -> str:
        where = "base" if obj.zone.kind is ZoneKind.BASE else \
            next(self._bf_name(bf) for bf in self.battlefields if bf.units is obj.zone)
        return f"{obj.card.name} ({self.names[obj.controller]}, {where})"

    def _describe_target(self, obj: CardInstance) -> str:
        if obj.card.is_unit:
            return self._describe_unit(obj)
        if obj.card.is_battlefield:
            return self._bf_name(self.battlefields[self.battlefield_index(obj)])
        return f"{obj.card.name} ({self.names[obj.controller]}, on the chain)"

    def _bf_name(self, bf: Battlefield) -> str:
        """A battlefield's name, plus whose it is when both players brought the same one."""
        name = bf.card.card.name
        if sum(b.card.card.name == name for b in self.battlefields) > 1:
            name += f" (brought by {self.names[bf.card.owner]})"
        return name

    def _opponent(self, seat: int) -> int:
        return (seat + 1) % len(self.players)

    def _log(self, text: str) -> None:
        self.log.append(text)

    # --- observation ----------------------------------------------------------

    def observation(self, viewer: int) -> dict[str, Any]:
        """Everything `viewer` is allowed to know (128). JSON-serializable."""
        players = []
        for seat, zones in enumerate(self.players):
            players.append({
                "seat": seat,
                "name": self.names[seat],
                "points": self.points[seat],
                "rune_pool": zones.rune_pool.view(),
                "zones": {z.kind.value: z.view(viewer) for z in zones.all()},
            })
        chain = []
        for item in self.chain_items:
            if item.obj is not None:
                entry = item.obj.view()
            else:
                card_id = (item.trigger or item.ability)[0]
                entry = {"ability": item.label, "card_id": card_id, "source_oid": item.source_oid}
            chain.append(dict(entry, chain_controller=item.controller))
        decision = None
        if self.decision is not None:
            d = self.decision
            visible = viewer == d.seat or not d.private
            decision = {"seat": d.seat, "kind": d.kind, "count": len(d.options),
                        "options": [_option_view(label, value) for label, value in d.options] if visible else []}
        return {
            "viewer": viewer,
            "turn": self.turn,
            "turn_player": self.turn_player,
            "turn_phase": self.turn_phase.value,
            "acting_player": self.acting_player,
            "priority": self.priority,
            "focus": self.focus,
            "showdown": self.showdown,
            "attacker": self.attacker,
            "assigning": self.assigning[0] if self.assigning else None,
            "decision": decision,
            "cards_played": list(self.cards_played),
            "scored_this_turn": sorted([seat, i] for seat, oid in self.scored
                                       for i, bf in enumerate(self.battlefields) if bf.card.oid == oid),
            "extra_turns": list(self.extra_turns),
            "phase": self.phase.value,
            "victory_score": self.victory_score,
            "winner": self.winner,
            "players": players,
            "battlefields": [bf.view(viewer) for bf in self.battlefields],
            "chain": chain,
            "unit_might": {str(u.oid): self.might(u) for u in self.units()},
        }


def _option_view(label: str, value: Any) -> dict[str, Any]:
    """A Decision option for observations: what it points at, if anything.
    Values are None (decline), True (accept), a battlefield index (which showdown
    starts), (object, oid), ("amount", n) or ("location", battlefield index or
    None for base)."""
    view: dict[str, Any] = {"label": label, "declines": value is None}
    if isinstance(value, tuple) and isinstance(value[0], CardInstance):
        view["oid"] = value[1]
        view["card_id"] = value[0].card.card_id
    elif isinstance(value, tuple) and value[0] == "amount":
        view["amount"] = value[1]
    elif isinstance(value, tuple) and value[0] == "location":
        view["location"] = value[1]
    elif isinstance(value, int) and not isinstance(value, bool):
        view["battlefield"] = value
    return view


def _trigger_script(item: ChainItem) -> TriggerScript:
    card_id, index = item.trigger
    return TRIGGERS[card_id][index]


def _rune_domain(rune: CardInstance) -> str:
    """164.2.b.1: the Power a recycled rune adds. Basic runes have one domain."""
    return min(d.value for d in rune.card.domains)


def _splits(total: int, limits: list[int]) -> list[list[int]]:
    """Every way to split `total` into len(limits) parts with part i <= limits[i]."""
    if not limits:
        return [[]] if total == 0 else []
    result = []
    for first in range(min(total, limits[0]) + 1):
        for rest in _splits(total - first, limits[1:]):
            result.append([first, *rest])
    return result


def _match(pips: list[frozenset[Domain]], units: list[Power]) -> list[Power]:
    """Largest set of Power `units` that pays the `pips` (bipartite matching)."""
    owner: dict[int, int] = {}          # unit index -> pip index

    def fits(unit: Power, pip: frozenset[Domain]) -> bool:
        return unit.domain == "A" or Domain(unit.domain) in pip

    def augment(p: int, seen: set[int]) -> bool:
        for u, unit in enumerate(units):
            if u in seen or not fits(unit, pips[p]):
                continue
            seen.add(u)
            if u not in owner or augment(owner[u], seen):
                owner[u] = p
                return True
        return False

    for p in range(len(pips)):
        augment(p, set())
    return [units[u] for u in owner]


def _build_payment(domains: list[str], ready: dict[str, list[CardInstance]],
                   tired: dict[str, list[CardInstance]], exhaust: list[int],
                   recycle: list[int], use_legend: bool) -> tuple[Payment, list[int]]:
    """Pick concrete runes for a per-domain split. Recycle already-exhausted runes
    first, then runes exhausted for this cost, and only then other ready runes."""
    exhaust_oids: list[int] = []
    recycle_oids: list[int] = []
    ready_after = []
    for d, ex, k in zip(domains, exhaust, recycle):
        r, t = ready.get(d, []), tired.get(d, [])
        exhausted_now = r[:ex]
        from_tired = t[:k]
        k -= len(from_tired)
        from_exhausted_now = exhausted_now[:k]
        k -= len(from_exhausted_now)
        from_ready = r[ex:ex + k]
        exhaust_oids += [o.oid for o in exhausted_now]
        recycle_oids += [o.oid for o in (*from_tired, *from_exhausted_now, *from_ready)]
        ready_after.append(len(r) - ex - len(from_ready))
    payment = Payment(tuple(sorted(exhaust_oids)), tuple(sorted(recycle_oids)), use_legend)
    return payment, ready_after


def play_game(game: Game, agents: list[Agent]) -> int:
    """Run a game to completion with one agent per seat. Returns the winning seat."""
    while not game.is_over:
        seat = game.acting_player
        legal = game.legal_actions(seat)
        game.step(agents[seat].act(game.observation(seat), legal))
    return game.winner


# The Origins meta decks, by short name (decks/<name>.json).
DECK_NAMES = ("kaisa", "annie", "master_yi", "miss_fortune")


def load_deck(name: str, pool: dict[str, CardDef] | None = None) -> Deck:
    pool = pool if pool is not None else load_card_pool(ROOT / "card_data")
    return Deck.load(ROOT / "decks" / f"{name}.json", pool)


def load_decks(names: tuple[str, ...] = DECK_NAMES) -> dict[str, Deck]:
    pool = load_card_pool(ROOT / "card_data")
    return {name: load_deck(name, pool) for name in names}


def load_demo_decks() -> tuple[Deck, Deck]:
    deck = load_deck("kaisa")
    return deck, deck


if __name__ == "__main__":
    a, b = load_demo_decks()
    assert not a.errors(), a.errors()
    wins = [0, 0]
    turns = 0
    for i in range(200):
        game = Game([a, b], seed=i)
        wins[play_game(game, [RandomAgent(i), RandomAgent(i + 1)])] += 1
        turns += game.turn
    print(f"200 games, wins by seat: {wins}, average length {turns / 200:.1f} turns")
