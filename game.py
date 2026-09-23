"""Game engine: decks, setup, the turn, paying costs, the chain and agents.

Implemented: setup with the mulligan (110-118), the phases of the turn (314-317),
runes and rune pools (160-168), paying costs (356-357), playing cards onto the
chain with priority (327-340, 349-359), the cleanup win check (323.1), Hold
scoring (467-471) and Burn Out (431).

Not yet: movement, showdowns, focus and combat, damage and death, triggered and
activated abilities, keywords, and spell effects. A spell can only be played
once it has an entry in SPELL_EFFECTS, so for now the real decks only play units.

The decision loop is `acting_player` / `legal_actions` / `step` / `observation`.
Everything that needs no decision runs inside `step`, and when a player's only
option is to pass priority or end their turn, the engine does it for them.
Game state is plain data (no generators or callbacks), so games deep-copy and
pickle.

    python3 game.py            # headless: random agent vs random agent
"""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol

from cards import CardDef, Domain, Keyword, deck_errors, load_card_pool
from zones import Battlefield, CardInstance, PlayerZones, Power, Zone, ZoneKind

ROOT = Path(__file__).parent
VICTORY_SCORE = 8          # 485.3
OPENING_HAND = 4           # 116
MULLIGAN_MAX = 2           # 117.1
CHANNEL_PER_TURN = 2       # 315.3.b
SECOND_PLAYER_EXTRA = 1    # 485.7: extra rune on the second player's first Channel Phase


# ---------------------------------------------------------------------------
# Card scripts
# ---------------------------------------------------------------------------

# card_id -> what the spell does when it resolves (359.3.d). The effects system
# will fill this in; until then only spells registered here can be played.
SpellEffect = Callable[["Game", "ChainItem"], None]
SPELL_EFFECTS: dict[str, SpellEffect] = {}

# Legends with "[E]: [Reaction] — Add [A]" that can help pay costs. The value is
# whether that Power can only be used to play spells.
LEGEND_POWER: dict[str, bool] = {
    "OGN-247": True,       # Kai'Sa, Daughter of the Void
}


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
             "deck": {"Main Board": [{"id": "OGN-039", "count": 1}, ...], "Side Board": [...]}}

        Main Board is one flat list, so entries are sorted by card type (103):
        the legend, the battlefields, the runes, and the main deck. One copy of
        the champion unit whose champion tag matches the legend is set aside as
        the Chosen Champion (103.2.a); any further copies stay in the main deck.
        The Side Board is ignored - sideboarding isn't a Core Rules deck zone.
        """
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
        board = spec["deck"]["Main Board"] if "deck" in spec else spec["Main Board"]
        cards: list[CardDef] = [pool[e["id"]] for e in board for _ in range(e.get("count", 1))]

        legends = [c for c in cards if c.is_legend]
        if len(legends) != 1:
            raise ValueError(f"decklist needs exactly 1 legend, found {[c.name for c in legends]}")
        legend = legends[0]
        champions = [c for c in cards if c.is_champion_unit and c.champion_tag == legend.champion_tag]
        if not champions:
            raise ValueError(f"decklist has no champion unit with the tag {legend.champion_tag!r}")
        champion = champions[0]

        main, runes, battlefields = [], [], []
        champion_taken = False
        for card in cards:
            if card is legend:
                continue
            if card.is_rune:
                runes.append(card)
            elif card.is_battlefield:
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
    PASS = "pass"              # 338.1.b: pass priority while a chain exists
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

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind.value, "label": self.label}
        if self.card_oid is not None:
            result["card_oid"] = self.card_oid
        if self.payment is not None:
            result["payment"] = self.payment.to_json()
        if self.kind is ActionKind.MULLIGAN:
            result["set_aside"] = list(self.set_aside)
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
    BEGINNING = "beginning"    # 315.2
    CHANNEL = "channel"        # 315.3
    DRAW = "draw"              # 315.4
    MAIN = "main"              # 316
    ENDING = "ending"          # 317


@dataclass(eq=False)
class ChainItem:
    """329: something on the chain. Only cards for now; abilities join later."""
    obj: CardInstance
    controller: int


class Game:
    def __init__(self, decks: list[Deck], names: list[str] | None = None, *, seed: int | None = None):
        if len(decks) != 2:
            raise ValueError("only 1v1 Duel (485) is supported")
        self.rng = random.Random(seed)
        self.seed = seed
        self.names = names or [f"Player {i + 1}" for i in range(len(decks))]
        self.decks = decks
        self.players = [PlayerZones(i) for i in range(len(decks))]
        self.points = [0] * len(decks)
        self.battlefields: list[Battlefield] = []
        self.set_aside: list[CardInstance] = []           # unused battlefields (485.5)
        self.chain = Zone(ZoneKind.CHAIN, None)
        self.chain_items: list[ChainItem] = []            # parallel to self.chain.objects
        self.turn = 0
        self.turn_player = 0
        self.turn_phase = TurnPhase.MULLIGAN
        self.first_player = 0
        self.phase = Phase.PLAYING
        self.winner: int | None = None
        self.priority: int | None = None                  # 312
        self.passes = 0                                   # consecutive passes on the chain (339.1)
        self.mulligan_queue: list[int] = []
        self.channel_phases = [0] * len(decks)            # for 485.7
        self.scored: set[tuple[int, int]] = set()         # (seat, battlefield oid) scored this turn (470)
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

    def _channel(self, seat: int, n: int, *, exhausted: bool = False) -> None:
        """430: put the top n runes of the Rune Deck into the base."""
        zones = self.players[seat]
        for rune in zones.rune_deck.top(n):                                     # 430.3
            self.move(rune, zones.base)
            rune.exhausted = exhausted                                          # 430.2.a

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
        return self.priority

    def legal_actions(self, seat: int | None = None) -> list[Action]:
        seat = self.acting_player if seat is None else seat
        if self.is_over or seat is None or seat != self.acting_player:
            return []
        if self.turn_phase is TurnPhase.MULLIGAN:
            actions = self._mulligan_options(seat)
        else:
            actions = self._play_options(seat)
            if self.chain_items:
                actions.append(Action(ActionKind.PASS, "Pass"))
            else:
                actions.append(Action(ActionKind.END_TURN, "End turn"))
        actions.append(Action(ActionKind.CONCEDE, "Concede"))
        return actions

    def step(self, action: Action) -> None:
        seat = self.acting_player
        if action not in self.legal_actions(seat):
            raise ValueError(f"illegal action {action} for seat {seat}")
        self._apply(seat, action)
        self._advance()

    def _apply(self, seat: int, action: Action) -> None:
        if action.kind is ActionKind.CONCEDE:
            self._log(f"{self.names[seat]} concedes")
            self._end(winner=self._opponent(seat))
        elif action.kind is ActionKind.MULLIGAN:
            self._mulligan(seat, action.set_aside)
        elif action.kind is ActionKind.PLAY_CARD:
            self._play_card(seat, action)
        elif action.kind is ActionKind.PASS:
            self._pass_priority(seat)
        elif action.kind is ActionKind.END_TURN:
            self._end_turn()
        self._cleanup()

    def _advance(self) -> None:
        """Take every action that isn't a real choice: when passing priority or
        ending the turn is all a player can do (conceding aside), do it for them."""
        while not self.is_over:
            seat = self.acting_player
            options = [a for a in self.legal_actions(seat) if a.kind is not ActionKind.CONCEDE]
            if len(options) != 1 or options[0].kind not in (ActionKind.PASS, ActionKind.END_TURN):
                return
            self._apply(seat, options[0])

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
            self._start_turn(self.first_player)                                 # 118

    # --- the turn (314-317) ---------------------------------------------------

    def _start_turn(self, seat: int) -> None:
        self.turn += 1
        self.turn_player = seat
        self.scored.clear()
        self._log(f"Turn {self.turn}: {self.names[seat]}")

        self.turn_phase = TurnPhase.AWAKEN                                      # 315.1
        for obj in self._on_board(seat):
            obj.exhausted = False

        self.turn_phase = TurnPhase.BEGINNING                                   # 315.2
        for bf in self.battlefields:                                            # 315.2.b: Hold
            if bf.controller == seat:
                self._score(seat, bf, conquer=False)
        if self._cleanup():
            return

        self.turn_phase = TurnPhase.CHANNEL                                     # 315.3
        n = CHANNEL_PER_TURN
        if seat != self.first_player and self.channel_phases[seat] == 0:
            n += SECOND_PLAYER_EXTRA                                            # 485.7
        self.channel_phases[seat] += 1
        self._channel(seat, n)

        self.turn_phase = TurnPhase.DRAW                                        # 315.4
        self.draw(seat)
        if self._cleanup():                                                     # a Burn Out can end the game
            return

        self.turn_phase = TurnPhase.MAIN                                        # 316
        for p in self.players:                                                  # 316.3
            p.rune_pool.empty()
        self.priority = seat                                                    # 312.2.a
        self.passes = 0

    def _end_turn(self) -> None:
        seat = self.turn_player
        self._log(f"{self.names[seat]} ends turn {self.turn}")
        self.turn_phase = TurnPhase.ENDING                                      # 317
        self.priority = None
        for p in self.players:
            for obj in self._on_board(p.seat):
                if obj.card.is_unit:
                    obj.damage = 0                                              # 317.2.b: heal all units
            p.rune_pool.empty()                                                 # 317.2.d
        self._start_turn(self._opponent(seat))                                  # 317.3

    def _on_board(self, seat: int) -> list[CardInstance]:
        """Every object `seat` controls in their base, their legend zone, and at battlefields."""
        zones = self.players[seat]
        objs = [*zones.base, *zones.legend]
        for bf in self.battlefields:
            objs += [o for o in bf.units if o.controller == seat]
        return objs

    # --- scoring and winning --------------------------------------------------

    def _score(self, seat: int, bf: Battlefield, *, conquer: bool) -> None:
        """469-471: Hold or Conquer a battlefield, at most once per battlefield per turn."""
        key = (seat, bf.card.oid)
        if key in self.scored:                                                  # 470
            return
        self.scored.add(key)
        how = "conquers" if conquer else "holds"
        scored_all = all((seat, b.card.oid) in self.scored for b in self.battlefields)
        if conquer and self.points[seat] >= VICTORY_SCORE - 1 and not scored_all:
            self._log(f"{self.names[seat]} {how} {bf.card.card.name} but draws instead of the final point")
            self.draw(seat)                                                     # 471.1.b.1
            return
        self.points[seat] += 1
        self._log(f"{self.names[seat]} {how} {bf.card.card.name}: {self.points[seat]} points")

    def _cleanup(self) -> bool:
        """323. Only step 1 (the win check) so far. Returns True if the game ended."""
        if not self.is_over:
            best = max(self.points)
            leaders = [seat for seat, p in enumerate(self.points) if p == best]
            if best >= VICTORY_SCORE and len(leaders) == 1:                     # 323.1 / 194.2
                self._end(leaders[0])
        return self.is_over

    def _end(self, winner: int) -> None:
        self.winner = winner
        self.phase = Phase.GAME_OVER
        self.priority = None
        self._log(f"{self.names[winner]} wins")

    # --- playing cards (349-359) ----------------------------------------------

    def _playable_now(self, seat: int, card: CardDef) -> bool:
        """Card has rules support, and 308-310 timing allows it."""
        if card.is_spell and card.card_id not in SPELL_EFFECTS:
            return False
        if not (card.is_permanent or card.is_spell):
            return False
        if self.chain_items:                                                    # Closed: 309.1.a
            return card.has_keyword(Keyword.REACTION)
        # Neutral Open: the turn player, in their Main Phase (310.1.a)
        return seat == self.turn_player and self.turn_phase is TurnPhase.MAIN

    def _play_options(self, seat: int) -> list[Action]:
        zones = self.players[seat]
        actions, seen = [], set()
        for obj in [*zones.hand, *zones.champion]:                              # 108.3.d
            key = (obj.zone.kind, obj.card.card_id)        # copies in one zone are interchangeable
            if key in seen or not self._playable_now(seat, obj.card):
                continue
            seen.add(key)
            for payment in self.payment_options(seat, obj.card):
                label = f"Play {obj.card.name}"
                if payment != Payment():
                    label += f" ({self._describe(seat, payment)})"
                actions.append(Action(ActionKind.PLAY_CARD, label, obj.oid, payment))
        return actions

    def _play_card(self, seat: int, action: Action) -> None:
        zones = self.players[seat]
        obj = next(o for o in [*zones.hand, *zones.champion] if o.oid == action.card_oid)
        card = obj.card
        self.move(obj, self.chain)                                              # 354
        item = ChainItem(obj, seat)
        self.chain_items.append(item)
        self._pay(seat, card, action.payment or Payment())                      # 357
        self._log(f"{self.names[seat]} plays {card.name}")
        if card.is_permanent:                                                   # 337.2: resolves at once
            self.chain_items.remove(item)
            self.move(obj, zones.base)                                          # 355.2.a: units to base for now
            obj.exhausted = card.is_unit                                        # 359.2.c-d
            self._after_resolve()
        else:
            self.priority = seat                                                # 337.4
            self.passes = 0

    def _pass_priority(self, seat: int) -> None:
        self.passes += 1
        if self.passes >= len(self.players):                                    # 339.1
            self._resolve_top()
        else:
            self.priority = self._opponent(seat)                                # 339.2

    def _resolve_top(self) -> None:
        """340.1: the newest item resolves, then the spell goes to its owner's trash."""
        item = self.chain_items.pop()
        obj = item.obj
        self._log(f"{obj.card.name} resolves")
        SPELL_EFFECTS[obj.card.card_id](self, item)
        if obj.zone is self.chain:
            self.move(obj, self.players[obj.owner].trash)                       # 359.3.d
        self._after_resolve()

    def _after_resolve(self) -> None:
        self.passes = 0
        if self.chain_items:
            self.priority = self.chain_items[-1].controller                     # 340.4
        elif self.turn_phase is TurnPhase.MAIN:
            self.priority = self.turn_player                                    # 335

    # --- paying costs (356-357) -----------------------------------------------

    def payment_options(self, seat: int, card: CardDef) -> list[Payment]:
        """Every sensible way `seat` can pay for `card` right now.

        Energy has no domain and an exhausted rune can still be recycled, so what
        a payment leaves behind comes down to how many runes stay ready, which
        domains were recycled, and whether the legend was used. Options are offered
        once per such outcome. Anything already in the pool is spent first, nothing
        is added beyond the cost, and options that leave fewer ready runes than
        another option with the same recycles are dropped.
        """
        zones = self.players[seat]
        pool = zones.rune_pool
        pips = [p.payable_with(card.domains) for p in card.cost.power]
        usable = [u for u in pool.power if card.is_spell or not u.spells_only]
        energy_needed = max(0, card.cost.energy - pool.energy)
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
                and (card.is_spell or not LEGEND_POWER[legend.card.card_id])):
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

    def _pay(self, seat: int, card: CardDef, payment: Payment) -> None:
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

        if pool.energy < card.cost.energy:
            raise ValueError(f"not enough Energy to play {card.name}")
        pips = [p.payable_with(card.domains) for p in card.cost.power]
        usable = [u for u in pool.power if card.is_spell or not u.spells_only]
        matched = _match(pips, usable)
        if len(matched) < len(pips):
            raise ValueError(f"not enough Power to play {card.name}")
        pool.energy -= card.cost.energy
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
        return {
            "viewer": viewer,
            "turn": self.turn,
            "turn_player": self.turn_player,
            "turn_phase": self.turn_phase.value,
            "acting_player": self.acting_player,
            "priority": self.priority,
            "phase": self.phase.value,
            "victory_score": VICTORY_SCORE,
            "winner": self.winner,
            "players": players,
            "battlefields": [bf.view(viewer) for bf in self.battlefields],
            "chain": [dict(item.obj.view(), chain_controller=item.controller) for item in self.chain_items],
        }


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


def load_demo_decks() -> tuple[Deck, Deck]:
    pool = load_card_pool(ROOT / "card_data")
    deck = Deck.load(ROOT / "decks" / "kaisa.json", pool)
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
