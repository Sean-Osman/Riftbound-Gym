"""Game framework: decks, players, setup, the decision loop and agents.

No gameplay rules yet. Setup puts every card in its starting zone (rules
111-116, mulligan skipped); then players pass turns and after
`placeholder_turns` turns a winner is picked at random. The decision loop
(`acting_player` / `legal_actions` / `step` / `observation`) is what real
rules and RL training will plug into later.

    python3 game.py            # headless: random agent vs random agent
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from cards import CardDef, deck_errors, load_card_pool
from zones import Battlefield, CardInstance, PlayerZones, Zone, ZoneKind

ROOT = Path(__file__).parent
VICTORY_SCORE = 8          # 485.3
OPENING_HAND = 4           # 116


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
    PASS = "pass"          # placeholder for "end turn"
    CONCEDE = "concede"    # 650


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    label: str

    def to_json(self) -> dict[str, str]:
        return {"kind": self.kind.value, "label": self.label}


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


class Game:
    def __init__(self, decks: list[Deck], names: list[str] | None = None, *,
                 seed: int | None = None, placeholder_turns: int = 6):
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
        self.placeholder_turns = placeholder_turns
        self.turn = 0
        self.turn_player = 0
        self.first_player = 0
        self.phase = Phase.PLAYING
        self.winner: int | None = None
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
        self.turn = 1

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
        """413. Burn Out (431) is not implemented yet: drawing from an empty deck does nothing."""
        zones = self.players[seat]
        for obj in zones.main_deck.top(n):
            self.move(obj, zones.hand)

    # --- decision loop --------------------------------------------------------

    @property
    def is_over(self) -> bool:
        return self.phase is Phase.GAME_OVER

    @property
    def acting_player(self) -> int | None:
        """The seat that must decide next, or None if the game is over."""
        return None if self.is_over else self.turn_player

    def legal_actions(self, seat: int | None = None) -> list[Action]:
        seat = self.acting_player if seat is None else seat
        if self.is_over or seat != self.acting_player:
            return []
        return [Action(ActionKind.PASS, "End turn"), Action(ActionKind.CONCEDE, "Concede")]

    def step(self, action: Action) -> None:
        seat = self.acting_player
        if action not in self.legal_actions(seat):
            raise ValueError(f"illegal action {action} for seat {seat}")
        if action.kind is ActionKind.CONCEDE:
            self._log(f"{self.names[seat]} concedes")
            self._end(winner=self._opponent(seat))
        elif action.kind is ActionKind.PASS:
            self._log(f"{self.names[seat]} ends turn {self.turn}")
            if self.turn >= self.placeholder_turns:
                self._log("placeholder: no rules yet, picking a random winner")
                self._end(winner=self.rng.randrange(len(self.players)))
            else:
                self.turn += 1
                self.turn_player = self._opponent(seat)

    def _end(self, winner: int) -> None:
        self.winner = winner
        self.phase = Phase.GAME_OVER
        self._log(f"{self.names[winner]} wins")

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
                "rune_pool": {"energy": zones.rune_pool.energy, "power": dict(zones.rune_pool.power)},
                "zones": {z.kind.value: z.view(viewer) for z in zones.all()},
            })
        return {
            "viewer": viewer,
            "turn": self.turn,
            "turn_player": self.turn_player,
            "acting_player": self.acting_player,
            "phase": self.phase.value,
            "victory_score": VICTORY_SCORE,
            "winner": self.winner,
            "players": players,
            "battlefields": [bf.view(viewer) for bf in self.battlefields],
            "chain": self.chain.view(viewer),
        }


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
    for i in range(200):
        wins[play_game(Game([a, b], seed=i), [RandomAgent(i), RandomAgent(i + 1)])] += 1
    print(f"200 games, wins by seat: {wins}")
