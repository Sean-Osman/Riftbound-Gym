"""Zones and in-game objects (Core Rules 105-108, 119-128).

No game rules run here. These are containers that know their owner, privacy
and ordering, plus CardInstance, the per-game object wrapping a static CardDef.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator

from cards import CardDef


class Privacy(Enum):
    """128.3-128.5."""
    SECRET = "secret"      # nobody may look (deck order)
    PRIVATE = "private"    # owner (or controller, on the board) only
    PUBLIC = "public"      # everyone


class ZoneKind(Enum):
    # The Board (107)
    BASE = "base"
    BATTLEFIELD = "battlefield"      # the units / gear at one battlefield
    FACEDOWN = "facedown"            # 107.3: one per battlefield, not a location
    LEGEND = "legend"                # 107.4
    # Non-Board zones (108)
    CHAIN = "chain"
    TRASH = "trash"
    CHAMPION = "champion"
    MAIN_DECK = "main_deck"
    RUNE_DECK = "rune_deck"
    BANISHMENT = "banishment"
    HAND = "hand"


BOARD_ZONES = frozenset({ZoneKind.BASE, ZoneKind.BATTLEFIELD, ZoneKind.FACEDOWN, ZoneKind.LEGEND})

# (privacy, ordered) for each zone kind
_ZONE_RULES: dict[ZoneKind, tuple[Privacy, bool]] = {
    ZoneKind.BASE: (Privacy.PUBLIC, False),          # 107.1.d
    ZoneKind.BATTLEFIELD: (Privacy.PUBLIC, False),   # 107.2.c
    ZoneKind.FACEDOWN: (Privacy.PRIVATE, False),     # 107.3.f: zone public, cards private
    ZoneKind.LEGEND: (Privacy.PUBLIC, False),
    ZoneKind.CHAIN: (Privacy.PUBLIC, True),          # 108.1.b, order matters
    ZoneKind.TRASH: (Privacy.PUBLIC, False),         # 108.2.c-d
    ZoneKind.CHAMPION: (Privacy.PUBLIC, False),      # 108.3.e
    ZoneKind.MAIN_DECK: (Privacy.SECRET, True),      # 108.4.d
    ZoneKind.RUNE_DECK: (Privacy.SECRET, True),      # 108.5.d
    ZoneKind.BANISHMENT: (Privacy.PUBLIC, False),    # 108.6.d-e
    ZoneKind.HAND: (Privacy.PRIVATE, False),         # 108.7.c-e
}

_object_ids = itertools.count(1)


@dataclass(eq=False)
class CardInstance:
    """One game object backed by a card or token (119-124).

    `oid` changes whenever the object moves to or from a non-Board zone,
    because it becomes a new object (124). Temporary state is cleared then too.
    """
    card: CardDef
    owner: int
    controller: int
    oid: int = field(default_factory=lambda: next(_object_ids))
    zone: Zone | None = None
    # temporary state (124.1-124.2)
    exhausted: bool = False
    damage: int = 0
    buffs: int = 0
    facedown: bool = False

    def become_new_object(self) -> None:
        """124 / 124.1: new identity, all temporary modifications dropped."""
        self.oid = next(_object_ids)
        self.controller = self.owner
        self.exhausted = False
        self.damage = 0
        self.buffs = 0
        self.facedown = False

    def view(self) -> dict[str, Any]:
        """Public face of the object, for observations / the UI."""
        c = self.card
        return {
            "oid": self.oid,
            "card_id": c.card_id,
            "name": c.name,
            "types": sorted(t.value for t in c.types),
            "domains": sorted(d.value for d in c.domains),
            "cost": str(c.cost) if c.is_main_deck_card else None,
            "might": c.might,
            "keywords": {k.value: v for k, v in c.keywords.items()},
            "rules_text": c.rules_text,
            "art_url": c.art_url,
            "owner": self.owner,
            "controller": self.controller,
            "exhausted": self.exhausted,
            "damage": self.damage,
            "buffs": self.buffs,
        }


class Zone:
    """An ordered or unordered pile of objects. Index -1 is the top of a deck."""

    def __init__(self, kind: ZoneKind, owner: int | None, *, capacity: int | None = None):
        self.kind = kind
        self.owner = owner                      # None for shared zones (the Chain)
        self.privacy, self.ordered = _ZONE_RULES[kind]
        self.capacity = capacity                # 107.3.b: facedown zones hold one card
        self.objects: list[CardInstance] = []

    @property
    def is_board(self) -> bool:
        return self.kind in BOARD_ZONES

    def __len__(self) -> int:
        return len(self.objects)

    def __iter__(self) -> Iterator[CardInstance]:
        return iter(self.objects)

    def __repr__(self) -> str:
        return f"Zone({self.kind.value}, owner={self.owner}, n={len(self)})"

    def top(self, n: int = 1) -> list[CardInstance]:
        return self.objects[-n:][::-1] if n else []

    def shuffle(self, rng: random.Random) -> None:
        rng.shuffle(self.objects)

    def can_see(self, viewer: int, obj: CardInstance | None = None) -> bool:
        if self.privacy is Privacy.PUBLIC:
            return True
        if self.privacy is Privacy.PRIVATE:
            # 128.4: controller on the board, owner everywhere else
            who = obj.controller if (obj is not None and self.is_board) else self.owner
            return who == viewer
        return False

    def view(self, viewer: int) -> dict[str, Any]:
        """What `viewer` may know about this zone (128). Counts are always public."""
        if self.privacy is Privacy.SECRET:
            visible = []
        else:
            visible = [o.view() if self.can_see(viewer, o) else {"hidden": True} for o in self.objects]
        return {"kind": self.kind.value, "owner": self.owner, "count": len(self), "objects": visible}


@dataclass(eq=False)
class Battlefield:
    """107.2-107.3: a battlefield location plus its facedown zone."""
    card: CardInstance
    units: Zone
    facedown: Zone
    controller: int | None = None
    contested: bool = False

    @classmethod
    def create(cls, card: CardInstance) -> Battlefield:
        return cls(card, Zone(ZoneKind.BATTLEFIELD, None), Zone(ZoneKind.FACEDOWN, None, capacity=1))

    def view(self, viewer: int) -> dict[str, Any]:
        return {
            "card": self.card.view(),
            "controller": self.controller,
            "contested": self.contested,
            "units": self.units.view(viewer)["objects"],
            "facedown": self.facedown.view(viewer),
        }


@dataclass
class RunePool:
    """165: available Energy and Power. Not a zone and not a game object."""
    energy: int = 0
    power: dict[str, int] = field(default_factory=dict)    # domain letter or "A" -> amount

    def empty(self) -> None:
        self.energy = 0
        self.power.clear()


class PlayerZones:
    """Every zone that belongs to one player."""

    def __init__(self, seat: int):
        self.seat = seat
        self.main_deck = Zone(ZoneKind.MAIN_DECK, seat)
        self.rune_deck = Zone(ZoneKind.RUNE_DECK, seat)
        self.hand = Zone(ZoneKind.HAND, seat)
        self.trash = Zone(ZoneKind.TRASH, seat)
        self.banishment = Zone(ZoneKind.BANISHMENT, seat)
        self.champion = Zone(ZoneKind.CHAMPION, seat)
        self.legend = Zone(ZoneKind.LEGEND, seat)
        self.base = Zone(ZoneKind.BASE, seat)
        self.rune_pool = RunePool()

    def all(self) -> list[Zone]:
        return [self.main_deck, self.rune_deck, self.hand, self.trash, self.banishment,
                self.champion, self.legend, self.base]

    def by_kind(self, kind: ZoneKind) -> Zone:
        for z in self.all():
            if z.kind is kind:
                return z
        raise KeyError(kind)
