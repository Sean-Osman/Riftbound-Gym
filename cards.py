"""Static card definitions for Riftbound.

A CardDef is the *printed* card: immutable and shared by every copy. Anything
that can change during play (damage, buffs, granted keywords, effects on
domain / might / cost, control, zone) belongs on the engine's per-object state,
not here.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Domain(Enum):
    """The six domains and the shorthand letters used for them in text."""
    FURY = "R"
    CALM = "G"
    MIND = "B"
    BODY = "O"
    CHAOS = "P"
    ORDER = "Y"


ALL_DOMAINS: frozenset[Domain] = frozenset(Domain)


class CardType(Enum):
    """Card types. A game object may have several of them at once."""
    UNIT = "unit"
    GEAR = "gear"
    SPELL = "spell"
    RUNE = "rune"
    BATTLEFIELD = "battlefield"
    LEGEND = "legend"


PERMANENT_TYPES = frozenset({CardType.UNIT, CardType.GEAR})
MAIN_DECK_TYPES = frozenset({CardType.UNIT, CardType.GEAR, CardType.SPELL})
NON_DECK_TYPES = frozenset({CardType.BATTLEFIELD, CardType.LEGEND})


class Supertype(Enum):
    CHAMPION = "champion"      # units only
    SIGNATURE = "signature"    # any card type


EQUIPMENT_TAG = "Equipment"    # gear carrying this tag are Equipment

KNOWN_TAGS: frozenset[str] = frozenset({
    "Ahri", "Akali", "Akshan", "Ambessa", "Anivia", "Annie", "Aphelios", "Ashe", "Azir",
    "Bandle City", "Bard", "Bilgewater", "Bird", "Blitzcrank", "Caitlyn", "Cat", "Darius",
    "Demacia", "Demon", "Diana", "Dog", "Dr. Mundo", "Dragon", "Draven", "Ekko", "Elite",
    "Equipment", "Evelynn", "Ezreal", "Fae", "Fiora", "Fizz", "Freljord", "Galio",
    "Gangplank", "Garen", "Heimerdinger", "Hwei", "Icathia", "Illaoi", "Ionia", "Irelia",
    "Ivern", "Ixtal", "Janna", "Jax", "Jayce", "Jhin", "Jinx", "Kai'Sa", "Karma", "Karthus",
    "Katarina", "Kathkan", "Kayle", "Kayn", "Kennen", "Kha'Zix", "Kog'Maw", "LeBlanc",
    "Lee Sin", "Leona", "Lillia", "Lucian", "Lux", "Malzahar", "Master Yi", "Mech", "Mel",
    "Miss Fortune", "Morgana", "Mount Targon", "Nami", "Nasus", "Nidalee", "Nilah",
    "Nocturne", "Noxus", "Ornn", "Piltover", "Pirate", "Poppy", "Poro", "Pyke", "Qiyana",
    "Recruit", "Rek'Sai", "Rell", "Renata Glasc", "Renekton", "Rengar", "Riven", "Rumble",
    "Sentinel", "Sett", "Shadow Isles", "Shen", "Shurima", "Sivir", "Sona", "Soraka",
    "Spider", "Spirit", "Swain", "Syndra", "Taric", "Teemo", "The Void", "Trifarian",
    "Tryndamere", "Twisted Fate", "Udyr", "Vayne", "Vex", "Vi", "Viktor", "Volibear",
    "Warwick", "Xerath", "Xin Zhao", "Yasuo", "Yone", "Yordle", "Yuumi", "Zaun", "Zed",
    "Zilean",
})


# ---------------------------------------------------------------------------
# Keywords
# ---------------------------------------------------------------------------

class Keyword(Enum):
    ACCELERATE = "Accelerate"
    ACTION = "Action"
    ASSAULT = "Assault"
    DEATHKNELL = "Deathknell"
    DEFLECT = "Deflect"
    GANKING = "Ganking"
    HIDDEN = "Hidden"
    LEGION = "Legion"
    REACTION = "Reaction"
    SHIELD = "Shield"
    TANK = "Tank"
    TEMPORARY = "Temporary"
    VISION = "Vision"
    EQUIP = "Equip"
    QUICK_DRAW = "Quick-Draw"
    REPEAT = "Repeat"
    WEAPONMASTER = "Weaponmaster"
    AMBUSH = "Ambush"
    HUNT = "Hunt"
    LEVEL = "Level"
    UNIQUE = "Unique"
    BACKLINE = "Backline"
    EMPOWER = "Empower"
    EMPOWERED = "Empowered"
    FLOW = "Flow"

    @classmethod
    def parse(cls, name: str) -> Keyword:
        """Case-insensitive"""
        key = name.replace("ﬂ", "fl").replace("_", "-").strip().lower()
        for kw in cls:
            if kw.value.lower() == key:
                return kw
        raise ValueError(f"unknown keyword {name!r}")


# Keywords printed as "Keyword X": X defaults to 1 when omitted, and several
# instances on the same object add their values together.
VALUED_KEYWORDS = frozenset({Keyword.ASSAULT, Keyword.DEFLECT, Keyword.SHIELD, Keyword.HUNT})

# Keywords whose meaning is tied to a specific ability or cost on the card.
# CardDef.keywords only records that the card has them, which other effects can
# check; the ability itself belongs in the card's script.
ABILITY_KEYWORDS = frozenset({
    Keyword.DEATHKNELL, Keyword.LEGION, Keyword.LEVEL, Keyword.EMPOWERED,
    Keyword.EQUIP, Keyword.REPEAT, Keyword.EMPOWER, Keyword.FLOW,
})

# Card types each keyword can be printed on. Keywords left out of this table
# (Action, Reaction, Deflect, Ambush, Unique, Repeat, Legion, Level, Empower,
# Empowered) are not restricted to any type.
_UNITS = frozenset({CardType.UNIT})
_KEYWORD_TYPES: dict[Keyword, frozenset[CardType]] = {
    Keyword.ACCELERATE: _UNITS,
    Keyword.ASSAULT: _UNITS,
    Keyword.DEATHKNELL: PERMANENT_TYPES,
    Keyword.GANKING: _UNITS,
    Keyword.HIDDEN: MAIN_DECK_TYPES,
    Keyword.SHIELD: _UNITS,
    Keyword.TANK: _UNITS,
    Keyword.TEMPORARY: PERMANENT_TYPES,
    Keyword.VISION: PERMANENT_TYPES,
    Keyword.EQUIP: frozenset({CardType.GEAR}),
    Keyword.QUICK_DRAW: frozenset({CardType.GEAR}),
    Keyword.WEAPONMASTER: _UNITS,
    Keyword.HUNT: _UNITS,
    Keyword.BACKLINE: _UNITS,
    Keyword.FLOW: frozenset({CardType.SPELL}),
}


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------

class Pip(Enum):
    """One Power symbol in a cost.

    CARD ([C]) stays symbolic and is resolved against the object's *current*
    domains when it is paid, because effects can change an object's domains and
    a copied cost keeps its [C].
    """
    FURY = "R"
    CALM = "G"
    MIND = "B"
    BODY = "O"
    CHAOS = "P"
    ORDER = "Y"
    ANY = "A"      # any domain
    CARD = "C"     # the card's own domain(s)

    def payable_with(self, domains: frozenset[Domain]) -> frozenset[Domain]:
        """Domains of Power that can pay this pip on an object with `domains`."""
        if self is Pip.ANY:
            return ALL_DOMAINS
        if self is Pip.CARD:
            # any of the card's domains, or any domain at all if it has none
            return domains if domains else ALL_DOMAINS
        return frozenset({Domain(self.value)})


@dataclass(frozen=True)
class Cost:
    """A card's Energy cost (the numeral) and Power cost (the symbols)."""
    energy: int = 0
    power: tuple[Pip, ...] = ()

    def __post_init__(self) -> None:
        if self.energy < 0:
            raise ValueError("energy cost can't be negative")

    @property
    def is_free(self) -> bool:
        return self.energy == 0 and not self.power

    @property
    def power_amount(self) -> int:
        """Number of Power symbols, e.g. for 'costs no more than [A]' checks."""
        return len(self.power)

    def power_options(self, domains: frozenset[Domain]) -> tuple[frozenset[Domain], ...]:
        """Resolve every pip against the paying object's current domains."""
        return tuple(p.payable_with(domains) for p in self.power)

    def __str__(self) -> str:
        return f"[{self.energy}]" + "".join(f"[{p.value}]" for p in self.power)


_POWER_GROUP = re.compile(r"(\d*)([RGBOPYAC])")


def parse_power(text: str) -> tuple[Pip, ...]:
    """Parse a Power cost. Every letter is one symbol; a leading number repeats it.

        ""      -> no power cost
        "B"     -> [B]
        "2B"    -> [B][B]
        "RG"    -> [R][G]
        "A"     -> [A]   (any domain)
        "C"     -> [C]   (the card's own domain(s), resolved at payment time)

    There are no hybrid symbols: a [C] on a multi-domain card is the only pip
    that can be paid with more than one domain.
    """
    text = text.replace(" ", "").replace("[", "").replace("]", "").upper()
    pips: list[Pip] = []
    pos = 0
    while pos < len(text):
        match = _POWER_GROUP.match(text, pos)
        if not match:
            raise ValueError(f"can't parse power cost {text!r} at position {pos}")
        count_str, letter = match.groups()
        count = int(count_str) if count_str else 1
        if count < 1:
            raise ValueError(f"power symbol count must be at least 1 in {text!r}")
        pips.extend([Pip(letter)] * count)
        pos = match.end()
    return tuple(pips)


ACCELERATE_COST = Cost(1, (Pip.CARD,))   # extra cost to have a unit enter ready
HIDE_COST = Cost(0, (Pip.ANY,))          # cost to hide a card with Hidden facedown


# ---------------------------------------------------------------------------
# CardDef
# ---------------------------------------------------------------------------

KeywordValue = int | None


@dataclass(frozen=True, eq=False)
class CardDef:
    """A printed card. Immutable and shared by every copy of that card.

    Equality and hashing use card_id. Reprints and alt arts have different
    card_ids but the same `name`, and the game treats them as the same card,
    so compare `name` for anything the rules care about.
    """

    card_id: str                                   # stable id, e.g. "OGN-042"
    short_name: str
    types: frozenset[CardType]                     # usually just one
    subtitle: str | None = None                    # "Jinx, Rebel" -> subtitle "Rebel"
    supertypes: frozenset[Supertype] = frozenset()
    tags: frozenset[str] = frozenset()             # e.g. {"Jinx", "Zaun"}
    # The tag that links a legend to its champion units and signature cards.
    # Kept apart from `tags` so a region tag like "Zaun" never counts as a match.
    champion_tag: str | None = None
    domains: frozenset[Domain] = frozenset()
    cost: Cost = Cost()                            # main deck cards only
    might: int | None = None                       # units only
    might_bonus: int | None = None                 # Equipment only
    # Printed keywords. Valued keywords (VALUED_KEYWORDS) map to their X,
    # already defaulted to 1; every other keyword maps to None.
    keywords: Mapping[Keyword, KeywordValue] = field(default_factory=dict)
    rules_text: str = ""                           # kept for logs and card scripts
    effect_text: str = ""                          # applies only while attached
    is_token: bool = False                         # tokens are game objects, not cards
    art_url: str = ""                              # display only, no gameplay meaning

    def __post_init__(self) -> None:
        cid = self.card_id
        if not self.types:
            raise ValueError(f"{cid}: a card needs at least one type")
        if self.is_unit and self.might is None:
            raise ValueError(f"{cid}: units must have might")
        if not self.is_unit and self.might is not None:
            raise ValueError(f"{cid}: only units have might")
        if self.might is not None and self.might < 0:
            raise ValueError(f"{cid}: printed might can't be negative")
        if Supertype.CHAMPION in self.supertypes and not self.is_unit:
            raise ValueError(f"{cid}: the champion supertype applies to units only")
        if self.might_bonus is not None and not self.is_equipment:
            raise ValueError(f"{cid}: only Equipment has a might bonus")
        if not self.is_main_deck_card and not self.cost.is_free:
            raise ValueError(f"{cid}: only main deck cards have a cost")
        if self.is_token and (self.domains or not self.cost.is_free):
            raise ValueError(f"{cid}: tokens have no domains or costs")
        if self.champion_tag is not None and self.champion_tag not in self.tags:
            raise ValueError(f"{cid}: champion_tag {self.champion_tag!r} must also be in tags")
        if (self.is_champion_unit or self.is_signature or self.is_legend) \
                and not self.is_token and self.champion_tag is None:
            raise ValueError(f"{cid}: legends, champions and signature cards need a champion_tag")

        keywords: dict[Keyword, KeywordValue] = {}
        for kw, value in dict(self.keywords).items():
            kw = kw if isinstance(kw, Keyword) else Keyword.parse(kw)
            allowed = _KEYWORD_TYPES.get(kw)
            if allowed is not None and not (self.types & allowed):
                raise ValueError(f"{cid}: {kw.value} can't be printed on {sorted(t.value for t in self.types)}")
            if kw in VALUED_KEYWORDS:
                value = 1 if value is None else value
                if not isinstance(value, int) or value < 1:
                    raise ValueError(f"{cid}: {kw.value} value must be a positive int, got {value!r}")
            elif value is not None:
                raise ValueError(f"{cid}: {kw.value} takes no value on the card; put it in the card script")
            keywords[kw] = value
        # freeze the keyword dict so the dataclass stays truly immutable
        object.__setattr__(self, "keywords", MappingProxyType(keywords))

    # --- identity / copying ---------------------------------------------------

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CardDef) and other.card_id == self.card_id

    def __hash__(self) -> int:
        return hash(self.card_id)

    def __copy__(self) -> CardDef:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> CardDef:
        # immutable and shared: cloning a game state must not clone its cards
        return self

    def __getstate__(self) -> dict[str, Any]:
        # mappingproxy can't be pickled, and multiprocess vec envs need this
        state = dict(self.__dict__)
        state["keywords"] = dict(self.keywords)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        for key, value in state.items():
            object.__setattr__(self, key, value)
        object.__setattr__(self, "keywords", MappingProxyType(state["keywords"]))

    @property
    def name(self) -> str:
        """The full name, '[Short Name], [Subtitle]'."""
        return f"{self.short_name}, {self.subtitle}" if self.subtitle else self.short_name

    # --- type checks (a card with several types answers True to each) ---------

    def is_type(self, t: CardType) -> bool:
        return t in self.types

    @property
    def is_unit(self) -> bool:
        return CardType.UNIT in self.types

    @property
    def is_gear(self) -> bool:
        return CardType.GEAR in self.types

    @property
    def is_spell(self) -> bool:
        return CardType.SPELL in self.types

    @property
    def is_rune(self) -> bool:
        return CardType.RUNE in self.types

    @property
    def is_battlefield(self) -> bool:
        return CardType.BATTLEFIELD in self.types

    @property
    def is_legend(self) -> bool:
        return CardType.LEGEND in self.types

    @property
    def is_permanent(self) -> bool:
        """Units and gear. Runes stay on the board too, but are not permanents."""
        return bool(self.types & PERMANENT_TYPES)

    @property
    def is_main_deck_card(self) -> bool:
        """Units, gear and spells. Card text saying "card" means one of these."""
        return bool(self.types & MAIN_DECK_TYPES)

    @property
    def is_equipment(self) -> bool:
        """Gear carrying the Equipment tag."""
        return self.is_gear and EQUIPMENT_TAG in self.tags

    @property
    def is_champion_unit(self) -> bool:
        return self.is_unit and Supertype.CHAMPION in self.supertypes

    @property
    def is_signature(self) -> bool:
        return Supertype.SIGNATURE in self.supertypes

    # --- rule-derived defaults ------------------------------------------------

    @property
    def enters_exhausted(self) -> bool:
        """Units enter exhausted; gear enter ready and runes are channeled ready.
        A unit that is also gear still enters exhausted. Accelerate and other
        effects can override this while the card is being played."""
        return self.is_unit

    @property
    def printed_mighty(self) -> bool:
        """A unit is Mighty at 5 or more might; off the board its printed might counts."""
        return self.might is not None and self.might >= 5

    def has_keyword(self, keyword: Keyword) -> bool:
        return keyword in self.keywords

    def keyword_value(self, keyword: Keyword) -> int:
        """Printed X of a valued keyword (Assault/Shield/Deflect/Hunt), 0 if absent."""
        if keyword not in VALUED_KEYWORDS:
            raise ValueError(f"{keyword.value} has no value")
        return self.keywords.get(keyword) or 0

    def __repr__(self) -> str:
        return f"CardDef({self.card_id!r}, {self.name!r})"


# ---------------------------------------------------------------------------
# Loading from JSON
# ---------------------------------------------------------------------------

def card_from_dict(data: dict[str, Any], *, check_tags: bool = True) -> CardDef:
    """Expected JSON shape (only card_id, short_name, types are required):
    {
      "card_id": "OGN-001", "short_name": "Jinx", "subtitle": "Rebel",
      "types": ["unit"], "supertypes": ["champion"],
      "tags": ["Jinx", "Zaun"], "champion_tag": "Jinx",
      "domains": ["R", "P"], "energy": 4, "power": "C", "might": 4,
      "keywords": {"Assault": 2, "Tank": null}, "rules_text": "...", "effect_text": ""
    }
    """
    tags = frozenset(data.get("tags", []))
    if check_tags and (unknown := tags - KNOWN_TAGS):
        raise ValueError(f"{data['card_id']}: unknown tags {sorted(unknown)}")
    return CardDef(
        card_id=data["card_id"],
        short_name=data["short_name"],
        subtitle=data.get("subtitle"),
        types=frozenset(CardType(t) for t in data["types"]),
        supertypes=frozenset(Supertype(s) for s in data.get("supertypes", [])),
        tags=tags,
        champion_tag=data.get("champion_tag"),
        domains=frozenset(Domain(d) for d in data.get("domains", [])),
        cost=Cost(data.get("energy", 0), parse_power(data.get("power", ""))),
        might=data.get("might"),
        might_bonus=data.get("might_bonus"),
        keywords={Keyword.parse(k): v for k, v in data.get("keywords", {}).items()},
        rules_text=data.get("rules_text", ""),
        effect_text=data.get("effect_text", ""),
        is_token=data.get("is_token", False),
        art_url=data.get("art_url", ""),
    )


def load_card_pool(folder: str | Path, *, check_tags: bool = True) -> dict[str, CardDef]:
    """Load every *.json file in a folder. Each file holds a list of card dicts."""
    pool: dict[str, CardDef] = {}
    for path in sorted(Path(folder).glob("*.json")):
        for entry in json.loads(path.read_text(encoding="utf-8")):
            card = card_from_dict(entry, check_tags=check_tags)
            if card.card_id in pool:
                raise ValueError(f"duplicate card_id {card.card_id} in {path}")
            pool[card.card_id] = card
    return pool


# ---------------------------------------------------------------------------
# Deck construction
# ---------------------------------------------------------------------------

MIN_MAIN_DECK = 40        # counting the Chosen Champion
MAX_COPIES = 3
MAX_SIGNATURES = 3
RUNE_DECK_SIZE = 12
DUEL_BATTLEFIELDS = 3


def in_domain_identity(card: CardDef, identity: frozenset[Domain]) -> bool:
    """Every domain on the card must appear in the legend's domain identity."""
    return card.domains <= identity


def deck_errors(
    legend: CardDef,
    champion: CardDef,
    main_deck: Iterable[CardDef],
    runes: Iterable[CardDef],
    battlefields: Iterable[CardDef],
    *,
    battlefield_count: int = DUEL_BATTLEFIELDS,
) -> list[str]:
    """Return every deck-construction violation (empty list = legal deck).

    `main_deck` excludes the Chosen Champion; it is counted in separately.
    """
    main = [champion, *main_deck]
    runes, battlefields = list(runes), list(battlefields)
    errors: list[str] = []
    identity = legend.domains
    tag = legend.champion_tag

    if not legend.is_legend or legend.is_token:
        errors.append(f"{legend.name} is not a legend")
    if not champion.is_champion_unit or champion.champion_tag != tag:
        errors.append(f"{champion.name} is not a champion unit with the tag {tag!r}")

    if len(main) < MIN_MAIN_DECK:
        errors.append(f"main deck has {len(main)} cards, needs at least {MIN_MAIN_DECK}")
    for card in main:
        if not card.is_main_deck_card or card.is_token:
            errors.append(f"{card.name} can't be in the main deck")
        elif not in_domain_identity(card, identity):
            errors.append(f"{card.name} is outside the domain identity")
    for name, n in Counter(c.name for c in main).items():
        unique = any(c.name == name and c.has_keyword(Keyword.UNIQUE) for c in main)
        limit = 1 if unique else MAX_COPIES
        if n > limit:
            errors.append(f"{n} copies of {name}, max {limit}")
    signatures = [c for c in main if c.is_signature]
    if len(signatures) > MAX_SIGNATURES:
        errors.append(f"{len(signatures)} signature cards, max {MAX_SIGNATURES}")
    for card in signatures:
        if card.champion_tag != tag:
            errors.append(f"signature {card.name} doesn't match the legend's tag {tag!r}")

    if len(runes) != RUNE_DECK_SIZE:
        errors.append(f"rune deck has {len(runes)} runes, needs {RUNE_DECK_SIZE}")
    for rune in runes:
        if not rune.is_rune:
            errors.append(f"{rune.name} is not a rune")
        elif not in_domain_identity(rune, identity):
            errors.append(f"{rune.name} is outside the domain identity")

    if len(battlefields) != battlefield_count:
        errors.append(f"{len(battlefields)} battlefields, this mode needs {battlefield_count}")
    for bf in battlefields:
        if not bf.is_battlefield:
            errors.append(f"{bf.name} is not a battlefield")
        elif not in_domain_identity(bf, identity):
            errors.append(f"{bf.name} is outside the domain identity")
    if len({bf.name for bf in battlefields}) != len(battlefields):
        errors.append("duplicate battlefield names")

    return errors


# ---------------------------------------------------------------------------
# Test-only vanilla pool
# ---------------------------------------------------------------------------

def make_vanilla_pool() -> dict[str, CardDef]:
    """Plain cards with no abilities, for building and fuzzing the core engine."""
    pool: dict[str, CardDef] = {}

    def add(card: CardDef) -> None:
        pool[card.card_id] = card

    for d in Domain:
        add(CardDef(f"RUNE-{d.value}", f"{d.name.title()} Rune",
                    types=frozenset({CardType.RUNE}), domains=frozenset({d})))
    for cost in range(1, 7):
        add(CardDef(f"VAN-{cost}", f"Vanilla {cost}", types=frozenset({CardType.UNIT}),
                    domains=frozenset({Domain.FURY}),
                    cost=Cost(cost), might=cost))
    for i in range(1, 4):
        add(CardDef(f"BF-{i}", f"Plain Battlefield {i}",
                    types=frozenset({CardType.BATTLEFIELD})))
    add(CardDef("LEG-TEST", "Tester", subtitle="Legend", types=frozenset({CardType.LEGEND}),
                tags=frozenset({"Tester"}), champion_tag="Tester",
                domains=frozenset({Domain.FURY, Domain.CALM})))
    add(CardDef("CHAMP-TEST", "Tester", subtitle="Champion", types=frozenset({CardType.UNIT}),
                supertypes=frozenset({Supertype.CHAMPION}), tags=frozenset({"Tester"}),
                champion_tag="Tester", domains=frozenset({Domain.FURY}),
                cost=Cost(3, parse_power("C")), might=4))
    return pool
