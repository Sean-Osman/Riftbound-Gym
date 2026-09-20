import copy
import pickle
from pathlib import Path

from cards import (
    ACCELERATE_COST, ALL_DOMAINS, CardDef, CardType, Cost, Domain, Keyword, Pip,
    Supertype, card_from_dict, deck_errors, in_domain_identity, load_card_pool,
    make_vanilla_pool, parse_power,
)


def expect_error(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_parse_power():
    assert parse_power("") == ()
    assert parse_power("2B") == (Pip.MIND, Pip.MIND)
    assert parse_power("RG") == (Pip.FURY, Pip.CALM)       # two pips, not one hybrid
    assert parse_power("[C][C]") == (Pip.CARD, Pip.CARD)
    expect_error(parse_power, "X")


def test_card_pip_resolution():
    # 135.2.e.6.b/c: [C] is any of the card's domains, or [A] if domainless
    assert Pip.CARD.payable_with(frozenset({Domain.CALM, Domain.CHAOS})) == {Domain.CALM, Domain.CHAOS}
    assert Pip.CARD.payable_with(frozenset()) == ALL_DOMAINS
    assert ACCELERATE_COST.power_options(frozenset({Domain.FURY})) == (frozenset({Domain.FURY}),)


def test_keywords():
    card = card_from_dict({
        "card_id": "T-1", "short_name": "Petty Officer", "types": ["unit"],
        "domains": ["Y"], "energy": 2, "might": 2,
        "keywords": {"Assault": None, "deflect": 2, "Tank": None},
    })
    assert card.keyword_value(Keyword.ASSAULT) == 1       # 807.1.b.3
    assert card.keyword_value(Keyword.DEFLECT) == 2
    assert card.keyword_value(Keyword.SHIELD) == 0
    assert card.has_keyword(Keyword.TANK)
    expect_error(card_from_dict, {"card_id": "T-2", "short_name": "x", "types": ["spell"],
                                  "keywords": {"Tank": None}})   # Tank is units only
    expect_error(card_from_dict, {"card_id": "T-3", "short_name": "x", "types": ["unit"],
                                  "might": 1, "keywords": {"Tnak": None}})


def test_validation():
    expect_error(CardDef, "X", "x", types=frozenset({CardType.SPELL}), might=3)
    expect_error(CardDef, "X", "x", types=frozenset({CardType.RUNE}), cost=Cost(1))
    expect_error(CardDef, "X", "x", types=frozenset({CardType.UNIT}), might=1,
                 is_token=True, domains=frozenset({Domain.FURY}))
    expect_error(CardDef, "X", "x", types=frozenset({CardType.GEAR}), might_bonus=2)
    expect_error(card_from_dict, {"card_id": "X", "short_name": "x", "types": ["unit"],
                                  "might": 1, "tags": ["Kaisa"]})   # typo of Kai'Sa


def test_pickle_and_deepcopy():
    pool = make_vanilla_pool()
    card = pool["CHAMP-TEST"]
    clone = pickle.loads(pickle.dumps(card))
    assert clone == card and clone.keywords == card.keywords
    assert copy.deepcopy(card) is card
    assert copy.deepcopy({"hand": [card]})["hand"][0] is card


def test_deck_errors():
    pool = make_vanilla_pool()
    legend, champ = pool["LEG-TEST"], pool["CHAMP-TEST"]
    main = [pool[f"VAN-{c}"] for c in range(1, 7) for _ in range(3)]
    main += [CardDef(f"FILL-{i}", f"Filler {i}", types=frozenset({CardType.UNIT}),
                     domains=frozenset({Domain.CALM}), cost=Cost(2), might=2) for i in range(21)]
    runes = [pool["RUNE-R"]] * 6 + [pool["RUNE-G"]] * 6
    bfs = [pool[f"BF-{i}"] for i in range(1, 4)]
    assert deck_errors(legend, champ, main, runes, bfs) == []

    too_many = main + [pool["VAN-1"]]
    assert any("copies" in e for e in deck_errors(legend, champ, too_many, runes, bfs))
    bad_runes = runes[:-1] + [pool["RUNE-B"]]
    assert any("identity" in e for e in deck_errors(legend, champ, main, bad_runes, bfs))
    wrong_champ = CardDef("C2", "Other", subtitle="Champ", types=frozenset({CardType.UNIT}),
                          supertypes=frozenset({Supertype.CHAMPION}), tags=frozenset({"Jinx"}),
                          champion_tag="Jinx", might=3)
    assert any("champion unit" in e for e in deck_errors(legend, wrong_champ, main, runes, bfs))


def test_kaisa_card_data():
    pool = load_card_pool(Path(__file__).parent / "card_data")
    by_name = {c.name: c for c in pool.values()}
    legend, champ = by_name["Daughter of the Void"], by_name["Kai'Sa, Survivor"]
    assert legend.domains == {Domain.FURY, Domain.MIND}
    assert champ.is_champion_unit and champ.champion_tag == legend.champion_tag == "Kai'Sa"
    assert not by_name["Cleave"].has_keyword(Keyword.ASSAULT)   # it grants Assault, doesn't have it
    assert by_name["Pouty Poro"].keyword_value(Keyword.DEFLECT) == 1
    assert by_name["Time Warp"].cost == Cost(10, parse_power("4B"))
    assert all(in_domain_identity(c, legend.domains) for c in pool.values())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
