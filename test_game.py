import copy
import pickle

from cards import CardDef, CardType, Cost, Domain, Keyword, make_vanilla_pool, parse_power
from game import (
    LEGEND_POWER, SPELLS, ActionKind, Deck, Game, Payment, RandomAgent, SpellScript, TurnPhase,
    load_demo_decks, play_game,
)


# --- test cards ----------------------------------------------------------------

def _spell(card_id, name, cost, keywords=None):
    return CardDef(card_id, name, types=frozenset({CardType.SPELL}), domains=frozenset({Domain.FURY}),
                   cost=cost, keywords=keywords or {})


SORCERY = _spell("TEST-SORCERY", "Test Sorcery", Cost(1))
REACTION = _spell("TEST-REACTION", "Test Reaction", Cost(1), {Keyword.REACTION: None})
ACTION = _spell("TEST-ACTION", "Test Action", Cost(1), {Keyword.ACTION: None})
CALM_SPELL = CardDef("TEST-CALM-SPELL", "Calm Spell", types=frozenset({CardType.SPELL}),
                     domains=frozenset({Domain.CALM}), cost=Cost(0, parse_power("G")))
DUAL_UNIT = CardDef("TEST-DUAL-UNIT", "Dual Unit", types=frozenset({CardType.UNIT}),
                    domains=frozenset({Domain.FURY, Domain.CALM}), cost=Cost(1, parse_power("C")), might=1)
CALM_UNIT = CardDef("TEST-CALM-UNIT", "Calm Unit", types=frozenset({CardType.UNIT}),
                    domains=frozenset({Domain.CALM}), cost=Cost(0, parse_power("G")), might=1)

for _card in (SORCERY, REACTION, ACTION, CALM_SPELL):
    SPELLS[_card.card_id] = SpellScript(lambda game, item: None)
LEGEND_POWER["LEG-TEST"] = True          # the test legend gets Kai'Sa's "Add [A] for spells"


def make_test_deck():
    pool = make_vanilla_pool()
    main = [pool["VAN-2"]] * 12 + [SORCERY] * 3 + [REACTION] * 3 + [ACTION] * 3 + [CALM_SPELL] * 3 + [CALM_UNIT] * 3 + [DUAL_UNIT] * 3 \
        + [pool[f"VAN-{c}"] for c in (1, 3, 4, 5, 6) for _ in range(3)]
    return Deck("Test", pool["LEG-TEST"], pool["CHAMP-TEST"], tuple(main),
                tuple([pool["RUNE-R"]] * 6 + [pool["RUNE-G"]] * 6),
                tuple(pool[f"BF-{i}"] for i in (1, 2, 3)))


# --- helpers -------------------------------------------------------------------

def new_game(seed=0, decks=None):
    return Game(list(decks or load_demo_decks()), seed=seed)


def keep_hands(g):
    while g.turn_phase is TurnPhase.MULLIGAN:
        g.step(next(a for a in g.legal_actions() if a.label == "Keep hand"))


def started(seed=0, decks=None):
    g = new_game(seed, decks)
    keep_hands(g)
    return g


def set_runes(g, seat, letters, exhausted=""):
    """Replace `seat`'s runes in play with one rune per letter (from their rune deck)."""
    zones = g.players[seat]
    for rune in zones.runes():
        g.move(rune, zones.rune_deck)
    runes = []
    for i, letter in enumerate(letters):
        rune = next(r for r in zones.rune_deck if r.card.domains == {Domain(letter)})
        g.move(rune, zones.base)
        rune.exhausted = exhausted[i:i + 1] == "x"
        runes.append(rune)
    return runes


def to_hand(g, seat, card_id):
    zones = g.players[seat]
    obj = next(o for o in [*zones.main_deck, *zones.hand] if o.card.card_id == card_id)
    g.move(obj, zones.hand)
    return obj


def plays(g, card_id, seat=None):
    seat = g.acting_player if seat is None else seat
    return [a for a in g.legal_actions(seat) if a.kind is ActionKind.PLAY_CARD
            and next(o for o in [*g.players[seat].hand, *g.players[seat].champion]
                     if o.oid == a.card_oid).card.card_id == card_id]


def ready_unit(g, seat, card_id, where=None):
    """Put a ready copy of a unit in `seat`'s base (or at a battlefield)."""
    zones = g.players[seat]
    obj = next(o for o in [*zones.main_deck, *zones.hand] if o.card.card_id == card_id)
    g.move(obj, zones.base if where is None else where.units)
    obj.exhausted = False
    return obj


def moves(g, dest=None):
    return [a for a in g.legal_actions() if a.kind is ActionKind.MOVE and a.destination == dest]


def clear_hand(g, seat):
    zones = g.players[seat]
    for obj in list(zones.hand):
        g.move(obj, zones.main_deck, bottom=True)


# --- setup and mulligan ----------------------------------------------------------

def test_demo_deck_is_legal():
    deck, _ = load_demo_decks()
    assert deck.errors() == []


def test_decklists_keep_runes_in_a_separate_rune_deck(tmp_path):
    import json
    from cards import load_card_pool
    from game import ROOT
    pool = load_card_pool(ROOT / "card_data")
    deck, _ = load_demo_decks()
    assert len(deck.runes) == 12 and all(c.is_rune for c in deck.runes)
    assert not any(c.is_rune for c in deck.main)

    spec = json.loads((ROOT / "decks" / "kaisa.json").read_text())
    spec["deck"]["Main Board"] += spec["deck"].pop("Rune Deck")
    path = tmp_path / "old_style.json"
    path.write_text(json.dumps(spec))
    try:
        Deck.load(path, pool)
    except ValueError as error:
        assert "Rune Deck" in str(error)
    else:
        raise AssertionError("runes in the Main Board should be rejected")


def test_setup_puts_cards_in_starting_zones():
    g = new_game()
    for seat, zones in enumerate(g.players):
        deck = g.decks[seat]
        assert len(zones.legend) == 1 and len(zones.champion) == 1
        assert len(zones.hand) == 4
        assert len(zones.main_deck) + len(zones.hand) == len(deck.main)
        assert len(zones.rune_deck) == 12
        assert all(o.owner == seat for z in zones.all() for o in z)
    assert len(g.battlefields) == 2 and len(g.set_aside) == 4
    assert not g.chain.objects


def test_mulligans_happen_in_turn_order_then_the_first_turn_starts():
    g = new_game()
    first, second = g.first_player, 1 - g.first_player
    assert g.turn_phase is TurnPhase.MULLIGAN and g.acting_player == first
    g.step(next(a for a in g.legal_actions() if a.label == "Keep hand"))
    assert g.acting_player == second
    g.step(next(a for a in g.legal_actions() if a.label == "Keep hand"))

    assert g.turn == 1 and g.turn_phase is TurnPhase.MAIN and g.acting_player == first
    assert len(g.players[first].hand) == 5          # drew in the Draw Phase
    assert len(g.players[first].runes()) == 2       # channeled 2
    assert len(g.players[second].hand) == 4


def test_mulligan_draws_replacements_and_recycles_the_set_aside_cards():
    g = new_game()
    seat = g.acting_player
    zones = g.players[seat]
    action = next(a for a in g.legal_actions() if len(a.set_aside) == 2)
    set_aside = [o for o in zones.hand if o.oid in action.set_aside]

    g.step(action)

    assert len(zones.hand) == 4
    assert all(o not in zones.hand.objects for o in set_aside)
    assert set(zones.main_deck.objects[:2]) == set(set_aside)     # bottom of the deck


def test_mulligan_options_skip_duplicate_card_sets():
    g = new_game()
    options = g.legal_actions()
    hand = {o.oid: o.card.card_id for o in g.players[g.acting_player].hand}
    keys = [tuple(sorted(hand[oid] for oid in a.set_aside)) for a in options if a.kind is ActionKind.MULLIGAN]
    assert len(keys) == len(set(keys)) and 1 < len(keys) <= 11


# --- the turn --------------------------------------------------------------------

def test_second_player_channels_an_extra_rune_on_their_first_turn():
    g = started()
    second = 1 - g.first_player
    clear_hand(g, g.first_player)
    g.step(next(a for a in g.legal_actions() if a.kind is ActionKind.END_TURN))
    assert g.turn_player == second
    assert len(g.players[second].runes()) == 3
    clear_hand(g, second)
    g.step(next(a for a in g.legal_actions() if a.kind is ActionKind.END_TURN))
    assert len(g.players[g.first_player].runes()) == 4      # back to 2 per turn


def test_awaken_readies_runes_units_and_legend():
    g = started()
    seat = g.acting_player
    zones = g.players[seat]
    for obj in [*zones.runes(), *zones.legend]:
        obj.exhausted = True
    clear_hand(g, seat)
    g.step(next(a for a in g.legal_actions() if a.kind is ActionKind.END_TURN))
    clear_hand(g, g.acting_player)
    g.step(next(a for a in g.legal_actions() if a.kind is ActionKind.END_TURN))
    assert g.turn_player == seat
    assert not any(o.exhausted for o in [*zones.runes(), *zones.legend])


def test_rune_pool_empties_at_the_end_of_the_turn():
    g = started()
    seat = g.acting_player
    g.players[seat].rune_pool.energy = 3
    g.step(next(a for a in g.legal_actions() if a.kind is ActionKind.END_TURN))
    assert g.players[seat].rune_pool.energy == 0


def test_when_ending_the_turn_is_the_only_option_it_happens_automatically():
    g = started()
    first = g.first_player
    clear_hand(g, 1 - first)
    g.players[1 - first].champion.objects.clear()
    g.players[1 - first].rune_deck.objects.clear()      # no runes, so nothing is affordable
    g.step(next(a for a in g.legal_actions() if a.kind is ActionKind.END_TURN))
    # the second player has nothing to play, so their turn ends on its own
    assert g.turn == 3 and g.turn_player == first


def test_burn_out_recycles_the_trash_and_gives_the_opponent_a_point():
    g = started()
    seat = g.acting_player
    zones = g.players[seat]
    trash = list(zones.main_deck)[:3]
    for obj in list(zones.main_deck):
        g.move(obj, zones.trash if obj in trash else zones.hand)

    g.draw(seat)

    assert g.points[1 - seat] == 1
    assert len(zones.trash) == 0 and len(zones.main_deck) == 2


def test_burn_out_with_an_empty_trash_still_gives_the_point():
    g = started()
    seat = g.acting_player
    zones = g.players[seat]
    for obj in list(zones.main_deck):
        g.move(obj, zones.hand)
    hand = len(zones.hand)
    g.draw(seat, 2)
    assert g.points[1 - seat] == 2 and len(zones.hand) == hand


def test_winning_needs_eight_points_and_the_lead():
    g = started()
    g.points = [8, 8]
    assert not g._cleanup()
    g.points = [9, 8]
    assert g._cleanup() and g.winner == 0


def test_holding_a_battlefield_scores_in_the_beginning_phase():
    g = started()
    other = 1 - g.acting_player
    g.battlefields[0].controller = other
    ready_unit(g, other, "OGN-013", g.battlefields[0])
    g.step(next(a for a in g.legal_actions() if a.kind is ActionKind.END_TURN))
    assert g.turn_player == other and g.points[other] == 1


def test_final_point_from_a_conquer_needs_every_battlefield_scored():
    g = started()
    seat = g.acting_player
    bf0, bf1 = g.battlefields
    g.points[seat] = 7
    hand = len(g.players[seat].hand)
    g._score(seat, bf0, conquer=True)
    assert g.points[seat] == 7 and len(g.players[seat].hand) == hand + 1     # drew instead

    g.scored.clear()
    g._score(seat, bf1, conquer=False)
    g._score(seat, bf0, conquer=True)
    assert g.points[seat] == 9


# --- paying costs ----------------------------------------------------------------

def test_playing_a_unit_exhausts_runes_and_it_enters_exhausted():
    g = started(decks=[make_test_deck()] * 2)
    seat = g.acting_player
    set_runes(g, seat, "RRGG")
    to_hand(g, seat, "VAN-2")
    copies = len([o for o in g.players[seat].hand if o.card.card_id == "VAN-2"])
    [action] = plays(g, "VAN-2")                 # one action, however many copies I hold

    g.step(action)

    base_units = [o for o in g.players[seat].base if o.card.card_id == "VAN-2"]
    assert len(base_units) == 1 and base_units[0].exhausted
    assert len([o for o in g.players[seat].hand if o.card.card_id == "VAN-2"]) == copies - 1
    assert sum(r.exhausted for r in g.players[seat].runes()) == 2


def test_payment_options_differ_only_in_what_gets_recycled():
    g = started(decks=[make_test_deck()] * 2)
    seat = g.acting_player
    set_runes(g, seat, "RRGG")
    to_hand(g, seat, "TEST-DUAL-UNIT")
    options = plays(g, "TEST-DUAL-UNIT")        # [1][C] on a Fury/Calm card: recycle either
    assert len(options) == 2
    assert len(plays(g, "VAN-2")) <= 1          # plain Energy: which runes to exhaust doesn't matter

    g.step(options[0])

    runes = g.players[seat].runes()
    assert len(runes) == 3                     # one rune recycled
    assert sum(not r.exhausted for r in runes) == 3   # the recycled rune gave the Energy too


def test_recycling_prefers_runes_that_are_already_exhausted():
    g = started(decks=[make_test_deck()] * 2)
    seat = g.acting_player
    spent, *_ = set_runes(g, seat, "RRGG", exhausted="x")
    [action] = plays(g, "CHAMP-TEST")
    assert action.payment.recycle == (spent.oid,)


def test_unaffordable_cards_are_not_offered():
    g = started(decks=[make_test_deck()] * 2)
    seat = g.acting_player
    set_runes(g, seat, "RR")
    to_hand(g, seat, "VAN-6")
    assert plays(g, "VAN-6") == [] and plays(g, "CHAMP-TEST") == []


def test_legend_power_only_pays_for_spells():
    g = started(decks=[make_test_deck()] * 2)
    seat = g.acting_player
    set_runes(g, seat, "RR")
    to_hand(g, seat, "TEST-CALM-SPELL")
    to_hand(g, seat, "TEST-CALM-UNIT")
    [spell] = plays(g, "TEST-CALM-SPELL")
    assert spell.payment == Payment(legend=True)
    assert plays(g, "TEST-CALM-UNIT") == []

    g.step(spell)
    assert g.players[seat].legend.objects[0].exhausted


def test_the_chosen_champion_is_played_from_the_champion_zone():
    g = started()
    seat = g.acting_player
    set_runes(g, seat, "RRRRBB")
    champion = g.players[seat].champion.objects[0]
    assert any(a.card_oid == champion.oid for a in g.legal_actions())


def test_every_card_in_the_kaisa_deck_is_scripted():
    deck, _ = load_demo_decks()
    assert all(c.card_id in SPELLS for c in deck.main if c.is_spell)


# --- the chain and priority ------------------------------------------------------

def chain_game():
    """Turn player holds a Test Sorcery; the opponent holds a Test Reaction; both can pay."""
    g = started(decks=[make_test_deck()] * 2)
    me, opp = g.acting_player, 1 - g.acting_player
    for seat in (me, opp):
        clear_hand(g, seat)
        g.players[seat].champion.objects.clear()
        set_runes(g, seat, "RR")
    to_hand(g, me, "TEST-SORCERY")
    to_hand(g, me, "VAN-1")          # so I still have a choice once the chain is done
    to_hand(g, opp, "TEST-REACTION")
    return g, me, opp


def test_a_spell_waits_on_the_chain_for_reactions():
    g, me, opp = chain_game()
    [sorcery] = plays(g, "TEST-SORCERY")
    g.step(sorcery)
    # I have nothing else to play, so I pass automatically and the opponent may react
    assert len(g.chain_items) == 1 and g.acting_player == opp
    assert {a.kind for a in g.legal_actions()} == {ActionKind.PLAY_CARD, ActionKind.PASS}

    g.step(next(a for a in g.legal_actions() if a.kind is ActionKind.PASS))

    assert not g.chain_items and "Test Sorcery resolves" in g.log
    assert any(o.card.card_id == "TEST-SORCERY" for o in g.players[me].trash)
    assert g.acting_player == me and g.turn_phase is TurnPhase.MAIN


def test_reactions_resolve_before_the_spell_they_respond_to():
    g, me, opp = chain_game()
    g.step(plays(g, "TEST-SORCERY")[0])
    g.step(plays(g, "TEST-REACTION")[0])
    assert not g.chain_items
    order = [line for line in g.log if line.endswith("resolves")]
    assert order == ["Test Reaction resolves", "Test Sorcery resolves"]


def test_only_reactions_can_be_played_while_a_chain_exists():
    g, me, opp = chain_game()
    to_hand(g, opp, "TEST-SORCERY")
    to_hand(g, opp, "VAN-1")
    g.step(plays(g, "TEST-SORCERY")[0])
    assert g.acting_player == opp
    assert plays(g, "TEST-SORCERY") == [] and plays(g, "VAN-1") == []
    assert plays(g, "TEST-REACTION")


def test_with_no_possible_reaction_a_spell_resolves_at_once():
    g, me, opp = chain_game()
    clear_hand(g, opp)
    g.step(plays(g, "TEST-SORCERY")[0])
    assert not g.chain_items and g.acting_player == me


def test_the_opponent_cannot_act_on_my_turn_without_a_chain():
    g, me, opp = chain_game()
    assert g.legal_actions(opp) == []


# --- observation, zones and the loop ---------------------------------------------

def test_observation_hides_private_and_secret_info():
    g = started()
    me = g.acting_player
    obs = g.observation(me)
    assert obs["turn_phase"] == "main"
    mine, theirs = obs["players"][me], obs["players"][1 - me]
    assert all("name" in c for c in mine["zones"]["hand"]["objects"])
    assert all(c == {"hidden": True} for c in theirs["zones"]["hand"]["objects"])
    assert theirs["zones"]["hand"]["count"] == 4
    assert mine["zones"]["main_deck"]["objects"] == [] and mine["zones"]["main_deck"]["count"] > 0


def test_runes_in_play_are_public():
    g = started()
    me = g.acting_player
    theirs = g.observation(1 - me)["players"][me]["zones"]["base"]["objects"]
    assert [c["name"] for c in theirs if "rune" in c["types"]]


def test_zone_changes_follow_056_and_124():
    g = new_game()
    obj = g.players[0].hand.objects[0]
    old_oid = obj.oid
    obj.damage = 2
    g.move(obj, g.players[1].trash)          # 056: goes to its owner's trash instead
    assert obj.zone is g.players[0].trash
    assert obj.oid != old_oid and obj.damage == 0   # 124: new object, state cleared


def test_random_games_end_with_a_winner():
    for seed in range(10):
        for decks in (None, [make_test_deck()] * 2):
            g = new_game(seed, decks)
            winner = play_game(g, [RandomAgent(seed), RandomAgent(seed + 100)])
            assert g.is_over and winner in (0, 1)
            assert g.points[winner] >= 8 and g.points[winner] > g.points[1 - winner]
            assert g.legal_actions() == []


def test_concede_is_only_offered_when_allowed():
    g = started()
    assert all(a.kind is not ActionKind.CONCEDE for a in g.legal_actions())

    g = Game(list(load_demo_decks()), seed=0, allow_concede=True)
    keep_hands(g)
    seat = g.acting_player
    g.step(next(a for a in g.legal_actions() if a.kind is ActionKind.CONCEDE))
    assert g.winner == 1 - seat


def test_game_state_can_be_cloned_and_pickled():
    g = started()
    clone = copy.deepcopy(g)
    clone.step(next(a for a in clone.legal_actions() if a.kind is ActionKind.END_TURN))
    assert g.turn == 1 and clone.turn >= 2
    assert pickle.loads(pickle.dumps(g)).observation(0)["turn"] == 1


def test_random_play_keeps_every_card_and_never_leaks_hidden_info():
    import json
    for seed in range(5):
        for decks in (None, [make_test_deck()] * 2):
            g = new_game(seed, decks)
            agents = [RandomAgent(seed), RandomAgent(seed + 1)]

            def count(seat):
                zones = g.players[seat]
                n = sum(len(z) for z in zones.all())
                n += sum(1 for bf in g.battlefields for o in bf.units if o.owner == seat)
                n += sum(1 for o in g.chain if o.owner == seat)
                return n

            totals = [count(0), count(1)]
            while not g.is_over:
                seat = g.acting_player
                for viewer in (0, 1):
                    obs = g.observation(viewer)
                    json.dumps(obs)
                    hand = obs["players"][1 - viewer]["zones"]["hand"]["objects"]
                    assert all(c == {"hidden": True} for c in hand)
                g.step(agents[seat].act(g.observation(seat), g.legal_actions(seat)))
                assert [count(0), count(1)] == totals


# --- movement, showdowns and focus ---------------------------------------------

def quiet_game():
    """Test decks; the turn player holds nothing, the opponent holds nothing, both have 2 runes."""
    g = started(decks=[make_test_deck()] * 2)
    me, opp = g.acting_player, 1 - g.acting_player
    for seat in (me, opp):
        clear_hand(g, seat)
        g.players[seat].champion.objects.clear()
        set_runes(g, seat, "RR")
    return g, me, opp


def test_moving_to_an_empty_battlefield_conquers_it_after_a_showdown():
    g, me, opp = quiet_game()
    unit = ready_unit(g, me, "VAN-2")
    to_hand(g, me, "VAN-1")                       # keep me from auto-ending the turn afterwards
    [move] = moves(g, 0)

    g.step(move)

    bf = g.battlefields[0]
    assert unit in bf.units.objects and unit.exhausted              # 144.2
    assert "Showdown at " + bf.card.card.name in g.log
    assert bf.controller == me and not bf.contested and g.showdown is None
    assert g.points[me] == 1 and g.acting_player == me              # conquered, back to my Main Phase


def test_identical_units_move_as_one_group_option():
    g, me, opp = quiet_game()
    ready_unit(g, me, "VAN-2")
    ready_unit(g, me, "VAN-2")
    ready_unit(g, me, "VAN-1")
    labels = [a.label for a in moves(g, 0)]
    assert len(labels) == 5                        # {0,1,2} Vanilla 2 x {0,1} Vanilla 1, minus moving none
    assert any("2x Vanilla 2" in label for label in labels)


def test_the_mover_gets_focus_and_focus_alternates_until_both_pass():
    g, me, opp = quiet_game()
    ready_unit(g, me, "VAN-2")
    to_hand(g, me, "TEST-ACTION")
    to_hand(g, opp, "TEST-REACTION")
    g.step(moves(g, 0)[0])

    assert g.showdown == 0 and g.focus == me and g.acting_player == me
    kinds = {a.kind for a in g.legal_actions()}
    assert ActionKind.MOVE not in kinds and ActionKind.END_TURN not in kinds     # 144.1.c
    assert plays(g, "TEST-ACTION")

    g.step(next(a for a in g.legal_actions() if a.kind is ActionKind.PASS))
    assert g.focus == opp and g.acting_player == opp
    g.step(next(a for a in g.legal_actions() if a.kind is ActionKind.PASS))
    assert g.showdown is None and g.battlefields[0].controller == me


def test_a_spell_in_a_showdown_passes_focus_once_its_chain_resolves():
    g, me, opp = quiet_game()
    ready_unit(g, me, "VAN-2")
    to_hand(g, me, "TEST-ACTION")
    to_hand(g, opp, "TEST-REACTION")
    g.step(moves(g, 0)[0])

    g.step(plays(g, "TEST-ACTION")[0])
    assert g.chain_items and g.acting_player == opp               # opp may react
    g.step(next(a for a in g.legal_actions() if a.kind is ActionKind.PASS))

    assert not g.chain_items and g.showdown == 0
    assert g.focus == opp and g.acting_player == opp              # 346
    g.step(next(a for a in g.legal_actions() if a.kind is ActionKind.PASS))
    # I have nothing left to play, so my pass is automatic and the showdown ends
    assert g.showdown is None and g.battlefields[0].controller == me


def test_only_actions_and_reactions_can_be_played_in_a_showdown():
    g, me, opp = quiet_game()
    ready_unit(g, me, "VAN-2")
    to_hand(g, me, "TEST-ACTION")
    to_hand(g, me, "TEST-SORCERY")
    to_hand(g, me, "VAN-1")
    g.step(moves(g, 0)[0])
    assert plays(g, "TEST-ACTION")
    assert plays(g, "TEST-SORCERY") == [] and plays(g, "VAN-1") == []


def test_leaving_a_battlefield_gives_up_control():
    g, me, opp = quiet_game()
    bf = g.battlefields[0]
    unit = ready_unit(g, me, "VAN-2", bf)
    bf.controller = me
    to_hand(g, me, "VAN-1")
    [move] = [a for a in moves(g, None) if a.units == (unit.oid,)]
    g.step(move)
    assert unit.zone is g.players[me].base and bf.controller is None     # 190.4.c


def test_units_can_be_played_to_a_battlefield_i_control():
    g, me, opp = quiet_game()
    bf = g.battlefields[0]
    ready_unit(g, me, "VAN-2", bf)
    bf.controller = me
    to_hand(g, me, "VAN-1")
    destinations = {a.destination for a in plays(g, "VAN-1")}
    assert destinations == {None, 0}
    g.step(next(a for a in plays(g, "VAN-1") if a.destination == 0))
    assert sum(o.card.card_id == "VAN-1" for o in bf.units) == 1 and bf.controller == me


# --- combat -------------------------------------------------------------------

def defended(opp_units, my_units):
    """Opponent controls battlefield 0 with `opp_units`; I have ready `my_units` in base."""
    g, me, opp = quiet_game()
    bf = g.battlefields[0]
    theirs = [ready_unit(g, opp, cid, bf) for cid in opp_units]
    bf.controller = opp
    mine = [ready_unit(g, me, cid) for cid in my_units]
    to_hand(g, me, "VAN-1")                       # keep me from auto-ending the turn afterwards
    return g, me, opp, bf, mine, theirs


def attack(g, units):
    oids = {u.oid for u in units}
    g.step(next(a for a in moves(g, 0) if set(a.units) == oids))


def test_attacking_a_defended_battlefield_wins_it_and_conquers():
    g, me, opp, bf, [mine], [theirs] = defended(["VAN-2"], ["VAN-3"])
    attack(g, [mine])
    assert "Combat at " + bf.card.card.name in " ".join(g.log)
    # neither side had a real choice, so the combat ran to the end on its own
    assert theirs.zone is g.players[opp].trash
    assert mine in bf.units.objects and mine.damage == 0          # healed after combat (466.1.a.1)
    assert bf.controller == me and g.points[me] == 1 and g.showdown is None


def test_a_defender_that_survives_keeps_the_battlefield():
    g, me, opp, bf, [mine], [theirs] = defended(["VAN-4"], ["VAN-2"])
    attack(g, [mine])
    assert mine.zone is g.players[me].trash
    assert theirs in bf.units.objects and theirs.damage == 0
    assert bf.controller == opp and g.points == [0, 0] and not bf.contested


def test_when_both_sides_die_the_battlefield_is_left_uncontrolled():
    g, me, opp, bf, [mine], [theirs] = defended(["VAN-2"], ["VAN-2"])
    attack(g, [mine])
    assert mine.zone is g.players[me].trash and theirs.zone is g.players[opp].trash
    assert bf.controller is None and g.points == [0, 0]


def test_the_attacker_has_focus_in_the_combat_showdown():
    g, me, opp, bf, [mine], _ = defended(["VAN-2"], ["VAN-3"])
    to_hand(g, me, "TEST-ACTION")
    attack(g, [mine])
    assert g.attacker == me and g.focus == me and g.acting_player == me
    assert plays(g, "TEST-ACTION")


def test_the_attacker_chooses_which_defenders_die():
    g, me, opp, bf, [mine], theirs = defended(["VAN-2", "VAN-3"], ["VAN-3"])
    attack(g, [mine])
    # 3 damage: kill the 2-Might unit (1 left over, not enough for the other) or the 3-Might unit
    options = g.legal_actions()
    assert g.acting_player == me
    kills = sorted(a.label for a in options if a.kind is ActionKind.ASSIGN_DAMAGE)
    assert kills == ["Kill Vanilla 2 (3 damage)", "Kill Vanilla 3 (3 damage)"]

    g.step(next(a for a in options if a.label.startswith("Kill Vanilla 3")))
    # the defenders deal 5 back, which kills my unit either way
    assert theirs[1].zone is g.players[opp].trash and theirs[0] in bf.units.objects
    assert mine.zone is g.players[me].trash and bf.controller == opp


def test_attackers_are_recalled_if_defenders_are_still_there():
    g, me, opp, bf, [mine], [theirs] = defended(["VAN-6"], ["VAN-2"])
    g.showdown, g.attacker = 0, me
    g.move(mine, bf.units)
    g._combat_resolution()
    assert mine.zone is g.players[me].base and mine.exhausted is False   # a recall, not a move
    assert bf.controller == opp


def test_damage_assignment_must_be_lethal_before_moving_on():
    g, me, opp, bf, mine, theirs = defended(["VAN-3", "VAN-3", "VAN-3"], ["VAN-5"])
    g.showdown, g.attacker = 0, me
    g.move(mine[0], bf.units)
    [assign] = g._assignment_options(me)       # identical defenders: a single option
    # 5 damage into three 3-Might units: exactly one dies, 2 spills onto another (465.2.c.3)
    assert sorted(n for _, n in assign.damage) == [2, 3]


# --- Kai'Sa deck cards -----------------------------------------------------------

CLEAVE, HEXTECH_RAY, VOID_SEEKER, FALLING_STAR = "OGN-004", "OGN-009", "OGN-024", "OGN-029"
SMOKE_SCREEN, STUPEFY, RETREAT, TIME_WARP = "OGN-093", "OGN-095", "OGN-104", "OGN-122"
DARIUS, KAISA, NOXUS, PORO = "OGN-027", "OGN-039", "OGN-012", "OGN-013"
RAVENBLOOM, WATCHER, SENTRY = "OGN-103", "OGN-116", "OGN-096"
REAVERS_ROW, STARTIPPED_PEAK, ARENAS_GREATEST = "OGN-285", "OGN-288", "OGN-290"


def kaisa_game(my_runes="RRRRBBBB", their_runes=""):
    """Kai'Sa mirror on my Main Phase: empty hands, chosen runes, plain battlefields."""
    g = started()
    me, opp = g.acting_player, 1 - g.acting_player
    for seat, runes in ((me, my_runes), (opp, their_runes)):
        clear_hand(g, seat)
        set_runes(g, seat, runes)
    for bf in g.battlefields:
        use_battlefield(g, bf, None)
    return g, me, opp


def use_battlefield(g, bf, card_id):
    """Swap `bf`'s card for a set-aside battlefield with `card_id` (None: a plain
    battlefield with no abilities). The old card goes to the set-aside pile."""
    from zones import CardInstance
    if card_id is None:
        new = CardInstance(make_vanilla_pool()["BF-1"], owner=bf.card.owner, controller=bf.card.owner)
    else:
        new = next(o for o in g.set_aside if o.card.card_id == card_id)
        g.set_aside.remove(new)
    g.set_aside.append(bf.card)
    bf.card = new


def champion(g, seat):
    return g.players[seat].champion.objects[0]


def act(g, prefix):
    g.step(next(a for a in g.legal_actions() if a.label.startswith(prefix)))


def test_hextech_ray_only_targets_units_at_battlefields_and_can_kill():
    g, me, opp = kaisa_game()
    in_base = ready_unit(g, opp, NOXUS)
    at_bf = ready_unit(g, opp, PORO, g.battlefields[0])
    to_hand(g, me, HEXTECH_RAY)
    to_hand(g, me, PORO)
    options = plays(g, HEXTECH_RAY)
    assert options and all(a.targets == (at_bf.oid,) for a in options)
    g.step(options[0])
    assert at_bf.zone is g.players[opp].trash and in_base.zone is g.players[opp].base


def test_void_seeker_deals_four_and_draws():
    g, me, opp = kaisa_game()
    target = ready_unit(g, opp, NOXUS, g.battlefields[0])
    to_hand(g, me, VOID_SEEKER)
    to_hand(g, me, PORO)
    hand = len(g.players[me].hand)
    g.step(plays(g, VOID_SEEKER)[0])
    assert target.zone is g.players[opp].trash
    assert len(g.players[me].hand) == hand - 1 + 1


def test_falling_star_can_hit_one_unit_twice_or_two_units():
    g, me, opp = kaisa_game("RRRRRR")
    darius = ready_unit(g, opp, DARIUS, g.battlefields[0])
    noxus = ready_unit(g, opp, NOXUS)
    to_hand(g, me, FALLING_STAR)
    to_hand(g, me, PORO)
    pairs = {tuple(sorted(a.targets)) for a in plays(g, FALLING_STAR)}
    assert pairs == {(darius.oid, darius.oid), (darius.oid, noxus.oid), (noxus.oid, noxus.oid)}
    g.step(next(a for a in plays(g, FALLING_STAR) if a.targets == (darius.oid, darius.oid)))
    assert darius.zone is g.players[opp].trash and noxus.zone is g.players[opp].base   # 6 damage on Darius


def test_smoke_screen_shrinks_to_a_minimum_of_one_until_the_turn_ends():
    g, me, opp = kaisa_game()
    darius = ready_unit(g, opp, DARIUS)
    to_hand(g, me, SMOKE_SCREEN)
    to_hand(g, me, PORO)
    g.step(next(a for a in plays(g, SMOKE_SCREEN) if a.targets == (darius.oid,)))
    assert g.might(darius) == 1
    assert g.observation(me)["unit_might"][str(darius.oid)] == 1
    g.step(next(a for a in g.legal_actions() if a.kind is ActionKind.END_TURN))
    assert g.might(darius) == 5


def test_stupefy_gives_minus_one_and_draws():
    g, me, opp = kaisa_game()
    noxus = ready_unit(g, opp, NOXUS)
    poro = ready_unit(g, opp, PORO)
    to_hand(g, me, STUPEFY)
    to_hand(g, me, PORO)
    hand = len(g.players[me].hand)
    g.step(next(a for a in plays(g, STUPEFY) if a.targets == (noxus.oid,)))
    assert g.might(noxus) == 3 and g.might(poro) == 2
    assert len(g.players[me].hand) == hand       # played one, drew one


def test_retreat_returns_a_friendly_unit_and_channels_an_exhausted_rune():
    g, me, opp = kaisa_game()
    mine = ready_unit(g, me, NOXUS, g.battlefields[0])
    theirs = ready_unit(g, opp, NOXUS, g.battlefields[1])
    to_hand(g, me, RETREAT)
    to_hand(g, me, PORO)
    options = plays(g, RETREAT)
    assert {a.targets for a in options} == {(mine.oid,)}      # friendly units only
    runes = len(g.players[me].runes())
    g.step(options[0])
    assert mine.zone is g.players[me].hand and theirs.zone is g.battlefields[1].units
    assert len(g.players[me].runes()) == runes + 1
    assert g.players[me].runes()[-1].exhausted


def test_a_target_that_leaves_before_resolution_is_skipped():
    g, me, opp = kaisa_game(their_runes="BB")
    theirs = ready_unit(g, opp, NOXUS, g.battlefields[0])
    to_hand(g, me, HEXTECH_RAY)
    to_hand(g, me, PORO)
    to_hand(g, opp, RETREAT)
    g.step(plays(g, HEXTECH_RAY)[0])
    assert g.acting_player == opp
    g.step(plays(g, RETREAT)[0])            # save the unit in response
    assert theirs.zone is g.players[opp].hand
    assert [line for line in g.log if line.endswith("resolves")][-2:] == ["Retreat resolves", "Hextech Ray resolves"]
    assert all(u.damage == 0 for u in g.units())


def test_cleave_adds_might_only_while_attacking():
    g, me, opp = kaisa_game()
    poro = ready_unit(g, me, PORO)
    defender = ready_unit(g, opp, NOXUS, g.battlefields[0])
    g.battlefields[0].controller = opp
    to_hand(g, me, CLEAVE)
    to_hand(g, me, PORO)
    g.step(next(a for a in plays(g, CLEAVE) if a.targets == (poro.oid,)))
    assert g.might(poro) == 2
    attack(g, [poro])
    # Pouty Poro attacks with 2 + Assault 3 = 5 and kills the 4-Might Noxus Hopeful
    assert defender.zone is g.players[opp].trash


def test_time_warp_gives_an_extra_turn_and_is_banished():
    g, me, opp = kaisa_game("RRRRRRBBBBBB")
    to_hand(g, me, TIME_WARP)
    to_hand(g, me, PORO)
    g.step(plays(g, TIME_WARP)[0])
    warp = next(o for o in g.players[me].banishment if o.card.card_id == TIME_WARP)
    assert warp and g.extra_turns == [me]
    turn = g.turn
    g.step(next(a for a in g.legal_actions() if a.kind is ActionKind.END_TURN))
    assert g.turn == turn + 1 and g.turn_player == me              # the extra turn
    clear_hand(g, me)
    g.step(next(a for a in g.legal_actions() if a.kind is ActionKind.END_TURN))
    assert g.turn_player == opp                                     # then the normal order resumes


def test_deflect_makes_opponents_pay_extra_power_to_target_it():
    g, me, opp = kaisa_game("RRRR")
    poro = ready_unit(g, opp, PORO, g.battlefields[0])
    noxus = ready_unit(g, opp, NOXUS, g.battlefields[1])
    to_hand(g, me, HEXTECH_RAY)
    card = next(o.card for o in g.players[me].hand if o.card.card_id == HEXTECH_RAY)
    assert len(g.total_cost(me, card, deflect=g._deflect(me, [poro])).power) == 2
    assert len(g.total_cost(me, card, deflect=g._deflect(me, [noxus])).power) == 1
    at_poro = [a for a in plays(g, HEXTECH_RAY) if a.targets == (poro.oid,)]
    assert at_poro and all(len(a.payment.recycle) + a.payment.legend == 2 for a in at_poro)


def test_accelerate_pays_extra_for_a_unit_that_enters_ready():
    g, me, opp = kaisa_game("RRRRBB")
    to_hand(g, me, PORO)
    fast = [a for a in plays(g, KAISA) if a.accelerate]
    assert fast and all(len(a.payment.exhaust) == 5 for a in fast)       # [4] + [1] extra
    g.step(fast[0])
    played = next(o for o in g.players[me].base if o.card.card_id == KAISA)
    assert not played.exhausted


def test_legion_discount_needs_another_card_played_this_turn():
    g, me, opp = kaisa_game("RRRRRR")
    to_hand(g, me, NOXUS)
    to_hand(g, me, PORO)
    card = next(o.card for o in g.players[me].hand if o.card.card_id == NOXUS)
    assert g.total_cost(me, card).energy == 4
    g.step(plays(g, PORO)[0])
    assert g.total_cost(me, card).energy == 2


def test_darius_grows_and_readies_on_the_second_card():
    g, me, opp = kaisa_game("RRRRRRBB")
    darius = ready_unit(g, me, DARIUS)
    darius.exhausted = True
    for _ in range(3):
        to_hand(g, me, PORO)
    g.step(plays(g, PORO)[0])
    assert g.might(darius) == 5 and darius.exhausted
    g.step(plays(g, PORO)[0])
    assert g.might(darius) == 7 and not darius.exhausted


def test_ravenbloom_student_grows_when_i_play_a_spell():
    g, me, opp = kaisa_game()
    student = ready_unit(g, me, RAVENBLOOM)
    target = ready_unit(g, opp, NOXUS)
    to_hand(g, me, STUPEFY)
    to_hand(g, me, PORO)
    g.step(next(a for a in plays(g, STUPEFY) if a.targets == (target.oid,)))
    assert g.might(student) == 3


def test_thousand_tailed_watcher_shrinks_enemy_units():
    g, me, opp = kaisa_game("RRRRRRBBB")
    darius = ready_unit(g, opp, DARIUS)
    poro = ready_unit(g, opp, PORO)
    mine = ready_unit(g, me, PORO)
    to_hand(g, me, WATCHER)
    g.step(plays(g, WATCHER)[0])
    assert g.might(darius) == 2 and g.might(poro) == 1 and g.might(mine) == 2


def test_kaisa_draws_when_she_conquers():
    g, me, opp = kaisa_game()
    kaisa = champion(g, me)
    g.move(kaisa, g.players[me].base)
    kaisa.exhausted = False
    to_hand(g, me, PORO)
    hand = len(g.players[me].hand)
    attack(g, [kaisa])
    assert g.battlefields[0].controller == me
    assert len(g.players[me].hand) == hand + 1


def test_watchful_sentry_draws_when_it_dies():
    g, me, opp = kaisa_game()
    ready_unit(g, opp, SENTRY, g.battlefields[0])
    g.battlefields[0].controller = opp
    darius = ready_unit(g, me, DARIUS)
    to_hand(g, me, PORO)
    hand = len(g.players[opp].hand)
    attack(g, [darius])
    assert len(g.players[opp].hand) == hand + 1


def test_the_arenas_greatest_gives_a_point_on_each_players_first_beginning_phase():
    for seed in range(50):
        g = new_game(seed)
        arenas = sum(bf.card.card.card_id == ARENAS_GREATEST for bf in g.battlefields)
        if arenas:
            break
    keep_hands(g)
    first = g.first_player
    assert g.points[first] == arenas
    g.players[first].rune_pool.energy = 0
    clear_hand(g, first)
    g.step(next(a for a in g.legal_actions() if a.kind is ActionKind.END_TURN))
    while g.turn_player != first or g.turn_phase is not TurnPhase.MAIN:     # back to my next turn
        g.step(next((a for a in g.legal_actions() if a.kind in (ActionKind.END_TURN, ActionKind.PASS,
                                                                 ActionKind.CHOOSE)), g.legal_actions()[0]))
    assert sum(line.startswith("Triggers: The Arena's Greatest") for line in g.log) == 2 * arenas


def test_startipped_peak_may_channel_a_rune_when_held():
    g, me, opp = kaisa_game()
    bf = g.battlefields[0]
    use_battlefield(g, bf, STARTIPPED_PEAK)
    ready_unit(g, me, PORO, bf)
    bf.controller = me
    g._score(me, bf, conquer=False)
    g._settle()
    assert g.decision is not None and g.acting_player == me
    runes = len(g.players[me].runes())
    g.step(next(a for a in g.legal_actions() if a.label == "Channel 1 rune exhausted"))
    assert len(g.players[me].runes()) == runes + 1


def test_reavers_row_lets_the_defender_pull_a_unit_back():
    g, me, opp = kaisa_game()
    bf = g.battlefields[0]
    use_battlefield(g, bf, REAVERS_ROW)
    theirs = ready_unit(g, opp, NOXUS, bf)
    bf.controller = opp
    mine = ready_unit(g, me, DARIUS)
    to_hand(g, me, PORO)
    attack(g, [mine])
    assert g.decision is not None and g.acting_player == opp
    g.step(next(a for a in g.legal_actions() if a.label.startswith("Move Noxus Hopeful")))
    assert theirs.zone is g.players[opp].base
    assert bf.controller == me                     # no defenders left, so I conquer


def test_focus_stays_put_after_a_triggered_ability_resolves():
    g, me, opp = kaisa_game()
    bf = g.battlefields[0]
    use_battlefield(g, bf, REAVERS_ROW)
    ready_unit(g, opp, NOXUS, bf)
    ready_unit(g, opp, PORO, bf)
    bf.controller = opp
    mine = ready_unit(g, me, DARIUS)
    to_hand(g, me, HEXTECH_RAY)
    attack(g, [mine])
    g.step(next(a for a in g.legal_actions() if a.label.startswith("Move Pouty Poro")))
    # the Reaver's Row trigger resolved; the attacker keeps focus (346.1)
    assert g.showdown == 0 and g.focus == me and g.acting_player == me


def test_the_turn_player_chooses_between_staged_showdowns():
    g, me, opp = kaisa_game()
    for bf in g.battlefields:
        ready_unit(g, me, PORO, bf)
        bf.contested, bf.contested_by = True, me
    g._settle()
    assert g.decision is not None and g.decision.kind == "showdown"
    assert len(g.legal_actions()) == 2
    act(g, "Start at")
    assert all(bf.controller == me for bf in g.battlefields)     # both resolve in turn
