import copy
import pickle

from game import ActionKind, Game, RandomAgent, load_demo_decks, play_game
from zones import ZoneKind


def new_game(seed=0):
    return Game(list(load_demo_decks()), seed=seed)


def test_demo_deck_is_legal():
    deck, _ = load_demo_decks()
    assert deck.errors() == []


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


def test_passing_begins_next_turn_with_a_draw():
    g = new_game()
    seat = g.acting_player
    opponent = 1 - seat
    before = len(g.players[opponent].hand)
    rune_before = len(g.players[opponent].rune_deck)
    g.step(next(a for a in g.legal_actions() if a.kind is ActionKind.PASS))
    assert g.turn == 2
    assert g.turn_player == opponent
    assert g.observation(seat)["turn_phase"] == "action"
    assert len(g.players[opponent].hand) == before + 1
    assert len(g.players[opponent].rune_deck) == rune_before - 2
    assert g.players[opponent].rune_pool.energy == 2
    assert sum(g.players[opponent].rune_pool.power.values()) == 2


def test_channeling_adds_power_for_the_top_runes_domain():
    g = new_game()
    seat = g.acting_player
    expected_domains = {domain.value for domain in g.players[seat].rune_deck.objects[-1].card.domains}

    g._channel_rune(seat)

    assert g.players[seat].rune_pool.power == {domain: 1 for domain in expected_domains}


def test_channeling_empty_rune_deck_is_a_noop():
    g = new_game()
    seat = g.acting_player
    zones = g.players[seat]
    zones.rune_deck.objects.clear()

    g._channel_rune(seat)

    assert not zones.rune_pool.power


def test_observation_shows_my_channeled_rune_and_hides_opponents():
    g = new_game()
    seat = g.acting_player
    opponent = 1 - seat
    g._channel_rune(seat)

    mine = g.observation(seat)["players"][seat]["rune_pool"]["runes"]
    theirs = g.observation(opponent)["players"][seat]["rune_pool"]["runes"]

    assert len(mine) == 1 and "name" in mine[0]
    assert theirs == [{"hidden": True}]


def test_starting_turn_readies_runes_and_restores_power():
    g = new_game()
    seat = g.acting_player
    g._channel_rune(seat)
    rune = g.players[seat].rune_pool.runes[0]
    rune.exhausted = True
    g.players[seat].rune_pool.power.clear()

    g._ready_runes(seat)

    assert not rune.exhausted
    assert sum(g.players[seat].rune_pool.power.values()) == len(rune.card.domains)


def test_play_card_spends_energy_and_moves_permanent_to_base():
    g = new_game()
    seat = g.acting_player
    zones = g.players[seat]
    zones.rune_pool.energy = 10
    g._channel_rune(seat)
    g._channel_rune(seat)
    card = next(obj for obj in zones.hand if obj.card.is_permanent and not obj.card.cost.power)
    action = next(a for a in g.legal_actions() if a.card_oid == card.oid)

    g.step(action)

    assert card.zone is zones.base
    assert card not in zones.hand
    assert zones.rune_pool.energy == 10 - card.card.cost.energy


def test_play_card_rejects_insufficient_energy_without_moving_card():
    g = new_game()
    seat = g.acting_player
    zones = g.players[seat]
    card = next(obj for obj in zones.hand if obj.card.is_permanent and not obj.card.cost.power)
    zones.rune_pool.energy = 10
    g._channel_rune(seat)
    g._channel_rune(seat)
    action = next(a for a in g.legal_actions(seat) if a.card_oid == card.oid)
    zones.rune_pool.energy = max(0, card.card.cost.energy - 1)
    energy_before = zones.rune_pool.energy

    try:
        g._play_card(seat, action)
    except ValueError as error:
        assert "not enough Energy" in str(error)
    else:
        raise AssertionError("expected insufficient Energy to be rejected")

    assert card.zone is zones.hand
    assert zones.rune_pool.energy == energy_before


def test_play_card_can_pay_power_cost_for_a_unit():
    g = new_game()
    seat = g.acting_player
    zones = g.players[seat]
    card = next(obj for obj in zones.hand if obj.card.is_unit and obj.card.cost.power)
    zones.rune_pool.energy = card.card.cost.energy
    while (not g._can_pay_power(card.card, zones.rune_pool.power)
           or not g._can_pay_costs(card.card, zones)):
        g._channel_rune(seat)
    action = next(a for a in g.legal_actions(seat) if a.card_oid == card.oid)
    runes_before = len(zones.rune_pool.runes)

    g.step(action)

    assert card.zone is zones.base
    assert len(zones.rune_pool.runes) == runes_before - len(card.card.cost.power)


def test_darius_spends_five_energy_and_exhausts_red_rune():
    g = new_game()
    seat = g.acting_player
    zones = g.players[seat]
    card = next(obj for obj in [*zones.hand, *zones.main_deck] if obj.card.short_name == "Darius")
    if card.zone is not zones.hand:
        g.move(card, zones.hand)
    red_rune = next(rune for rune in zones.rune_deck if any(domain.value == "R" for domain in rune.card.domains))
    runes = [red_rune, *(rune for rune in zones.rune_deck if rune is not red_rune)][:5]
    for rune in runes:
        zones.rune_deck.objects.remove(rune)
        rune.zone = None
        zones.rune_pool.runes.append(rune)
        for domain in rune.card.domains:
            zones.rune_pool.power[domain.value] = zones.rune_pool.power.get(domain.value, 0) + 1
    zones.rune_pool.energy = 10
    action = next(a for a in g.legal_actions(seat) if a.card_oid == card.oid)

    g.step(action)

    assert card.zone is zones.base
    assert zones.rune_pool.energy == 5
    assert red_rune.zone is zones.rune_deck
    assert red_rune in zones.rune_deck.objects
    assert sum(rune.exhausted for rune in zones.rune_pool.runes) == 4


def test_observation_hides_private_and_secret_info():
    g = new_game()
    obs = g.observation(0)
    assert obs["turn_phase"] == "action"
    me, opp = obs["players"]
    assert all("name" in c for c in me["zones"]["hand"]["objects"])
    assert all(c == {"hidden": True} for c in opp["zones"]["hand"]["objects"])
    assert opp["zones"]["hand"]["count"] == 4
    assert me["zones"]["main_deck"]["objects"] == [] and me["zones"]["main_deck"]["count"] > 0


def test_zone_changes_follow_056_and_124():
    g = new_game()
    obj = g.players[0].hand.objects[0]
    old_oid = obj.oid
    obj.damage = 2
    g.move(obj, g.players[1].trash)          # 056: goes to its owner's trash instead
    assert obj.zone is g.players[0].trash
    assert obj.oid != old_oid and obj.damage == 0   # 124: new object, state cleared


def test_loop_ends_with_a_winner():
    for seed in range(20):
        g = new_game(seed)
        winner = play_game(g, [RandomAgent(seed), RandomAgent(seed + 100)])
        assert g.is_over and winner in (0, 1)
        assert g.legal_actions() == []


def test_concede():
    g = new_game()
    seat = g.acting_player
    g.step(next(a for a in g.legal_actions() if a.kind is ActionKind.CONCEDE))
    assert g.winner == 1 - seat


def test_game_state_can_be_cloned_and_pickled():
    g = new_game()
    clone = copy.deepcopy(g)
    clone.step(clone.legal_actions()[0])
    assert g.turn == 1 and clone.turn == 2
    assert pickle.loads(pickle.dumps(g)).observation(0)["turn"] == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
