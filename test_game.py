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


def test_observation_hides_private_and_secret_info():
    g = new_game()
    obs = g.observation(0)
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
