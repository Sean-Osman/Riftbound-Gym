"""The Annie, Master Yi and Miss Fortune decks: every card and the rules they
need (Hidden, Ganking, gear, stuns, counters, Tank and Shield, choices made on
resolution, the Ending Step, replacement effects). Kai'Sa cards are covered in
test_game.py and test_interactions.py; this file reuses their helpers.
"""

import copy
import itertools
import json
import pickle

from game import DECK_NAMES, ActionKind, Game, RandomAgent, TurnPhase, load_decks
from scripts import ACTIVATED, SPELLS, TRIGGERS
from test_game import clear_hand, keep_hands, moves, plays, ready_unit, set_runes, to_hand, use_battlefield

DECKS = load_decks()

ANNIE_LEGEND, ANNIE, VI, FIGHT_OR_FLIGHT, GUST, REBUKE, RIDE_THE_WIND = \
    "OGS-017", "OGS-010", "OGN-036", "OGN-168", "OGN-169", "OGN-172", "OGN-173"
SNEAKY, SCRAPHEAP, STACKED_DECK, MERCHANT, DREAMING_TREE, ZAUN, FLASH, PORO, CLEAVE = \
    "OGN-176", "OGN-182", "OGN-183", "OGN-185", "OGN-292", "OGN-298", "OGS-011", "OGN-013", "OGN-004"
YI, CHARM, DEFY, FIND_YOUR_CENTER, RUNE_PRISON, DISCIPLINE, WIND_WALL, FAEFOLK, ZHONYA = \
    "OGS-009", "OGN-043", "OGN-045", "OGN-047", "OGN-050", "OGN-058", "OGN-064", "OGN-075", "OGN-077"
WHITEFLAME, CHALLENGE, MOBILIZE, CATALYST, DRAKE, SABOTAGE, VOLIBEAR, AURORA, DEADBLOOM = \
    "OGN-082", "OGN-128", "OGN-134", "OGN-138", "OGN-142", "OGN-156", "OGN-158", "OGN-160", "OGN-161"
ASPIRANTS_CLIMB, OBELISK, VILEMAW = "OGN-276", "OGN-284", "OGN-295"
MF, PRIMAL_STRENGTH, ACCEPTABLE_LOSSES, TROVE, MINDSPLITTER, SOULGORGER, INVERT, BULLET_TIME = \
    "OGN-162", "OGN-154", "OGN-179", "OGN-186", "OGN-192", "OGN-196", "OGN-201", "OGN-268"
WATCHER = "OGN-116"


# --- helpers -------------------------------------------------------------------

def setup(mine, theirs="kaisa", my_runes="", their_runes="", stall=False):
    """Seat 0 plays `mine` and is in their first Main Phase with an empty hand, the
    chosen runes and plain battlefields. Seat 1 plays `theirs`. `stall` puts a ready
    unit in my base so my turn doesn't end on its own when I run out of things to do."""
    for seed in range(100):
        g = Game([DECKS[mine], DECKS[theirs]], seed=seed)
        keep_hands(g)
        if g.acting_player == 0 and g.turn_phase is TurnPhase.MAIN:
            break
    for seat, runes in ((0, my_runes), (1, their_runes)):
        clear_hand(g, seat)
        set_runes(g, seat, runes)
    for bf in g.battlefields:
        use_battlefield(g, bf, None)
    if stall:
        unit = next(o for o in g.players[0].main_deck if o.card.is_unit)
        g.move(unit, g.players[0].base)
    return g, 0, 1


def gear(g, seat, card_id):
    """Put a copy of a gear into `seat`'s base."""
    zones = g.players[seat]
    obj = next(o for o in [*zones.main_deck, *zones.hand] if o.card.card_id == card_id)
    g.move(obj, zones.base)
    return obj


def ready_champion(g, seat, where=None):
    """Put `seat`'s Chosen Champion on the board, ready."""
    obj = g.players[seat].champion.objects[0]
    if where is not None and where.controller is None and not where.units.objects:
        where.controller = seat
    g.move(obj, g.players[seat].base if where is None else where.units)
    obj.exhausted = False
    return obj


def labeled(g, prefix, kind=None):
    return [a for a in g.legal_actions() if a.label.startswith(prefix) and (kind is None or a.kind is kind)]


def act(g, prefix, kind=None):
    g.step(labeled(g, prefix, kind)[0])


def pass_until_empty(g):
    while g.chain_items and g.decision is None:
        g.step(next(a for a in g.legal_actions() if a.kind is ActionKind.PASS))


def to_turn_of(g, seat):
    """End turns until it's `seat`'s Main Phase with nothing pending."""
    while not (g.turn_player == seat and g.turn_phase is TurnPhase.MAIN and g.acting_player == seat
               and not g.chain_items and g.decision is None):
        legal = g.legal_actions()
        pick = next((a for a in legal if a.kind is ActionKind.END_TURN), None) \
            or next((a for a in legal if a.kind is ActionKind.PASS), None) or legal[0]
        g.step(pick)


def ids(zone):
    return [o.card.card_id for o in zone]


def top_of_deck(g, seat, *card_ids):
    """Put these cards on top of the Main Deck, the last one on top."""
    zones = g.players[seat]
    for cid in card_ids:
        obj = next(o for o in [*zones.main_deck, *zones.hand, *zones.trash] if o.card.card_id == cid)
        g.move(obj, zones.main_deck)


# --- decks ---------------------------------------------------------------------

def test_the_meta_decks_are_legal_and_fully_scripted():
    for name, deck in DECKS.items():
        assert deck.errors() == [], name
        for card in {deck.champion, *deck.main}:
            if card.is_spell:
                assert card.card_id in SPELLS, card.name
            elif card.rules_text:
                has_script = card.card_id in TRIGGERS or card.card_id in ACTIVATED
                passive = card.card_id in {SNEAKY, DEADBLOOM, YI, ZHONYA, VI, PORO}   # keywords / passives
                assert has_script or passive or not card.rules_text.strip() or card.keywords, card.name


def test_random_games_in_every_pairing_keep_cards_and_hide_information():
    for a, b in itertools.combinations_with_replacement(DECK_NAMES, 2):
        for seed in range(3):
            g = Game([DECKS[a], DECKS[b]], seed=seed)
            agents = [RandomAgent(seed), RandomAgent(seed + 1)]

            def count(seat):
                zones = g.players[seat]
                n = sum(len(z) for z in zones.all())
                n += sum(1 for bf in g.battlefields for o in [*bf.units, *bf.facedown] if o.owner == seat)
                n += sum(1 for o in g.chain if o.owner == seat)
                return n

            totals = [count(0), count(1)]
            while not g.is_over:
                seat = g.acting_player
                for viewer in (0, 1):
                    obs = g.observation(viewer)
                    json.dumps(obs)
                    assert all(c == {"hidden": True} for c in obs["players"][1 - viewer]["zones"]["hand"]["objects"])
                    for bf in obs["battlefields"]:
                        for c in bf["facedown"]["objects"]:
                            assert "card_id" not in c or c["controller"] == viewer
                    d = obs["decision"]
                    if d is not None and d["seat"] != viewer and g.decision.private:
                        assert d["options"] == []
                g.step(agents[seat].act(g.observation(seat), g.legal_actions(seat)))
                assert [count(0), count(1)] == totals, (a, b, seed)
            assert g.winner is not None


def test_games_with_the_new_decks_copy_and_pickle():
    g = Game([DECKS["annie"], DECKS["miss_fortune"]], seed=3)
    agents = [RandomAgent(0), RandomAgent(1)]
    for _ in range(80):
        if g.is_over:
            break
        g.step(agents[g.acting_player].act(g.observation(g.acting_player), g.legal_actions()))
    clone = pickle.loads(pickle.dumps(g))
    assert copy.deepcopy(g).observation(0) == g.observation(0) == clone.observation(0)


# --- Hidden (421, 811) -------------------------------------------------------------

def test_hiding_costs_any_power_and_the_card_stays_secret():
    g, me, opp = setup("annie", my_runes="RP")
    ready_unit(g, me, PORO, g.battlefields[0])
    to_hand(g, me, FIGHT_OR_FLIGHT)
    hides = [a for a in g.legal_actions() if a.kind is ActionKind.HIDE]
    assert hides and all(a.destination == 0 for a in hides)       # only where I control
    g.step(hides[0])
    assert len(g.runes(me)) == 1                                   # [A]: one rune recycled
    theirs = g.observation(opp)["battlefields"][0]["facedown"]
    assert theirs["count"] == 1 and theirs["objects"] == [{"hidden": True}]
    assert not any("Fight or Flight" in line for line in g.log)
    assert not labeled(g, "Play Fight or Flight from hidden")      # not on the turn it was hidden


def test_a_hidden_card_plays_for_free_on_a_later_turn_targeting_only_its_battlefield():
    g, me, opp = setup("annie", my_runes="P")
    mine = ready_unit(g, me, PORO, g.battlefields[0])
    theirs = ready_unit(g, opp, PORO, g.battlefields[1])
    to_hand(g, me, FIGHT_OR_FLIGHT)
    act(g, "Hide Fight or Flight")
    to_turn_of(g, opp)
    to_turn_of(g, me)
    options = labeled(g, "Play Fight or Flight from hidden")
    assert options and all(a.targets == (mine.oid,) for a in options)
    assert all(a.payment.exhaust == () and a.payment.recycle == () for a in options)
    assert theirs.oid not in {t for a in options for t in a.targets}
    g.step(options[0])
    pass_until_empty(g)
    assert mine.zone is g.players[me].base


def test_hidden_cards_are_removed_when_their_battlefield_is_lost():
    g, me, opp = setup("annie", my_runes="P")
    poro = ready_unit(g, me, PORO, g.battlefields[0])
    to_hand(g, me, CLEAVE)                                          # something to keep the turn going
    ff = to_hand(g, me, FIGHT_OR_FLIGHT)
    act(g, "Hide Fight or Flight")
    g.step(next(a for a in moves(g, None) if a.units == (poro.oid,)))
    assert g.battlefields[0].controller is None
    assert FIGHT_OR_FLIGHT in ids(g.players[me].trash) and not g.battlefields[0].facedown.objects
    assert ff.zone is g.players[me].trash


def test_hidden_zhonyas_hourglass_is_played_to_its_battlefield_then_recalled():
    g, me, opp = setup("master_yi", my_runes="G")
    ready_unit(g, me, FAEFOLK, g.battlefields[0])
    to_hand(g, me, ZHONYA)
    act(g, "Hide Zhonya's Hourglass")
    to_turn_of(g, opp)
    to_turn_of(g, me)
    act(g, "Play Zhonya's Hourglass from hidden")
    assert ZHONYA in ids(g.players[me].base)                       # 457.1: recalled in the cleanup


# --- movement --------------------------------------------------------------------

def test_ganking_units_can_move_between_battlefields():
    g, me, opp = setup("annie")
    vi = ready_unit(g, me, VI, g.battlefields[0])
    poro = ready_unit(g, me, PORO, g.battlefields[1])
    assert any(vi.oid in a.units for a in moves(g, 1))
    assert not any(poro.oid in a.units for a in moves(g, 0))
    g.step(next(a for a in moves(g, 1) if a.units == (vi.oid,)))
    assert vi.zone is g.battlefields[1].units


def test_miss_fortunes_legend_gives_a_unit_ganking_this_turn():
    g, me, opp = setup("miss_fortune", stall=True)
    drake = ready_unit(g, me, MINDSPLITTER, g.battlefields[0])
    ready_unit(g, me, MINDSPLITTER, g.battlefields[1])
    assert not any(drake.oid in a.units for a in moves(g, 1))
    act(g, "Miss Fortune, Bounty Hunter: give a unit Ganking", ActionKind.ACTIVATE)
    pass_until_empty(g)
    assert g.legend(me).exhausted
    assert any(drake.oid in a.units for a in moves(g, 1))


def test_units_cannot_leave_vilemaws_lair_for_base():
    g, me, opp = setup("annie", "master_yi", my_runes="PPP")
    use_battlefield(g, g.battlefields[0], VILEMAW)
    vi = ready_unit(g, me, VI, g.battlefields[0])
    assert not moves(g, None)
    assert moves(g, 1)                                              # Ganking still works
    to_hand(g, me, RIDE_THE_WIND)
    assert all(a.destination is not None for a in plays(g, RIDE_THE_WIND))
    assert g.move_units(me, [vi], None) == []


def test_charm_moves_an_enemy_unit_and_ride_the_wind_moves_and_readies():
    g, me, opp = setup("master_yi", my_runes="GG")
    theirs = ready_unit(g, opp, PORO, g.battlefields[0])
    to_hand(g, me, CHARM)
    g.step(next(a for a in plays(g, CHARM) if a.destination is None))
    pass_until_empty(g)
    assert theirs.zone is g.players[opp].base

    g, me, opp = setup("annie", my_runes="PPP")
    poro = ready_unit(g, me, PORO)
    poro.exhausted = True
    to_hand(g, me, RIDE_THE_WIND)
    g.step(next(a for a in plays(g, RIDE_THE_WIND) if a.destination == 1))
    pass_until_empty(g)
    assert poro.zone is g.battlefields[1].units and not poro.exhausted


def test_flash_moves_up_to_two_friendly_units_to_base():
    g, me, opp = setup("annie", my_runes="PP")
    a = ready_unit(g, me, PORO, g.battlefields[0])
    b = ready_unit(g, me, VI, g.battlefields[1])
    to_hand(g, me, FLASH)
    options = plays(g, FLASH)
    assert {len(x.targets) for x in options} == {1, 2}
    g.step(next(x for x in options if len(x.targets) == 2))
    pass_until_empty(g)
    assert a.zone is b.zone is g.players[me].base


def test_units_play_to_open_or_occupied_enemy_battlefields_when_allowed():
    g, me, opp = setup("annie", my_runes="PPP")
    ready_unit(g, opp, PORO, g.battlefields[1])
    to_hand(g, me, SNEAKY)
    assert {a.destination for a in plays(g, SNEAKY)} == {None, 0}      # the open battlefield
    g, me, opp = setup("master_yi", my_runes="OOOOOOGGGG")
    ready_unit(g, opp, PORO, g.battlefields[1])
    to_hand(g, me, DEADBLOOM)
    assert {a.destination for a in plays(g, DEADBLOOM)} == {None, 1}   # an occupied enemy one
    g.step(next(a for a in plays(g, DEADBLOOM) if a.destination == 1))
    assert PORO in ids(g.players[opp].trash)                           # it attacked and won


# --- combat ------------------------------------------------------------------------

def test_tank_must_be_killed_first_and_shield_helps_it_defend():
    g, me, opp = setup("master_yi", "miss_fortune")
    bf = g.battlefields[0]
    voli = ready_unit(g, opp, VOLIBEAR, bf)
    their_drake = ready_unit(g, opp, DRAKE, bf)
    mine = [ready_unit(g, me, DRAKE), ready_unit(g, me, FAEFOLK)]
    assert g.might(voli) == 10
    g.step(next(a for a in moves(g, 0) if set(a.units) == {u.oid for u in mine}))
    # 16 damage: enough for the 10 Might Drake alone, but Tank (13 with Shield 3) comes first
    assert VOLIBEAR in ids(g.players[opp].trash)
    assert their_drake.zone is bf.units


def test_a_stunned_unit_deals_no_combat_damage():
    g, me, opp = setup("master_yi", "miss_fortune", my_runes="GGG")
    bf = g.battlefields[0]
    theirs = ready_unit(g, opp, DRAKE, bf)
    mine = ready_unit(g, me, DRAKE)
    to_hand(g, me, RUNE_PRISON)
    g.step(plays(g, RUNE_PRISON)[0] if len(plays(g, RUNE_PRISON)) == 1 else
           next(a for a in plays(g, RUNE_PRISON) if a.targets == (theirs.oid,)))
    pass_until_empty(g)
    assert theirs.stunned
    g.step(next(a for a in moves(g, 0) if a.units == (mine.oid,)))
    assert mine.zone is bf.units and DRAKE in ids(g.players[opp].trash)


def test_master_yi_gives_a_lone_defender_two_might():
    g, me, opp = setup("kaisa", "master_yi")
    bf = g.battlefields[0]
    faefolk = ready_unit(g, opp, FAEFOLK, bf)
    watcher = ready_unit(g, me, WATCHER)
    g.step(next(a for a in moves(g, 0) if a.units == (watcher.oid,)))
    assert faefolk.zone is bf.units and WATCHER in ids(g.players[me].trash)   # 7 < 6 + 2


def test_zhonyas_hourglass_saves_a_dying_unit_instead():
    g, me, opp = setup("master_yi", "miss_fortune", my_runes="OOO", stall=True)
    zhonya = gear(g, me, ZHONYA)
    faefolk = ready_unit(g, me, FAEFOLK)
    drake = ready_unit(g, opp, DRAKE, g.battlefields[0])
    to_hand(g, me, CHALLENGE)
    g.step(next(a for a in plays(g, CHALLENGE) if a.targets[0] == faefolk.oid))
    pass_until_empty(g)
    assert zhonya.zone is g.players[me].trash
    assert faefolk.zone is g.players[me].base and faefolk.exhausted and faefolk.damage == 0
    assert drake.damage == 6 and drake.zone is g.battlefields[0].units   # 6 < 10: it survives
    assert not any("Tasty Faefolk: Deathknell" in line for line in g.log)


# --- counters and the chain ------------------------------------------------------

def test_defy_counters_only_cheap_spells_and_wind_wall_counters_anything():
    g, me, opp = setup("master_yi", my_runes="GGGGGGOOO", stall=True)
    unit = ready_unit(g, me, FAEFOLK)
    for cid in (DISCIPLINE, WIND_WALL, DEFY):
        to_hand(g, me, cid)
    g.step(next(a for a in plays(g, DISCIPLINE) if a.targets == (unit.oid,)))
    discipline = g.chain_items[-1].obj
    g.step(next(a for a in plays(g, WIND_WALL) if a.targets == (discipline.oid,)))
    wind_wall = g.chain_items[-1].obj
    defy_targets = {a.targets for a in plays(g, DEFY)}
    assert (discipline.oid,) in defy_targets and (wind_wall.oid,) not in defy_targets
    hand = len(g.players[me].hand)
    pass_until_empty(g)
    assert DISCIPLINE in ids(g.players[me].trash) and g.might(unit) == 6
    assert len(g.players[me].hand) == hand                          # countered: no draw
    assert any("Discipline is countered" in line for line in g.log)


# --- choices on resolution -------------------------------------------------------

def test_stacked_deck_choice_is_private():
    g, me, opp = setup("annie", my_runes="P")
    top_of_deck(g, me, PORO, VI, GUST)
    to_hand(g, me, STACKED_DECK)
    act(g, "Play Stacked Deck")
    pass_until_empty(g)
    assert g.decision is not None and g.decision.private
    assert g.observation(opp)["decision"]["options"] == []
    assert g.observation(opp)["decision"]["count"] == 3
    act(g, "Put Vi")
    assert ids(g.players[me].hand) == [VI]
    assert ids(g.players[me].main_deck)[:2] and VI not in ids(g.players[me].main_deck)[-2:]


def test_traveling_merchant_discards_then_draws_when_it_moves():
    g, me, opp = setup("annie")
    merchant = ready_unit(g, me, MERCHANT)
    to_hand(g, me, GUST)
    to_hand(g, me, CLEAVE)
    g.step(next(a for a in moves(g, 0) if a.units == (merchant.oid,)))
    assert g.decision is not None and g.decision.seat == me and g.decision.private
    act(g, "Discard Gust")
    assert GUST in ids(g.players[me].trash) and len(g.players[me].hand) == 2


def test_mindsplitter_discards_a_revealed_card_and_scrapheap_draws():
    g, me, opp = setup("miss_fortune", "annie", my_runes="PPPPOOOO", stall=True)
    to_hand(g, opp, SCRAPHEAP)
    to_hand(g, opp, GUST)
    to_hand(g, me, MINDSPLITTER)
    act(g, "Play Mindsplitter")
    pass_until_empty(g)
    assert not g.decision.private and any("reveals their hand" in line for line in g.log)
    act(g, "Discard Scrapheap")
    pass_until_empty(g)
    assert SCRAPHEAP in ids(g.players[opp].trash)
    assert len(g.players[opp].hand) == 2                           # Gust plus Scrapheap's draw


def test_sabotage_recycles_a_non_unit_card():
    g, me, opp = setup("master_yi", "annie", my_runes="OO", stall=True)
    to_hand(g, opp, PORO)
    to_hand(g, opp, GUST)
    to_hand(g, me, SABOTAGE)
    act(g, "Play Sabotage")
    pass_until_empty(g)
    assert ids(g.players[opp].hand) == [PORO] and ids(g.players[opp].main_deck)[0] == GUST


def test_bullet_time_pays_any_amount_to_hit_every_enemy_unit_there():
    g, me, opp = setup("miss_fortune", my_runes="OOPP")
    enemies = [ready_unit(g, opp, PORO, g.battlefields[0]) for _ in range(2)]
    mine = ready_unit(g, me, DRAKE, g.battlefields[0])
    g.battlefields[0].controller = None
    to_hand(g, me, BULLET_TIME)
    g.step(next(a for a in plays(g, BULLET_TIME) if a.targets == (g.battlefields[0].card.oid,)))
    pass_until_empty(g)
    amounts = [o["amount"] for o in g.observation(me)["decision"]["options"]]
    assert amounts == [0, 1, 2, 3, 4]
    act(g, "Pay 2 Power")
    assert all(e.zone is g.players[opp].trash for e in enemies) and mine.damage == 0
    assert len(g.runes(me)) == 2


def test_acceptable_losses_makes_each_player_kill_a_gear():
    g, me, opp = setup("miss_fortune", "master_yi", my_runes="P", stall=True)
    trove = gear(g, me, TROVE)
    zhonya = gear(g, opp, ZHONYA)
    to_hand(g, me, ACCEPTABLE_LOSSES)
    act(g, "Play Acceptable Losses")
    pass_until_empty(g)
    assert trove.zone is g.players[me].trash and zhonya.zone is g.players[opp].trash
    assert len(g.players[me].hand) == 1                           # Treasure Trove left: draw 1


def test_invert_timelines_discards_every_hand_and_draws_four():
    g, me, opp = setup("miss_fortune", "annie", my_runes="PPPP")
    to_hand(g, opp, GUST)
    to_hand(g, me, INVERT)
    to_hand(g, me, BULLET_TIME)
    act(g, "Play Invert Timelines")
    pass_until_empty(g)
    assert len(g.players[me].hand) == len(g.players[opp].hand) == 4
    assert GUST in ids(g.players[opp].trash) and BULLET_TIME in ids(g.players[me].trash)


# --- triggers --------------------------------------------------------------------

def test_annies_legend_readies_two_runes_at_the_end_of_the_turn():
    g, me, opp = setup("annie", my_runes="RRPP")
    for rune in g.runes(me):
        rune.exhausted = True
    g.step(next(a for a in g.legal_actions() if a.kind is ActionKind.END_TURN))
    to_turn_of(g, opp)
    assert sum(not r.exhausted for r in g.runes(me)) == 2


def test_dazzling_aurora_plays_the_first_unit_from_the_deck_at_the_end_of_the_turn():
    g, me, opp = setup("master_yi")
    gear(g, me, AURORA)
    top_of_deck(g, me, DRAKE, MOBILIZE)
    g.step(next(a for a in g.legal_actions() if a.kind is ActionKind.END_TURN))
    to_turn_of(g, opp)
    drake = next(o for o in g.players[me].base if o.card.card_id == DRAKE)
    assert ids(g.players[me].main_deck)[0] == MOBILIZE              # recycled to the bottom
    assert any("reveals Mobilize, Mountain Drake" in line for line in g.log)
    assert drake.controller == me


def test_whiteflame_protector_gives_a_unit_eight_might():
    g, me, opp = setup("master_yi", my_runes="GGGGGGOOOO", stall=True)
    faefolk = ready_unit(g, me, FAEFOLK)
    to_hand(g, me, WHITEFLAME)
    act(g, "Play Whiteflame Protector")
    act(g, "+8 Might to Tasty Faefolk")
    pass_until_empty(g)
    assert g.might(faefolk) == 14


def test_tasty_faefolk_deathknell_channels_and_draws():
    g, me, opp = setup("master_yi", "miss_fortune", my_runes="OO", stall=True)
    faefolk = ready_unit(g, me, FAEFOLK)
    ready_unit(g, opp, DRAKE, g.battlefields[0])
    to_hand(g, me, CHALLENGE)
    g.step(next(a for a in plays(g, CHALLENGE) if a.targets[0] == faefolk.oid))
    pass_until_empty(g)
    assert faefolk.zone is g.players[me].trash
    assert len(g.runes(me)) == 3 and len(g.players[me].hand) == 1   # 2 - 1 recycled + 2 channeled


def test_volibear_draws_when_the_opponent_moves_to_another_battlefield():
    g, me, opp = setup("kaisa", "miss_fortune", stall=True)
    ready_unit(g, opp, VOLIBEAR, g.battlefields[0])
    poro = ready_unit(g, me, PORO)
    hand = len(g.players[opp].hand)
    g.step(next(a for a in moves(g, 1) if len(a.units) == 1))      # the stall unit is a Poro too
    pass_until_empty(g)
    assert len(g.players[opp].hand) == hand + 1


def test_miss_fortune_readies_something_the_first_time_she_moves():
    g, me, opp = setup("miss_fortune", my_runes="OO", stall=True)
    captain = ready_champion(g, me, g.battlefields[0])
    g.battlefields[1].controller = me
    for rune in g.runes(me):
        rune.exhausted = True
    g.step(next(a for a in moves(g, 1) if a.units == (captain.oid,)))
    assert [o["label"] for o in g.observation(me)["decision"]["options"]][-1] == "Don't ready anything"
    act(g, "Ready Body Rune")
    pass_until_empty(g)
    assert sum(not r.exhausted for r in g.runes(me)) == 1


def test_annie_stubborn_returns_a_spell_from_the_trash():
    g, me, opp = setup("annie", my_runes="PPPPP")
    g.move(next(o for o in g.players[me].main_deck if o.card.card_id == GUST), g.players[me].trash)
    act(g, "Play Annie, Stubborn")
    pass_until_empty(g)
    assert ids(g.players[me].hand) == [GUST]


def test_soulgorger_plays_a_unit_from_the_trash_paying_only_power():
    g, me, opp = setup("miss_fortune", my_runes="PPPPPPOOOOOO", stall=True)
    g.move(next(o for o in g.players[me].main_deck if o.card.card_id == MINDSPLITTER), g.players[me].trash)
    to_hand(g, me, SOULGORGER)
    act(g, "Play Soulgorger")
    act(g, "Play Mindsplitter")
    pass_until_empty(g)
    assert MINDSPLITTER in ids(g.players[me].base)
    assert len(g.runes(me)) == 12 - 2 - 2                          # PP for each


def test_vi_recycles_a_card_from_the_trash_for_might():
    g, me, opp = setup("annie")
    vi = ready_unit(g, me, VI, g.battlefields[0])
    assert not labeled(g, "Vi, Destructive", ActionKind.ACTIVATE)  # nothing to recycle
    g.move(next(o for o in g.players[me].main_deck if o.card.card_id == GUST), g.players[me].trash)
    act(g, "Vi, Destructive", ActionKind.ACTIVATE)
    pass_until_empty(g)
    assert g.might(vi) == 4 and not g.players[me].trash.objects


def test_treasure_trove_kills_itself_to_draw_and_channel():
    g, me, opp = setup("miss_fortune", my_runes="P")
    gear(g, me, TROVE)
    act(g, "Treasure Trove", ActionKind.ACTIVATE)
    pass_until_empty(g)
    assert TROVE in ids(g.players[me].trash)
    assert len(g.players[me].hand) == 1 and len(g.runes(me)) == 1  # the Chaos rune was recycled


def test_scrapheap_draws_when_played():
    g, me, opp = setup("annie", my_runes="PP")
    to_hand(g, me, SCRAPHEAP)
    act(g, "Play Scrapheap")
    pass_until_empty(g)
    assert len(g.players[me].hand) == 1


# --- spells ------------------------------------------------------------------------

def test_gust_only_returns_small_units_at_battlefields():
    g, me, opp = setup("annie", my_runes="P")
    small = ready_unit(g, opp, PORO, g.battlefields[0])
    ready_unit(g, opp, WATCHER, g.battlefields[1])
    ready_unit(g, opp, PORO)
    to_hand(g, me, GUST)
    assert {a.targets for a in plays(g, GUST)} == {(small.oid,)}


def test_challenge_makes_two_units_fight():
    g, me, opp = setup("master_yi", my_runes="OO")
    drake = ready_unit(g, me, DRAKE)
    poro = ready_unit(g, opp, PORO)
    to_hand(g, me, CHALLENGE)
    g.step(plays(g, CHALLENGE)[0])
    pass_until_empty(g)
    assert poro.zone is g.players[opp].trash and drake.damage == 2


def test_find_your_center_is_cheaper_when_an_opponent_is_close_to_winning():
    g, me, opp = setup("master_yi", my_runes="G")
    to_hand(g, me, FIND_YOUR_CENTER)
    assert not plays(g, FIND_YOUR_CENTER)
    g.points[opp] = 5
    act(g, "Play Find Your Center")
    pass_until_empty(g)
    assert len(g.players[me].hand) == 1 and len(g.runes(me)) == 2


def test_mobilize_and_catalyst_draw_when_they_cannot_channel():
    g, me, opp = setup("master_yi", my_runes="OOOOOO")
    for rune in list(g.players[me].rune_deck):
        g.move(rune, g.players[me].trash)
    to_hand(g, me, CATALYST)
    act(g, "Play Catalyst of Aeons")
    pass_until_empty(g)
    assert len(g.players[me].hand) == 1


# --- battlefields ------------------------------------------------------------------

def test_aspirants_climb_raises_the_victory_score():
    g, me, opp = setup("master_yi")
    use_battlefield(g, g.battlefields[0], ASPIRANTS_CLIMB)
    assert g.victory_score == 9 and g.observation(me)["victory_score"] == 9
    g.points[me] = 8
    g._cleanup()
    assert not g.is_over


def test_the_dreaming_tree_draws_the_first_time_each_turn():
    g, me, opp = setup("annie", my_runes="RRRR")
    use_battlefield(g, g.battlefields[0], DREAMING_TREE)
    unit = ready_unit(g, me, PORO, g.battlefields[0])
    for _ in range(2):
        to_hand(g, me, CLEAVE)
    act(g, "Play Cleave")
    pass_until_empty(g)
    assert len(g.players[me].hand) == 2
    act(g, "Play Cleave")
    pass_until_empty(g)
    assert len(g.players[me].hand) == 1 and g.might(unit) == 2


def test_zaun_warrens_discards_then_draws_on_a_conquer():
    g, me, opp = setup("annie")
    use_battlefield(g, g.battlefields[0], ZAUN)
    poro = ready_unit(g, me, PORO)
    to_hand(g, me, GUST)
    g.step(next(a for a in moves(g, 0) if a.units == (poro.oid,)))
    pass_until_empty(g)
    assert GUST in ids(g.players[me].trash) and len(g.players[me].hand) == 1


def test_obelisk_of_power_channels_a_rune_on_each_first_beginning_phase():
    g = Game([DECKS["master_yi"], DECKS["master_yi"]], seed=0)
    obelisks = sum(bf.card.card.card_id == OBELISK for bf in g.battlefields)
    keep_hands(g)
    first = g.turn_player
    assert len(g.runes(first)) == 2 + obelisks


# --- tools ---------------------------------------------------------------------

def test_the_sim_plays_the_chosen_decks():
    import sim
    session = sim.Session("random", human_seat=1, decks=("annie", "master_yi"))
    assert session.game.decks[1].legend.card_id == ANNIE_LEGEND
    assert session.game.decks[0].champion.card_id == YI


def test_matchup_intervals():
    from matchups import wilson
    lo, hi = wilson(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60
    assert wilson(0, 0) == (0.0, 1.0)
