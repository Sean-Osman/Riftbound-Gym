"""Cards interacting with each other and with the rules: responses on the chain,
spells during combat, triggers stacking, and effects that change combat outcomes.

Single-card behavior is tested in test_game.py; these tests reuse its helpers.
"""

from game import ActionKind, TurnPhase
from test_game import (
    ARENAS_GREATEST, CLEAVE, DARIUS, FALLING_STAR, HEXTECH_RAY, KAISA, NOXUS, PORO, RAVENBLOOM,
    REAVERS_ROW, RETREAT, SENTRY, SMOKE_SCREEN, STUPEFY, TIME_WARP, VOID_SEEKER, WATCHER,
    attack, champion, clear_hand, kaisa_game, keep_hands, new_game, plays, ready_unit, set_runes,
    to_hand, use_battlefield,
)


def pass_(g):
    g.step(next(a for a in g.legal_actions() if a.kind is ActionKind.PASS))


def resolve_chain(g):
    """Both players pass until the chain is empty (for when someone still holds a Reaction)."""
    while g.chain_items:
        pass_(g)


def resolved(g):
    return [line[:-len(" resolves")] for line in g.log if line.endswith(" resolves")]


def defended_by(g, opp, *card_ids):
    """The opponent controls battlefield 0 with these units."""
    bf = g.battlefields[0]
    units = [ready_unit(g, opp, cid, bf) for cid in card_ids]
    bf.controller = opp
    return bf, units


# --- damage and Might changes stacking -------------------------------------------

def test_shrinking_a_damaged_unit_kills_it():
    g, me, opp = kaisa_game("RRRRRRBB")
    darius = ready_unit(g, opp, DARIUS, g.battlefields[0])
    to_hand(g, me, HEXTECH_RAY)
    to_hand(g, me, SMOKE_SCREEN)
    to_hand(g, me, PORO)
    g.step(plays(g, HEXTECH_RAY)[0])
    resolve_chain(g)                  # I still hold a Reaction, so I pass by hand
    assert darius.damage == 3 and darius.zone is g.battlefields[0].units     # 3 < 5 Might: survives
    g.step(next(a for a in plays(g, SMOKE_SCREEN) if a.targets == (darius.oid,)))
    resolve_chain(g)
    assert darius.zone is g.players[opp].trash                               # now 1 Might with 3 damage


def test_watcher_turns_earlier_damage_lethal():
    g, me, opp = kaisa_game("RRRRRRBBBB")
    darius = ready_unit(g, opp, DARIUS, g.battlefields[0])
    to_hand(g, me, HEXTECH_RAY)
    to_hand(g, me, WATCHER)
    g.step(plays(g, HEXTECH_RAY)[0])
    g.step(plays(g, WATCHER)[0])
    assert darius.zone is g.players[opp].trash                               # 5 - 3 = 2 Might, 3 damage


def test_damage_heals_at_the_end_of_the_turn():
    g, me, opp = kaisa_game()
    darius = ready_unit(g, opp, DARIUS, g.battlefields[0])
    to_hand(g, me, HEXTECH_RAY)
    g.step(plays(g, HEXTECH_RAY)[0])
    g.step(next(a for a in g.legal_actions() if a.kind is ActionKind.END_TURN))
    assert darius.damage == 0


def test_two_shrinks_stop_at_the_minimum():
    g, me, opp = kaisa_game("RRRRBBBB")
    darius = ready_unit(g, opp, DARIUS)
    to_hand(g, me, SMOKE_SCREEN)
    to_hand(g, me, STUPEFY)
    to_hand(g, me, PORO)
    g.step(next(a for a in plays(g, STUPEFY) if a.targets == (darius.oid,)))
    resolve_chain(g)
    g.step(next(a for a in plays(g, SMOKE_SCREEN) if a.targets == (darius.oid,)))
    resolve_chain(g)
    assert g.might(darius) == 1



def test_triggers_resolve_before_the_spell_that_caused_them():
    g, me, opp = kaisa_game("RRRRBBBB")
    student = ready_unit(g, me, RAVENBLOOM)
    target = ready_unit(g, opp, NOXUS)
    to_hand(g, me, SMOKE_SCREEN)
    to_hand(g, me, STUPEFY)
    # Smoke Screen my own Student: its +1 trigger resolves first (2 -> 3),
    # then -4 to a minimum of 1 (3 -> 1)
    g.step(next(a for a in plays(g, SMOKE_SCREEN) if a.targets == (student.oid,)))
    resolve_chain(g)
    assert resolved(g)[-2:] == ["Ravenbloom Student: +1 Might this turn", "Smoke Screen"]
    assert g.might(student) == 1
    # a later spell's +1 applies on top of the shrink
    g.step(next(a for a in plays(g, STUPEFY) if a.targets == (target.oid,)))
    assert g.might(student) == 2


def test_falling_star_kills_a_student_that_its_own_trigger_grew():
    g, me, opp = kaisa_game("RRRRRR")
    student = ready_unit(g, me, RAVENBLOOM)
    other = ready_unit(g, opp, NOXUS)
    to_hand(g, me, FALLING_STAR)
    to_hand(g, me, PORO)
    g.step(next(a for a in plays(g, FALLING_STAR) if sorted(a.targets) == sorted((student.oid, other.oid))))
    # Student grew to 3 before Falling Star resolved, and 3 damage is still lethal
    assert student.zone is g.players[me].trash
    assert other.damage == 3 and other.zone is g.players[opp].base


# --- responses on the chain ------------------------------------------------------

def test_both_players_stack_reactions_and_they_resolve_newest_first():
    g, me, opp = kaisa_game("RRRRBBBB", their_runes="BBBB")
    mine = ready_unit(g, me, NOXUS, g.battlefields[0])
    theirs = ready_unit(g, opp, DARIUS, g.battlefields[1])
    to_hand(g, me, HEXTECH_RAY)
    to_hand(g, me, STUPEFY)
    to_hand(g, opp, SMOKE_SCREEN)
    g.step(next(a for a in plays(g, HEXTECH_RAY) if a.targets == (theirs.oid,)))
    assert g.acting_player == me                     # I hold Stupefy, so I get the first chance
    pass_(g)
    assert g.acting_player == opp
    g.step(next(a for a in plays(g, SMOKE_SCREEN) if a.targets == (mine.oid,)))
    assert g.acting_player == me                     # I may respond to their response
    g.step(next(a for a in plays(g, STUPEFY) if a.targets == (theirs.oid,)))
    assert resolved(g)[-3:] == ["Stupefy", "Smoke Screen", "Hextech Ray"]
    assert g.might(mine) == 1 and g.might(theirs) == 4 and theirs.damage == 3


def test_retreat_saves_a_unit_from_a_trigger():
    g, me, opp = kaisa_game("RRRRRRBBBB", their_runes="BB")
    darius = ready_unit(g, opp, DARIUS, g.battlefields[0])
    to_hand(g, me, HEXTECH_RAY)
    to_hand(g, me, WATCHER)
    to_hand(g, opp, RETREAT)
    g.step(plays(g, HEXTECH_RAY)[0])
    assert g.acting_player == opp
    pass_(g)                                             # let the Ray resolve: 3 damage on Darius
    assert darius.damage == 3
    g.step(plays(g, WATCHER)[0])
    # the Watcher's -3 trigger would kill Darius; the opponent returns him in response
    assert g.acting_player == opp and g.chain_items
    g.step(plays(g, RETREAT)[0])
    assert darius.zone is g.players[opp].hand
    assert "Thousand-Tailed Watcher: enemy units get -3 Might this turn" in resolved(g)


def test_darius_counts_reactions_played_on_the_opponents_turn():
    g, me, opp = kaisa_game("RRRRBBBB", their_runes="BBBBBB")
    darius = ready_unit(g, opp, DARIUS)
    darius.exhausted = True
    mine = ready_unit(g, me, NOXUS)
    ready_unit(g, me, PORO, g.battlefields[0])
    to_hand(g, me, HEXTECH_RAY)
    to_hand(g, me, PORO)
    to_hand(g, opp, STUPEFY)
    to_hand(g, opp, STUPEFY)
    g.step(plays(g, HEXTECH_RAY)[0])
    g.step(next(a for a in plays(g, STUPEFY) if a.targets == (mine.oid,)))
    assert g.might(darius) == 5 and darius.exhausted
    g.step(next(a for a in plays(g, STUPEFY) if a.targets == (mine.oid,)))
    # their second card this turn (even though it's my turn): Darius grows and readies
    assert g.might(darius) == 7 and not darius.exhausted


def test_kaisas_legend_pays_power_for_a_reaction_on_the_opponents_turn():
    g, me, opp = kaisa_game("RRRR", their_runes="RR")
    ready_unit(g, opp, NOXUS, g.battlefields[0])
    darius = ready_unit(g, me, DARIUS)
    to_hand(g, me, HEXTECH_RAY)
    to_hand(g, me, PORO)
    to_hand(g, opp, SMOKE_SCREEN)      # [2][B], and they only have Fury runes
    g.step(plays(g, HEXTECH_RAY)[0])
    assert g.acting_player == opp
    [smoke] = [a for a in plays(g, SMOKE_SCREEN) if a.targets == (darius.oid,)]
    assert smoke.payment.legend
    g.step(smoke)
    assert g.might(darius) == 1 and g.players[opp].legend.objects[0].exhausted


# --- spells during combat --------------------------------------------------------

def test_the_defender_shrinks_the_attacker_during_the_combat_showdown():
    g, me, opp = kaisa_game(their_runes="BBB")
    bf, [noxus] = defended_by(g, opp, NOXUS)
    darius = ready_unit(g, me, DARIUS)
    to_hand(g, me, PORO)
    to_hand(g, opp, SMOKE_SCREEN)
    attack(g, [darius])
    # I had nothing to play, so focus passed to the defender
    assert g.attacker == me and g.focus == opp and g.acting_player == opp
    g.step(next(a for a in plays(g, SMOKE_SCREEN) if a.targets == (darius.oid,)))
    # 1-Might Darius loses to the 4-Might defender
    assert darius.zone is g.players[me].trash and noxus.zone is bf.units
    assert bf.controller == opp and g.showdown is None


def test_retreating_the_only_attacker_ends_the_combat_without_damage():
    g, me, opp = kaisa_game()
    bf, [noxus] = defended_by(g, opp, NOXUS)
    poro = ready_unit(g, me, PORO)
    to_hand(g, me, RETREAT)
    to_hand(g, me, NOXUS)
    attack(g, [poro])
    assert g.focus == me
    g.step(next(a for a in plays(g, RETREAT) if a.targets == (poro.oid,)))
    assert poro.zone is g.players[me].hand
    assert noxus.zone is bf.units and noxus.damage == 0 and bf.controller == opp
    assert g.showdown is None and g.points == [0, 0]


def test_hextech_ray_during_combat_removes_a_defender_before_damage():
    g, me, opp = kaisa_game()
    bf, [poro, noxus] = defended_by(g, opp, PORO, NOXUS)
    darius = ready_unit(g, me, DARIUS)
    to_hand(g, me, HEXTECH_RAY)
    attack(g, [darius])
    # Hextech Ray on Pouty Poro costs an extra [A] because of Deflect
    g.step(next(a for a in plays(g, HEXTECH_RAY) if a.targets == (poro.oid,)))
    # Darius (5) then kills Noxus (4) and survives its 4 damage
    assert poro.zone is g.players[opp].trash and noxus.zone is g.players[opp].trash
    assert darius.zone is bf.units and bf.controller == me and g.points[me] == 1


def test_cleave_on_a_defender_does_nothing():
    g, me, opp = kaisa_game(their_runes="RR")
    bf, [noxus] = defended_by(g, opp, NOXUS)
    poro = ready_unit(g, me, PORO)
    to_hand(g, me, PORO)
    to_hand(g, opp, CLEAVE)
    attack(g, [poro])
    g.step(next(a for a in plays(g, CLEAVE) if a.targets == (noxus.oid,)))
    assert noxus.granted == {"Assault": 3} and g.might(noxus) == 4        # not an attacker
    assert poro.zone is g.players[me].trash and noxus.zone is bf.units


def test_assault_makes_the_attacker_both_hit_harder_and_survive():
    g, me, opp = kaisa_game()
    bf, [darius] = defended_by(g, opp, DARIUS)
    noxus = ready_unit(g, me, NOXUS)
    to_hand(g, me, CLEAVE)
    to_hand(g, me, PORO)
    attack(g, [noxus])
    g.step(next(a for a in plays(g, CLEAVE) if a.targets == (noxus.oid,)))
    # Might is both damage dealt and toughness: 4 + Assault 3 = 7 kills Darius (5),
    # and Darius's 5 damage isn't lethal to a 7-Might attacker (142.4.b)
    assert darius.zone is g.players[opp].trash
    assert noxus.zone is bf.units and noxus.damage == 0 and bf.controller == me
    assert g.might(noxus) == 4                   # no longer attacking once combat ends


def test_watcher_shrink_decides_a_later_combat():
    g, me, opp = kaisa_game("RRRRRRBBB")
    bf, [darius] = defended_by(g, opp, DARIUS)
    poro = ready_unit(g, me, PORO)
    noxus = ready_unit(g, me, NOXUS)
    to_hand(g, me, WATCHER)
    g.step(plays(g, WATCHER)[0])
    assert g.might(darius) == 2
    attack(g, [poro])
    # 2 vs 2: both die
    assert darius.zone is g.players[opp].trash and poro.zone is g.players[me].trash
    assert noxus.zone is g.players[me].base


def test_deathknell_and_conquer_triggers_after_a_combat():
    g, me, opp = kaisa_game()
    bf, [sentry] = defended_by(g, opp, SENTRY)
    kaisa = champion(g, me)
    g.move(kaisa, g.players[me].base)
    kaisa.exhausted = False
    to_hand(g, me, PORO)
    their_hand, my_hand = len(g.players[opp].hand), len(g.players[me].hand)
    attack(g, [kaisa])
    assert sentry.zone is g.players[opp].trash and bf.controller == me
    assert len(g.players[opp].hand) == their_hand + 1                      # Deathknell
    assert len(g.players[me].hand) == my_hand + 1                          # Kai'Sa's conquer draw


def test_the_defender_chooses_which_attacker_dies():
    g, me, opp = kaisa_game()
    bf, [darius] = defended_by(g, opp, DARIUS)
    noxus = ready_unit(g, me, NOXUS)
    poro = ready_unit(g, me, PORO)
    to_hand(g, me, PORO)
    attack(g, [noxus, poro])
    # my 6 kills Darius; Darius's 5 can kill either Noxus (4) or Poro (2) but not both
    assert g.acting_player == opp
    labels = sorted(a.label for a in g.legal_actions())
    assert labels == ["Kill Noxus Hopeful (5 damage)", "Kill Pouty Poro (5 damage)"]
    g.step(next(a for a in g.legal_actions() if a.label.startswith("Kill Noxus")))
    assert noxus.zone is g.players[me].trash and poro.zone is bf.units and bf.controller == me


def test_accelerated_kaisa_attacks_the_turn_she_is_played():
    g, me, opp = kaisa_game("RRRRBB")
    bf, [poro] = defended_by(g, opp, PORO)
    to_hand(g, me, PORO)
    g.step(next(a for a in plays(g, KAISA) if a.accelerate))
    kaisa = next(o for o in g.players[me].base if o.card.card_id == KAISA)
    hand = len(g.players[me].hand)
    attack(g, [kaisa])
    assert poro.zone is g.players[opp].trash and bf.controller == me
    assert len(g.players[me].hand) == hand + 1


def test_reavers_row_retreat_hands_the_battlefield_to_the_attacker():
    g, me, opp = kaisa_game()
    bf = g.battlefields[0]
    use_battlefield(g, bf, REAVERS_ROW)
    noxus = ready_unit(g, opp, NOXUS, bf)
    bf.controller = opp
    kaisa = champion(g, me)
    g.move(kaisa, g.players[me].base)
    kaisa.exhausted = False
    to_hand(g, me, PORO)
    hand = len(g.players[me].hand)
    attack(g, [kaisa])
    g.step(next(a for a in g.legal_actions() if a.label.startswith("Move Noxus Hopeful")))
    assert noxus.zone is g.players[opp].base and noxus.damage == 0
    assert bf.controller == me and g.points[me] == 1
    assert len(g.players[me].hand) == hand + 1                             # Kai'Sa's conquer draw


# --- costs ---------------------------------------------------------------------

def test_deflect_is_paid_for_each_time_a_unit_is_chosen():
    g, me, opp = kaisa_game("RRRRRR")
    poro = ready_unit(g, opp, PORO)
    to_hand(g, me, FALLING_STAR)
    card = next(o.card for o in g.players[me].hand if o.card.card_id == FALLING_STAR)
    twice = g.total_cost(me, card, deflect=g._deflect(me, [poro, poro]))
    assert twice.energy == 2 and len(twice.power) == 4           # [2][R][R] + [A][A]
    # six Fury runes: exhaust 2 and recycle 4 of them
    assert any(a.targets == (poro.oid, poro.oid) for a in plays(g, FALLING_STAR))


def test_my_own_deflect_unit_costs_nothing_extra_to_target():
    g, me, opp = kaisa_game("R")
    poro = ready_unit(g, me, PORO)
    to_hand(g, me, STUPEFY)
    assert g._deflect(me, [poro]) == 0
    assert any(a.targets == (poro.oid,) for a in plays(g, STUPEFY))


def test_legion_counts_spells_played_earlier_in_the_turn():
    g, me, opp = kaisa_game("RRR")
    target = ready_unit(g, opp, NOXUS)
    to_hand(g, me, STUPEFY)
    to_hand(g, me, NOXUS)
    assert plays(g, NOXUS) == []                                  # [4] with 3 runes
    g.step(next(a for a in plays(g, STUPEFY) if a.targets == (target.oid,)))
    [noxus] = plays(g, NOXUS)
    assert len(noxus.payment.exhaust) == 2                        # now [2]


# --- turn structure --------------------------------------------------------------

def test_an_extra_turn_is_not_a_first_turn():
    """Time Warp's extra turn doesn't retrigger The Arena's Greatest."""
    for seed in range(50):
        g = new_game(seed)
        if any(bf.card.card.card_id == ARENAS_GREATEST for bf in g.battlefields):
            break
    keep_hands(g)
    me = g.acting_player
    points = g.points[me]
    clear_hand(g, me)
    set_runes(g, me, "RRRRRRBBBBBB")
    to_hand(g, me, TIME_WARP)
    to_hand(g, me, PORO)
    g.step(plays(g, TIME_WARP)[0])
    g.step(next(a for a in g.legal_actions() if a.kind is ActionKind.END_TURN))
    while g.turn_phase is not TurnPhase.MAIN:
        g.step(g.legal_actions()[0])
    assert g.turn_player == me and g.points[me] == points
    assert sum(line.startswith("Triggers: The Arena's Greatest") for line in g.log) == \
        sum(bf.card.card.card_id == ARENAS_GREATEST for bf in g.battlefields)


def test_banished_time_warp_stays_out_of_a_burn_out():
    g, me, opp = kaisa_game("RRRRRRBBBBBB")
    to_hand(g, me, TIME_WARP)
    to_hand(g, me, PORO)
    g.step(plays(g, TIME_WARP)[0])
    zones = g.players[me]
    for obj in list(zones.main_deck):
        g.move(obj, zones.hand)
    g.draw(me)
    assert all(o.card.card_id != TIME_WARP for o in zones.main_deck)
    assert any(o.card.card_id == TIME_WARP for o in zones.banishment)


def test_the_final_point_from_a_conquer_needs_both_battlefields():
    g, me, opp = kaisa_game()
    g.points[me] = 7
    kaisa = champion(g, me)
    g.move(kaisa, g.players[me].base)
    kaisa.exhausted = False
    to_hand(g, me, PORO)
    hand = len(g.players[me].hand)
    attack(g, [kaisa])
    # conquering one battlefield at 7 points draws a card instead of winning,
    # and Kai'Sa's own conquer trigger draws another
    assert not g.is_over and g.points[me] == 7
    assert len(g.players[me].hand) == hand + 2


def test_stupefy_still_draws_when_its_target_is_saved():
    g, me, opp = kaisa_game(their_runes="BB")
    theirs = ready_unit(g, opp, NOXUS, g.battlefields[0])
    to_hand(g, me, STUPEFY)
    to_hand(g, me, PORO)
    to_hand(g, opp, RETREAT)
    hand = len(g.players[me].hand)
    g.step(next(a for a in plays(g, STUPEFY) if a.targets == (theirs.oid,)))
    g.step(plays(g, RETREAT)[0])
    # the target left, so the -1 does nothing (359.3.e), but "Draw 1" still happens (055)
    assert theirs.zone is g.players[opp].hand
    assert len(g.players[me].hand) == hand - 1 + 1


def test_returning_watchful_sentry_to_hand_is_not_a_death():
    g, me, opp = kaisa_game("BB")
    sentry = ready_unit(g, me, SENTRY, g.battlefields[0])
    to_hand(g, me, RETREAT)
    to_hand(g, me, PORO)
    hand = len(g.players[me].hand)
    g.step(plays(g, RETREAT)[0])
    assert sentry.zone is g.players[me].hand
    assert len(g.players[me].hand) == hand - 1 + 1               # the Sentry itself, no Deathknell draw
    assert not any("Deathknell" in line for line in g.log)
