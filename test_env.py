"""Tests for the RL environment (env.py), the action cache and GreedyAgent."""

import copy
import json
import random

import numpy as np
import pytest

from agents import GreedyAgent
from env import (ACTION_DIM, GLOBAL_DIM, MAX_TOKENS, N_POINTERS, TOKEN_DIM, CardVocab, Encoder,
                 OpponentEnv, RiftboundEnv, UNKNOWN)
from game import ActionKind, Game, RandomAgent, load_demo_decks, play_game


def random_positions(n_games=5, every=7):
    """Games mid-play, sampled every few decisions."""
    decks = list(load_demo_decks())
    for seed in range(n_games):
        g = Game(decks, seed=seed)
        rng = random.Random(seed)
        k = 0
        while not g.is_over:
            if k % every == 0:
                yield g
            g.step(rng.choice(g.legal_actions()))
            k += 1


def test_cached_actions_play_the_same_games():
    decks = list(load_demo_decks())
    for seed in range(10):
        games = [Game(decks, seed=seed), Game(decks, seed=seed, cache_actions=True)]
        winners = [play_game(g, [RandomAgent(seed), RandomAgent(seed + 1)]) for g in games]
        assert winners[0] == winners[1]
        assert games[0].log == games[1].log


def test_cached_actions_are_copies_and_still_reject_illegal_actions():
    g = Game(list(load_demo_decks()), seed=0, cache_actions=True)
    legal = g.legal_actions()
    legal.clear()
    assert g.legal_actions()
    while not any(a.kind is ActionKind.PLAY_CARD for a in g.legal_actions()):
        g.step(g.legal_actions()[-1])
    played = next(a for a in g.legal_actions() if a.kind is ActionKind.PLAY_CARD)
    g.step(played)
    assert played not in g.legal_actions()
    with pytest.raises(ValueError):
        g.step(played)         # the card has left the hand


def test_observation_has_what_the_encoder_needs_and_stays_json():
    for g in random_positions(3):
        obs = g.observation(g.acting_player)
        json.dumps(obs)
        assert obs["cards_played"] == g.cards_played
        for entry in obs["chain"]:
            if "ability" in entry:
                assert entry["card_id"]
        if obs["decision"] is not None:
            assert len(obs["decision"]["options"]) == len(g.decision.options)


def test_encoding_shapes_and_pointers():
    enc = Encoder(CardVocab.from_pool([c.card_id for c in load_demo_decks()[0].main]))
    seen_kinds = set()
    for g in random_positions(5, every=3):
        seat = g.acting_player
        legal = g.legal_actions()
        e = enc.encode(g.observation(seat), legal)
        assert e.tokens.shape == (MAX_TOKENS, TOKEN_DIM) and e.tokens.dtype == np.float32
        assert e.glob.shape == (GLOBAL_DIM,)
        assert e.actions.shape == (len(legal), ACTION_DIM)
        assert e.pointers.shape == (len(legal), N_POINTERS)
        assert np.isfinite(e.tokens).all() and np.isfinite(e.glob).all() and np.isfinite(e.actions).all()
        n = int(e.mask.sum())
        assert e.mask[:n].all() and not e.mask[n:].any()          # real tokens first
        assert ((e.pointers == -1) | ((e.pointers >= 0) & (e.pointers < n))).all()
        for a, p in zip(legal, e.pointers):
            seen_kinds.add(a.kind)
            if a.kind is ActionKind.PLAY_CARD:
                assert p[0] >= 0                                    # the card being played
            if a.kind is ActionKind.MOVE:
                assert (p[:min(len(a.units), N_POINTERS)] >= 0).all()
        rows = {r.tobytes() + p.tobytes() for r, p in zip(e.actions, e.pointers)}
        assert len(rows) == len(legal), "two legal actions encode the same"
    assert {ActionKind.PLAY_CARD, ActionKind.MOVE, ActionKind.END_TURN, ActionKind.MULLIGAN} <= seen_kinds


def test_encoding_never_depends_on_hidden_information():
    """Swapping the opponent's hand with their deck, and reshuffling both decks,
    must not change what the viewer's encoding contains."""
    enc = Encoder(CardVocab.from_pool([c.card_id for c in load_demo_decks()[0].main]))
    checked = 0
    for g in random_positions(5, every=5):
        viewer = g.acting_player
        other = g.players[1 - viewer]
        if not other.hand.objects or not other.main_deck.objects:
            continue
        before = enc.encode(g.observation(viewer), g.legal_actions())
        h = copy.deepcopy(g)
        o = h.players[1 - viewer]
        o.hand.objects, o.main_deck.objects = o.main_deck.objects[:len(o.hand)], \
            o.hand.objects + o.main_deck.objects[len(o.hand):]
        for obj in [*o.hand.objects, *o.main_deck.objects]:
            obj.zone = o.hand if obj in o.hand.objects else o.main_deck
        rng = random.Random(checked)
        rng.shuffle(o.main_deck.objects)
        rng.shuffle(h.players[viewer].main_deck.objects)
        rng.shuffle(o.rune_deck.objects)
        after = enc.encode(h.observation(viewer), h.legal_actions())
        for field in ("cards", "tokens", "mask", "glob", "actions", "pointers"):
            assert np.array_equal(getattr(before, field), getattr(after, field)), field
        checked += 1
    assert checked >= 10


def test_vocab_keeps_indices_when_extended():
    vocab = CardVocab(["OGN-001", "OGN-009"])
    bigger = vocab.extended(["OGN-005", "OGN-009"])
    assert bigger["OGN-001"] == vocab["OGN-001"] and bigger["OGN-009"] == vocab["OGN-009"]
    assert bigger["OGN-005"] == len(bigger) - 1
    assert vocab["nope"] == UNKNOWN


def test_env_plays_to_the_end_with_opposite_rewards():
    env = RiftboundEnv(seed=3)
    rng = random.Random(3)
    while not env.done:
        assert env.legal == env.game.legal_actions()
        env.step(rng.randrange(len(env.legal)))
    assert env.winner in (0, 1)
    assert env.reward(0) == -env.reward(1) != 0


def test_env_cuts_long_games_off_as_draws():
    env = RiftboundEnv(seed=0, max_decisions=10)
    while not env.done:
        env.step(0)
    assert env.truncated and env.decisions == 10
    assert env.reward(0) == env.reward(1) == 0


def test_env_reset_is_reproducible():
    env = RiftboundEnv()
    runs = []
    for _ in range(2):
        env.reset(7)
        trace = []
        while not env.done:
            trace.append(env.encode().glob.tobytes())
            env.step(len(env.legal) - 1)
        runs.append(trace)
    assert runs[0] == runs[1]


def test_opponent_env_only_hands_over_the_learners_decisions():
    env = OpponentEnv(RandomAgent(0), seat=1)
    enc, info = env.reset(0)
    steps, reward, done = 0, 0.0, False
    while not done:
        assert env.env.acting_player == 1
        assert len(enc.actions) == len(info["legal"])
        enc, reward, terminated, truncated, info = env.step(0)
        done = terminated or truncated
        steps += 1
    assert enc is None and reward in (-1.0, 1.0) and steps > 5


def test_greedy_agent_beats_random():
    decks = list(load_demo_decks())
    wins = 0
    for seed in range(20):
        seat = seed % 2
        agents = [RandomAgent(seed), RandomAgent(seed)]
        agents[seat] = GreedyAgent(seed)
        wins += play_game(Game(decks, seed=seed, cache_actions=True), agents) == seat
    assert wins >= 16


def test_training_env_never_offers_concede():
    env = RiftboundEnv()
    for seed in range(10):
        env.reset(seed)
        rng = random.Random(seed)
        while not env.done:
            assert all(a.kind is not ActionKind.CONCEDE for a in env.legal)
            env.step(rng.randrange(len(env.legal)))


def test_sim_offers_concede_to_the_human_but_not_the_agent():
    import sim

    class Recorder:
        name = "recorder"

        def __init__(self):
            self.saw_concede = False
            self.rng = random.Random(0)

        def act(self, observation, legal):
            self.saw_concede |= any(a.kind is ActionKind.CONCEDE for a in legal)
            return self.rng.choice(legal)

    session = sim.Session("random", human_seat=0)
    recorder = session.agent = Recorder()
    rng = random.Random(1)
    while not session.game.is_over:
        legal = session.game.legal_actions(0)
        assert legal[-1].kind is ActionKind.CONCEDE
        session.act(rng.randrange(len(legal) - 1))          # anything but Concede
    assert not recorder.saw_concede
