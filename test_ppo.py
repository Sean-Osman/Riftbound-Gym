"""Tests for the PPO model and trainer (ppo.py). Skipped without torch."""

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import ppo  # noqa: E402
from env import RiftboundEnv  # noqa: E402
from game import ActionKind, Game, RandomAgent, load_demo_decks  # noqa: E402


def encodings(n=40):
    env = RiftboundEnv(seed=1)
    rng = random.Random(1)
    out = []
    while not env.done and len(out) < n:
        out.append(env.encode())
        env.step(rng.randrange(len(env.legal)))
    return env, out


def test_logits_cover_exactly_the_legal_actions():
    env, encs = encodings()
    model = ppo.build_model(ppo.Config(d_model=32, layers=1, heads=2), env.encoder.vocab)
    logits, value = model(ppo.Batch(encs))
    assert value.shape == (len(encs),)
    for row, e in zip(logits, encs):
        k = len(e.actions)
        assert torch.isfinite(row[:k]).all()
        assert (row[k:] <= -1e8).all()
        assert torch.softmax(row, -1)[k:].sum() < 1e-6


def test_batching_does_not_change_the_outputs():
    env, encs = encodings(10)
    model = ppo.build_model(ppo.Config(d_model=32, layers=1, heads=2), env.encoder.vocab).eval()
    with torch.no_grad():
        together, v = model(ppo.Batch(encs))
        for i, e in enumerate(encs):
            alone, v1 = model(ppo.Batch([e]))
            k = len(e.actions)
            assert torch.allclose(alone[0, :k], together[i, :k], atol=1e-5)
            assert torch.allclose(v1[0], v[i], atol=1e-5)


def test_gae_with_only_a_final_reward():
    traj = [ppo.Sample(None, 0, 0.0, value) for value in (0.2, -0.1, 0.5)]
    ppo._gae(traj, 1.0, gamma=1.0, lam=1.0)
    assert [round(s.ret, 6) for s in traj] == [1.0, 1.0, 1.0]           # lambda=1: Monte Carlo
    assert round(traj[-1].advantage, 6) == 0.5
    ppo._gae(traj, -1.0, gamma=1.0, lam=0.0)
    assert round(traj[0].advantage, 6) == round(-0.1 - 0.2, 6)          # lambda=0: one-step TD
    assert round(traj[-1].advantage, 6) == -1.5


def test_point_shaping_credits_points_at_once_and_cancels_over_a_game():
    """Potentials 0 -> 0.2 -> 0.5 (the lead grew twice), then a win. Monte Carlo
    returns are the result minus the potential where each decision was made, so the
    step after scoring sees the gain right away, and from the first decision the
    shaping adds nothing: winning is still all that counts."""
    traj = [ppo.Sample(None, 0, 0.0, 0.0, potential=p) for p in (0.0, 0.2, 0.5)]
    ppo._gae(traj, 1.0, gamma=1.0, lam=1.0)
    assert [round(s.ret, 6) for s in traj] == [1.0, 0.8, 0.5]
    ppo._gae(traj, 1.0, gamma=1.0, lam=0.0)                              # one-step rewards
    assert [round(s.advantage, 6) for s in traj] == [0.2, 0.3, 0.5]      # +0.2, +0.3, then 1 - 0.5
    flat = [ppo.Sample(None, 0, 0.0, 0.0) for _ in range(3)]
    ppo._gae(flat, 1.0, gamma=1.0, lam=1.0)
    assert round(flat[0].ret, 6) == 1.0                                  # the same as with shaping above


def test_point_potential_is_the_lead_over_the_victory_score():
    env = RiftboundEnv(seed=0)
    env.game.points = [5, 1]
    assert ppo.point_potential(env, 0, 0.5) == 0.5 * 4 / 8
    assert ppo.point_potential(env, 1, 0.5) == -0.5 * 4 / 8
    assert ppo.point_potential(env, 0, 0.0) == 0.0


def test_training_runs_saves_resumes_and_the_agent_plays(tmp_path, monkeypatch):
    monkeypatch.setattr(ppo, "CHECKPOINTS", tmp_path)
    cfg = ppo.Config(run="t", iterations=2, games=4, workers=0, d_model=32, layers=1, heads=2,
                     minibatch=64, epochs=1, snapshot_every=1, eval_every=2, eval_games=2,
                     greedy_frac=0.25, pool_frac=0.25, device="cpu", point_reward=0.5)
    ppo.Trainer(cfg).train()
    lines = (tmp_path / "t" / "log.jsonl").read_text().splitlines()
    assert len(lines) == 2 and "eval_vs_greedy" in lines[-1]
    assert len(list((tmp_path / "t" / "pool").glob("*.pt"))) == 2
    with pytest.raises(FileExistsError):
        ppo.Trainer(cfg)
    cfg.iterations = 3
    cfg.lr = 1e-4
    trainer = ppo.Trainer(cfg, resume=True)
    assert trainer.iteration == 2
    assert trainer.opt.param_groups[0]["lr"] == 1e-4          # not the saved optimizer's 3e-4
    trainer.train()
    assert len((tmp_path / "t" / "log.jsonl").read_text().splitlines()) == 3

    agent = ppo.PPOAgent(tmp_path / "t" / "latest.pt")
    assert agent.name.endswith("#3")
    g = Game(list(load_demo_decks()), seed=5, allow_concede=True)
    while not g.is_over:                            # offered Concede, but never takes it
        seat = g.acting_player
        player = agent if seat == 0 else RandomAgent(5)
        action = player.act(g.observation(seat), g.legal_actions(seat))
        assert action.kind is not ActionKind.CONCEDE
        g.step(action)


def test_updates_stop_early_once_kl_passes_the_target(tmp_path, monkeypatch):
    monkeypatch.setattr(ppo, "CHECKPOINTS", tmp_path)
    base = dict(games=4, workers=0, d_model=32, layers=1, heads=2, minibatch=32, epochs=4,
                lr=1e-2, device="cpu")
    capped = ppo.Trainer(ppo.Config(run="capped", target_kl=1e-6, **base))
    samples = [s for r in capped._play(capped._specs()) for s in r["samples"]]
    assert capped.update(samples)["update_frac"] < 1
    free = ppo.Trainer(ppo.Config(run="free", target_kl=0.0, **base))
    assert free.update(samples)["update_frac"] == 1


def test_duel_trains_both_sides_saves_resumes_and_evaluates(tmp_path, monkeypatch):
    import duel
    monkeypatch.setattr(duel, "CHECKPOINTS", tmp_path)
    cfg = ppo.Config(run="d", iterations=2, games=8, workers=1, d_model=32, layers=1, heads=2,
                     minibatch=64, epochs=1, snapshot_every=1, eval_every=2, device="cpu")
    duel.DuelTrainer(cfg, ("annie", "master_yi"), init=None, resume=False).train(eval_games=4)
    lines = (tmp_path / "d" / "log.jsonl").read_text().splitlines()
    assert len(lines) == 2 and "eval_a_win_rate" in lines[-1]
    for deck in ("annie", "master_yi"):
        assert len(list((tmp_path / "d" / deck / "pool").glob("*.pt"))) == 2
        assert ppo.PPOAgent(tmp_path / "d" / deck / "latest.pt").name.endswith("#2")
    with pytest.raises(FileExistsError):
        duel.DuelTrainer(cfg, ("annie", "master_yi"), init=None, resume=False)
    cfg.iterations = 3
    trainer = duel.DuelTrainer(cfg, ("annie", "master_yi"), init=None, resume=True)
    assert trainer.iteration == 2
    specs = trainer._specs()                    # with snapshots, some games are against the pool
    assert {s.a_source for s in specs} - {"live"} and {s.b_source for s in specs} - {"live"}
    trainer.train(eval_games=4)
    assert len((tmp_path / "d" / "log.jsonl").read_text().splitlines()) == 3
