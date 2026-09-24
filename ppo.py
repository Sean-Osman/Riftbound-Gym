"""PPO self-play on the Kai'Sa mirror.

    python3 ppo.py --run kaisa-v1                   # train (Ctrl-C is safe: latest.pt is saved each iteration)
    python3 ppo.py --run kaisa-v1 --resume          # continue a run
    python3 ppo.py --run smoke --iterations 2 --games 8 --workers 2 --eval-games 8   # quick check
    python3 sim.py --agent ppo:PPOAgent             # play the newest checkpoint in the browser

The policy scores every legal action (see env.py), so the action space can be any
size. Each iteration, rollout workers play `--games` games and send back the
learner's decisions with advantages already computed; the learner then does a
few epochs of clipped PPO on them.

Opponents (per game): the current policy against itself (both seats are
training data), a saved past version from the pool (only the learner's seat is
training data), or GreedyAgent. Playing old versions keeps the policy from
forgetting how to beat strategies it has moved away from.

Everything for a run lives in checkpoints/<run>/: latest.pt, pool/*.pt snapshots
and log.jsonl (one line per iteration).
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents import GreedyAgent
from env import (ACTION_DIM, GLOBAL_DIM, N_POINTERS, TOKEN_DIM, CardVocab, Encoded, Encoder,
                 RiftboundEnv)
from game import ROOT, Action, RandomAgent

CHECKPOINTS = ROOT / "checkpoints"


@dataclass
class Config:
    run: str = "kaisa-mirror"
    seed: int = 0
    iterations: int = 1000
    games: int = 64                 # games per iteration
    workers: int = max(1, (os.cpu_count() or 2) - 1)   # 0 plays in this process
    # opponent mix; the rest of the games are self-play
    pool_frac: float = 0.2
    greedy_frac: float = 0.1
    snapshot_every: int = 10
    pool_size: int = 20
    # model
    d_model: int = 128
    layers: int = 2
    heads: int = 4
    # PPO
    lr: float = 3e-4
    epochs: int = 4
    minibatch: int = 512
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    gamma: float = 1.0              # episodes are short and only the result counts
    lam: float = 0.95
    max_decisions: int = 500
    # evaluation against the scripted baselines
    eval_every: int = 10
    eval_games: int = 100
    device: str = "auto"            # for the learner; rollouts always run on CPU


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Batch:
    """Encoded decisions as tensors. Tokens are padded per batch; actions are packed
    into one flat list (decisions have ~8 actions on average but up to ~170, so
    padding them would waste most of the work) and only the logits get padded."""

    def __init__(self, items: Sequence[Encoded], device: str | torch.device = "cpu"):
        n_tokens = max(1, max(int(e.mask.sum()) for e in items))     # real tokens come first
        counts = [len(e.actions) for e in items]
        t = lambda a: torch.from_numpy(a).to(device)
        self.cards = t(np.stack([e.cards[:n_tokens] for e in items]))
        self.tokens = t(np.stack([e.tokens[:n_tokens] for e in items]))
        self.mask = t(np.stack([e.mask[:n_tokens] for e in items]))
        self.glob = t(np.stack([e.glob for e in items]))
        self.actions = t(np.concatenate([e.actions for e in items]))
        self.pointers = t(np.concatenate([e.pointers for e in items]))
        self.action_batch = t(np.repeat(np.arange(len(items)), counts))
        self.action_slot = t(np.concatenate([np.arange(n) for n in counts]))
        self.action_mask = t(np.arange(max(counts))[None, :] < np.array(counts)[:, None])


class PolicyNet(nn.Module):
    """A small transformer over the card tokens plus a global token. Each legal
    action is embedded from its features and the tokens it points at, then scored
    against the state. The value head reads the global token."""

    def __init__(self, vocab_size: int, d_model: int = 128, layers: int = 2, heads: int = 4):
        super().__init__()
        d = d_model
        self.card_emb = nn.Embedding(vocab_size, d, padding_idx=0)
        self.token_proj = nn.Linear(TOKEN_DIM, d)
        self.glob_proj = nn.Sequential(nn.Linear(GLOBAL_DIM, d), nn.ReLU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.action_proj = nn.Sequential(nn.Linear(ACTION_DIM + N_POINTERS * d, 2 * d), nn.ReLU(),
                                         nn.Linear(2 * d, d))
        self.score = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, 1))
        self.value = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, b: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits [B, max actions] with padding at -1e9, value [B])."""
        x = self.card_emb(b.cards) + self.token_proj(b.tokens)
        g = self.glob_proj(b.glob).unsqueeze(1)
        keep = torch.cat([torch.ones_like(b.mask[:, :1]), b.mask], dim=1)
        h = self.encoder(torch.cat([g, x], dim=1), src_key_padding_mask=~keep)
        state, toks = h[:, 0], h[:, 1:]
        pointed = toks[b.action_batch[:, None], b.pointers.clamp(min=0)]          # [N, P, d]
        pointed = pointed * (b.pointers >= 0).unsqueeze(-1)
        a = self.action_proj(torch.cat([b.actions, pointed.flatten(1)], dim=-1))
        s = state[b.action_batch]
        scores = self.score(torch.cat([s, a, s * a], dim=-1)).squeeze(-1)
        logits = scores.new_full(b.action_mask.shape, -1e9)
        logits = logits.index_put((b.action_batch, b.action_slot), scores)
        return logits, self.value(state).squeeze(-1)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")                 # about 2x the CPU on an M2
    return torch.device("cpu")


def build_model(cfg: Config, vocab: CardVocab) -> PolicyNet:
    return PolicyNet(len(vocab), cfg.d_model, cfg.layers, cfg.heads)


def load_checkpoint(path: str | Path) -> tuple[PolicyNet, Config, CardVocab, dict[str, Any]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config(**{k: v for k, v in ckpt["config"].items() if k in {f.name for f in fields(Config)}})
    vocab = CardVocab(ckpt["vocab"])
    model = build_model(cfg, vocab)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, vocab, ckpt


def _pick(logits: torch.Tensor, rng: np.random.Generator, greedy: bool) -> tuple[int, float]:
    logp = F.log_softmax(logits, dim=-1)
    if greedy:
        i = int(torch.argmax(logp))
    else:
        i = int(rng.choice(len(logp), p=logp.exp().double().numpy() / float(logp.exp().double().sum())))
    return i, float(logp[i])


# ---------------------------------------------------------------------------
# Agent (for the sim and matchups)
# ---------------------------------------------------------------------------

def newest_checkpoint() -> Path:
    found = sorted(CHECKPOINTS.glob("*/latest.pt"), key=lambda p: p.stat().st_mtime)
    if not found:
        raise FileNotFoundError(f"no checkpoints/*/latest.pt yet; train with python3 ppo.py first")
    return found[-1]


class PPOAgent:
    """A trained policy as an Agent. Loads `path`, else $RIFTBOUND_CHECKPOINT, else
    the newest checkpoints/*/latest.pt. Picks the most likely action unless `sample`."""

    def __init__(self, path: str | Path | None = None, *, sample: bool = False, seed: int | None = None):
        path = path or os.environ.get("RIFTBOUND_CHECKPOINT") or newest_checkpoint()
        self.model, _, vocab, ckpt = load_checkpoint(path)
        self.encoder = Encoder(vocab)
        self.sample = sample
        self.rng = np.random.default_rng(seed)
        self.name = f"PPO {Path(path).parent.name} #{ckpt.get('iteration', '?')}"

    @torch.no_grad()
    def act(self, observation: dict[str, Any], legal: list[Action]) -> Action:
        logits, _ = self.model(Batch([self.encoder.encode(observation, legal)]))
        i, _ = _pick(logits[0], self.rng, greedy=not self.sample)
        return legal[i]


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    enc: Encoded
    action: int
    logp: float
    value: float
    advantage: float = 0.0
    ret: float = 0.0


@dataclass
class GameSpec:
    seed: int
    opponent: str           # "self", "greedy", "random" or a pool checkpoint path
    learner: int            # the learner's seat (both seats learn in self-play)
    record: bool = True     # False for evaluation games
    greedy: bool = False    # learner picks its most likely action (evaluation)


_worker: dict[str, Any] = {}


def _init_worker(cfg: dict[str, Any], vocab: list[str]) -> None:
    torch.set_num_threads(1)
    c = Config(**cfg)
    v = CardVocab(vocab)
    _worker.update(cfg=c, vocab=v, model=build_model(c, v).eval(), pool={},
                   env=RiftboundEnv(vocab=v, max_decisions=c.max_decisions))


def _opponent_model(path: str) -> PolicyNet:
    pool = _worker["pool"]
    if path not in pool:
        if len(pool) > 64:
            pool.clear()
        pool[path] = load_checkpoint(path)[0]
    return pool[path]


def _run_games(weights: dict[str, np.ndarray], specs: list[GameSpec]) -> list[dict[str, Any]]:
    """Play games in a worker. Returns, per game, the recorded samples (with
    advantages) and the result from the learner's point of view."""
    model: PolicyNet = _worker["model"]
    model.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()})
    cfg: Config = _worker["cfg"]
    env: RiftboundEnv = _worker["env"]
    results = []
    for spec in specs:
        rng = np.random.default_rng(spec.seed)
        env.reset(spec.seed)
        players: list[Any] = [model, model]
        other = 1 - spec.learner
        if spec.opponent == "greedy":
            players[other] = GreedyAgent(spec.seed)
        elif spec.opponent == "random":
            players[other] = RandomAgent(spec.seed)
        elif spec.opponent != "self":
            players[other] = _opponent_model(spec.opponent)
        learning = {spec.learner} if spec.opponent != "self" else {0, 1}
        trajectories: dict[int, list[Sample]] = {0: [], 1: []}
        with torch.no_grad():
            while not env.done:
                seat = env.acting_player
                player = players[seat]
                if not isinstance(player, nn.Module):
                    env.step(env.legal.index(player.act(env.observation(), env.legal)))
                    continue
                enc = env.encode()
                logits, value = player(Batch([enc]))
                greedy = spec.greedy and seat == spec.learner
                i, logp = _pick(logits[0], rng, greedy)
                if spec.record and player is model and seat in learning:
                    trajectories[seat].append(Sample(enc, i, logp, float(value[0])))
                env.step(i)
        samples = []
        for seat in learning:
            traj = trajectories[seat]
            _gae(traj, env.reward(seat), cfg.gamma, cfg.lam)
            samples += traj
        results.append({"opponent": spec.opponent if spec.opponent in ("self", "greedy", "random") else "pool",
                        "reward": env.reward(spec.learner), "decisions": env.decisions,
                        "truncated": env.truncated, "samples": samples})
    return results


def _gae(traj: list[Sample], reward: float, gamma: float, lam: float) -> None:
    """Generalized advantage estimation over one seat's decisions. The only reward
    is the result, after the seat's last decision."""
    advantage, next_value = 0.0, 0.0
    for t in reversed(range(len(traj))):
        r = reward if t == len(traj) - 1 else 0.0
        delta = r + gamma * next_value - traj[t].value
        advantage = delta + gamma * lam * advantage
        traj[t].advantage = advantage
        traj[t].ret = advantage + traj[t].value
        next_value = traj[t].value


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, cfg: Config, *, resume: bool = False):
        self.cfg = cfg
        self.dir = CHECKPOINTS / cfg.run
        self.pool_dir = self.dir / "pool"
        self.iteration = 0
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.rng = random.Random(cfg.seed)
        if resume:
            self.model, _, self.vocab, ckpt = load_checkpoint(self.dir / "latest.pt")
            self.model.train()
            self.iteration = ckpt["iteration"]
            self.rng.setstate(ckpt["rng"])
        else:
            if (self.dir / "latest.pt").exists():
                raise FileExistsError(f"{self.dir} already has a run; pass --resume or pick another --run")
            self.vocab = RiftboundEnv().encoder.vocab
            self.model = build_model(cfg, self.vocab)
        self.device = pick_device(cfg.device)
        self.model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, eps=1e-5)
        if resume and "optimizer" in ckpt:
            self.opt.load_state_dict(ckpt["optimizer"])
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.executor = None
        if cfg.workers > 0:
            ctx = mp.get_context("spawn")
            self.executor = ctx.Pool(cfg.workers, _init_worker, (asdict(cfg), self.vocab.card_ids))
        else:
            _init_worker(asdict(cfg), self.vocab.card_ids)

    def close(self) -> None:
        if self.executor is not None:
            self.executor.terminate()
            self.executor = None

    def _weights(self) -> dict[str, np.ndarray]:
        return {k: v.detach().cpu().numpy() for k, v in self.model.state_dict().items()}

    def _play(self, specs: list[GameSpec]) -> list[dict[str, Any]]:
        weights = self._weights()
        if self.executor is None:
            return _run_games(weights, specs)
        n = max(1, min(len(specs), self.cfg.workers * 2))
        chunks = [specs[i::n] for i in range(n)]
        return [r for part in self.executor.starmap(_run_games, [(weights, c) for c in chunks]) for r in part]

    def _specs(self) -> list[GameSpec]:
        cfg = self.cfg
        pool = sorted(self.pool_dir.glob("*.pt"))[-cfg.pool_size:]
        specs = []
        for k in range(cfg.games):
            seed = (cfg.seed * 1_000_003 + self.iteration * cfg.games + k) % (1 << 31)
            roll = self.rng.random()
            if roll < cfg.greedy_frac:
                opponent = "greedy"
            elif roll < cfg.greedy_frac + cfg.pool_frac and pool:
                opponent = str(self.rng.choice(pool))
            else:
                opponent = "self"
            specs.append(GameSpec(seed, opponent, learner=k % 2))
        return specs

    def evaluate(self, games: int | None = None) -> dict[str, float]:
        """Win rate of the greedy (argmax) policy against the scripted agents,
        alternating seats. Evaluation seeds never overlap training seeds."""
        games = games or self.cfg.eval_games
        specs = [GameSpec(10**9 + k, opp, learner=k % 2, record=False, greedy=True)
                 for opp in ("random", "greedy") for k in range(games)]
        out: dict[str, list[float]] = {}
        for r in self._play(specs):
            out.setdefault(r["opponent"], []).append(r["reward"] > 0)
        return {f"eval_vs_{k}": float(np.mean(v)) for k, v in out.items()}

    def update(self, samples: list[Sample]) -> dict[str, float]:
        cfg = self.cfg
        self.model.train()
        stats: dict[str, list[float]] = {}
        order = np.arange(len(samples))
        np_rng = np.random.default_rng(self.iteration)
        for _ in range(cfg.epochs):
            np_rng.shuffle(order)
            for start in range(0, len(order), cfg.minibatch):
                mb = [samples[i] for i in order[start:start + cfg.minibatch]]
                dev = self.device
                batch = Batch([s.enc for s in mb], dev)
                act = torch.tensor([s.action for s in mb], device=dev)
                old_logp = torch.tensor([s.logp for s in mb], dtype=torch.float32, device=dev)
                adv = torch.tensor([s.advantage for s in mb], dtype=torch.float32, device=dev)
                ret = torch.tensor([s.ret for s in mb], dtype=torch.float32, device=dev)
                adv = (adv - adv.mean()) / (adv.std() + 1e-8) if len(mb) > 1 else adv
                logits, value = self.model(batch)
                logp_all = F.log_softmax(logits, dim=-1)
                logp = logp_all.gather(1, act[:, None]).squeeze(1)
                entropy = -(logp_all.exp() * logp_all).masked_fill(~batch.action_mask, 0).sum(-1).mean()
                ratio = (logp - old_logp).exp()
                pg = -torch.min(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv).mean()
                v_loss = 0.5 * F.mse_loss(value, ret)
                loss = pg + cfg.vf_coef * v_loss - cfg.ent_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    for k, v in (("policy_loss", pg), ("value_loss", v_loss), ("entropy", entropy),
                                 ("approx_kl", ((ratio - 1) - (logp - old_logp)).mean()),
                                 ("clip_frac", ((ratio - 1).abs() > cfg.clip).float().mean())):
                        stats.setdefault(k, []).append(float(v))
        values = np.array([s.value for s in samples])
        returns = np.array([s.ret for s in samples])
        var = returns.var()
        result = {k: float(np.mean(v)) for k, v in stats.items()}
        result["explained_variance"] = float(1 - (returns - values).var() / var) if var > 0 else 0.0
        self.model.eval()
        return result

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        weights = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        torch.save({"model": weights, "optimizer": self.opt.state_dict(),
                    "config": asdict(self.cfg), "vocab": self.vocab.card_ids,
                    "iteration": self.iteration, "rng": self.rng.getstate()}, tmp)
        tmp.replace(path)

    def train(self) -> None:
        cfg = self.cfg
        log = open(self.dir / "log.jsonl", "a")
        (self.dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
        try:
            while self.iteration < cfg.iterations:
                t0 = time.time()
                results = self._play(self._specs())
                samples = [s for r in results for s in r["samples"]]
                t1 = time.time()
                stats = self.update(samples)
                self.iteration += 1
                row: dict[str, Any] = {"iteration": self.iteration, "games": len(results),
                                       "samples": len(samples),
                                       "decisions_per_s": sum(r["decisions"] for r in results) / (t1 - t0),
                                       "rollout_s": t1 - t0, "update_s": time.time() - t1,
                                       "game_length": float(np.mean([r["decisions"] for r in results])),
                                       "truncated": sum(r["truncated"] for r in results), **stats}
                for opp in ("greedy", "pool"):
                    rewards = [r["reward"] for r in results if r["opponent"] == opp]
                    if rewards:
                        row[f"win_vs_{opp}"] = float(np.mean([x > 0 for x in rewards]))
                if self.iteration % cfg.snapshot_every == 0:
                    self.save(self.pool_dir / f"iter_{self.iteration:06d}.pt")
                if self.iteration % cfg.eval_every == 0:
                    row.update(self.evaluate())
                self.save(self.dir / "latest.pt")
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(_format(row), flush=True)
        finally:
            log.close()
            self.close()


def _format(row: dict[str, Any]) -> str:
    keys = ["iteration", "samples", "decisions_per_s", "game_length", "policy_loss", "value_loss",
            "entropy", "approx_kl", "clip_frac", "explained_variance", "win_vs_greedy", "win_vs_pool",
            "eval_vs_random", "eval_vs_greedy"]
    parts = []
    for k in keys:
        if k in row:
            v = row[k]
            parts.append(f"{k}={v:.3f}" if isinstance(v, float) and not math.isnan(v) else f"{k}={v}")
    return " ".join(parts)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(Config):
        flag = "--" + f.name.replace("_", "-")
        parser.add_argument(flag, type=type(f.default), default=f.default)
    parser.add_argument("--resume", action="store_true", help="continue checkpoints/<run>/latest.pt")
    parser.add_argument("--eval-only", action="store_true", help="evaluate the run's latest.pt and exit")
    args = vars(parser.parse_args(argv))
    resume, eval_only = args.pop("resume"), args.pop("eval_only")
    cfg = Config(**args)
    trainer = Trainer(cfg, resume=resume or eval_only)
    if eval_only:
        try:
            print(trainer.evaluate())
        finally:
            trainer.close()
        return
    trainer.train()


if __name__ == "__main__":
    main()
