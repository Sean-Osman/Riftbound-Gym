"""Adversarial training for one matchup: one policy per deck, each trained against the other.

    .venv/bin/python duel.py --run annie-vs-yi --decks annie,master_yi --init checkpoints/meta-v1/latest.pt
    .venv/bin/python duel.py --run annie-vs-yi --resume                  # continue
    .venv/bin/python duel.py --run annie-vs-yi --eval-only --eval-games 400

Side A pilots the first deck and side B the second. Each iteration plays --games
games with the decks alternating seats:

- half: current A against current B; both sides learn from them
- a quarter: current A against a past snapshot of B; only A learns
- a quarter: current B against a past snapshot of A; only B learns

Playing past snapshots keeps a side from forgetting how to beat strategies the
other side has moved away from, so the two don't just chase each other in circles.
Both sides get the same number of games and updates. Every --eval-every
iterations the current policies play --eval-games head-to-head games (sampling
their actions, as in matchups.py) for a win rate with a 95% interval, and each
side also plays GreedyAgent piloting the other deck, to track skill.

--init warm-starts both sides from a ppo.py checkpoint (e.g. the shared meta policy).
Each side is saved as an ordinary ppo.py checkpoint in checkpoints/<run>/<deck>/, so
PPOAgent and the sim can load it; log.jsonl has one line per iteration.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from agents import GreedyAgent
from env import CardVocab, RiftboundEnv
from game import load_decks
from matchups import wilson
from ppo import (CHECKPOINTS, Batch, Config, PolicyNet, Sample, _gae, _pick, build_model, load_checkpoint,
                 pick_device, point_potential, ppo_update)


@dataclass
class DuelSpec:
    seed: int
    a_seat: int                # the seat side A plays; B has the other
    a_source: str              # "live", "greedy" or a snapshot path
    b_source: str
    record: bool = True        # False for evaluation games


_worker: dict[str, Any] = {}


def _init_worker(cfg: dict[str, Any], vocab: list[str], decks: tuple[str, str]) -> None:
    torch.set_num_threads(1)
    c = Config(**cfg)
    v = CardVocab(vocab)
    pool = load_decks(decks)
    _worker.update(cfg=c, live={"a": build_model(c, v).eval(), "b": build_model(c, v).eval()},
                   snapshots={}, decks=(pool[decks[0]], pool[decks[1]]),
                   env=RiftboundEnv(vocab=v, max_decisions=c.max_decisions))


def _snapshot(path: str) -> PolicyNet:
    snapshots = _worker["snapshots"]
    if path not in snapshots:
        if len(snapshots) > 32:
            snapshots.clear()
        snapshots[path] = load_checkpoint(path)[0]
    return snapshots[path]


def _run_games(weights: dict[str, dict[str, np.ndarray]], specs: list[DuelSpec]) -> list[dict[str, Any]]:
    """Play games in a worker. Returns, per game, each live side's samples and who won."""
    for side, w in weights.items():
        _worker["live"][side].load_state_dict({k: torch.from_numpy(v) for k, v in w.items()})
    cfg: Config = _worker["cfg"]
    env: RiftboundEnv = _worker["env"]
    deck_a, deck_b = _worker["decks"]
    results = []
    for spec in specs:
        rng = np.random.default_rng(spec.seed)
        env.reset(spec.seed, [deck_a, deck_b] if spec.a_seat == 0 else [deck_b, deck_a])
        side_of = {spec.a_seat: "a", 1 - spec.a_seat: "b"}
        players: dict[int, Any] = {}
        for seat, side in side_of.items():
            source = spec.a_source if side == "a" else spec.b_source
            players[seat] = (_worker["live"][side] if source == "live" else
                             GreedyAgent(spec.seed) if source == "greedy" else _snapshot(source))
        learning = {seat for seat, side in side_of.items()
                    if spec.record and (spec.a_source if side == "a" else spec.b_source) == "live"}
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
                i, logp = _pick(logits[0], rng, greedy=False)
                if seat in learning:
                    trajectories[seat].append(Sample(enc, i, logp, float(value[0]),
                                                     potential=point_potential(env, seat, cfg.point_reward)))
                env.step(i)
        samples: dict[str, list[Sample]] = {"a": [], "b": []}
        for seat in learning:
            _gae(trajectories[seat], env.reward(seat), cfg.gamma, cfg.lam)
            samples[side_of[seat]] += trajectories[seat]
        winner = None if env.winner is None else side_of[env.winner]
        results.append({"spec": spec, "winner": winner, "samples": samples,
                        "decisions": env.decisions, "truncated": env.truncated})
    return results


class Side:
    """One deck's policy: model, optimizer and snapshot pool."""

    def __init__(self, name: str, deck: str, cfg: Config, model: PolicyNet, vocab: CardVocab,
                 directory: Path, device: torch.device, optimizer_state: dict | None = None):
        self.name, self.deck = name, deck
        self.cfg = replace(cfg, decks=deck)
        self.model, self.vocab, self.dir, self.device = model.to(device), vocab, directory, device
        self.pool_dir = directory / "pool"
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, eps=1e-5)
        if optimizer_state is not None:
            self.opt.load_state_dict(optimizer_state)
            for group in self.opt.param_groups:
                group["lr"] = cfg.lr

    def weights(self) -> dict[str, np.ndarray]:
        return {k: v.detach().cpu().numpy() for k, v in self.model.state_dict().items()}

    def pool(self) -> list[Path]:
        return sorted(self.pool_dir.glob("*.pt"))[-self.cfg.pool_size:]

    def save(self, path: Path, iteration: int) -> None:
        tmp = path.with_suffix(".tmp")
        weights = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        torch.save({"model": weights, "optimizer": self.opt.state_dict(), "config": asdict(self.cfg),
                    "vocab": self.vocab.card_ids, "iteration": iteration,
                    "rng": random.Random(iteration).getstate()}, tmp)
        tmp.replace(path)


class DuelTrainer:
    def __init__(self, cfg: Config, decks: tuple[str, str], *, init: str | None, resume: bool):
        self.cfg, self.decks = cfg, decks
        self.dir = CHECKPOINTS / cfg.run
        self.iteration = 0
        self.rng = random.Random(cfg.seed)
        torch.manual_seed(cfg.seed)
        device = pick_device(cfg.device)
        self.sides: dict[str, Side] = {}
        for name, deck in zip("ab", decks):
            directory = self.dir / deck
            latest = directory / "latest.pt"
            if resume:
                model, _, vocab, ckpt = load_checkpoint(latest)
                self.iteration = ckpt["iteration"]
                opt_state = ckpt.get("optimizer")
            else:
                if latest.exists():
                    raise FileExistsError(f"{self.dir} already has a run; pass --resume or pick another --run")
                if init:
                    model, _, vocab, _ = load_checkpoint(init)
                else:
                    vocab = RiftboundEnv().encoder.vocab
                    model = build_model(cfg, vocab)
                opt_state = None
            model.train()
            self.sides[name] = Side(name, deck, cfg, model, vocab, directory, device, opt_state)
        vocab = self.sides["a"].vocab
        self.executor = mp.get_context("spawn").Pool(cfg.workers, _init_worker,
                                                     (asdict(cfg), vocab.card_ids, decks))

    def close(self) -> None:
        self.executor.terminate()

    def _play(self, specs: list[DuelSpec]) -> list[dict[str, Any]]:
        weights = {name: side.weights() for name, side in self.sides.items()}
        n = max(1, min(len(specs), self.cfg.workers * 2))
        chunks = [specs[i::n] for i in range(n)]
        return [r for part in self.executor.starmap(_run_games, [(weights, c) for c in chunks]) for r in part]

    def _specs(self) -> list[DuelSpec]:
        cfg = self.cfg
        pools = {name: [str(p) for p in side.pool()] for name, side in self.sides.items()}
        specs = []
        for k in range(cfg.games):
            seed = (cfg.seed * 1_000_003 + self.iteration * cfg.games + k) % (1 << 31)
            a_seat = k % 2
            roll = k % 4                                   # 0-1: live vs live, 2: A vs old B, 3: B vs old A
            if roll == 2 and pools["b"]:
                specs.append(DuelSpec(seed, a_seat, "live", self.rng.choice(pools["b"])))
            elif roll == 3 and pools["a"]:
                specs.append(DuelSpec(seed, a_seat, self.rng.choice(pools["a"]), "live"))
            else:
                specs.append(DuelSpec(seed, a_seat, "live", "live"))
        return specs

    def evaluate(self, games: int) -> dict[str, Any]:
        """Head-to-head (current A vs current B) plus each side against GreedyAgent
        piloting the other deck. Evaluation seeds never overlap training seeds."""
        specs = [DuelSpec(10**9 + k, k % 2, "live", "live", record=False) for k in range(games)]
        specs += [DuelSpec(2 * 10**9 + k, k % 2, "live", "greedy", record=False) for k in range(games // 2)]
        specs += [DuelSpec(2 * 10**9 + 10**6 + k, k % 2, "greedy", "live", record=False) for k in range(games // 2)]
        results = self._play(specs)
        h2h = [r for r in results if r["spec"].a_source == r["spec"].b_source == "live" and r["winner"]]
        a_wins = sum(r["winner"] == "a" for r in h2h)
        lo, hi = wilson(a_wins, len(h2h))
        vs_greedy_a = [r["winner"] == "a" for r in results if r["spec"].b_source == "greedy" and r["winner"]]
        vs_greedy_b = [r["winner"] == "b" for r in results if r["spec"].a_source == "greedy" and r["winner"]]
        return {"eval_a_win_rate": a_wins / len(h2h) if h2h else None, "eval_a_ci95": [lo, hi],
                "eval_games": len(h2h), "eval_a_vs_greedy": float(np.mean(vs_greedy_a)) if vs_greedy_a else None,
                "eval_b_vs_greedy": float(np.mean(vs_greedy_b)) if vs_greedy_b else None}

    def train(self, eval_games: int) -> None:
        cfg = self.cfg
        (self.dir / "config.json").write_text(json.dumps({**asdict(cfg), "duel_decks": self.decks}, indent=2))
        log = open(self.dir / "log.jsonl", "a")
        try:
            while self.iteration < cfg.iterations:
                t0 = time.time()
                results = self._play(self._specs())
                t1 = time.time()
                row: dict[str, Any] = {"iteration": self.iteration + 1, "games": len(results),
                                       "decisions_per_s": sum(r["decisions"] for r in results) / (t1 - t0),
                                       "truncated": sum(r["truncated"] for r in results)}
                live = [r for r in results if r["spec"].a_source == r["spec"].b_source == "live" and r["winner"]]
                row["train_a_win_rate"] = float(np.mean([r["winner"] == "a" for r in live])) if live else None
                for name, side in self.sides.items():
                    samples = [s for r in results for s in r["samples"][name]]
                    stats = ppo_update(side.model, side.opt, side.cfg, samples, self.iteration, side.device)
                    row.update({f"{name}_{k}": v for k, v in stats.items()
                                if k in ("policy_loss", "value_loss", "entropy", "approx_kl", "update_frac")})
                    row[f"{name}_samples"] = len(samples)
                self.iteration += 1
                row["update_s"] = time.time() - t1
                if self.iteration % cfg.eval_every == 0:
                    row.update(self.evaluate(eval_games))
                for side in self.sides.values():
                    if self.iteration % cfg.snapshot_every == 0:
                        side.save(side.pool_dir / f"iter_{self.iteration:06d}.pt", self.iteration)
                    side.save(side.dir / "latest.pt", self.iteration)
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(_format(row, self.decks), flush=True)
        finally:
            log.close()
            self.close()


def _format(row: dict[str, Any], decks: tuple[str, str]) -> str:
    parts = [f"iteration={row['iteration']}", f"decisions_per_s={row['decisions_per_s']:.0f}"]
    if row.get("train_a_win_rate") is not None:
        parts.append(f"train_{decks[0]}_wins={row['train_a_win_rate']:.3f}")
    for name in "ab":
        parts.append(f"{name}_entropy={row[f'{name}_entropy']:.3f}")
    if row.get("eval_a_win_rate") is not None:
        lo, hi = row["eval_a_ci95"]
        parts.append(f"eval_{decks[0]}_wins={row['eval_a_win_rate']:.3f} ({lo:.2f}-{hi:.2f})")
        parts.append(f"{decks[0]}_vs_greedy={row['eval_a_vs_greedy']:.2f}")
        parts.append(f"{decks[1]}_vs_greedy={row['eval_b_vs_greedy']:.2f}")
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--decks", default="annie,master_yi", help="two deck names: side A, side B")
    parser.add_argument("--init", help="ppo.py checkpoint to warm-start both sides from")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--point-reward", type=float, default=0.0,
                        help="shaping weight for the point lead (see ppo.py Config.point_reward)")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    decks = tuple(d.strip() for d in args.decks.split(","))
    if len(decks) != 2 or decks[0] == decks[1]:
        parser.error("--decks needs two different decks")
    base = Config()
    if args.init:
        base = load_checkpoint(args.init)[1]                  # keep the model's shape
    cfg = replace(base, run=args.run, seed=args.seed, iterations=args.iterations, games=args.games,
                  workers=args.workers, lr=args.lr, eval_every=args.eval_every, decks=",".join(decks),
                  point_reward=args.point_reward)
    trainer = DuelTrainer(cfg, decks, init=args.init, resume=args.resume or args.eval_only)
    if args.eval_only:
        try:
            print(json.dumps(trainer.evaluate(args.eval_games), indent=2))
        finally:
            trainer.close()
        return
    trainer.train(args.eval_games)


if __name__ == "__main__":
    main()
