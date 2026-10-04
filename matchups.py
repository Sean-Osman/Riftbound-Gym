"""Matchup matrix: how each deck does against each other deck.

    .venv/bin/python matchups.py checkpoints/meta-v1/latest.pt --games 200
    .venv/bin/python matchups.py --greedy --games 200          # scripted baseline, no policy

Both seats are played by the same agent (a trained policy sampling its actions,
or GreedyAgent), so a win rate measures the decks rather than the pilots. Every
pairing is played with the decks alternating seats, since going first matters.
Games cut off at `--max-decisions` count as draws and are left out of the rates.
Writes <out>.json and prints a markdown table of row-deck win rates with 95%
intervals.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

from game import DECK_NAMES, Game, load_decks

_worker: dict[str, Any] = {}


def _init(checkpoint: str | None, names: tuple[str, ...]) -> None:
    import torch
    torch.set_num_threads(1)
    _worker["decks"] = load_decks(names)
    _worker["checkpoint"] = checkpoint


def _agent(seed: int) -> Any:
    if _worker["checkpoint"] is None:
        from agents import GreedyAgent
        return GreedyAgent(seed)
    from ppo import PPOAgent
    if "ppo" not in _worker:
        _worker["ppo"] = PPOAgent(_worker["checkpoint"], sample=True, seed=seed)
    agent = _worker["ppo"]
    import numpy as np
    agent.rng = np.random.default_rng(seed)
    return agent


def _play(job: tuple[str, str, int, int]) -> tuple[str, str, int | None, int]:
    """One game: deck `a` in seat `a_seat`. Returns (a, b, winner seat's deck side, a_seat)."""
    a, b, seed, a_seat = job
    decks = _worker["decks"]
    pair = [decks[a], decks[b]] if a_seat == 0 else [decks[b], decks[a]]
    g = Game(pair, seed=seed, cache_actions=True)
    agent = _agent(seed)
    for _ in range(_worker.get("max_decisions", 500)):
        if g.is_over:
            break
        seat = g.acting_player
        g.step(agent.act(g.observation(seat), g.legal_actions(seat)))
    if not g.is_over:
        return a, b, None, a_seat
    return a, b, int(g.winner == a_seat), a_seat


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a win rate."""
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", nargs="?", help="a ppo.py checkpoint (omit with --greedy)")
    parser.add_argument("--greedy", action="store_true", help="GreedyAgent pilots both seats")
    parser.add_argument("--games", type=int, default=200, help="games per pairing")
    parser.add_argument("--decks", default=",".join(DECK_NAMES))
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out", help="output path without extension (default: next to the checkpoint)")
    args = parser.parse_args()
    if not args.greedy and not args.checkpoint:
        parser.error("give a checkpoint or --greedy")
    names = tuple(d.strip() for d in args.decks.split(","))
    pairings = [(a, b) for i, a in enumerate(names) for b in names[i:]]
    jobs = [(a, b, 7_000_000 + k, k % 2) for a, b in pairings for k in range(args.games)]
    checkpoint = None if args.greedy else args.checkpoint
    with mp.get_context("spawn").Pool(args.workers, _init, (checkpoint, names)) as pool:
        results = pool.map(_play, jobs, chunksize=4)

    table: dict[str, dict[str, Any]] = {}
    for a, b in pairings:
        rows = [r for r in results if r[0] == a and r[1] == b]
        decided = [r for r in rows if r[2] is not None]
        wins = sum(r[2] for r in decided)
        first = [r for r in decided if (r[3] == 0) == bool(r[2])]            # games won by seat 0
        lo, hi = wilson(wins, len(decided))
        table[f"{a}|{b}"] = {"deck": a, "opponent": b, "games": len(rows), "decided": len(decided),
                             "wins": wins, "win_rate": wins / len(decided) if decided else None,
                             "ci95": [lo, hi], "seat0_win_rate": len(first) / len(decided) if decided else None}

    label = "greedy" if args.greedy else str(Path(args.checkpoint))
    out = Path(args.out) if args.out else (Path("matchups-greedy") if args.greedy
                                           else Path(args.checkpoint).with_name("matchups"))
    out.with_suffix(".json").write_text(json.dumps({"pilot": label, "games_per_pairing": args.games,
                                                    "pairings": table}, indent=2))
    print(f"Pilot: {label}, {args.games} games per pairing (row deck's win rate vs column deck)\n")
    print("| | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for a in names:
        cells = []
        for b in names:
            if a == b:
                e = table[f"{a}|{b}"]
                cells.append(f"mirror (seat 0 wins {e['seat0_win_rate']:.0%})" if e["decided"] else "mirror")
                continue
            key, flip = (f"{a}|{b}", False) if f"{a}|{b}" in table else (f"{b}|{a}", True)
            e = table[key]
            if not e["decided"]:
                cells.append("n/a")
                continue
            rate, (lo, hi) = e["win_rate"], e["ci95"]
            if flip:
                rate, lo, hi = 1 - rate, 1 - hi, 1 - lo
            cells.append(f"{rate:.0%} ({lo:.0%}-{hi:.0%})")
        print(f"| **{a}** | " + " | ".join(cells) + " |")
    draws = sum(e["games"] - e["decided"] for e in table.values())
    print(f"\n{draws} games cut off as draws. Wrote {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
