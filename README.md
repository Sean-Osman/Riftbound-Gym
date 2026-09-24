# RiftboundGym

A reinforcement-learning environment for the **Riftbound** trading card game.

The goal of RiftboundGym is to train agents with **PPO** (Proximal Policy
Optimization) that can pilot competitive "meta" decks, then pit those agents
against each other to measure which matchups are favored and which are not. As
a side effect, the trained agents double as practice opponents you can play
against in the browser.

> **Status: early.** Setup and the mulligan, the full turn structure, runes and
> paying costs, the chain with priority and Reactions, moving units, showdowns
> with focus, combat, conquering and holding battlefields, and Burn Out are in
> place. Card abilities, keywords and spell effects are not, so only units can
> be played. See [DEVLOG.md](DEVLOG.md) for what changed and the known pitfalls.
> There is no PPO training code yet.

## Why

Deckbuilders argue about matchups from a handful of games, and tournament data
only covers the pairings that happened to show up in the meta. A
rules-accurate simulator plus self-play agents can play thousands of games for
*any* pairing of decks and turn "I think Kai'Sa beats X" into a number with an
error bar. The long-term dream is a full matchup spread for a format, the kind
of thing a public analytics site could be built on.

## Scope

Riftbound is a lot harder to simulate than something like chess. A turn is not
a single move: a player can take many actions per turn, and the opponent can
respond with Reactions in the middle of it. Who gets to act next is decided by
focus, and card effects stack up on a chain. To keep this tractable, the scope
is deliberately narrow and grows in stages:

- **1v1 Duel only.** No multiplayer modes.
- **The Origins format.** The Origins set, including Proving Grounds cards
  (which are part of Origins). Promo and alternate-art printings are dropped
  because they are the same cards with different art. Later sets (Spiritforged,
  Unleashed, Vendetta, …) come after Origins works end to end.
- **A small, fixed pool of real decks.** The first milestone is a single
  mirror match: [Kai'Sa](decks/kaisa.json) vs. Kai'Sa. After that come a
  handful more Origins meta decklists (Master Yi is a likely next one, since it
  introduces hidden cards), then the rest of the meta.
- **Best-of-one games.** No sideboarding and no Bo3 matches.
- **No deckbuilding.** Agents play fixed decklists. Having an agent *build*
  decks is a separate, much later problem.

## Validation: can it rediscover the meta?

The Origins format already played out, so its results are known and can be
used to check this project's work. If the engine is rules-accurate and the
agents play well, their matchup results should roughly match how the decks
performed in major tournaments at the end of the Origins season.

For that reason, **cards and battlefields that were banned stay legal here.**
Draven and the battlefields *The Arena's Greatest* and *Reaver's Row* are all
playable (the starter Kai'Sa list runs both battlefields). A
strong result would be the agents independently finding that those cards are
too strong. In other words, the model should make us want to ban Draven.

## Design principles

- **Rules-accurate.** The official Core Rules (July 16, 2026) are the spec, and
  code comments cite rule numbers (e.g. `# 485.3`). When a card contradicts the
  rules, the card wins (e.g. an extra-turn spell like Time Warp overrides the
  normal turn order).
- **Deterministic and replayable.** All randomness (shuffles, battlefield
  selection, first player) comes from one seeded RNG, so any game can be
  replayed exactly from its seed and action list. Training should be made
  deterministic as well where possible.
- **Watchable.** A browser view lets you see exactly what an agent is doing, and
  play against it yourself.
- **Python first.** The engine and RL code are in Python for fast iteration
  and access to its ML and plotting libraries. If training throughput becomes
  the bottleneck, the hot loop may later be ported to Rust.

## Quick start

Requires Python 3.10+. The engine has no third-party dependencies.

```bash
git clone https://github.com/Sean-Osman/riftbound-thing.git
cd riftbound-thing

python3 game.py                         # 200 headless games, random vs random
python3 sim.py                          # play in the browser at http://localhost:8765
```

Play against your own agent (any class with a `name` and an `act(observation, legal)` method):

```bash
python3 sim.py --agent my_module:MyAgent --seat 1 --port 9000
```

Run the tests (they are plain pytest-style functions):

```bash
pip install pytest && python3 -m pytest
```

## How it works

The engine exposes a small decision loop that both the browser sim and future
RL training plug into:

```python
from game import Game, RandomAgent, load_demo_decks, play_game

game = Game(list(load_demo_decks()), seed=0)
while not game.is_over:
    seat = game.acting_player
    obs = game.observation(seat)        # only what this seat is allowed to see
    legal = game.legal_actions(seat)    # list[Action]
    game.step(agent.act(obs, legal))
```

- `observation(seat)` is JSON-serializable and hides private information
  (the opponent's hand, deck order, etc.), so agents can't cheat.
- `legal_actions(seat)` enumerates every legal move, which maps directly
  onto a masked discrete action space for PPO.
- `acting_player` is whoever must decide next: the player with priority, which
  switches to the opponent while spells wait on the chain for Reactions.
- Games are seeded and deterministic, and `Game` objects can be deep-copied
  and pickled.

The engine is modeled on the rules' own vocabulary. Every place a card can be
is a zone: the main deck, hand, base, trash, battlefields, face-down zones,
the chain and so on. Cards are objects that move between zones, and each
player owns their own set of zones.

## Project layout

| Path | What it is |
| --- | --- |
| `cards.py` | Static card definitions (`CardDef`), costs, domains, keywords, deck-legality checks |
| `zones.py` | Per-game object state (`CardInstance`), zones, battlefields, rune pool, visibility |
| `game.py` | `Deck`, `Game` (setup, turn structure, decision loop), `Action`, `Agent`, `RandomAgent` |
| `sim.py` + `sim/index.html` | Local web UI for playing against an agent |
| `import_sheet.py` | Converts a tab of the card spreadsheet (`.xlsx`) into card JSON |
| `card_data/` | Card JSON loaded by `cards.load_card_pool()` |
| `decks/` | Decklists: card IDs and counts, looked up in the card pool |
| `test_*.py` | Tests |
| `DEVLOG.md` | What changed and known pitfalls; update it with every engine change |
| `Riftbound-Core-Rules-*.pdf` | The official Core Rules the engine follows |

## Card data

Cards are maintained in a spreadsheet, which is then converted into JSON. Only
the cards used by the supported decks need to be filled in. Decklists are
separate JSON files of card IDs and counts, so every deck loads the same way
from one shared card pool.

Domains are written with one letter each, taken from their color:

| Fury | Calm | Mind | Body | Chaos | Order |
| --- | --- | --- | --- | --- | --- |
| `R` | `G` | `B` | `O` | `P` | `Y` |

Energy costs are plain numbers. Power costs are strings with one letter per
symbol, and a number in front repeats that symbol:

| Power cost | Meaning |
| --- | --- |
| `B` | one Mind |
| `4B` | four Mind (e.g. Time Warp) |
| `RG` | one Fury and one Calm |
| `A` | one of any domain |
| `C` | one of the card's own domains (e.g. Fury *or* Calm on a Fury/Calm card) |

The source spreadsheet only lists one domain per card, so dual-domain cards
like legends are fixed via `OVERRIDES` in `import_sheet.py`.

To add a deck:

1. Fill in any missing cards in the spreadsheet, then convert the tab:
   ```bash
   python3 import_sheet.py "~/Downloads/Riftbound Collection.xlsx" "Kaisa Deck" card_data/kaisa_deck.json
   ```
2. Add the decklist to `decks/` in the same format as `decks/kaisa.json`.
3. Check legality with `Deck.load(path, pool).errors()`.

## Roadmap

**Phase 1: Rules engine** (in progress)
- [x] Card model, deck legality, zones, visibility
- [x] Setup, mulligan and the phases of the turn
- [x] Runes, the rune pool and paying costs
- [x] The chain, priority and Reactions
- [x] Moving units, showdowns and focus, conquering battlefields
- [x] Combat
- [x] Victory at 8 points, Hold scoring and burning out (running out of cards)
- [x] Browser sim and pluggable agent interface
- [x] Random legal-move agent
- [ ] Damage, death and "this turn" effects
- [ ] Spell effects, and triggered, activated and static abilities
- [ ] Keywords (Accelerate, Legion, Deflect, Assault, …)
- [ ] Kai'Sa mirror fully playable under the real rules

**Phase 2: Training**
- [ ] Gymnasium-style environment wrapper with observation encoding and action masks
- [ ] PPO self-play on the Kai'Sa mirror
- [ ] Trained agents selectable as opponents in the sim

**Phase 3: Meta analysis**
- [ ] Add the rest of the Origins meta decks
- [ ] Matchup matrix across decks
- [ ] Compare with end-of-season Origins tournament results

**Later / maybe**
- [ ] Later sets and formats
- [ ] Port performance-critical parts to Rust
- [ ] Public matchup analytics site
- [ ] Deckbuilding agents

## Contributing

Work is tracked in GitHub issues, and [DEVLOG.md](DEVLOG.md) records changes and known pitfalls. Claim or open an issue before starting, so
two people don't end up building the same thing. Work on a branch and open a pull
request against `main`. Please include tests for any change to game logic, and
cite the Core Rules section a behavior comes from in a comment.

## Disclaimer

RiftboundGym is an unofficial fan project and is not endorsed by or affiliated with
Riot Games. Riftbound, League of Legends and all related names, card text and
artwork are trademarks or copyrights of Riot Games, Inc.
