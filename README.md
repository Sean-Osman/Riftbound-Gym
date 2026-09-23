# RiftboundGym

A reinforcement-learning environment for the **Riftbound** trading card game.

The goal of RiftboundGym is to train agents with **PPO** that can pilot competitive
"meta" decks, then pit those agents against each other to measure which
matchups are favored and which are not. As a side effect, the trained agents
double as practice opponents you can play against in the browser.

> **Status: early.** Deck loading, game setup, zones, runes/energy and playing
> permanents are in place. Combat, spells, abilities, scoring and the
> showdown/chain system are not implemented yet, so games currently end with
> a placeholder random winner after a fixed number of turns. There is no PPO
> training code yet.

## Why

Deckbuilders argue about matchups from a handful of games. A rules-accurate
simulator plus self-play agents can play thousands of games per matchup and
turn "I think Kai'Sa beats X" into a number with an error bar.

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
- Games are seeded and deterministic, and `Game` objects can be deep-copied
  and pickled.

## Project layout

| Path | What it is |
| --- | --- |
| `cards.py` | Static card definitions (`CardDef`), costs, domains, keywords, deck-legality checks |
| `zones.py` | Per-game object state (`CardInstance`), zones, battlefields, rune pool, visibility |
| `game.py` | `Deck`, `Game` (setup, turn structure, decision loop), `Action`, `Agent`, `RandomAgent` |
| `sim.py` + `sim/index.html` | Local web UI for playing against an agent |
| `import_sheet.py` | Converts a tab of a card-collection `.xlsx` into card JSON |
| `card_data/` | Card JSON loaded by `cards.load_card_pool()` |
| `decks/` | Decklists (card IDs and counts) |
| `test_*.py` | Tests |
| `Riftbound-Core-Rules-*.pdf` | The official Core Rules the engine follows. Code comments cite rule numbers (e.g. `# 485.3`) |

## Adding cards and decks

1. Export or maintain your card list in a spreadsheet, then convert one tab:
   ```bash
   python3 import_sheet.py "~/Downloads/Riftbound Collection.xlsx" "Kaisa Deck" card_data/kaisa_deck.json
   ```
2. Add a decklist to `decks/` in the same format as `decks/kaisa.json`.
3. Check legality with `Deck.load(path, pool).errors()`.

## Roadmap

- [x] Card model, deck legality, zones, visibility
- [x] Setup, turn structure, runes/energy, playing permanents
- [x] Browser sim and pluggable agent interface
- [ ] Movement, combat, showdowns and the chain
- [ ] Spells, triggered and activated abilities
- [ ] Scoring, conquering/holding battlefields, victory at 8 points
- [ ] Gymnasium-style env wrapper with observation encoding and action masks
- [ ] PPO self-play training
- [ ] Matchup matrix across meta decks
- [ ] Trained agents selectable as opponents in the sim

## Contributing

Work on a branch and open a pull request against `main`. Please include tests
for any change to game logic, and cite the Core Rules section a behavior comes
from in a comment.

## Disclaimer

RiftboundGym is an unofficial fan project and is not endorsed by or affiliated with
Riot Games. Riftbound, League of Legends and all related names, card text and
artwork are trademarks or copyrights of Riot Games, Inc.
