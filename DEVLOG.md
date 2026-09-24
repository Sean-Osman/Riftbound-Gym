# Dev log

What changed, why, and what to watch out for. Newest entries first. Add an entry
with every change to the engine, and keep **Known pitfalls** current: remove an
item when it's fixed and add new ones as you find them.

Rule numbers refer to `Riftbound-Core-Rules-RUP4-July-16-2026.pdf`.

---

## Known pitfalls

### Rules that are simplified or missing
- **No card abilities yet.** Triggered, activated and static abilities are missing,
  and so are Deathknell (Watchful Sentry), battlefield abilities (*The Arena's Greatest*,
  *Startipped Peak*, *Reaver's Row*) and the champion/unit triggers (Darius,
  Ravenbloom Student, Kai'Sa conquer draw, Thousand-Tailed Watcher).
- **No spell effects.** Spells are unplayable until they have an entry in
  `SPELL_EFFECTS`, so the real deck plays units only. Every Action/Reaction in the
  Kai'Sa deck is a spell, so showdowns in real games always auto-resolve.
- **No keywords.** Accelerate, Legion, Deflect, Assault, Tank, Backline and Shield
  are not implemented. `CardDef.keywords` records them but nothing reads them
  except `REACTION` / `ACTION` for timing.
- **No "this turn" effects.** There is nowhere to store temporary Might changes or
  granted keywords, so the Ending Phase only heals units and empties rune pools.
- **Burn Out interpretation (431.3).** If the trash is also empty,
  drawing from an empty deck burns out once per card and draws nothing. The rule
  could be read as burning out repeatedly. This is a judgment call.
- **Multiple staged showdowns (323.12 / 461.1).** The turn player should choose
  which showdown or combat to start. The engine starts the lowest-index battlefield.
  This can't come up yet, because one move can only stage one showdown.
- **Focus after a chain (346.1).** Focus should *not* pass when the chain was
  started by a triggered or Add ability. There are no triggers yet, so the
  exception isn't implemented.
- **Recall of attackers (466.1.a.2).** Attackers are recalled only if defenders
  survive. With plain Might combat that can't happen (whichever side has less total
  Might loses all its units), so it is only reachable once damage prevention exists.
  It is tested by calling `_combat_resolution` directly.
- **Leftover combat damage** goes onto the first surviving enemy unit (by object
  id). That is harmless now because all units heal right after combat, but it will
  matter once effects react to damage being dealt.
- **Runes with several domains.** `_rune_domain` takes the first domain letter. Basic
  runes have one domain, but a dual-domain rune would need a choice.
- **The Hidden / facedown zone** exists but nothing uses it yet (the Master Yi deck
  will need it).

### Engine and API design
- **Auto-passing.** `step()` passes priority, passes focus, ends the turn or assigns
  damage for a player whenever that is their *only* option (conceding aside). Two
  consequences:
  - In the sim, your turn can end on its own when you can't afford anything. It
    shows in the log, but it's surprising.
  - An agent never sees these forced steps, so it can't count on getting a decision
    at every priority window.
- **Action deduplication.** Copies of a card in the same zone produce one action, and
  so do interchangeable units in moves and damage assignment. Payment options are
  merged when they leave the same result (number of ready runes, recycled domains,
  legend use). This relies on two facts: Energy has no domain, and exhausted runes
  can still be recycled. An effect that cares *which* runes are ready (for example
  "ready a Fury rune") would break that assumption.
- **Combinatorial action counts.** Group moves and damage assignment enumerate
  subsets of units. That is fine for the Kai'Sa mirror, but a wide board could produce
  hundreds of actions. For PPO it's worth switching to step-by-step selection (pick
  units one at a time, then confirm).
- **Object ids** come from a process-wide counter and change whenever a card moves to
  or from a non-board zone (rule 124). Don't store oids across games or use them to
  replay a game. Replay from the seed plus the list of chosen action *indices*.
- **Card-script registries are global.** `SPELL_EFFECTS` and `LEGEND_POWER` are module
  dicts keyed by card id. `test_game.py` registers test-only ids in them at import
  time. That's harmless now, but tests must never register a real card id.
- **The log is public.** The sim shows `Game.log` to both players, so it must never
  contain hidden information (it logs how many cards were mulliganed, not which).
- **Pickling / deep copy.** `Game` has to stay plain data (no generators, lambdas or
  open files on the instance), because search and rollouts depend on copying it.
  Registry functions live at module level for this reason.

### Data and tooling
- **Decklists.** `Deck.load` expects runes in a separate `"Rune Deck"` section and
  rejects runes in `"Main Board"`. Decklists exported from other sites may need to be
  split by hand.
- **Card data** comes from `import_sheet.py`. The sheet has one domain column, so
  dual-domain cards need `OVERRIDES`. Only the Kai'Sa deck's cards are in
  `card_data/`.
- **pytest isn't a declared dependency** (there is no requirements file). Install
  it with `pip install pytest`.
- **The sim server** keeps a single game in memory. Restarting it loses the game in
  progress.

---

## 2026-09-23: Concede is off by default

- `Game(..., allow_concede=False)` is the default, so RL agents never see a Concede
  action. A random early policy would concede at some point in most games, and a
  learned policy could learn to give up in positions it could win, which would skew
  matchup win rates. The browser sim passes `allow_concede=True` for humans.

## 2026-09-23: Combat

- Moving onto a battlefield with enemy units now starts a **combat**, which removes
  the temporary block on doing so. The player who contested the battlefield is the
  attacker and has focus in the combat showdown (464).
- When the showdown closes, each side assigns its total Might as damage, attacker
  first (465). Damage must be lethal on one unit before moving to the next, so the
  real choice is **which enemy units die**. Each distinct set of kills is one
  `ASSIGN_DAMAGE` action, and a forced assignment happens automatically.
- **Combat cleanup (466):** units with lethal damage die, all units heal, and attackers
  are recalled if defenders remain. A lone survivor takes control, which is a
  conquer if that player hasn't scored the battlefield this turn. If no units
  remain, the battlefield becomes uncontrolled.
- **Cleanups** now kill units with lethal damage (323.5). A non-combat showdown
  becomes a combat if enemy units arrive (323.14).
- Added `might(obj)` (printed Might plus buffs).
- The sim shows the attacker and a damage-assignment status. Duplicate battlefield
  names now say "(brought by X)".
- Tests: attacker wins and conquers, defender holds, both sides die, attacker has
  focus, choosing kills, lethal-first assignment, recall.

## 2026-09-23: Moves, battlefield control, showdowns and focus

- **Standard Moves (144):** ready units move from base to a battlefield or back,
  several at once, and moving exhausts them.
- Moving onto a battlefield you don't control contests it (450). The next cleanup
  opens a **showdown**, and the mover gets **focus** (345).
- In a showdown only Action/Reaction cards can be played. Focus passes after each
  pass, or once a spell's chain resolves (346-347). When both players pass, a lone
  occupant takes control, which is a conquer (348.2).
- **Control:** you lose a battlefield when your units leave it (190.4.c), and units
  can be played to battlefields you control (355.2.a).

## 2026-09-23: Separate Rune Deck in decklists

- `decks/kaisa.json` has a `"Rune Deck"` section. `Deck.load` rejects runes in the
  Main Board and non-runes in the Rune Deck (103.3).

## 2026-09-23: Runes, costs, turn structure and the chain

- **Runes and costs:** runes are board objects in the base. Exhausting one adds [1]
  and recycling one adds Power of its domain (164). Kai'Sa's legend adds [A] for
  spells (`LEGEND_POWER`). Playing a card means picking a payment option, and the
  pool empties at the start of the Main Phase and at end of turn (167).
- **Turn structure:** mulligan in turn order (117), then the Awaken, Beginning (Hold
  scoring), Channel (+1 rune on the second player's first turn, 485.7), Draw, Main and
  Ending phases (315-317).
- **The chain:** spells wait on the chain while priority passes, only Reactions can
  be played in response, and items resolve newest first (327-340).
- **Winning:** Burn Out (431), the conquer final-point rule (471.1.b), and the cleanup
  win check (8+ points and the lead). These replaced the placeholder random winner.
- Units enter exhausted, and the Chosen Champion can be played from its zone.
- Runes on the board are public in observations (109).

## 2026-09-23: Project docs

- Named the project **RiftboundGym** and added the README (goal, scope, the plan to
  validate against the Origins meta, roadmap), `CLAUDE.md` and `.gitignore`. Stopped
  tracking `__pycache__`.
