# Dev log

What changed, why, and what to watch out for. Newest entries first. Add an entry
with every change to the engine, and keep **Known pitfalls** current: remove an
item when it's fixed and add new ones as you find them.

Rule numbers refer to `Riftbound-Core-Rules-RUP4-July-16-2026.pdf`.

---

## Known pitfalls

### Rules that are simplified or missing
- **Only the Kai'Sa deck is scripted.** `scripts.py` covers exactly the cards in
  `decks/kaisa.json`. Any other spell is unplayable and any other unit has no
  abilities. Keywords outside the deck (Tank, Backline, Shield, Hidden, Ganking, ...)
  are not implemented.
- **Order of simultaneous triggers (383.3.d).** The controller should choose the
  order. The engine puts the turn player's triggers first (as the rules say) but
  orders each player's own triggers by board position. Only matters with several
  simultaneous triggers, e.g. two Ravenbloom Students.
- **Triggers during combat resolve after the combat ends.** Rules 466.2/466.4 resolve
  them before the result is decided and before control is established. Today's
  triggers there (Deathknell draw, Kai'Sa's conquer draw) don't change the result,
  but a trigger that moves or kills units would.
- **Assault designation timing.** Units count as attackers the moment they are at the
  battlefield under the attacker's control. A unit arriving mid-combat should only
  get the designation at the next cleanup (464.2.c.3.a). Nothing in the deck can
  bring a unit into a combat yet.
- **"To a minimum of N"** is applied when Might is calculated: a reduction can't take
  Might below N, and never raises a unit that is already lower. Effects are
  applied in the order they were created; there are no layers (473) yet.
- **Additional costs are hard-coded.** Accelerate is always `[1][C]` (805.1.a),
  Deflect is `[A]` per time chosen (so Falling Star on the same Pouty Poro twice
  costs 2), and Legion is a flat Energy discount from `LEGION_DISCOUNT`.
- **Both players can bring *The Arena's Greatest*.** Each copy triggers, so with two in
  play each player gains 2 points on their first Beginning Phase. That follows the
  card text, but it's a big swing worth knowing about.
- **Burn Out interpretation (431.3).** If the trash is also empty,
  drawing from an empty deck burns out once per card and draws nothing. The rule
  could be read as burning out repeatedly. This is a judgment call.
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
  so do interchangeable units (same card, place and state, see `_state_key`) in moves,
  targets and damage assignment. Payment options are
  merged when they leave the same result (number of ready runes, recycled domains,
  legend use). This relies on two facts: Energy has no domain, and exhausted runes
  can still be recycled. An effect that cares *which* runes are ready (for example
  "ready a Fury rune") would break that assumption.
- **Combinatorial action counts.** Group moves, damage assignment and spell targets
  multiply with payment options. The most seen in 300 random Kai'Sa games was 168
  actions at once (mean 8, p99 46); a wide board could produce many more. The PPO
  policy scores each legal action, so any count works, but cost grows with it. If
  boards get much wider, switch to step-by-step selection (card, then targets, then
  payment).
- **Speed.** Enumerating every action is the main cost (`payment_options`). Without
  `cache_actions`, `step` enumerates them about 4.5 times per decision (validation,
  then auto-passing). With it, a single process plays about 1,000 decisions/s
  including encoding.
- **`cache_actions=True` assumes only `step` changes the game.** The cache is dropped
  when an action is applied; anything that edits state directly (tests, scripts poking
  at a Game) must leave it off (the default) or it will see stale actions.
- **Object ids** come from a process-wide counter and change whenever a card moves to
  or from a non-board zone (rule 124). Don't store oids across games or use them to
  replay a game. Replay from the seed plus the list of chosen action *indices*.
- **Card-script registries are global.** `SPELLS`, `TRIGGERS`, `LEGEND_POWER` and
  `LEGION_DISCOUNT` in `scripts.py` are module dicts keyed by card id. `test_game.py`
  registers test-only ids in them at import time. That's harmless now, but tests
  must never register a real card id. Chain items refer to trigger scripts by
  `(card_id, index)`, never by function, so reordering a card's triggers changes
  what pickled games point at.
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
- **Test setup must be reachable.** Tests build positions by hand. A position
  that couldn't happen in a real game (units on an uncontrolled battlefield outside
  a showdown, damage left over from an earlier turn) makes the engine do surprising
  but correct things in its next cleanup. Use the helpers in `test_game.py`.
- **Test battlefields.** `kaisa_game()` in the tests swaps in plain battlefields
  (a vanilla `BF-1` card) so battlefield abilities don't interfere. Use
  `use_battlefield()` to put a specific one back.
- **Dependencies.** The engine is stdlib-only; `requirements.txt` adds numpy, torch
  and pytest for RL and tests. Use a venv (`.venv/` is gitignored). Run the sim with
  the venv's Python when loading `ppo:PPOAgent`, since the system Python has no torch.
- **The sim server** keeps a single game in memory. Restarting it loses the game in
  progress.

### RL (env.py, ppo.py)
- **The card vocabulary is saved in each checkpoint.** `CardVocab` maps card ids to
  embedding rows. A new run takes every card in `card_data/`; to use new cards with an
  old checkpoint, extend its vocab (`CardVocab.extended`, which appends) and grow the
  embedding. Never re-sort it.
- **Changing the encoding breaks checkpoints.** `TOKEN_FEATURES`, `GLOBAL_FEATURES` and
  `ACTION_FEATURES` fix the input sizes. Adding a feature means retraining (or copying
  weights by hand).
- **Encoding reads only the observation.** It must never take anything from `Game`
  directly; `test_encoding_never_depends_on_hidden_information` swaps the opponent's
  hand with their deck and checks the encoding doesn't change.
- **Two actions must never encode the same.** If they did, the policy couldn't tell
  them apart. `test_encoding_shapes_and_pointers` checks it on random positions; a new
  action field needs a matching feature.
- **Token cap.** Only the first `MAX_TOKENS` (64) tokens are kept. Hands and the board
  come first and trash piles last; random Kai'Sa games peak around 41.
- **Draws.** Games cut off at `max_decisions` (500) give both seats 0. None happened in
  300 random games.

---

## 2026-09-24: RL environment, scripted baseline and PPO trainer

- **`env.py`**: `Encoder` turns an observation plus the legal actions into arrays:
  card tokens (card index + 39 features), a 75-number global vector, and one row per
  legal action (39 features + 4 pointers into the tokens for the card played, targets,
  movers, ...). Runes are summed per domain; trash and banishment become one token per
  distinct card with a count; the opponent's hand is only a count. `RiftboundEnv`
  drives both seats one decision at a time; `OpponentEnv` is a Gymnasium-style
  single-agent view against any Agent.
- **`agents.py`**: `GreedyAgent`, a scripted baseline (play the most expensive card,
  aim spells sensibly, attack where it wins). Beats RandomAgent 95% over 200 games.
- **`ppo.py`**: a transformer policy that scores each legal action, parallel rollout
  workers, PPO with GAE (gamma 1, lambda 0.95), self-play with a pool of past snapshots
  plus some GreedyAgent games, evaluation against Random and Greedy every 10
  iterations, checkpoints and a JSONL log. `PPOAgent` plays a checkpoint in the sim.
  On an M2 (7 rollout workers, learner on MPS) an iteration of 64 games (~4,000
  decisions) takes about 5 s.
- **Engine:**
  - `Game(..., cache_actions=True)` reuses legal actions between steps. That cuts
    action enumeration about 4x and gives the same games (tested).
  - Observations now include public information the encoder needs: `cards_played`
    (Legion, Darius), `scored_this_turn` (470), the source `card_id` of abilities on
    the chain, and what each Decision option points at.
- `requirements.txt` added. Tests: `test_env.py` (encoding shapes, pointers, distinct
  action rows, no hidden-information leak, cache equivalence, env rewards and cutoff,
  GreedyAgent vs Random) and `test_ppo.py` (masking, batching invariance, GAE, a
  train/save/resume/play round trip).

## 2026-09-23: Interaction tests

- New `test_interactions.py` (28 tests) covers cards working together:
  - damage and Might changes stacking (a shrink on a damaged unit kills it, and
    shrinks stop at the minimum)
  - trigger vs spell ordering on the chain (Ravenbloom Student)
  - both players stacking Reactions, and responding to a trigger (Retreat vs
    Thousand-Tailed Watcher)
  - Darius counting cards played on the opponent's turn
  - Kai'Sa's legend paying for a Reaction
  - spells during a combat showdown (shrinking or removing units, Retreating the
    attacker, Cleave on either side)
  - the defender choosing which attacker dies
  - Deathknell and conquer triggers after combat
  - Accelerated Kai'Sa attacking the turn she's played
  - Reaver's Row handing the battlefield over
  - Deflect paid per target, Legion after a spell
  - extra turns and first-turn effects, banished cards and Burn Out, and the
    final-point rule
- The tests found no engine bugs, but they corrected two of my rules assumptions:
  - A Reaction in the acting player's own hand means passing isn't automatic.
  - Assault raises Might, and Might is also toughness: an attacker with Assault is
    harder to kill, not just harder-hitting (142.4.b).
- Checked by deliberately breaking the engine three ways (printed Might instead of
  current Might, first-in-first-out chain resolution, no Deathknell). The tests
  caught each one.
- The test helper `ready_unit()` now gives a player control of an empty
  battlefield they place a unit on. Otherwise the next cleanup would (correctly)
  contest it and start a showdown, a state a real game can't reach.

## 2026-09-23: Every Kai'Sa card, triggers, keywords and extra turns

- **`scripts.py`** holds what each card does: `SPELLS` (effects and targets),
  `TRIGGERS` (triggered abilities), `LEGEND_POWER` and `LEGION_DISCOUNT`. Every card in
  `decks/kaisa.json` is scripted.
- **Targets (355.5-355.10):** spells pick their targets as they're played (a
  `targets` tuple on the action). Targets are checked again on resolution, and a target
  that left the board, or no longer qualifies, is skipped (359.3.e). A spell with no
  legal target can't be played (355.8).
- **Spells:** Hextech Ray, Void Seeker, Falling Star (two targets, possibly the same
  unit), Smoke Screen, Stupefy, Cleave, Retreat, and Time Warp (an extra turn, then the
  card is banished).
- **"This turn" effects:** Might changes (with "to a minimum of N") and granted
  keywords live on the unit and expire in the Ending Phase (317.2.c). `Game.might()`
  gives current Might.
- **Triggered abilities (382-383):** events (`played`, `conquer`, `hold`, `defend`,
  `dies`, `beginning`) queue abilities, which go on the chain with priority like
  spells. "You may" abilities ask their controller first with a new `CHOOSE` action.
  Focus doesn't pass when a trigger started the chain (346.1). Scripted: Darius,
  Ravenbloom Student, Thousand-Tailed Watcher, Kai'Sa's conquer draw, Watchful
  Sentry's Deathknell, *The Arena's Greatest*, *Startipped Peak* and *Reaver's Row*.
- **Keywords:** Accelerate (an optional `[1][C]` to enter ready), Legion (Noxus Hopeful
  costs 2 less), Deflect (`[A]` more to target), Assault (+Might while attacking).
- **The start of the turn is resumable:** each step waits for its chain to resolve
  (335), so Beginning Phase and Hold triggers can be responded to.
- **Additional turns (734-738)** are queued, and the regular turn order resumes
  afterwards.
- **Loose ends fixed:** the turn player now chooses which staged showdown starts
  (323.12 / 461.1). The sim shows current Might (green/red when changed), damage,
  abilities on the chain, targets in action labels, and pending choices.
- Tests for every card and mechanic above (77 tests). Games pickle and deep-copy
  mid-chain, mid-choice and mid-damage-assignment.

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
