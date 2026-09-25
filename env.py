"""RL environment: turns observations and legal actions into arrays for a policy.

The encoder only reads `Game.observation(seat)` and the legal `Action`s, never the
Game itself, so it can't see anything the seat isn't allowed to know.

An encoded state is a set of card tokens plus a vector of global numbers:

- tokens: one per visible card that matters (hands, base units, legends, the
  Chosen Champions, battlefields and the units on them, the chain), plus one
  token per distinct card in each trash and banishment with a count. Runes are
  summarized in the global vector instead. Opponent hand cards are hidden and
  only counted. Each token has a card index (for an embedding) and features.
- global: points, phase, priority/focus, rune pools, zone counts, ...

Legal actions are encoded one row each: numeric features plus up to
`N_POINTERS` pointers into the tokens (the card played, its targets, the units
moving, ...). A policy scores each row against the state, so the number of
actions can differ at every decision. Object ids are never features: they come
from a process-wide counter and change whenever a card changes zones (124).

    env = RiftboundEnv(seed=0)
    while not env.done:
        state = env.encode()                 # for env.acting_player
        env.step(policy(state))              # an index into env.legal
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from cards import CardDef, Domain, load_card_pool
from game import ROOT, Action, ActionKind, Agent, Deck, Game, TurnPhase, load_demo_decks

MAX_TOKENS = 64
N_POINTERS = 4
MAX_BATTLEFIELDS = 3
MAX_DECISIONS = 500          # games are cut off (a draw) after this many decisions

DOMAINS = [d.value for d in Domain]                         # R G B O P Y
POWER_KEYS = DOMAINS + ["A"]
TURN_PHASES = [p.value for p in TurnPhase]
ACTION_KINDS = list(ActionKind)
TOKEN_ZONES = ["hand", "base", "legend", "champion", "battlefield", "battlefield_card",
               "chain", "trash", "banishment"]
CARD_TYPES = ["unit", "spell", "gear", "battlefield", "legend"]

PAD, UNKNOWN = 0, 1          # card indices with no card behind them


class CardVocab:
    """Card id -> embedding index. Saved with every checkpoint: adding cards to the
    pool appends indices, so earlier checkpoints keep their meaning."""

    def __init__(self, card_ids: Sequence[str]):
        self.card_ids = list(card_ids)
        self.index = {cid: i + 2 for i, cid in enumerate(self.card_ids)}

    @classmethod
    def from_pool(cls, pool: dict[str, CardDef] | Sequence[str]) -> CardVocab:
        return cls(sorted(pool))

    def extended(self, card_ids: Sequence[str]) -> CardVocab:
        return CardVocab(self.card_ids + [c for c in sorted(card_ids) if c not in self.index])

    def __len__(self) -> int:
        return len(self.card_ids) + 2

    def __getitem__(self, card_id: str | None) -> int:
        return self.index.get(card_id, UNKNOWN) if card_id else UNKNOWN


# --- feature layouts ---------------------------------------------------------

def _layout(names: list[str]) -> dict[str, int]:
    return {n: i for i, n in enumerate(names)}


TOKEN_FEATURES = _layout(
    ["mine", "theirs", "owned"]
    + [f"zone_{z}" for z in TOKEN_ZONES]
    + [f"bf_{i}" for i in range(MAX_BATTLEFIELDS)]
    + [f"type_{t}" for t in CARD_TYPES]
    + ["exhausted", "damage", "might", "printed_might", "might_change", "buffs", "granted",
       "count", "chain_depth", "ability", "attacking", "bf_uncontrolled", "bf_contested",
       "bf_contested_by_me", "bf_showdown", "bf_scored_by_me", "bf_scored_by_them",
       "energy_cost", "power_cost"]
)
TOKEN_DIM = len(TOKEN_FEATURES)


def _player_features(prefix: str) -> list[str]:
    return ([f"{prefix}_{n}" for n in ("points", "hand", "deck", "rune_deck", "trash", "runes",
                                       "legend_exhausted", "cards_played", "extra_turns")]
            + [f"{prefix}_ready_{d}" for d in DOMAINS]
            + [f"{prefix}_exhausted_{d}" for d in DOMAINS])


GLOBAL_FEATURES = _layout(
    ["points_diff", "turn", "my_turn", "priority_mine", "focus_mine", "focus_none",
     "showdown", "combat", "attacker_mine", "assigning_mine", "decision", "decision_trigger",
     "decision_showdown", "chain_length", "chain_top_mine", "chain_by_me",
     "pool_energy", "pool_spells_only"]
    + [f"pool_{k}" for k in POWER_KEYS]
    + [f"phase_{p}" for p in TURN_PHASES]
    + _player_features("me") + _player_features("them")
)
GLOBAL_DIM = len(GLOBAL_FEATURES)

ACTION_FEATURES = _layout(
    [f"kind_{k.value}" for k in ACTION_KINDS]
    + ["to_base"] + [f"to_bf_{i}" for i in range(MAX_BATTLEFIELDS)]
    + ["accelerate", "exhaust", "recycle", "legend", "ready_after", "runes_after"]
    + [f"recycle_{d}" for d in DOMAINS]
    + ["units", "unit_might", "targets", "targets_mine", "targets_theirs", "set_aside"]
    + [f"damage_{i}" for i in range(N_POINTERS)]
    + [f"choice_{i}" for i in range(4)] + ["declines"]
)
ACTION_DIM = len(ACTION_FEATURES)


@dataclass
class Encoded:
    """One decision, as arrays. `pointers` index into the tokens; -1 means none."""
    cards: np.ndarray        # int64 [MAX_TOKENS]
    tokens: np.ndarray       # float32 [MAX_TOKENS, TOKEN_DIM]
    mask: np.ndarray         # bool [MAX_TOKENS], True for real tokens
    glob: np.ndarray         # float32 [GLOBAL_DIM]
    actions: np.ndarray      # float32 [n_actions, ACTION_DIM]
    pointers: np.ndarray     # int64 [n_actions, N_POINTERS]


_COST = re.compile(r"\[(\d+|[A-Z])\]")


def _cost(text: str | None) -> tuple[int, int]:
    """'[2][R]' -> (2 Energy, 1 Power)."""
    energy = power = 0
    for part in _COST.findall(text or ""):
        if part.isdigit():
            energy += int(part)
        else:
            power += 1
    return energy, power


class Encoder:
    def __init__(self, vocab: CardVocab):
        self.vocab = vocab

    def encode(self, obs: dict[str, Any], legal: Sequence[Action]) -> Encoded:
        me = obs["viewer"]
        tokens = np.zeros((MAX_TOKENS, TOKEN_DIM), np.float32)
        cards = np.zeros(MAX_TOKENS, np.int64)
        oid_token: dict[int, int] = {}
        bf_token: dict[int, int] = {}
        might = {int(k): v for k, v in obs["unit_might"].items()}
        scored = {(seat, i) for seat, i in obs["scored_this_turn"]}
        rows: list[tuple[str | None, dict[str, float], int | None]] = []   # card id, features, oid

        def card_row(view: dict[str, Any], zone: str, **extra: float) -> None:
            f: dict[str, float] = {"mine": view["controller"] == me, "theirs": view["controller"] != me,
                                   "owned": view["owner"] == me, f"zone_{zone}": 1.0}
            for t in view["types"]:
                if t in CARD_TYPES:
                    f[f"type_{t}"] = 1.0
            printed = view["might"] or 0
            current = might.get(view["oid"], printed)
            energy, power = _cost(view["cost"])
            f.update(exhausted=view["exhausted"], damage=view["damage"] / 4, might=current / 8,
                     printed_might=printed / 8, might_change=(current - printed) / 4,
                     buffs=view["buffs"], granted=sum(v or 0 for v in view["granted"].values()) / 4,
                     energy_cost=energy / 8, power_cost=power / 4)
            f.update(extra)
            rows.append((view["card_id"], f, view["oid"]))

        players = obs["players"]
        order = [me, 1 - me]
        for seat in order:                                  # the cards that matter most come first
            zones = players[seat]["zones"]
            for view in zones["hand"]["objects"]:
                if not view.get("hidden"):
                    card_row(view, "hand")
            for view in zones["base"]["objects"]:
                if "rune" not in view["types"]:
                    card_row(view, "base")
            for zone in ("legend", "champion"):
                for view in zones[zone]["objects"]:
                    card_row(view, zone)
        for i, bf in enumerate(obs["battlefields"][:MAX_BATTLEFIELDS]):
            controller = bf["controller"]
            f = {"bf_uncontrolled": controller is None, "bf_contested": bf["contested"],
                 "bf_contested_by_me": bf["contested_by"] == me, "bf_showdown": obs["showdown"] == i,
                 "bf_scored_by_me": (me, i) in scored, "bf_scored_by_them": (1 - me, i) in scored,
                 f"bf_{i}": 1.0}
            card_row(bf["card"], "battlefield_card", **f)
            rows[-1][1].update(mine=controller == me, theirs=controller not in (None, me))
            bf_token[i] = len(rows) - 1
            for view in bf["units"]:
                attacking = obs["attacker"] is not None and obs["showdown"] == i and view["controller"] == obs["attacker"]
                card_row(view, "battlefield", **{f"bf_{i}": 1.0, "attacking": attacking})
        chain = obs["chain"]
        for depth, entry in enumerate(reversed(chain)):
            if "ability" in entry:
                f = {"mine": entry["chain_controller"] == me, "theirs": entry["chain_controller"] != me,
                     "zone_chain": 1.0, "ability": 1.0, "chain_depth": depth / 4}
                rows.append((entry["card_id"], f, None))
            else:
                card_row(entry, "chain", chain_depth=depth / 4)
                rows[-1][1].update(mine=entry["chain_controller"] == me, theirs=entry["chain_controller"] != me)
        for seat in order:                                  # piles: one token per distinct card
            for zone in ("trash", "banishment"):
                counts: dict[str, int] = {}
                for view in players[seat]["zones"][zone]["objects"]:
                    counts[view["card_id"]] = counts.get(view["card_id"], 0) + 1
                for card_id, n in sorted(counts.items()):
                    rows.append((card_id, {"mine": seat == me, "theirs": seat != me, "owned": seat == me,
                                           f"zone_{zone}": 1.0, "count": n / 4}, None))

        for t, (card_id, f, oid) in enumerate(rows[:MAX_TOKENS]):
            cards[t] = self.vocab[card_id]
            for name, value in f.items():
                tokens[t, TOKEN_FEATURES[name]] = float(value)
            if oid is not None:
                oid_token[oid] = t
        mask = np.zeros(MAX_TOKENS, bool)
        mask[:min(len(rows), MAX_TOKENS)] = True

        runes = self._runes(obs)
        glob = self._global(obs, me, runes)
        actions = np.zeros((len(legal), ACTION_DIM), np.float32)
        pointers = np.full((len(legal), N_POINTERS), -1, np.int64)
        for i, action in enumerate(legal):
            self._action(action, obs, me, runes, oid_token, bf_token, tokens, might,
                         actions[i], pointers[i])
        return Encoded(cards, tokens, mask, glob, actions, pointers)

    @staticmethod
    def _runes(obs: dict[str, Any]) -> dict[int, tuple[int, str, bool]]:
        """rune oid -> (controller, domain, exhausted), for every rune on the board."""
        runes = {}
        for p in obs["players"]:
            for view in p["zones"]["base"]["objects"]:
                if "rune" in view["types"]:
                    runes[view["oid"]] = (view["controller"], min(view["domains"]), view["exhausted"])
        return runes

    def _global(self, obs: dict[str, Any], me: int, runes: dict[int, tuple[int, str, bool]]) -> np.ndarray:
        g = np.zeros(GLOBAL_DIM, np.float32)

        def put(name: str, value: float) -> None:
            g[GLOBAL_FEATURES[name]] = float(value)

        players = obs["players"]
        put("points_diff", (players[me]["points"] - players[1 - me]["points"]) / 8)
        put("turn", obs["turn"] / 30)
        put("my_turn", obs["turn_player"] == me)
        put("priority_mine", obs["priority"] == me)
        put("focus_mine", obs["focus"] == me)
        put("focus_none", obs["focus"] is None)
        put("showdown", obs["showdown"] is not None)
        put("combat", obs["attacker"] is not None)
        put("attacker_mine", obs["attacker"] == me)
        put("assigning_mine", obs["assigning"] == me)
        decision = obs["decision"]
        put("decision", decision is not None)
        put("decision_trigger", decision is not None and decision["kind"] == "trigger")
        put("decision_showdown", decision is not None and decision["kind"] == "showdown")
        put("chain_length", len(obs["chain"]) / 4)
        put("chain_top_mine", bool(obs["chain"]) and obs["chain"][-1]["chain_controller"] == me)
        put("chain_by_me", any(e["chain_controller"] == me for e in obs["chain"]))
        pool = players[me]["rune_pool"]
        put("pool_energy", pool["energy"] / 8)
        for key, n in pool["power"].items():
            domain, _, rest = key.partition(" ")
            g[GLOBAL_FEATURES[f"pool_{domain}"]] += n / 4
            if rest:
                g[GLOBAL_FEATURES["pool_spells_only"]] += n / 4
        put(f"phase_{obs['turn_phase']}", 1)
        for prefix, seat in (("me", me), ("them", 1 - me)):
            zones = players[seat]["zones"]
            put(f"{prefix}_points", players[seat]["points"] / 8)
            put(f"{prefix}_hand", zones["hand"]["count"] / 10)
            put(f"{prefix}_deck", zones["main_deck"]["count"] / 40)
            put(f"{prefix}_rune_deck", zones["rune_deck"]["count"] / 12)
            put(f"{prefix}_trash", zones["trash"]["count"] / 20)
            put(f"{prefix}_legend_exhausted", any(v["exhausted"] for v in zones["legend"]["objects"]))
            put(f"{prefix}_cards_played", obs["cards_played"][seat] / 4)
            put(f"{prefix}_extra_turns", obs["extra_turns"].count(seat))
            mine = [r for r in runes.values() if r[0] == seat]
            put(f"{prefix}_runes", len(mine) / 12)
            for _, domain, exhausted in mine:
                g[GLOBAL_FEATURES[f"{prefix}_{'exhausted' if exhausted else 'ready'}_{domain}"]] += 1 / 6
        return g

    def _action(self, action: Action, obs: dict[str, Any], me: int, runes: dict[int, tuple[int, str, bool]],
                oid_token: dict[int, int], bf_token: dict[int, int], tokens: np.ndarray,
                might: dict[int, int], row: np.ndarray, ptr: np.ndarray) -> None:
        def put(name: str, value: float) -> None:
            row[ACTION_FEATURES[name]] = float(value)

        def point(slot: int, oid: int | None) -> None:
            if slot < N_POINTERS and oid is not None:
                ptr[slot] = oid_token.get(oid, -1)

        put(f"kind_{action.kind.value}", 1)
        kind = action.kind
        if kind in (ActionKind.PLAY_CARD, ActionKind.MOVE):
            if action.destination is None:
                put("to_base", 1)
            elif action.destination < MAX_BATTLEFIELDS:
                put(f"to_bf_{action.destination}", 1)
        if kind is ActionKind.PLAY_CARD:
            point(0, action.card_oid)
            for slot, oid in enumerate(action.targets[:N_POINTERS - 1], start=1):
                point(slot, oid)
            put("accelerate", action.accelerate)
            put("targets", len(action.targets) / 2)
            owners = [tokens[oid_token[o], TOKEN_FEATURES["mine"]] for o in action.targets if o in oid_token]
            put("targets_mine", sum(owners) / 2)
            put("targets_theirs", (len(owners) - sum(owners)) / 2)
            pay = action.payment
            if pay is not None:
                mine = {oid: r for oid, r in runes.items() if r[0] == me}
                used = set(pay.exhaust) | {o for o in pay.recycle if not mine.get(o, (0, "", True))[2]}
                ready = sum(1 for r in mine.values() if not r[2])
                put("exhaust", len(pay.exhaust) / 6)
                put("recycle", len(pay.recycle) / 6)
                put("legend", pay.legend)
                put("ready_after", (ready - len(used)) / 12)
                put("runes_after", (len(mine) - len(pay.recycle)) / 12)
                for oid in pay.recycle:
                    if oid in mine:
                        row[ACTION_FEATURES[f"recycle_{mine[oid][1]}"]] += 1 / 3
        elif kind is ActionKind.MOVE:
            for slot, oid in enumerate(action.units[:N_POINTERS]):
                point(slot, oid)
            put("units", len(action.units) / 4)
            put("unit_might", sum(might.get(o, 0) for o in action.units) / 10)
        elif kind is ActionKind.ASSIGN_DAMAGE:
            for slot, (oid, amount) in enumerate(action.damage[:N_POINTERS]):
                point(slot, oid)
                put(f"damage_{slot}", amount / 8)
        elif kind is ActionKind.MULLIGAN:
            for slot, oid in enumerate(action.set_aside[:N_POINTERS]):
                point(slot, oid)
            put("set_aside", len(action.set_aside) / 2)
        elif kind is ActionKind.CHOOSE:
            if action.choice is not None and action.choice < 4:
                put(f"choice_{action.choice}", 1)
            options = (obs["decision"] or {}).get("options", [])
            if action.choice is not None and action.choice < len(options):
                option = options[action.choice]
                put("declines", option["declines"])
                if "oid" in option:
                    point(0, option["oid"])
                elif "battlefield" in option and option["battlefield"] in bf_token:
                    ptr[0] = bf_token[option["battlefield"]]


# --- environments --------------------------------------------------------------

class RiftboundEnv:
    """Both seats are driven by the caller, one decision at a time. `step` takes an
    index into `legal`. A game that runs past `max_decisions` is cut off as a draw."""

    def __init__(self, decks: Sequence[Deck] | None = None, *, seed: int | None = None,
                 vocab: CardVocab | None = None, max_decisions: int = MAX_DECISIONS):
        self.decks = list(decks) if decks is not None else list(load_demo_decks())
        if vocab is None:
            vocab = CardVocab.from_pool(load_card_pool(ROOT / "card_data"))
        self.encoder = Encoder(vocab)
        self.max_decisions = max_decisions
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        self.game = Game(self.decks, seed=seed, cache_actions=True)
        self.decisions = 0
        self.truncated = False
        self.legal = self.game.legal_actions()

    @property
    def done(self) -> bool:
        return self.game.is_over or self.truncated

    @property
    def acting_player(self) -> int | None:
        return None if self.done else self.game.acting_player

    @property
    def winner(self) -> int | None:
        return self.game.winner

    def observation(self) -> dict[str, Any]:
        return self.game.observation(self.acting_player)

    def encode(self) -> Encoded:
        return self.encoder.encode(self.observation(), self.legal)

    def step(self, index: int) -> None:
        if self.done:
            raise ValueError("the game is over")
        self.game.step(self.legal[index])
        self.decisions += 1
        self.truncated = not self.game.is_over and self.decisions >= self.max_decisions
        self.legal = [] if self.done else self.game.legal_actions()

    def reward(self, seat: int) -> float:
        """+1 for a win, -1 for a loss, 0 while playing or after a cutoff."""
        if self.game.winner is None:
            return 0.0
        return 1.0 if self.game.winner == seat else -1.0


class OpponentEnv:
    """Gymnasium-style single-agent view: the learner plays `seat` and `opponent`
    (any Agent) plays the other side inside `step`.

    reset(seed) -> (encoded, info); step(index) -> (encoded, reward, terminated,
    truncated, info). `info["legal"]` holds the legal Actions behind the rows."""

    def __init__(self, opponent: Agent, *, seat: int = 0, **kwargs: Any):
        self.env = RiftboundEnv(**kwargs)
        self.opponent = opponent
        self.seat = seat

    def reset(self, seed: int | None = None) -> tuple[Encoded | None, dict[str, Any]]:
        self.env.reset(seed)
        return self._handoff()

    def step(self, index: int) -> tuple[Encoded | None, float, bool, bool, dict[str, Any]]:
        self.env.step(index)
        encoded, info = self._handoff()
        return encoded, self.env.reward(self.seat), self.env.game.is_over, self.env.truncated, info

    def _handoff(self) -> tuple[Encoded | None, dict[str, Any]]:
        env = self.env
        while not env.done and env.acting_player != self.seat:
            action = self.opponent.act(env.observation(), env.legal)
            env.step(env.legal.index(action))
        if env.done:
            return None, {"legal": []}
        return env.encode(), {"legal": env.legal}
