"""Local browser sim: play against an agent.

    python3 sim.py                                  # you vs RandomAgent on http://localhost:8765
    python3 sim.py --agent my_module:MyAgent        # any class with .act(observation, legal)
    python3 sim.py --seat 1 --port 9000

Stdlib only. The server is single-threaded and the agent moves synchronously
right after yours, so there is no locking.
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from game import ActionKind, Agent, Game, RandomAgent, load_demo_decks

STATIC = Path(__file__).parent / "sim"


def load_agent(spec: str) -> Agent:
    if spec == "random":
        return RandomAgent()
    module, _, cls = spec.partition(":")
    return getattr(importlib.import_module(module), cls)()


class Session:
    def __init__(self, agent_spec: str, human_seat: int):
        self.agent_spec = agent_spec
        self.human_seat = human_seat
        self.new_game()

    def new_game(self, seed: int | None = None) -> None:
        self.seed = random.randrange(1 << 30) if seed is None else seed
        self.agent = load_agent(self.agent_spec)
        names = ["You", f"Agent ({self.agent.name})"]
        if self.human_seat == 1:
            names.reverse()
        self.game = Game(list(load_demo_decks()), names, seed=self.seed, allow_concede=True)
        self.run_agent()

    def run_agent(self) -> None:
        g = self.game
        while not g.is_over and g.acting_player != self.human_seat:
            seat = g.acting_player
            # Concede is only for the human: agents are trained without it (Game's
            # default), so a policy would score an action it has never seen.
            legal = [a for a in g.legal_actions(seat) if a.kind is not ActionKind.CONCEDE]
            g.step(self.agent.act(g.observation(seat), legal))

    def act(self, index: int) -> None:
        legal = self.game.legal_actions(self.human_seat)
        if not 0 <= index < len(legal):
            raise ValueError(f"no legal action {index}")
        self.game.step(legal[index])
        self.run_agent()

    def state(self) -> dict[str, Any]:
        g = self.game
        return {
            "seed": self.seed,
            "human_seat": self.human_seat,
            "observation": g.observation(self.human_seat),
            "legal_actions": [a.to_json() for a in g.legal_actions(self.human_seat)],
            "log": g.log,
        }


def make_handler(session: Session) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data: Any, code: int = 200) -> None:
            self._send(code, json.dumps(data).encode(), "application/json")

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self._json(session.state())
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            try:
                if self.path == "/api/action":
                    session.act(int(body["index"]))
                elif self.path == "/api/new":
                    session.new_game(body.get("seed"))
                else:
                    return self._send(404, b"not found", "text/plain")
            except (ValueError, KeyError) as e:
                return self._json({"error": str(e)}, 400)
            self._json(session.state())

        def log_message(self, fmt: str, *args: Any) -> None:
            pass   # keep the terminal quiet

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--agent", default="random", help="'random' or module:Class")
    parser.add_argument("--seat", type=int, default=0, choices=[0, 1], help="your seat")
    args = parser.parse_args()
    session = Session(args.agent, args.seat)
    server = HTTPServer(("127.0.0.1", args.port), make_handler(session))
    print(f"Riftbound sim on http://localhost:{args.port}  (agent: {args.agent})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
