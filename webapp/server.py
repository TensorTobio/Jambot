"""Local browser front end for manually trying the agent.

Manual testing convenience only - not part of the official evaluator or the
submission. Wraps the same starter.agent.Agent used by the evaluator behind
a tiny JSON API and serves a static chat page.

Usage:
    python3 webapp/server.py
    python3 webapp/server.py --port 8787 --catalog data/catalog.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from starter.agent import Agent  # noqa: E402

MAX_TURNS = 10


def load_catalog_display(catalog_path: Path) -> dict[str, dict]:
    display: dict[str, dict] = {}
    with catalog_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            asin = str(row["parent_asin"])
            display[asin] = {
                "title": str(row.get("title") or ""),
                "price": row.get("price"),
                "store": str(row.get("store") or ""),
                "average_rating": row.get("average_rating"),
                "rating_number": row.get("rating_number"),
            }
    return display


def load_samples(dataset_path: Path) -> list[dict]:
    # public_set.jsonl ships ground_truth.parent_asin openly - only the
    # organizer's private 800 sessions are actually hidden - so this is not
    # a leak, just using data already meant for local development/testing.
    samples = []
    with dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            samples.append({
                "sample_id": row.get("sample_id"),
                "scenario_type": row.get("scenario_type"),
                "difficulty_bucket": row.get("difficulty_bucket"),
                "user_profile": row.get("user_profile"),
                "target_asin": str((row.get("ground_truth") or {}).get("parent_asin", "")),
            })
    return samples


DEFAULT_PROFILE = {
    "purchase_frequency": "3-4 prior purchases",
    "average_prior_rating": 4.5,
    "rating_style": "usually positive",
    "preference_tags": ["comfort", "fit"],
    "summary": "Prior purchases emphasize comfort and fit; ratings are usually positive.",
}


class AppState:
    def __init__(self, catalog_path: Path, dataset_path: Path) -> None:
        print("Building catalog index (a few seconds)...", flush=True)
        self.agent = Agent(str(catalog_path))
        self.display = load_catalog_display(catalog_path)
        self.samples = load_samples(dataset_path)
        self.samples_by_id = {s["sample_id"]: s for s in self.samples}
        self.sessions: dict[str, dict] = {}
        print("Ready.", flush=True)

    def new_session(self, sample_id: str | None) -> dict:
        session_id = uuid.uuid4().hex[:12]
        target_asin = None
        if sample_id and sample_id in self.samples_by_id:
            sample = self.samples_by_id[sample_id]
            profile = sample["user_profile"]
            target_asin = sample.get("target_asin") or None
        else:
            profile = DEFAULT_PROFILE
            sample_id = None
        self.agent.reset(session_id, profile)
        self.sessions[session_id] = {"turn": 0, "target_asin": target_asin, "hit": False}
        target = None
        if target_asin:
            info = self.display.get(target_asin, {})
            target = {"parent_asin": target_asin, "title": info.get("title", "")}
        return {
            "session_id": session_id,
            "profile": profile,
            "sample_id": sample_id,
            "target": target,
        }

    def send_message(self, session_id: str, message: str) -> dict:
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(f"unknown session_id {session_id!r}")
        if session["turn"] >= MAX_TURNS:
            return {"error": "session already reached turn 10"}
        session["turn"] += 1
        turn = session["turn"]
        response = self.agent.respond(session_id, message, turn, 10)
        recommendations = []
        target_asin = session.get("target_asin")
        hit_rank = None
        for rank, rec in enumerate(response.get("recommendations") or [], start=1):
            asin = str(rec.get("parent_asin", ""))
            info = self.display.get(asin, {})
            recommendations.append({
                "parent_asin": asin,
                "title": info.get("title", ""),
                "price": info.get("price"),
                "store": info.get("store", ""),
                "average_rating": info.get("average_rating"),
                "rating_number": info.get("rating_number"),
                "is_target": bool(target_asin) and asin == target_asin,
            })
            if target_asin and asin == target_asin and hit_rank is None:
                hit_rank = rank
        # Matches the real scoring rule exactly: first turn the target
        # appears anywhere in that turn's (up to 10) recommendations.
        already_hit = session.get("hit", False)
        if hit_rank is not None and not already_hit:
            session["hit"] = True
        return {
            "turn": turn,
            "max_turns": MAX_TURNS,
            "message": response.get("message", ""),
            "ask_attribute": response.get("ask_attribute"),
            "recommendations": recommendations,
            "ended": turn >= MAX_TURNS,
            "hit": hit_rank is not None and not already_hit,
            "hit_rank": hit_rank if not already_hit else None,
            "has_target": target_asin is not None,
        }


STATE: AppState | None = None
INDEX_HTML: bytes = b""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # quieter logs
        pass

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw or b"{}")

    def do_GET(self) -> None:  # noqa: N802
        assert STATE is not None
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(INDEX_HTML)))
            self.end_headers()
            self.wfile.write(INDEX_HTML)
            return
        if self.path == "/api/samples":
            self._send_json({"samples": STATE.samples})
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        assert STATE is not None
        try:
            if self.path == "/api/session":
                body = self._read_json()
                self._send_json(STATE.new_session(body.get("sample_id")))
                return
            if self.path == "/api/message":
                body = self._read_json()
                session_id = str(body.get("session_id", ""))
                message = str(body.get("message", ""))
                if not session_id or not message:
                    self._send_json({"error": "session_id and message are required"}, 400)
                    return
                self._send_json(STATE.send_message(session_id, message))
                return
        except KeyError as exc:
            self._send_json({"error": str(exc)}, 404)
            return
        except Exception as exc:  # defensive: never crash the demo server on bad input
            self._send_json({"error": str(exc)}, 500)
            return
        self.send_response(404)
        self.end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description="Local browser front end for the agent")
    parser.add_argument("--catalog", default=str(ROOT / "data" / "catalog.jsonl"))
    parser.add_argument("--dataset", default=str(ROOT / "data" / "public_set.jsonl"))
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    global STATE, INDEX_HTML
    INDEX_HTML = (Path(__file__).resolve().parent / "index.html").read_bytes()
    STATE = AppState(Path(args.catalog), Path(args.dataset))

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Serving on http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
