"""Local browser front end for trying the agents by hand.

Manual testing and demo convenience only - not part of the official evaluator or
the submission. It wraps the same ``Agent`` classes the evaluator drives behind
a small JSON API and serves a static chat page.

Three agents are selectable at runtime, all behind the identical contract:

    rules    starter.agent            constraint + category + BM25, deterministic
    llm      api_call_agent.agent     Claude Haiku 4.5 rephrase -> retrieve -> rerank
    keyword  starter.agent_keyword    generic NLP/FTS5 agent (earlier variant)

Agents are constructed lazily on first use, and ``rules``/``llm`` share one
``CatalogIndex`` so switching between them costs nothing.

Usage:
    python webapp/server.py
    python webapp/server.py --port 8787 --agent llm
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    MAX_TURNS,
    TOP_K,
    coarse_category,
    customer_reply,
    initial_message,
    materialize_hidden_fields,
    normalize_recommendations,
)

DEFAULT_PROFILE = {
    "purchase_frequency": "3-4 prior purchases",
    "average_prior_rating": 4.5,
    "rating_style": "usually positive",
    "preference_tags": ["comfort", "fit"],
    "summary": "Prior purchases emphasize comfort and fit; ratings are usually positive.",
}

AGENT_INFO = {
    "rules": {
        "label": "Rule-based (starter)",
        "detail": "Constraint reverse-index + category filter + BM25. Deterministic, no API. TechnicalScore 0.953 on the public set.",
    },
    "llm": {
        "label": "Claude Haiku 4.5 (api_call_agent)",
        "detail": "Haiku rephrases the turn, retrieval shortlists 30, Haiku reranks to the top 10. Needs ANTHROPIC_API_KEY.",
    },
    "keyword": {
        "label": "Keyword/NLP (earlier variant)",
        "detail": "Generic phrase extraction over FTS5, no exact-constraint route. TechnicalScore 0.680 on the public set.",
    },
}


def load_catalog(catalog_path: Path) -> tuple[dict[str, dict], dict[str, list[str]], dict[str, dict]]:
    """One pass: card display data, category paths, and the raw products.

    ``products`` is what the evaluator's ``materialize_hidden_fields`` needs to
    rebuild a session's hidden intent card for auto-play.
    """
    display: dict[str, dict] = {}
    categories: dict[str, list[str]] = {}
    products: dict[str, dict] = {}
    with catalog_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            asin = str(row["parent_asin"])
            display[asin] = {
                "title": str(row.get("title") or ""),
                "price": row.get("price"),
                "store": str(row.get("store") or ""),
                "average_rating": row.get("average_rating"),
                "rating_number": row.get("rating_number"),
            }
            categories[asin] = [str(v) for v in (row.get("categories") or [])]
            products[asin] = row
    return display, categories, products


def _reset_options(options: dict) -> dict:
    """Demo overrides accepted from the browser, whitelisted and normalised.

    These change how the agent *behaves in this demo*, never how it is scored -
    the evaluator calls ``reset()`` with two positional arguments and never
    reaches this path. Measured cost of each is in the CHANGELOG.
    """
    policy = str(options.get("ask_policy") or "other").lower()
    if policy not in ("other", "split", "rotate"):
        policy = "other"
    return {
        "ask_policy": policy,
        "always_recommend": bool(options.get("always_recommend")),
    }


def load_samples(dataset_path: Path) -> list[dict]:
    # public_set.jsonl ships ground_truth.parent_asin openly - only the
    # organizer's private 800 sessions are actually hidden - so this is not a
    # leak, just using data already meant for local development/testing.
    samples: list[dict] = []
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
                "_raw": row,
            })
    return samples


class AppState:
    def __init__(self, catalog_path: Path, dataset_path: Path) -> None:
        self.catalog_path = catalog_path
        print("Loading catalog...", flush=True)
        self.display, self.categories, self.products = load_catalog(catalog_path)
        self.catalog_ids = set(self.display)
        self.samples = load_samples(dataset_path)
        self.samples_by_id = {s["sample_id"]: s for s in self.samples}

        self._agents: dict[str, object] = {}
        self._index = None            # shared CatalogIndex for rules + llm
        self._lock = threading.Lock()
        self.sessions: dict[str, dict] = {}
        print(f"Ready. {len(self.catalog_ids)} products.", flush=True)

    # -- agents -----------------------------------------------------------
    def available_agents(self) -> list[dict]:
        from api_call_agent.llm_client import api_key

        out = []
        for key, info in AGENT_INFO.items():
            entry = {"id": key, **info, "available": True, "loaded": key in self._agents}
            if key == "llm" and not api_key():
                entry["available"] = False
                entry["detail"] += "  (ANTHROPIC_API_KEY is not set - it would fall back to the rule-based path.)"
            out.append(entry)
        return out

    def agent(self, name: str):
        name = name if name in AGENT_INFO else "rules"
        with self._lock:
            if name in self._agents:
                return self._agents[name]
            print(f"Building agent {name!r} (first use, ~20s)...", flush=True)
            if name == "rules":
                from starter.agent import Agent as RuleAgent

                agent = RuleAgent(str(self.catalog_path))
                self._index = agent.index
            elif name == "llm":
                from api_call_agent.agent import Agent as ApiAgent

                if self._index is None:
                    from starter.retrieval import CatalogIndex

                    self._index = CatalogIndex(self.catalog_path)
                agent = ApiAgent(str(self.catalog_path), index=self._index)
            else:
                from starter.agent_keyword import Agent as KeywordAgent

                agent = KeywordAgent(str(self.catalog_path))
            self._agents[name] = agent
            print(f"Agent {name!r} ready.", flush=True)
            return agent

    def usage(self) -> dict | None:
        agent = self._agents.get("llm")
        if agent is None:
            return None
        try:
            return agent.usage_report()
        except Exception:
            return None

    # -- presentation -----------------------------------------------------
    def decorate(self, asins: list[str], target_asin: str | None) -> list[dict]:
        cards = []
        for asin in asins:
            info = self.display.get(asin, {})
            cards.append({
                "parent_asin": asin,
                "title": info.get("title", ""),
                "price": info.get("price"),
                "store": info.get("store", ""),
                "average_rating": info.get("average_rating"),
                "rating_number": info.get("rating_number"),
                "is_target": bool(target_asin) and asin == target_asin,
            })
        return cards

    # -- manual chat ------------------------------------------------------
    def new_session(self, sample_id: str | None, agent_name: str, options: dict | None = None) -> dict:
        session_id = uuid.uuid4().hex[:12]
        target_asin = None
        if sample_id and sample_id in self.samples_by_id:
            sample = self.samples_by_id[sample_id]
            profile = sample["user_profile"]
            target_asin = sample.get("target_asin") or None
        else:
            profile = DEFAULT_PROFILE
            sample_id = None
        agent_name = agent_name if agent_name in AGENT_INFO else "rules"
        options = options or {}
        self.agent(agent_name).reset(session_id, profile, **_reset_options(options))
        self.sessions[session_id] = {
            "turn": 0, "target_asin": target_asin, "hit": False, "agent": agent_name,
        }
        target = None
        if target_asin:
            info = self.display.get(target_asin, {})
            target = {"parent_asin": target_asin, "title": info.get("title", "")}
        return {
            "session_id": session_id,
            "profile": profile,
            "sample_id": sample_id,
            "agent": agent_name,
            "agent_label": AGENT_INFO[agent_name]["label"],
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
        agent = self.agent(session["agent"])
        response = agent.respond(session_id, message, turn, TOP_K)

        ranked = normalize_recommendations(response.get("recommendations"), self.catalog_ids)
        target_asin = session.get("target_asin")
        hit_rank = ranked.index(target_asin) + 1 if target_asin and target_asin in ranked else None
        already_hit = session.get("hit", False)
        if hit_rank is not None and not already_hit:
            session["hit"] = True

        return {
            "turn": turn,
            "max_turns": MAX_TURNS,
            "message": response.get("message", ""),
            "ask_attribute": response.get("ask_attribute"),
            "recommendations": self.decorate(ranked, target_asin),
            "withheld": not ranked,
            "usage": response.get("usage"),
            "llm_usage": self.usage() if session["agent"] == "llm" else None,
            "ended": turn >= MAX_TURNS,
            "hit": hit_rank is not None and not already_hit,
            "hit_rank": hit_rank if not already_hit else None,
            "has_target": target_asin is not None,
        }

    # -- auto-play (the evaluator's own simulated customer) ---------------
    def autoplay(self, sample_id: str, agent_name: str, options: dict | None = None) -> dict:
        """Run a full scored session, driven by the simulator, and return the transcript.

        This uses the evaluator's own ``initial_message`` / ``customer_reply`` /
        override logic, so what the browser shows is exactly what the scorer saw.
        """
        entry = self.samples_by_id.get(sample_id)
        if entry is None:
            raise KeyError(f"unknown sample_id {sample_id!r}")
        sample = entry["_raw"]
        agent_name = agent_name if agent_name in AGENT_INFO else "rules"
        agent = self.agent(agent_name)

        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, self.products)
        effective = {**sample, "intent_card": card, "behavior": behavior}

        session_id = f"autoplay_{uuid.uuid4().hex[:8]}"
        agent.reset(session_id, sample["user_profile"], **_reset_options(options or {}))
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        message = initial_message(effective, coarse_category(self.categories.get(target, [])), disclosed)

        turns: list[dict] = []
        hit_turn: int | None = None
        hit_rank: int | None = None
        for turn in range(1, MAX_TURNS + 1):
            response = agent.respond(session_id, message, turn, TOP_K)
            ranked = normalize_recommendations(response.get("recommendations"), self.catalog_ids)
            counted = override_applied and target in ranked
            turns.append({
                "turn": turn,
                "user": message,
                "message": response.get("message", ""),
                "ask_attribute": response.get("ask_attribute"),
                "recommendations": self.decorate(ranked, target),
                "withheld": not ranked,
                "scored_hit": counted,
                "override_pending": not override_applied,
            })
            if counted:
                hit_turn = turn
                hit_rank = ranked.index(target) + 1
                break
            if turn == MAX_TURNS:
                break
            override = effective.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                if override.get("new_value"):
                    disclosed.add(str(override["new_value"]))
                message = str(override.get("message", "Actually, please ignore my earlier preference."))
            else:
                message, boundary_used = customer_reply(
                    effective, response.get("ask_attribute"), disclosed, boundary_used
                )

        info = self.display.get(target, {})
        return {
            "sample_id": sample_id,
            "agent": agent_name,
            "agent_label": AGENT_INFO[agent_name]["label"],
            "scenario_type": sample["scenario_type"],
            "difficulty_bucket": sample.get("difficulty_bucket"),
            "profile": sample.get("user_profile") or {},
            "target": {"parent_asin": target, "title": info.get("title", "")},
            "intent_card": card,
            "turns": turns,
            "hit": hit_turn is not None,
            "hit_turn": hit_turn,
            "hit_rank": hit_rank,
            "reciprocal_rank": 0.0 if hit_rank is None else round(1.0 / hit_rank, 4),
            "llm_usage": self.usage() if agent_name == "llm" else None,
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
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(INDEX_HTML)))
            self.end_headers()
            self.wfile.write(INDEX_HTML)
            return
        if self.path == "/api/samples":
            self._send_json({
                "samples": [{k: v for k, v in s.items() if k != "_raw"} for s in STATE.samples]
            })
            return
        if self.path == "/api/agents":
            self._send_json({"agents": STATE.available_agents()})
            return
        if self.path == "/api/usage":
            self._send_json({"llm_usage": STATE.usage()})
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        assert STATE is not None
        try:
            if self.path == "/api/session":
                body = self._read_json()
                self._send_json(STATE.new_session(
                    body.get("sample_id"), str(body.get("agent") or "rules"), body.get("options")
                ))
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
            if self.path == "/api/autoplay":
                body = self._read_json()
                sample_id = str(body.get("sample_id", ""))
                if not sample_id:
                    self._send_json({"error": "sample_id is required"}, 400)
                    return
                self._send_json(STATE.autoplay(
                    sample_id, str(body.get("agent") or "rules"), body.get("options")
                ))
                return
        except KeyError as exc:
            self._send_json({"error": str(exc)}, 404)
            return
        except Exception as exc:  # defensive: never crash the demo server on bad input
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 500)
            return
        self.send_response(404)
        self.end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description="Local browser front end for the agents")
    parser.add_argument("--catalog", default=str(ROOT / "data" / "catalog.jsonl"))
    parser.add_argument("--dataset", default=str(ROOT / "data" / "public_set.jsonl"))
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--agent", default="rules", choices=sorted(AGENT_INFO),
                        help="agent to warm up at startup (all remain selectable in the UI)")
    args = parser.parse_args()

    global STATE, INDEX_HTML
    INDEX_HTML = (Path(__file__).resolve().parent / "index.html").read_bytes()
    STATE = AppState(Path(args.catalog), Path(args.dataset))
    STATE.agent(args.agent)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Serving on http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
