"""Interactive REPL for manually trying the agent against the real catalog.

This is a manual testing convenience, not part of the official evaluator.

Usage:
    python3 try_agent.py
    python3 try_agent.py --sample public_0006   # preload a public_set.jsonl session's profile
"""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from starter.agent import Agent

DEFAULT_PROFILE = {
    "purchase_frequency": "3-4 prior purchases",
    "average_prior_rating": 4.5,
    "rating_style": "usually positive",
    "preference_tags": ["comfort", "fit"],
    "summary": "Prior purchases emphasize comfort and fit; ratings are usually positive.",
}


def load_titles(catalog_path: str) -> dict[str, str]:
    titles: dict[str, str] = {}
    with Path(catalog_path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            titles[str(row["parent_asin"])] = str(row.get("title") or "")
    return titles


def find_profile(sample_id: str, dataset_path: str) -> dict:
    with Path(dataset_path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("sample_id") == sample_id:
                return row["user_profile"]
    raise SystemExit(f"sample_id {sample_id!r} not found in {dataset_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manually chat with the agent")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--sample", default=None, help="preload a public_set.jsonl sample_id's user_profile")
    parser.add_argument("--top", type=int, default=10, help="how many recommendations to print per turn")
    args = parser.parse_args()

    print("Loading catalog and building index (a few seconds)...")
    agent = Agent(args.catalog)
    titles = load_titles(args.catalog)

    profile = find_profile(args.sample, args.dataset) if args.sample else DEFAULT_PROFILE
    session_id = f"manual_{uuid.uuid4().hex[:8]}"
    agent.reset(session_id, profile)

    print(f"\nSession {session_id} | profile: {json.dumps(profile)}")
    print(
        "Type your shopping message each turn, or a number to pick that item from the "
        "last list shown (demo-only - the real protocol has no such action, see below). "
        "Type 'quit' to stop.\n"
    )

    last_recs: list[dict] = []
    turn = 1
    while turn <= 10:
        try:
            user_message = input(f"[turn {turn}] you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_message.lower() in {"quit", "exit"}:
            break
        if not user_message:
            continue

        picked = user_message.rstrip(".").strip()
        if picked.isdigit() and last_recs:
            index = int(picked)
            if 1 <= index <= len(last_recs):
                asin = last_recs[index - 1].get("parent_asin", "")
                title = titles.get(asin, "(unknown)")
                print(f"  You picked #{index}: {asin}  {title[:90]}")
                print(
                    "  Demo-only: in real scoring there is no 'select' action - the session\n"
                    "  ends automatically the instant the evaluator's hidden target parent_asin\n"
                    "  (which neither you nor the agent ever sees) appears anywhere in the\n"
                    "  recommendations respond() returns. Nothing you type can confirm or end\n"
                    "  it directly; only a real match against that hidden ID does.\n"
                )
                break
            print(f"  (no item #{index} in the last list - it only had {len(last_recs)})\n")
            continue

        response = agent.respond(session_id, user_message, turn, 10)
        print(f"  agent> {response['message']}")
        if response.get("ask_attribute"):
            print(f"  (asking about: {response['ask_attribute']})")
        last_recs = response.get("recommendations") or []
        for rank, rec in enumerate(last_recs[: args.top], start=1):
            asin = rec.get("parent_asin", "")
            title = titles.get(asin, "(unknown)")
            print(f"    {rank}. {asin}  {title[:90]}")
        print()
        turn += 1
    else:
        print("Reached turn 10 - in real scoring this session would now be scored a miss.")

    print("Session ended.")


if __name__ == "__main__":
    main()
