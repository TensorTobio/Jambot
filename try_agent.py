"""Hands-on driver for the agent - for demos and eyeballing behaviour.

Three modes:

    python try_agent.py                     # chat with the agent yourself
    python try_agent.py --sample public_0001  # replay one labelled session
    python try_agent.py --sample random --n 5 # replay 5 random sessions

Add ``--api`` to drive the Haiku-in-the-loop agent from ``api_call_agent/``
instead of the rule-based one. That is the mode to record for the demo: same
scored answers, but the ``agent:`` lines are written by the model rather than
by a template. It makes live API calls and costs a fraction of a cent a session.

The replay mode drives the agent with the *real* simulator functions imported
from evaluator/local_evaluator.py, so a transcript here is exactly what the
scorer saw. Nothing in this file is imported by the agent; it is a tool.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)
from starter.agent import Agent  # noqa: E402

CATALOG = str(ROOT / "data" / "catalog.jsonl")
DATASET = str(ROOT / "data" / "public_set.jsonl")


def short(text: str, width: int = 78) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 1] + "…"


def show(recommendations: list[dict], products: dict, target: str | None, limit: int = 5) -> None:
    if not recommendations:
        print("      (holding back - not confident enough yet)")
        return
    for rank, item in enumerate(recommendations[:limit], 1):
        asin = item["parent_asin"]
        product = products.get(asin, {})
        mark = "  <== TARGET" if target and asin == target else ""
        price = product.get("price")
        price_text = f" ${price}" if price not in (None, "") else ""
        print(f"      {rank:>2}. {asin}{price_text}  {short(product.get('title', ''), 60)}{mark}")
    if len(recommendations) > limit:
        print(f"          ... {len(recommendations) - limit} more")


def replay(agent: Agent, sample: dict, catalog_ids, categories, products) -> None:
    """Drive one labelled session exactly the way the evaluator does."""
    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}

    print("=" * 92)
    print(f"{sample['sample_id']}  [{sample['scenario_type']} / {sample.get('difficulty_bucket', '?')}]")
    print(f"target : {target}  {short(products[target].get('title', ''), 60)}")
    print(f"hidden : hard={card['hard_constraints']}")
    print(f"         soft={card['soft_preferences']}")
    print(f"profile: {sample['user_profile'].get('summary', '')}")
    print("-" * 92)

    session_id = f"try_{sample['sample_id']}"
    agent.reset(session_id, sample["user_profile"])
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    message = initial_message(effective, coarse_category(categories.get(target, [])), disclosed)

    for turn in range(1, MAX_TURNS + 1):
        print(f"  turn {turn}")
        print(f"    user : {short(message)}")
        response = agent.respond(session_id, message, turn, TOP_K)
        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
        print(f"    agent: {short(response.get('message', ''))}   [ask_attribute={response.get('ask_attribute')!r}]")
        show([{"parent_asin": a} for a in ranked], products, target)

        if override_applied and target in ranked:
            rank = ranked.index(target) + 1
            print("-" * 92)
            print(f"  HIT on turn {turn} at rank {rank}   (reciprocal rank {1 / rank:.3f})")
            return
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
    print("-" * 92)
    print("  MISS (target never entered the top 10)")


def chat(agent: Agent, products: dict) -> None:
    print("\nType as the customer. Blank line or 'quit' to exit.")
    print("Try:  I'm looking for Dresses Casual, but I'm still exploring.")
    print("then: For that, what matters is: polyester; color: black.\n")
    session_id = "manual"
    agent.reset(session_id, {"preference_tags": ["fit", "comfort", "durability"]})
    turn = 1
    while turn <= MAX_TURNS:
        try:
            message = input(f"you ({turn}/10) > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not message or message.lower() in {"quit", "exit"}:
            break
        response = agent.respond(session_id, message, turn, TOP_K)
        print(f"    agent: {response['message']}   [ask_attribute={response['ask_attribute']!r}]")
        show(response["recommendations"], products, None)
        print()
        turn += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Try the TechJam agent by hand")
    parser.add_argument("--sample", help="a sample_id from public_set.jsonl, or 'random'")
    parser.add_argument("--n", type=int, default=1, help="how many random samples to replay")
    parser.add_argument("--scenario", help="filter random replays: buying/browsing/intent_override/boundary")
    parser.add_argument("--catalog", default=CATALOG)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument(
        "--api", action="store_true",
        help="use the Haiku-in-the-loop agent (live API calls) instead of the rule-based one",
    )
    args = parser.parse_args()

    print("building catalog index (about 20s)...", flush=True)
    catalog_ids, categories, products = catalog_index(args.catalog)
    if args.api:
        from api_call_agent.agent import Agent as ApiAgent

        agent = ApiAgent(args.catalog)
        print(f"agent: api_call_agent ({agent.client.model})", flush=True)
    else:
        agent = Agent(args.catalog)
    print(f"ready: {len(catalog_ids)} products\n", flush=True)

    if not args.sample:
        chat(agent, products)
        return

    samples = load_jsonl(args.dataset)
    if args.scenario:
        samples = [s for s in samples if s["scenario_type"] == args.scenario]
    if args.sample == "random":
        chosen = random.sample(samples, min(args.n, len(samples)))
    else:
        chosen = [s for s in samples if s["sample_id"] == args.sample]
        if not chosen:
            print(f"no sample with id {args.sample!r}")
            return
    for sample in chosen:
        replay(agent, sample, catalog_ids, categories, products)
        print()


if __name__ == "__main__":
    main()
