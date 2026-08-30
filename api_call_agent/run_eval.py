"""Evaluate the API agent with the official scorer.

Defaults to a 20-session subset because a full 200-session run is roughly
2,000 turns x 2 model calls. Responses are cached on disk, so re-running after
an unrelated code change is free.

    python -m api_call_agent.run_eval                     # 20 sessions
    python -m api_call_agent.run_eval --limit 200         # the whole public set
    python -m api_call_agent.run_eval --compare           # vs the rule-based agent
    python -m api_call_agent.run_eval --no-rerank         # stage 1 only
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402

from api_call_agent.agent import Agent as ApiAgent  # noqa: E402
from api_call_agent.llm_client import DEFAULT_MODEL, ClaudeClient, api_key  # noqa: E402
from api_call_agent.rerank import MODEL_WEIGHT  # noqa: E402

CORE = ("hit_rate_at_10", "mrr", "mttc", "efficiency", "recommended_technical_score")


def summarise(name: str, result: dict) -> None:
    print(f"\n=== {name} ===")
    for key in CORE:
        print(f"  {key:<28} {result[key]}")
    print("  per scenario:")
    for scenario, metrics in result["scenario_metrics"].items():
        print(
            f"    {scenario:<16} n={metrics['sample_count']:<4} "
            f"HR={metrics['hit_rate_at_10']:<8} MRR={round(metrics['mrr'], 4):<8} "
            f"MTTC={metrics['mttc']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Haiku-in-the-loop agent")
    parser.add_argument("--catalog", default=str(ROOT / "data" / "catalog.jsonl"))
    parser.add_argument("--dataset", default=str(ROOT / "data" / "public_set.jsonl"))
    parser.add_argument("--output", default=str(Path(__file__).parent / "results_api.json"))
    parser.add_argument("--limit", type=int, default=20, help="sessions to run (default 20)")
    parser.add_argument("--seed", type=int, default=7, help="sampling seed for the subset")
    parser.add_argument("--scenario", help="restrict to one scenario_type")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--candidates", type=int, default=30, help="shortlist size sent to the model")
    parser.add_argument("--no-cache", action="store_true", help="ignore the on-disk response cache")
    parser.add_argument("--no-rephrase", action="store_true", help="skip stage 1")
    parser.add_argument("--no-rerank", action="store_true", help="skip stage 3")
    parser.add_argument("--no-reply", action="store_true", help="skip stage 4 (template wording)")
    parser.add_argument(
        "--reply-tokens", type=int, default=70,
        help="output cap for the customer-facing sentence (default 70)",
    )
    parser.add_argument(
        "--model-weight", type=float, default=None,
        help="how loud stage 3's vote is against retrieval (default 0.2; 0 disables it)",
    )
    parser.add_argument("--compare", action="store_true", help="also score the rule-based agent on the same sessions")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not api_key():
        print(
            "WARNING: ANTHROPIC_API_KEY is not set - every model call will fail and\n"
            "         the agent will fall back to deterministic ranking.\n"
        )

    samples = load_jsonl(args.dataset)
    if args.scenario:
        samples = [s for s in samples if s["scenario_type"] == args.scenario]
    if args.limit and args.limit < len(samples):
        samples = random.Random(args.seed).sample(samples, args.limit)

    client = ClaudeClient(model=args.model, use_cache=not args.no_cache, verbose=args.verbose)
    print(f"sessions: {len(samples)}   model: {client.model}   shortlist: {args.candidates}"
          f"   reply cap: {'off' if args.no_reply else args.reply_tokens}")

    print("building catalog index...", flush=True)
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = ApiAgent(
        args.catalog,
        client=client,
        use_rephrase=not args.no_rephrase,
        use_rerank=not args.no_rerank,
        use_reply=not args.no_reply,
        candidate_pool=args.candidates,
        reply_tokens=args.reply_tokens,
        model_weight=MODEL_WEIGHT if args.model_weight is None else args.model_weight,
        verbose=args.verbose,
    )
    print("running (this makes live API calls)...", flush=True)
    result = evaluate(agent, samples, catalog_ids, categories, products)
    usage = agent.usage_report()
    result["llm_usage"] = usage

    summarise(f"api_call_agent ({client.model})", result)
    print("\n  LLM usage:")
    for key, value in usage.items():
        if key != "stages":
            print(f"    {key:<20} {value}")
    print("")
    print("  stage activity:")
    for key, value in usage["stages"].items():
        print(f"    {key:<28} {value}")
    stages = usage["stages"]
    if stages["rerank_skipped_determined"]:
        print(
            f"    -> {stages['rerank_skipped_determined']} rerank call(s) skipped: "
            "the evidence already fixed the top 10"
        )
    if stages["constraints_rejected_not_verbatim"]:
        print(
            f"    -> {stages['constraints_rejected_not_verbatim']} model constraint(s) demoted to "
            "keywords for not quoting the customer verbatim"
        )

    if args.compare:
        from starter.agent import Agent as RuleAgent

        print("\nrunning the rule-based agent on the same sessions...", flush=True)
        baseline = evaluate(RuleAgent(args.catalog), samples, catalog_ids, categories, products)
        summarise("starter (rule-based, no LLM)", baseline)
        delta = round(
            result["recommended_technical_score"] - baseline["recommended_technical_score"], 6
        )
        print(f"\n  TechnicalScore delta (api - rules): {delta:+}")
        result["comparison_rule_based"] = {key: baseline[key] for key in CORE}

    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
