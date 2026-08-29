"""Ablation: ask-policy x always-recommend, scored by the official evaluator.

Answers two questions with numbers instead of intuition:

1. Does asking a *different, distinguishing* question each turn beat asking
   "other" every turn?
2. Does printing a top-10 after every question beat holding one back until the
   evidence is sharp?

Builds the catalog index once and reuses it across all runs.

    python sweep_policy.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from starter.agent import Agent  # noqa: E402
from starter.retrieval import CatalogIndex  # noqa: E402

CATALOG = str(ROOT / "data" / "catalog.jsonl")
DATASET = str(ROOT / "data" / "public_set.jsonl")

samples = load_jsonl(DATASET)
catalog_ids, categories, products = catalog_index(CATALOG)
print("building index once...", flush=True)
index = CatalogIndex(CATALOG)
print(f"index ready: {len(index.asins)} products\n", flush=True)

CONFIGS = [
    ("other  + hold back  (shipped)", "other", False),
    ("other  + always show", "other", True),
    ("split  + hold back", "split", False),
    ("split  + always show", "split", True),
    ("rotate + hold back", "rotate", False),
    ("rotate + always show", "rotate", True),
]

header = f"{'config':<32} {'HR@10':>7} {'MRR':>8} {'MTTC':>7} {'Eff':>7} {'TS':>8}"
print(header)
print("-" * len(header))

rows = []
for label, policy, always in CONFIGS:
    agent = Agent(CATALOG, index=index, ask_policy=policy, always_recommend=always)
    result = evaluate(agent, samples, catalog_ids, categories, products)
    rows.append({"config": label, "ask_policy": policy, "always_recommend": always,
                 "HR": result["hit_rate_at_10"], "MRR": result["mrr"],
                 "MTTC": result["mttc"], "Eff": result["efficiency"],
                 "TS": result["recommended_technical_score"],
                 "scenario_metrics": result["scenario_metrics"]})
    print(f"{label:<32} {result['hit_rate_at_10']:>7.4f} {result['mrr']:>8.4f} "
          f"{result['mttc']:>7.3f} {result['efficiency']:>7.4f} "
          f"{result['recommended_technical_score']:>8.4f}", flush=True)

best = max(rows, key=lambda r: r["TS"])
print(f"\nbest: {best['config']}  TS={best['TS']:.4f}")
Path(ROOT / "results_policy_ablation.json").write_text(
    json.dumps(rows, indent=2) + "\n", encoding="utf-8"
)
print("wrote results_policy_ablation.json")
