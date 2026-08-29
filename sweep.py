"""Weight/threshold sweep against the official local evaluator.

Builds the catalog index once and re-runs the untouched evaluator for each
configuration, so every number reported here comes from the real scorer.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from starter import agent as agent_mod  # noqa: E402
from starter import retrieval as retrieval_mod  # noqa: E402

CATALOG = "data/catalog.jsonl"
DATASET = "data/public_set.jsonl"

samples = load_jsonl(DATASET)
catalog_ids, categories, products = catalog_index(CATALOG)
shared = agent_mod.Agent(CATALOG)
print(f"index built: {len(shared.index.asins)} products", flush=True)

GRID = {
    "FORCE_RECOMMEND_TURN": [1, 2, 3, 4],
    "CONFIDENT_TIE": [1, 3, 10],
}

rows = []
keys = list(GRID)
for combo in itertools.product(*(GRID[k] for k in keys)):
    for key, value in zip(keys, combo):
        setattr(agent_mod, key, value)
    result = evaluate(shared, samples, catalog_ids, categories, products)
    row = {
        **dict(zip(keys, combo)),
        "HR": result["hit_rate_at_10"],
        "MRR": result["mrr"],
        "MTTC": result["mttc"],
        "TS": result["recommended_technical_score"],
    }
    rows.append(row)
    print(json.dumps(row), flush=True)

rows.sort(key=lambda r: -r["TS"])
print("\nBEST:")
for row in rows[:5]:
    print(json.dumps(row))
