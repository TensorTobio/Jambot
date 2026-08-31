"""Weight/schedule sweep against the official local evaluator.

Builds the catalog index once and re-runs the untouched evaluator for each
configuration, so every number reported here comes from the real scorer.

Two knobs are swept, because they are the two that move the score:

* ``SHOW_SCHEDULE`` - how many products each turn is allowed to show. This
  buys Efficiency and MRR at the same time; see ``starter/agent.py``.
* ``W_POPULARITY`` / ``W_POPULARITY_N`` - the purchase prior that breaks ties
  once the structured routes run out of discriminating power.
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

SCHEDULES = [(10,), (1, 10), (1, 1, 10), (1, 1, 1, 10), (1, 1, 1, 1, 10), (1, 1, 2, 10)]
GRID = {
    "W_POPULARITY": [3.0, 10.0, 20.0],
    "W_POPULARITY_N": [0.0, 20.0, 40.0],
}

rows = []
keys = list(GRID)
for schedule in SCHEDULES:
    agent_mod.SHOW_SCHEDULE = schedule
    for combo in itertools.product(*(GRID[k] for k in keys)):
        for key, value in zip(keys, combo):
            setattr(retrieval_mod, key, value)
        result = evaluate(shared, samples, catalog_ids, categories, products)
        row = {
            "SHOW_SCHEDULE": list(schedule),
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
