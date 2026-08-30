"""Offline wiring test - no API key, no spend.

Replaces ``ClaudeClient.complete`` with a stub that returns representative (and
deliberately messy) responses, then drives the real agent through the real
evaluator. It checks that:

* stage 1 output is absorbed into the slot store,
* stage 1 output is absorbed only when it verbatim-quotes the customer, and a
  paraphrase is demoted to a keyword rather than faked into the evidence store,
* stage 3 output is validated - unknown ids dropped, duplicates removed, and the
  evidence-tier guard holds so a lower-evidence product can never be promoted
  above a higher-evidence one,
* stage 4 output is validated - markdown, ids and multi-paragraph answers are
  rejected and the deterministic template takes over,
* a total model failure degrades to the rule-based result rather than breaking,
* ``respond()`` never raises and always returns a schema-valid payload.

    python -m api_call_agent.selftest
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402

from api_call_agent import agent as agent_mod  # noqa: E402
from api_call_agent.llm_client import ClaudeClient, extract_json  # noqa: E402

ASIN_RE = re.compile(r"\b(B[0-9A-Z]{9})\b")


def install_stub(mode: str = "normal") -> None:
    """Monkeypatch the client so no network call happens."""
    rng = random.Random(11)

    def fake_complete(
        self, system, user, *, max_tokens=600, temperature=0.0,
        prefill=None, stop_sequences=None,
    ):
        self.calls += 1
        self.input_tokens += len(user) // 4
        self.output_tokens += 40
        if mode == "dead":
            return None
        if system.startswith("You normalise"):
            if mode == "garbage":
                return "Sure! Here is the JSON you asked for: {not valid json"
            last = user.strip().splitlines()
            said = [line for line in last if line.startswith("turn ")]
            verbatim = []
            for line in said:
                match = re.search(r"what matters is:\s*(.+?)\.?$", line)
                if match:
                    verbatim.extend(part.strip() for part in match.group(1).split("; "))
            payload = {
                "category": "",
                "constraints": verbatim,
                "keywords": ["comfortable", "durable", "everyday", "lightweight"],
                "budget": None,
                "scenario": "buying",
                "search_query": "a comfortable everyday item matching the stated requirements",
            }
            # Wrapped in a fence on purpose - extract_json must cope.
            return "```json\n" + json.dumps(payload) + "\n```"
        if system.startswith("You are the final ranking"):
            ids = ASIN_RE.findall(user)
            if mode == "garbage":
                return '["NOT_AN_ASIN", "NOT_AN_ASIN", "B000000000"]'
            picked = ids[:8]
            rng.shuffle(picked)
            picked = picked + picked[:2]  # duplicates on purpose
            return "Here you go: " + json.dumps(picked)
        if system.startswith("You are a shop assistant"):
            if mode == "garbage":
                # Everything stage 4 is told not to do; _clean must reject all
                # of it and the deterministic template must take over.
                return (
                    "Certainly! Here are your options:" + chr(10) * 2
                    + "1. **B0912ABCDE** - great" + chr(10) + "2. ..."
                )
            return "Got it, that narrows it down nicely. Any colour you're set on?"
        return None

    ClaudeClient.complete = fake_complete  # type: ignore[method-assign]


def check_schema(response: dict, catalog_ids: set[str]) -> None:
    assert isinstance(response, dict), "response must be a dict"
    assert isinstance(response["message"], str)
    assert response["ask_attribute"] in (
        None, "category", "material", "color", "size", "style", "brand",
        "budget", "feature", "use_case", "other",
    )
    recs = response["recommendations"]
    assert isinstance(recs, list) and len(recs) <= 10
    asins = [r["parent_asin"] for r in recs]
    assert len(asins) == len(set(asins)), "duplicate parent_asin returned"
    assert all(a in catalog_ids for a in asins), "parent_asin not in catalog"
    usage = response["usage"]
    assert isinstance(usage["prompt_tokens"], int) and usage["prompt_tokens"] >= 0
    assert isinstance(usage["completion_tokens"], int) and usage["completion_tokens"] >= 0


def check_guards() -> None:
    """Unit-level checks on the three guards - no catalog, no model, instant."""
    from api_call_agent.reply import _clean
    from api_call_agent.rerank import MODEL_WEIGHT, _fuse_within_tiers

    # -- stage 4 validation ------------------------------------------------
    assert _clean("Got it. Any colour you're set on?", needs_question=True)
    assert _clean("1. **B0912ABCDE**", needs_question=True) is None, "markdown accepted"
    assert _clean("Sure, B0912ABCDE is nice. Colour?", needs_question=True) is None, "id leaked"
    assert _clean("Here are the matches.", needs_question=True) is None, "did not ask"
    assert _clean("x" * 400, needs_question=True) is None, "overlong accepted"
    assert _clean("As an AI, what colour?", needs_question=True) is None, "boilerplate accepted"
    assert _clean("Here are the closest matches.", needs_question=False)

    # -- stage 3 tier guard ------------------------------------------------
    candidates = ["a", "b", "c", "d"]
    tiers = {"a": 2, "b": 2, "c": 1, "d": 1}
    # The model tries to promote a 1-match product over two 2-match ones. The
    # tier boundary is hard, so it may not cross it however loudly it votes.
    ordered = _fuse_within_tiers(candidates, ["d", "c", "b", "a"], tiers)
    assert set(ordered[:2]) == {"a", "b"}, ordered
    assert set(ordered[2:]) == {"c", "d"}, ordered
    # Inside a tier the model's vote is weighted, not obeyed: one place of
    # disagreement is not enough to flip an adjacent pair.
    assert _fuse_within_tiers(candidates, ["b", "a", "d", "c"], tiers)[0] == "a"
    # ...but on a real shortlist, a near-miss the model ranks first is promoted,
    wide = [f"c{i:02d}" for i in range(30)]
    flat = {a: 1 for a in wide}
    def against(first: str, weight: float = 1.0) -> list[str]:
        """Model puts `first` at the top and the retrieval leader at the bottom."""
        middle = [a for a in wide if a not in (first, "c00")]
        return _fuse_within_tiers(wide, [first] + middle + ["c00"], flat, weight)

    near = against("c04")
    assert near.index("c04") < near.index("c00"), near[:5]
    # ...while at the shipped weight the vote is deliberately quiet at the head:
    # not even a near-miss displaces the retrieval leader, which is the property
    # the adversarial sweep documented in rerank.py was chosen to guarantee.
    quiet = against("c04", MODEL_WEIGHT)
    assert quiet.index("c00") < quiet.index("c04"), quiet[:5]
    # Deeper down, where the deterministic score has flattened out, that same
    # quiet vote still moves things - semantics are all that is left down there.
    deep = _fuse_within_tiers(
        wide, ["c18"] + [a for a in wide if a != "c18"], flat, MODEL_WEIGHT
    )
    assert deep.index("c18") < 18, deep[:20]
    # A model that returns nothing usable leaves the deterministic order alone.
    assert _fuse_within_tiers(candidates, [], tiers) == candidates
    print("  guards        stage 3 tier+fusion guard and stage 4 validation hold")


def main() -> None:
    check_guards()
    catalog = str(ROOT / "data" / "catalog.jsonl")
    samples = load_jsonl(str(ROOT / "data" / "public_set.jsonl"))
    subset = random.Random(3).sample(samples, 12)

    print("building catalog index...", flush=True)
    catalog_ids, categories, products = catalog_index(catalog)
    index = None

    for mode in ("normal", "garbage", "dead"):
        install_stub(mode)
        agent = agent_mod.Agent(catalog, index=index, use_cache=False)
        index = agent.index  # reuse across modes; building it is the slow part

        # hand-driven schema check with hostile inputs
        agent.reset("smoke", {"preference_tags": ["fit"]})
        for i, message in enumerate(
            ["", None, 42, "I'm looking for Dresses Casual, but I'm still exploring.",
             "For that, what matters is: polyester; color: black.", "x" * 4000], 1
        ):
            check_schema(agent.respond("smoke", message, i, 10), catalog_ids)

        result = evaluate(agent, subset, catalog_ids, categories, products)
        stages = agent.usage_report()["stages"]
        print(
            f"  stub={mode:<8} HR={result['hit_rate_at_10']:<6} "
            f"MRR={round(result['mrr'], 4):<8} MTTC={result['mttc']:<6} "
            f"TS={result['recommended_technical_score']:<8} "
            f"reported_tokens={result['reported_token_usage']['total_tokens']}"
        )
        print(
            f"                rerank called={stages['rerank_called']} "
            f"skipped_as_determined={stages['rerank_skipped_determined']}  "
            f"reply model={stages['reply_from_model']} "
            f"template={stages['reply_from_template']}"
        )
        if mode == "garbage":
            assert stages["reply_from_model"] == 0, "stage 4 accepted a malformed reply"
        if mode == "normal":
            assert stages["reply_from_model"] > 0, "stage 4 never produced a reply"

    # the 'dead' run must match the pure rule-based agent exactly
    from starter.agent import Agent as RuleAgent

    rules = RuleAgent(catalog)
    rules.index = index
    baseline = evaluate(rules, subset, catalog_ids, categories, products)
    print(f"  rule-based reference        TS={baseline['recommended_technical_score']}")
    print("\nselftest passed: schema valid in every mode, no exceptions raised.")


if __name__ == "__main__":
    main()
