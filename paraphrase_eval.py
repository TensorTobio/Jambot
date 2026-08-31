"""Robustness harness: how much score survives if the organizer rewords the customer?

``docs/competition_specification.md`` says the simulator's phrasing may be
paraphrased on the private set, and that paraphrasing "cannot decide
correctness" - so the *information* a turn carries is fixed, only the wording
moves. This harness reproduces exactly that: it calls the untouched evaluator's
own ``initial_message`` / ``customer_reply``, then rewrites the sentence they
return. Disclosure logic, the hidden card and the scoring are therefore
bit-identical to the official run; only the surface text differs.

The evaluator file is never modified - the wrappers are installed at runtime on
the imported module, and only inside this script.

    python paraphrase_eval.py             # all levels
    python paraphrase_eval.py --level L2  # one level
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evaluator import local_evaluator as ev  # noqa: E402
from starter.agent import Agent  # noqa: E402
from starter.retrieval import CatalogIndex  # noqa: E402

CATALOG = str(ROOT / "data" / "catalog.jsonl")
DATASET = str(ROOT / "data" / "public_set.jsonl")

# --- the frames, as emitted by evaluator/local_evaluator.py ----------------
RE_BUY = re.compile(r"^I'm looking for (.+?)\. A key requirement is: (.+?)\.$")
RE_BROWSE = re.compile(r"^I'm looking for (.+?), but I'm still exploring\.$")
RE_OPEN_OV = re.compile(r"^I'm looking for (.+?)\. (.+)$")
RE_DISCLOSE = re.compile(r"^For that, what matters is: (.+)\.$")
RE_NO_ADD = re.compile(r"^I don't have an additional preference for (.+?)\.$")
RE_NO_PREF = re.compile(r"^I don't have a preference for (.+?); please use your judgment\.$")
RE_NUDGE = re.compile(r"^Those options are not quite right yet\.")
RE_OVERRIDE = re.compile(r"^Actually, ignore my earlier preference\. What I need is: (.+?)\.$")

# level -> frame -> templates. The {v}/{cat}/{a} slots are filled with the
# verbatim values the simulator produced, so no level invents or destroys
# information - it only changes how the sentence is worded.
BANK = {
    "L1": {  # light: contractions, filler, reordered clauses
        "buy": ("Hi, I need {cat}. One thing that matters: {v}.",
                "I'm after {cat} - it has to be {v}.",
                "Looking for {cat}. Key thing is {v}."),
        "browse": ("I'm after {cat} - just browsing for now.",
                   "Show me some {cat}? Still deciding.",
                   "I want {cat}, though I'm not sure exactly what yet."),
        "open_ov": ("I want {cat}. {v}",
                    "Shopping for {cat} today. {v}",
                    "I need {cat}. {v}"),
        "disclose": ("What matters to me: {v}.",
                     "A couple of things: {v}.",
                     "Here is what counts: {v}."),
        "no_add": ("Nothing else comes to mind for {a}.",
                   "That is all I have on {a}.",
                   "No more thoughts on {a}."),
        "no_pref": ("No strong feelings on {a} - your call.",
                    "I really do not mind about {a}, you pick.",
                    "{a} is up to you honestly."),
        "nudge": ("Not quite there yet. Ask me something specific.",
                  "Hmm, none of those. What do you want to know?"),
        "override": ("Scratch that. What I really need is {v}.",
                     "Forget what I said - it needs to be {v}.",
                     "Change of plan: {v} is the must-have."),
    },
    "L2": {  # heavy: different sentence shapes, chattier, values still verbatim
        "buy": ("hey there! so i'm hunting for {cat} and honestly the big thing for me is {v}. what have you got?",
                "Been meaning to buy {cat} for ages. Non-negotiable: {v}. Ideas?",
                "ok so. {cat}. must be {v}. go."),
        "browse": ("hey! just poking around at {cat}, nothing firm yet, surprise me?",
                   "Window shopping for {cat} really. Not sure what I want.",
                   "so i'm thinking about {cat} maybe? open to suggestions"),
        "open_ov": ("hi! after some {cat}. {v}",
                    "Right, {cat} please. {v}",
                    "im looking at {cat} btw. {v}"),
        "disclose": ("oh yeah - {v}. that's the sort of thing i mean",
                     "Good question. {v}. Does that narrow it down?",
                     "hmm, i'd say {v}"),
        "no_add": ("nope, nothing more on {a} sorry",
                   "Can't think of anything else about {a} to be honest.",
                   "that's me tapped out on {a}"),
        "no_pref": ("honestly {a}? no preference at all, you choose",
                    "Zero opinion on {a}. Whatever you think is best.",
                    "dont care about {a} tbh"),
        "nudge": ("nah, none of those really. ask me something?",
                  "Not seeing it. What else do you need from me?"),
        "override": ("actually hold on - forget that. {v} is what i'm after",
                     "Hmm, I've changed my mind. Make it {v} instead.",
                     "wait no. {v}. that's the one"),
    },
}
# L3 = L2 wording, and the quoted values are case-folded and de-punctuated too,
# so exact string lookup against the catalog stops working.
BANK["L3"] = BANK["L2"]
# L4 = L3, and the category is degraded as well: lower-cased and cut to its last
# word, so "Accessories Belts" arrives as "belts". This is the worst case that
# still respects the specification - the turn carries the same information, it
# is simply not quoting the catalog any more.
BANK["L4"] = BANK["L2"]


def _mangle(value: str) -> str:
    """L3+ only: the kind of damage a rewriter does to a quoted attribute."""
    out = value.lower().replace(":", " ").replace("-", " ")
    return re.sub(r"\s+", " ", out).strip(" .,;")


def _degrade_category(category: str) -> str:
    """L4 only: the customer names the category loosely instead of quoting it."""
    words = str(category).split()
    return (words[-1] if words else str(category)).lower()


def make_paraphraser(level: str, seed: int = 0):
    bank = BANK[level]
    rng = random.Random(seed)
    mangle = _mangle if level in ("L3", "L4") else (lambda v: v)
    degrade = _degrade_category if level == "L4" else (lambda c: c)

    def pick(kind: str) -> str:
        options = bank[kind]
        return options[rng.randrange(len(options))]

    def rewrite(text: str) -> str:
        match = RE_BUY.match(text)
        if match:
            return pick("buy").format(cat=degrade(match.group(1)), v=mangle(match.group(2)))
        match = RE_BROWSE.match(text)
        if match:
            return pick("browse").format(cat=degrade(match.group(1)))
        match = RE_DISCLOSE.match(text)
        if match:
            payload = "; ".join(mangle(part) for part in match.group(1).split("; "))
            return pick("disclose").format(v=payload)
        match = RE_OVERRIDE.match(text)
        if match:
            return pick("override").format(v=mangle(match.group(1)))
        match = RE_NO_PREF.match(text)
        if match:
            return pick("no_pref").format(a=match.group(1))
        match = RE_NO_ADD.match(text)
        if match:
            return pick("no_add").format(a=match.group(1))
        if RE_NUDGE.match(text):
            return pick("nudge")
        match = RE_OPEN_OV.match(text)
        if match:
            return pick("open_ov").format(cat=degrade(match.group(1)), v=mangle(match.group(2)))
        return text

    return rewrite


def run(level, agent, samples, catalog_ids, categories, products) -> dict:
    """Score one level by wrapping the evaluator's own customer functions."""
    if level == "L0":
        return ev.evaluate(agent, samples, catalog_ids, categories, products)

    rewrite = make_paraphraser(level)
    original_initial = ev.initial_message
    original_reply = ev.customer_reply
    original_behavior = ev.behavior_for

    def initial_message(sample, category, disclosed):
        return rewrite(original_initial(sample, category, disclosed))

    def customer_reply(sample, ask_attribute, disclosed, boundary_used):
        text, used = original_reply(sample, ask_attribute, disclosed, boundary_used)
        return rewrite(text), used

    def behavior_for(scenario, card, rng):
        # The override line is built here and read straight off the sample, so
        # it never passes through customer_reply - reword it at the source.
        behavior = original_behavior(scenario, card, rng)
        if "override" in behavior:
            behavior["override"]["message"] = rewrite(behavior["override"]["message"])
        return behavior

    ev.initial_message = initial_message
    ev.customer_reply = customer_reply
    ev.behavior_for = behavior_for
    try:
        return ev.evaluate(agent, samples, catalog_ids, categories, products)
    finally:
        ev.initial_message = original_initial
        ev.customer_reply = original_reply
        ev.behavior_for = original_behavior


LABELS = {
    "L0": "L0 original frames (control)",
    "L1": "L1 light  reworded carrier",
    "L2": "L2 heavy  reworded carrier",
    "L3": "L3 heavy  + values mangled",
    "L4": "L4 heavy  + values + category",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Paraphrase robustness harness")
    parser.add_argument("--level", action="append", choices=list(LABELS), default=None)
    parser.add_argument("--output", default="results_paraphrase.json")
    args = parser.parse_args()
    levels = args.level or list(LABELS)

    samples = ev.load_jsonl(DATASET)
    catalog_ids, categories, products = ev.catalog_index(CATALOG)
    print("building index once...", flush=True)
    index = CatalogIndex(CATALOG)
    print(f"index ready: {len(index.asins)} products\n", flush=True)

    header = f"{'level':<30} {'HR@10':>7} {'MRR':>8} {'MTTC':>7} {'TS':>8}"
    print(header)
    print("-" * len(header))
    rows = []
    for level in levels:
        agent = Agent(CATALOG, index=index)
        result = run(level, agent, samples, catalog_ids, categories, products)
        rows.append({
            "level": level, "label": LABELS[level],
            "HR": result["hit_rate_at_10"], "MRR": result["mrr"],
            "MTTC": result["mttc"], "TS": result["recommended_technical_score"],
            "scenario_metrics": result["scenario_metrics"],
        })
        print(f"{LABELS[level]:<30} {result['hit_rate_at_10']:>7.4f} "
              f"{result['mrr']:>8.4f} {result['mttc']:>7.3f} "
              f"{result['recommended_technical_score']:>8.4f}", flush=True)
    Path(ROOT / args.output).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
