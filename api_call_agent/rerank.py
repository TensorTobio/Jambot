"""Stage 3 - order the shortlist by the evidence the conversation has produced.

The catalogue has 50,000 products and no prompt holds them, so retrieval hands
the model a shortlist (default 30). Two things changed the shape of this stage:

**The model is shown the evidence, not just the products.** Every shortlist row
carries which of the customer's stated requirements that product *verifiably*
matches - an exact hit against the catalogue's own constraint strings - plus how
its price sits against a stated budget. Ordering is then a judgement about the
rows the evidence cannot separate, which is the only part a language model does
better than the scorer.

**The model votes; it does not overrule.** Candidates are grouped by how many
stated requirements they exactly match, and the model's opinion is fused with
the retrieval order *inside* each group by weighted reciprocal rank. So a
product that verifiably matches two requirements can never be pushed below one
that matches one, and inside a tie the retrieval order still wins a straight
disagreement - the model has to be consistent to move something. That is the
precision guarantee: the model can improve a ranking the evidence left
ambiguous, and is bounded in how much it can damage one.

Everything after that is still validated hard - unknown ids and duplicates are
dropped, and a short list is topped up from the deterministic ordering.
"""

from __future__ import annotations

from .llm_client import ClaudeClient, extract_json

SYSTEM = """You are the final ranking stage of a shopping assistant.

You get the customer's requirements and a shortlist. Each row states how many of those requirements that product VERIFIABLY matches, and which ones. Return the best 10, best first.

Return ONLY a JSON array of product IDs, e.g. ["B0912ABCDE", "B07XYZ1234"].

Ranking rules, in order of importance:
1. More verified matches beats fewer. Never rank a lower-match product above a higher-match one.
2. Among products with the SAME number of verified matches - which is where your judgement actually decides the order - prefer the one whose title and attributes fit the customer's category, use case and still-unmatched requirements most closely.
3. A stated hard requirement outranks a general preference.
4. If a budget is stated, prefer prices near it.
5. Break remaining ties with overall relevance.
6. Use only IDs from the shortlist. Never invent one. No duplicates. Exactly 10 entries unless the shortlist is shorter."""

MAX_TITLE = 110
MAX_FEATURE = 90


def evidence_tiers(index, candidates: list[str], known: list[str]) -> dict[str, int]:
    """How many stated requirements each candidate exactly matches."""
    tiers: dict[str, int] = {}
    for asin in candidates:
        cset = index.constraint_set.get(asin, frozenset())
        tiers[asin] = sum(1 for value in known if value in cset)
    return tiers


def _describe(index, asin: str, known: list[str], budget: float | None) -> str:
    title = index.title.get(asin, "")[:MAX_TITLE]
    cset = index.constraint_set.get(asin, frozenset())
    matched = [value for value in known if value in cset]
    match_text = f"matches {len(matched)}/{len(known)}"
    if matched:
        match_text += " (" + "; ".join(m[:MAX_FEATURE] for m in matched[:3]) + ")"

    price = index.price.get(asin)
    if price is None:
        price_text = "price n/a"
    elif budget:
        gap = (price - budget) / budget * 100.0
        price_text = f"${price:g} ({gap:+.0f}% vs budget)"
    else:
        price_text = f"${price:g}"

    category = index.category.get(asin, "")
    # Attributes the customer has not spoken to yet - this is the material the
    # model actually reasons over once the verified matches have tied.
    matched_set = set(matched)
    other = [c for c in index.constraints.get(asin, []) if c not in matched_set]
    other_text = " | ".join(c[:MAX_FEATURE] for c in other[:3])
    return f"{asin} :: {match_text} :: {title} :: {category} :: {price_text} :: {other_text}"


# Rank-fusion constants. ``RRF_K`` flattens the head of the curve so the top few
# positions are not absurdly dominant; ``MODEL_WEIGHT`` is how loud the model's
# vote is next to the deterministic score.
#
# 0.2 is not a guess. Driving the whole evaluator with a deliberately *random*
# reranker - the worst model there is - over 40 public sessions gives:
#
#     weight   0.0     0.2     0.4     0.6     1.0
#     MRR      0.9583  0.9583  0.9271  0.9104  0.8938
#
# 0.2 is the largest weight at which an adversarial model does no damage at all,
# so it is the most influence we can hand the model for free. Raise it only with
# a measured win from the real model behind it (``--model-weight``).
#
# The shape of that influence is worth knowing: because reciprocal rank is steep
# at the head and flat in the tail, a 0.2 vote can barely move rank 1 but can
# move something several places at rank 10+ - it bites exactly where the
# deterministic score has stopped discriminating, which is where semantics are
# all that is left.
RRF_K = 10.0
MODEL_WEIGHT = 0.2


def _fuse_within_tiers(
    candidates: list[str],
    model_order: list[str],
    tiers: dict[str, int],
    model_weight: float = MODEL_WEIGHT,
) -> list[str]:
    """Merge the model's ordering into the deterministic one, tier by tier.

    Two guards, one hard and one soft.

    The hard one is the **evidence tier**: candidates are grouped by how many
    stated requirements they exactly match, and fusion happens only inside a
    group. A product that verifiably matches two requirements can never fall
    below one that matches one, whatever the model returns.

    The soft one is **rank fusion** inside the group. The model does not replace
    the retrieval order, it votes against it - reciprocal-rank scores from both
    orderings are summed, with the model's weighted at ``model_weight``. The
    deterministic order carries real signal the tier count throws away (disclosure
    position, price agreement, category, popularity), so discarding it wholesale
    on the model's say-so is how a reranker loses points. This way a confident,
    consistent model still promotes what it likes several places, while a
    confused one perturbs the order slightly instead of shredding it.

    ``candidates`` arrives in deterministic score order, which is already tier
    descending - an exact constraint match outweighs every other signal in the
    scorer.
    """
    det = {asin: i for i, asin in enumerate(candidates)}
    mod = {asin: i for i, asin in enumerate(model_order)}
    unranked = len(model_order)

    groups: list[list[str]] = []
    last_tier: int | None = None
    for asin in candidates:
        tier = tiers.get(asin, 0)
        if last_tier is not None and tier == last_tier:
            groups[-1].append(asin)
        else:
            groups.append([asin])
            last_tier = tier

    def score(asin: str) -> float:
        return (
            1.0 / (RRF_K + det[asin])
            + model_weight / (RRF_K + mod.get(asin, unranked))
        )

    ordered: list[str] = []
    for group in groups:
        # -score first, deterministic position as the stable tie-break.
        ordered.extend(sorted(group, key=lambda a: (-score(a), det[a])))
    return ordered


def rerank(
    client: ClaudeClient,
    index,
    candidates: list[str],
    state,
    *,
    top_k: int = 10,
    model_weight: float = MODEL_WEIGHT,
) -> list[str]:
    """Return an ordered list of ``top_k`` asins. Falls back to ``candidates``."""
    if not candidates:
        return []
    if len(candidates) <= 1:
        return candidates[:top_k]

    known = [c for c in state.constraints if index.is_known_constraint(c)]
    tiers = evidence_tiers(index, candidates, known)
    budget = state.budget

    shortlist = "\n".join(_describe(index, asin, known, budget) for asin in candidates)
    requirements = "\n".join(f"- {c}" for c in state.constraints) or "- (none stated yet)"
    keywords = ", ".join(getattr(state, "llm_keywords", []) or []) or "none"
    tags = ", ".join(state.profile_tags) or "none"

    user = (
        f"Customer is shopping for: {getattr(state, 'raw_category', '') or state.category or 'unspecified'}\n"
        f"Stated requirements (verbatim):\n{requirements}\n"
        f"Related terms: {keywords}\n"
        f"Budget: {budget if budget is not None else 'not stated'}\n"
        f"Long-term shopper preferences: {tags}\n\n"
        "Shortlist (id :: verified matches :: title :: category :: price :: other attributes):\n"
        f"{shortlist}\n\n"
        f"Return the best {min(top_k, len(candidates))} ids as a JSON array."
    )

    text = client.complete(SYSTEM, user, max_tokens=400, prefill="[")
    ordered = extract_json(text, list)

    allowed = set(candidates)
    model_order: list[str] = []
    if isinstance(ordered, list):
        for item in ordered:
            asin = (
                str(item.get("parent_asin", "")).strip()
                if isinstance(item, dict)
                else str(item).strip()
            )
            if asin in allowed and asin not in model_order:
                model_order.append(asin)

    if not model_order:
        return candidates[:top_k]
    return _fuse_within_tiers(candidates, model_order, tiers, model_weight)[:top_k]


def should_rerank(index, candidates: list[str], state, top_k: int = 10) -> bool:
    """Is there anything left for the model to decide?

    Under the tier guard the model can only act where the evidence is tied. If
    every candidate inside the top-k window sits alone in its own evidence tier,
    and no candidate outside the window shares a tier with the last one inside
    it, then the answer is already fully determined and the call is pure spend.
    """
    if len(candidates) <= 1:
        return False
    known = [c for c in state.constraints if index.is_known_constraint(c)]
    if not known:
        # Nothing verified yet: every candidate is tier 0, so it is all judgement.
        return True
    tiers = evidence_tiers(index, candidates, known)
    window = candidates[:top_k]
    if len(candidates) > top_k and tiers[candidates[top_k]] == tiers[window[-1]]:
        return True  # membership of the top-k is still contestable
    counts: dict[int, int] = {}
    for asin in window:
        counts[tiers[asin]] = counts.get(tiers[asin], 0) + 1
    return any(count > 1 for count in counts.values())
