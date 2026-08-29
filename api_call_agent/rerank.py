"""Stage 3 - Haiku picks and orders the final top 10 from a shortlist.

The catalogue has 50,000 products and no prompt holds them, so retrieval hands
the model a shortlist (default 30) and the model does the semantic judgement:
which of these actually satisfy what the customer asked for, best first.

The output is validated hard - unknown ids, duplicates and short lists are
repaired against the deterministic ordering, so a confused response can only
ever degrade to the rule-based answer, never below it.
"""

from __future__ import annotations

from .llm_client import ClaudeClient, extract_json

SYSTEM = """You are the final ranking stage of a shopping assistant.

You get the customer's requirements and a numbered shortlist of candidate products. Return the 10 that best satisfy the requirements, best first.

Return ONLY a JSON array of product IDs, e.g. ["B0912ABCDE", "B07XYZ1234"].

Ranking rules, in order of importance:
1. A product that satisfies a requirement the customer stated explicitly beats one that only looks similar.
2. Requirements stated as hard needs outrank general preferences.
3. If the customer named a budget, prefer products near that price.
4. Break remaining ties with overall relevance to the stated category and use.
5. Use only IDs from the shortlist. Never invent one. No duplicates. Exactly 10 entries unless the shortlist is shorter."""

MAX_TITLE = 110
MAX_FEATURE = 90


def _describe(index, asin: str) -> str:
    title = index.title.get(asin, "")[:MAX_TITLE]
    price = index.price.get(asin)
    price_text = f"${price:g}" if price is not None else "price n/a"
    category = index.category.get(asin, "")
    features = index.constraints.get(asin, [])[:3]
    feature_text = " | ".join(f[:MAX_FEATURE] for f in features)
    return f"{asin} :: {title} :: {category} :: {price_text} :: {feature_text}"


def rerank(
    client: ClaudeClient,
    index,
    candidates: list[str],
    state,
    *,
    top_k: int = 10,
) -> list[str]:
    """Return an ordered list of ``top_k`` asins. Falls back to ``candidates``."""
    if not candidates:
        return []
    if len(candidates) <= 1:
        return candidates[:top_k]

    shortlist = "\n".join(_describe(index, asin) for asin in candidates)
    requirements = "\n".join(f"- {c}" for c in state.constraints) or "- (none stated yet)"
    keywords = ", ".join(getattr(state, "llm_keywords", []) or []) or "none"
    budget = state.budget if state.budget is not None else "not stated"
    tags = ", ".join(state.profile_tags) or "none"

    user = (
        f"Customer is shopping for: {getattr(state, 'raw_category', '') or state.category or 'unspecified'}\n"
        f"Stated requirements (verbatim):\n{requirements}\n"
        f"Related terms: {keywords}\n"
        f"Budget: {budget}\n"
        f"Long-term shopper preferences: {tags}\n\n"
        f"Shortlist (id :: title :: category :: price :: key attributes):\n{shortlist}\n\n"
        f"Return the best {min(top_k, len(candidates))} ids as a JSON array."
    )

    text = client.complete(SYSTEM, user, max_tokens=400, prefill="[")
    ordered = extract_json(text, list)

    allowed = set(candidates)
    result: list[str] = []
    if isinstance(ordered, list):
        for item in ordered:
            asin = str(item).strip() if not isinstance(item, dict) else str(item.get("parent_asin", "")).strip()
            if asin in allowed and asin not in result:
                result.append(asin)
                if len(result) >= top_k:
                    break

    # Repair: top up from the deterministic ordering so we never return short.
    if len(result) < top_k:
        for asin in candidates:
            if asin not in result:
                result.append(asin)
                if len(result) >= top_k:
                    break
    return result[:top_k]
