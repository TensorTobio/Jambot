"""Stage 1 - rephrase the customer's turn into a structured search query.

This is the primary job of the LLM in this track. A raw customer turn is
conversational and partial ("For that, what matters is: Rubber sole; Imported");
the retrieval layer wants a category, a set of constraints, an expanded keyword
bag, and a budget. Haiku does that conversion, carrying the whole dialogue so
far so that pronouns, corrections and overrides resolve correctly.

One rule matters more than any other and is repeated in the prompt: **quote
constraint phrases verbatim**. The retrieval index is keyed on the customer's
exact wording, so a helpful paraphrase destroys an exact match. Paraphrases,
synonyms and inferred attributes are still valuable - they go in a separate
``keywords`` field that feeds the fuzzy/BM25 route instead.
"""

from __future__ import annotations

from .llm_client import ClaudeClient, extract_json

SYSTEM = """You normalise a shopping conversation into a structured search query for a product catalogue of clothing, shoes and jewellery.

Return ONLY a JSON object with these keys:
  "category"    string  - the product category the customer named, copied exactly as they said it. "" if they never named one.
  "constraints" array of strings - requirements the customer stated, each copied VERBATIM from their words.
  "keywords"    array of strings - your own expansion: synonyms, materials, styles, use cases and other words likely to appear in a matching product listing. 5-15 short entries.
  "budget"      number or null - a price the customer named, as a plain number.
  "scenario"    one of "buying", "browsing", "intent_override", "boundary".
  "search_query" string - one sentence describing what the customer wants, for a keyword search engine.

CRITICAL RULES
1. Every entry of "constraints" must be an exact substring of what the customer actually typed. Do not fix spelling, expand abbreviations, change punctuation or capitalisation, merge two requirements, or split one. Copy and paste.
2. Put all of your own interpretation in "keywords" instead. Never in "constraints".
3. Carry forward constraints from earlier turns as well as the newest turn.
4. If the customer says to ignore an earlier preference, keep the earlier one anyway AND add the new one - both describe the same product they have in mind.
5. If the customer says they have no preference, add nothing for that turn.
6. Output the JSON object and nothing else."""


def _dialogue(messages: list[str]) -> str:
    lines = []
    for index, message in enumerate(messages, 1):
        lines.append(f"turn {index} customer: {message}")
    return "\n".join(lines)


def rephrase(
    client: ClaudeClient,
    messages: list[str],
    profile: dict,
    *,
    known_constraints: list[str] | None = None,
) -> dict | None:
    """Return the structured query dict, or ``None`` if the call failed."""
    tags = ", ".join(str(t) for t in (profile.get("preference_tags") or [])) or "none"
    carried = known_constraints or []
    carried_text = "\n".join(f"- {c}" for c in carried) or "(none yet)"

    user = (
        f"Conversation so far:\n{_dialogue(messages)}\n\n"
        f"Constraints already extracted in earlier turns (keep these):\n{carried_text}\n\n"
        f"Anonymised shopper profile - long-term preferences: {tags}\n\n"
        "Produce the JSON object."
    )

    text = client.complete(SYSTEM, user, max_tokens=700, prefill="{")
    data = extract_json(text, dict)
    if not isinstance(data, dict):
        return None

    def as_list(value) -> list[str]:
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    budget = data.get("budget")
    try:
        budget = float(budget) if budget not in (None, "", "null") else None
    except (TypeError, ValueError):
        budget = None

    return {
        "category": str(data.get("category") or "").strip(),
        "constraints": as_list(data.get("constraints"))[:12],
        "keywords": as_list(data.get("keywords"))[:15],
        "budget": budget,
        "scenario": str(data.get("scenario") or "").strip().lower(),
        "search_query": str(data.get("search_query") or "").strip(),
    }
