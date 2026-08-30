"""Stage 4 - write the customer-facing sentence.

``message`` is the only part of the response a human ever reads, and it is the
only part the scorer does not look at. That asymmetry is the whole design here:
the model is allowed to choose the *words*, and nothing else. ``ask_attribute``
and ``recommendations`` are still decided by the deterministic policy, so a bad
reply costs a nicer sentence and never a point of MRR.

Three guarantees:

* **Bounded.** ``max_tokens`` is tiny (70 by default, roughly two sentences),
  so a turn that ignores the brief is cheap to throw away. A ``\n\n`` stop
  sequence would be the natural second guard and the API rejects it - a stop
  sequence must contain non-whitespace - so the cap does that work alone.
* **Grounded.** The prompt carries what we actually know - the customer's own
  last requirement and the title of the product we are about to show - so the
  sentence refers to real things instead of inventing them.
* **Validated.** :func:`_clean` rejects markdown, JSON, ids, links,
  multi-paragraph answers and anything that forgot to ask the question. A
  rejected reply falls back to ``SessionState.clarification`` - the same
  template the rule-based agent ships - so the worst case is the old behaviour.
"""

from __future__ import annotations

import re

from .llm_client import ClaudeClient

SYSTEM = """You are a shop assistant replying to a customer in a live chat. Write ONE reply.

Voice:
- Two short sentences at most. Never a list, never a heading, never an emoji.
- Contractions, plain words, the tone of a person who works in the shop.
- Acknowledge what they just told you in your own words - do not repeat their sentence back.
- Vary your opening. You are shown your own last replies in this conversation; never start a reply the same way twice, and never reuse a sentence pattern you already used.
- If products are being shown, refer to them naturally ("these", "the leather ones"). Never print an id, a price you were not given, or a product you were not given.
- Then ask for exactly the one detail named in ASK. One question mark, at the end.
- No apologising, no "as an AI", no "certainly", no restating the whole conversation.

Good: "Rubber soles it is - that rules out most of the dress styles. Any colour you're set on?"
Good: "Got it, these four all run in cotton. What are you mainly wearing it for?"
Good: "Hand wash only, noted. Is there a colour that would rule one out for you?"
Bad: "Certainly! I have noted your preference for rubber soles. Here are some options: 1. ..."
Bad: two replies in a row that both open "So you're after ..." - repetition is the loudest tell that a machine is talking.

Write only the reply text."""

MAX_CHARS = 240
NEWLINE = chr(10)
_ASIN_RE = re.compile(r"\bB[0-9A-Z]{9}\b")
_BANNED = ("http", "```", "as an ai", "i'm an ai", "language model")
_ATTRIBUTE_WORDING = {
    "material": "the material or fabric they want",
    "color": "the colour they want",
    "size": "the size or fit they need",
    "style": "the style or cut they want",
    "brand": "a brand they prefer",
    "budget": "roughly what they want to spend",
    "feature": "a specific feature it has to have",
    "use_case": "what they will use it for",
    "category": "which kind of item exactly they mean",
    "other": "any other detail that would rule an option in or out",
}


def _clean(text: str | None, *, needs_question: bool) -> str | None:
    """Return a safe one-paragraph reply, or ``None`` to use the template."""
    if not text:
        return None
    reply = " ".join(str(text).split())
    reply = reply.strip().strip('"').strip("'").strip()
    if not reply or len(reply) > MAX_CHARS:
        return None
    lowered = reply.lower()
    if any(token in lowered for token in _BANNED):
        return None
    # Structure the model was told not to produce - a strong signal it ignored
    # the brief, so the template is the safer answer.
    if any(ch in reply for ch in "{}[]`*#|") or _ASIN_RE.search(reply):
        return None
    if needs_question and "?" not in reply:
        return None
    if not needs_question and "?" in reply:
        return None
    # Non-ASCII beyond ordinary typography (emoji, decorative marks).
    if any(ord(ch) > 0x2019 for ch in reply):
        return None
    return reply


def compose_reply(
    client: ClaudeClient,
    state,
    attribute: str | None,
    *,
    shown: int = 0,
    lead_title: str = "",
    max_tokens: int = 70,
) -> str | None:
    """One short, human reply. ``None`` means the caller should use the template."""
    recent = [m for m in state.messages[-2:] if m]
    if not recent:
        return None

    latest = state.constraints[-1] if state.constraints else ""
    earlier = "; ".join(state.constraints[:-1][-3:]) or "nothing yet"
    if attribute:
        ask = _ATTRIBUTE_WORDING.get(attribute, _ATTRIBUTE_WORDING["other"])
    else:
        ask = "nothing - do not ask a question, just hand over the results"

    if shown:
        showing = f"{shown} product(s), the closest being: {lead_title[:90] or 'a close match'}"
    else:
        showing = "nothing yet - you are still narrowing it down"

    # Repetition is what makes a generated reply read as generated, and
    # the model cannot avoid an opening it cannot see. Show it its own.
    said_before = getattr(state, "replies", [])[-2:]
    avoid = ""
    if said_before:
        history = NEWLINE.join(f"- {said}" for said in said_before)
        avoid = "Your own last replies - do NOT open like these again:" + NEWLINE
        avoid += history + NEWLINE + NEWLINE

    user = (
        f"Customer wants: {getattr(state, 'raw_category', '') or state.category or 'unspecified'}\n"
        f"They just said: {recent[-1][:300]}\n"
        f"Newest requirement you now know: {latest or 'none yet'}\n"
        f"Already known: {earlier}\n"
        f"You are showing: {showing}\n"
        f"{avoid}"
        f"ASK: {ask}\n\n"
        "Write the reply."
    )

    text = client.complete(
        SYSTEM,
        user,
        max_tokens=max_tokens,
        temperature=0.7,          # the one place variety is worth more than determinism
    )
    return _clean(text, needs_question=attribute is not None)
