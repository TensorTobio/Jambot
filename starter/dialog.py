"""Dialog layer: intent routing, slot store and ask-policy.

The simulated customer speaks in a small, fixed set of sentence frames. This
module recognises each of them, extracts the constraint strings they carry, and
accumulates those into a per-session slot store. It also decides which
``ask_attribute`` to request next and writes the natural-language clarification
that goes back in ``message``.
"""

from __future__ import annotations

import re

ALLOWED_ATTRIBUTES = (
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
)

MATERIALS = (
    "cotton", "polyester", "nylon", "leather", "wool",
    "spandex", "silk", "rayon", "fabric",
)

# --- the simulator's sentence frames -------------------------------------
RE_OPEN_BUYING = re.compile(r"^I'm looking for (?P<cat>.+?)\. A key requirement is:\s*(?P<value>.+?)\.?\s*$")
RE_OPEN_BROWSING = re.compile(r"^I'm looking for (?P<cat>.+?), but I'm still exploring\.?\s*$")
RE_OPEN_OVERRIDE = re.compile(r"^I'm looking for (?P<cat>.+?)\.\s+(?P<value>.+?)\s*$")
RE_DISCLOSE = re.compile(r"what matters is:\s*(?P<payload>.+?)\.?\s*$", re.I)
RE_OVERRIDE = re.compile(r"ignore my earlier preference\.\s*What I need is:\s*(?P<value>.+?)\.?\s*$", re.I)
RE_NO_PREFERENCE = re.compile(
    r"I don't have (?P<additional>an additional )?preference for\s+(?P<attr>[a-z_]+)", re.I
)
RE_NO_SIGNAL = re.compile(r"not quite right yet", re.I)
RE_BUDGET = re.compile(r"budget around \$?\s*([0-9]+(?:\.[0-9]+)?)", re.I)

# --- free-form cues -------------------------------------------------------
# The frames above are the simulator as it ships. The specification permits the
# organizer to paraphrase the customer on the private set, which would leave
# every frame above matching nothing. These cue patterns are the fallback: they
# look for the *intent* of a turn rather than its exact sentence shape, and are
# only consulted when no frame matched.
RE_FREE_NO_PREF = re.compile(
    r"no preference|don'?t (?:really )?mind|do not mind|your call|up to you"
    r"|don'?t care|dont care|zero opinion|no strong feelings|no opinion"
    r"|you (?:choose|pick|decide)|whatever you think|surprise me",
    re.I,
)
RE_FREE_EXHAUSTED = re.compile(
    r"nothing (?:else|more|further)|can'?t think of anything|cannot think of anything"
    r"|that'?s all|that is all|tapped out|no more thoughts|nothing left|no other",
    re.I,
)
RE_FREE_OVERRIDE = re.compile(
    r"actually|forget (?:what|that|it)|scratch that|changed my mind|change of plan"
    r"|instead|wait no|hold on|never mind|nevermind|on second thought",
    re.I,
)


def classify_constraint(value: str) -> str:
    """Same classifier the simulator uses to decide which asks a value answers."""
    lowered = value.lower()
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return "budget"
    if any(material in lowered for material in MATERIALS):
        return "material"
    if any(word in lowered for word in ("color", "black", "white", "blue", "red", "pink", "green")):
        return "color"
    if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")):
        return "size"
    if any(word in lowered for word in ("department", "style", "fit", "sleeve", "neck")):
        return "style"
    if any(word in lowered for word in ("hiking", "running", "gym", "winter", "outdoor", "work")):
        return "use_case"
    return "feature"


class SessionState:
    """Short-term memory for one shopping session.

    Constraints are only ever *added*. Even the "ignore my earlier preference"
    override does not invalidate what came before: every string the customer
    utters is drawn verbatim from the one hidden target product, so an earlier
    soft preference remains a true fact about that product. Erasing it would
    throw away signal.
    """

    def __init__(self, session_id: str, user_profile: dict, index) -> None:
        self.session_id = session_id
        self.index = index
        self.profile = user_profile if isinstance(user_profile, dict) else {}
        self.profile_tags = [
            str(tag).lower() for tag in (self.profile.get("preference_tags") or [])
        ]
        self.category: str | None = None
        # False when the category was guessed by token overlap rather than read
        # from the message verbatim; the ranker trusts it less. See
        # ``starter.retrieval.W_CATEGORY_FUZZY``.
        self.category_confident = True
        self.constraints: list[str] = []          # ordered, de-duplicated
        self._seen: set[str] = set()
        self.messages: list[str] = []
        self.scenario: str = "unknown"
        self.override_seen = False
        self.budget: float | None = None
        self.asked: list[str] = []
        self.unhelpful: set[str] = set()           # attributes that came back empty
        self.exhausted = False                     # customer has nothing left to add
        # Per-session demo overrides, set by Agent.reset(); see starter/agent.py.
        self.ask_policy = "other"
        self.always_recommend = False

    # -- ingestion --------------------------------------------------------
    def add_constraint(self, value: str) -> None:
        value = re.sub(r"\s+", " ", str(value)).strip(" -;,.\t\n")
        if not value or value in self._seen:
            return
        self._seen.add(value)
        self.constraints.append(value)
        match = RE_BUDGET.search(value)
        if match:
            try:
                self.budget = float(match.group(1))
            except ValueError:
                pass

    def _split_payload(self, payload: str) -> list[str]:
        """Split a disclosure into constraint strings, verified against the catalog.

        The simulator joins values with "; ", but a value may itself contain a
        semicolon, so a naive split can shred a real constraint. Anything that
        resolves to a known catalog constraint is kept intact; the rest is
        re-joined greedily and finally handed over as fuzzy text.
        """
        payload = payload.strip()
        if not payload:
            return []
        if self.index.is_known_constraint(payload):
            return [payload]
        parts = [part.strip() for part in payload.split("; ") if part.strip()]
        if not parts:
            return []
        resolved: list[str] = []
        buffer: list[str] = []
        for part in parts:
            if self.index.is_known_constraint(part):
                if buffer:
                    joined = "; ".join(buffer)
                    resolved.append(joined)
                    buffer = []
                resolved.append(part)
            else:
                buffer.append(part)
                joined = "; ".join(buffer)
                if self.index.is_known_constraint(joined):
                    resolved.append(joined)
                    buffer = []
        if buffer:
            resolved.append("; ".join(buffer))
        return resolved

    def observe(self, user_message: str, turn: int) -> None:
        text = str(user_message or "").strip()
        self.messages.append(text)
        if not text:
            return
        if not self._observe_frames(text, turn):
            # No known frame matched. Either the organizer paraphrased the
            # customer, or this is a turn shape we have never seen; fall back to
            # reading the message for its content rather than its structure.
            self._observe_freeform(text)

    def _observe_frames(self, text: str, turn: int) -> bool:
        """The simulator as shipped. Returns True when a frame matched."""
        # 1. Opening turn: always carries the coarse category, sometimes a value.
        if turn == 1 or self.category is None:
            match = RE_OPEN_BUYING.match(text)
            if match:
                self.scenario = "buying"
                self._set_category(match.group("cat"))
                self.add_constraint(match.group("value"))
                return True
            match = RE_OPEN_BROWSING.match(text)
            if match:
                self.scenario = "browsing"
                self._set_category(match.group("cat"))
                return True
            match = RE_OPEN_OVERRIDE.match(text)
            if match and text.startswith("I'm looking for"):
                # "I'm looking for {cat}. {old soft preference}"
                self.scenario = "intent_override"
                self._set_category(match.group("cat"))
                self.add_constraint(match.group("value"))
                return
            if text.startswith("I'm looking for"):
                self._set_category(text[len("I'm looking for"):])
                return True

        # 2. Mid-session frames.
        match = RE_OVERRIDE.search(text)
        if match:
            self.scenario = "intent_override"
            self.override_seen = True
            for value in self._split_payload(match.group("value")):
                self.add_constraint(value)
            return True

        match = RE_DISCLOSE.search(text)
        if match:
            for value in self._split_payload(match.group("payload")):
                self.add_constraint(value)
            return True

        match = RE_NO_PREFERENCE.search(text)
        if match:
            attribute = (match.group("attr") or "").lower()
            if match.group("additional"):
                # "no additional preference for X" - X is spent. When X is
                # "other" that means the whole hidden card is drained.
                self.unhelpful.add(attribute)
                if attribute == "other":
                    self.exhausted = True
            else:
                self.scenario = "boundary" if self.scenario == "unknown" else self.scenario
            return True

        if RE_NO_SIGNAL.search(text):
            return True
        return False

    def _observe_freeform(self, text: str) -> None:
        """Read a turn by content when its sentence shape is unrecognised.

        Four things are worth recovering, in rising order of value:

        1. the coarse category, if we still do not have one - the opening line
           names it whatever the wording, and it is the single most valuable
           field on turn 1 of a Browsing session;
        2. an override cue, so the scenario is labelled correctly;
        3. a "no preference" / "nothing more" cue, which tells us the customer
           is out of information and there is no later turn worth waiting for;
        4. any catalog constraint quoted in the message.

        Nothing here can raise, and nothing here fires when a frame matched, so
        the shipped-simulator behaviour is untouched.
        """
        scan = text
        if self.category is None:
            # Verbatim taxonomy string first; token overlap only if that fails.
            matched = self.index.match_category(text)
            if matched is None:
                matched = self.index.match_category_fuzzy(text)
                if matched is not None:
                    self.category_confident = False
            if matched:
                self.category = matched
                self.raw_category = matched
        # The category name is not a disclosure - "Jewelry Necklaces" must not
        # donate "Jewelry" as a constraint - so hide it before scanning.
        if self.category:
            scan = re.sub(re.escape(self.category), " ", scan, flags=re.I)

        if RE_FREE_OVERRIDE.search(text):
            self.scenario = "intent_override"
            self.override_seen = True

        spent = RE_FREE_EXHAUSTED.search(text)
        no_pref = RE_FREE_NO_PREF.search(text)
        if spent or no_pref:
            lowered = text.lower()
            named = [a for a in ALLOWED_ATTRIBUTES if a in lowered]
            if spent:
                # "nothing more on X" - X is drained. "other" drains the card.
                for attribute in named:
                    self.unhelpful.add(attribute)
                if "other" in named or not named:
                    self.exhausted = True
            elif self.scenario == "unknown":
                self.scenario = "boundary"

        for value in self.index.find_constraints(scan):
            self.add_constraint(value)

    def _set_category(self, raw: str) -> None:
        raw = str(raw).strip().strip(".,")
        matched = self.index.match_category(raw)
        self.category = matched or (self.category or None)
        if matched is None:
            # Keep the raw phrase around for the keyword route.
            self.raw_category = raw
        else:
            self.raw_category = matched

    # -- outputs ----------------------------------------------------------
    def query_text(self) -> str:
        parts = [getattr(self, "raw_category", "") or (self.category or "")]
        parts.extend(self.constraints)
        return " ".join(part for part in parts if part)

    def next_ask(self, candidates: list[str] | None = None, policy: str = "other") -> str | None:
        """Which attribute to ask about next.

        Two policies, both legal, one measurably better:

        ``"other"`` (default, and what we ship) - in the simulator's
        ``customer_reply`` a value is disclosed when
        ``attribute == "other" or classify_constraint(value) == attribute``.
        ``"other"`` therefore matches *every* undisclosed constraint and returns
        the cap of two per turn, where a specific attribute returns at most two
        *of that one type* - usually zero or one. It is a strict superset, so it
        cannot be beaten on disclosure rate.

        ``"split"`` - the information-gain policy: pick the attribute whose
        values most evenly partition the current candidate pool. This is the
        textbook-correct thing to do against a real shopper, and it makes the
        conversation feel varied, but against this simulator it drains the
        hidden card more slowly. Measured cost is in the CHANGELOG.

        Either way, once the customer says they have nothing more to add we
        rotate through specific attributes - free, and it occasionally shakes
        loose a value the "other" branch had already consumed.
        """
        rotation = ("material", "color", "style", "size", "budget", "feature", "use_case", "other")
        if self.exhausted:
            attribute = rotation[len(self.asked) % len(rotation)]
            self.asked.append(attribute)
            return attribute

        attribute = "other"
        if policy == "split" and candidates:
            scores = self.index.attribute_split(candidates, self._seen)
            # Never re-ask an attribute the customer has already emptied.
            for name in self.unhelpful:
                scores.pop(name, None)
            # Halve the score each time we have already asked something, so the
            # conversation moves on instead of hammering the largest bucket.
            for name in list(scores):
                scores[name] *= 0.5 ** self.asked.count(name)
            if scores:
                attribute = max(scores, key=lambda k: scores[k])
        elif policy == "rotate":
            attribute = rotation[len(self.asked) % len(rotation)]

        self.asked.append(attribute)
        return attribute

    # -- clarification wording -------------------------------------------
    # The attribute we ask for is a scoring decision; the sentence we say is
    # not. Rotating the phrasing and naming what we already know costs nothing
    # and stops the transcript reading like a stuck record.
    _OPENERS = (
        "",
        "Got it.",
        "That helps.",
        "Noted.",
        "Understood.",
        "Right.",
    )
    _PHRASES = {
        "other": (
            "What else matters for this one - material, colour, fit, or budget?",
            "Anything else I should hold you to - a fabric, a colour, a price ceiling?",
            "Is there another must-have I should factor in?",
            "What other detail would rule an option in or out for you?",
        ),
        "material": (
            "Do you have a material preference?",
            "Any fabric you want - or want to avoid?",
        ),
        "color": (
            "Any colour you are set on?",
            "Is there a colour that would rule an option out?",
        ),
        "style": (
            "What style or fit are you after?",
            "Should this lean more fitted or more relaxed?",
        ),
        "size": (
            "Any sizing requirement I should respect?",
            "Is there a size or width I should stick to?",
        ),
        "budget": (
            "Roughly what budget are you working with?",
            "Is there a price you would rather not go past?",
        ),
        "feature": (
            "Is there a specific feature it has to have?",
            "Any detail it absolutely needs - pockets, lining, a closure?",
        ),
        "use_case": (
            "What will you mainly be using it for?",
            "Where do you picture wearing this most?",
        ),
        "category": ("Which kind of item exactly are you after?",),
        "brand": ("Any brand you prefer?",),
    }

    def clarification(self, attribute: str | None, count: int) -> str:
        turn = len(self.asked)
        if attribute is None:
            return "Here are the closest matches I found."

        options = self._PHRASES.get(attribute, self._PHRASES["other"])
        question = options[turn % len(options)]
        opener = self._OPENERS[turn % len(self._OPENERS)]

        known = ""
        if self.constraints and turn > 1:
            latest = self.constraints[-1]
            if len(latest) <= 48:
                known = f'I have "{latest}" down. '

        if count:
            # The ask-policy shows a single best guess on the early turns, so
            # "Here are 1 that fit" is the common case, not an edge case.
            if count == 1:
                lead = "Here's the closest match I've found so far."
            else:
                lead = f"Here are {count} that fit so far."
            if turn > 1 and known:
                lead = f"{known}{lead}"
            return f"{lead} {question}".strip()
        parts = [opener, known.strip(), question]
        return " ".join(part for part in parts if part).strip()
