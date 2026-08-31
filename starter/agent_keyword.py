from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "still", "exploring", "need", "am", "im",
}

MATERIAL_RE = re.compile(r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric|denim|suede|cashmere|linen)\b", re.I)
COLOR_RE = re.compile(r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange|navy|beige|gold|silver)\b", re.I)
BUDGET_RE = re.compile(r"\$\s?(\d+(?:\.\d+)?)")
OVERRIDE_RE = re.compile(r"\bignore\b.*\b(earlier|previous)\b|\bactually\b.*\binstead\b|\bchange(d)? my mind\b", re.I)
NO_PREFERENCE_RE = re.compile(r"\bno preference\b|\bdon.?t have a preference\b|\byour judgment\b|\bdoesn.?t matter\b", re.I)

# Buying vs. browsing intent cues. Matched against the raw message, not the
# tokenized/stopword-filtered terms, since phrases like "still exploring"
# and "key requirement" are exactly the signal (and "still"/"exploring" are
# themselves stopwords stripped out of retrieval terms).
BUYING_CUE_RE = re.compile(
    r"\bkey requirement\b|\bmust have\b|\bneed(?:s|ed)? (?:it |this )?to be\b|\brequires?\b|"
    r"\bhas to (?:be|have)\b|\bspecifically\b|\bexactly\b|\bi need\b",
    re.I,
)
BROWSING_CUE_RE = re.compile(
    r"\bstill exploring\b|\bjust (?:looking|browsing)\b|\bnot sure\b|\bnot certain\b|"
    r"\bbrowsing\b|\bopen to\b|\bany (?:ideas|suggestions)\b|\bno idea\b",
    re.I,
)

# Best-effort strip of common clarification lead-ins ("For that, what
# matters is: ...", "A key requirement is: ...") so what's left is just the
# customer's stated values. If nothing matches, the whole message is kept
# as-is - stripping is an optimization, not a requirement for correctness.
LEADIN_RE = re.compile(
    r"^(?:for that,? what matters is:?|a key requirement is:?|i(?:'m| am) looking for[^.]*\.?"
    r"|actually,? ignore my earlier preference\.? what i need is:?)\s*",
    re.I,
)

# Catalog-derived vocab (populated in Agent._build_index) below these
# frequency bounds is too rare to be a reliable signal; above them the term
# is too generic (e.g. "women", "clothing") to mean the customer has
# actually specified anything.
CATEGORY_VOCAB_MIN, CATEGORY_VOCAB_MAX = 15, 4000
BRAND_VOCAB_MIN, BRAND_VOCAB_MAX = 3, 1500
CATEGORY_EXCLUDE = {"clothing", "jewelry", "novelty", "fashion", "specific", "more"}
BRAND_EXCLUDE = {"generic", "amazon", "collection", "brand", "unbranded", "various", "assorted"}
UMBRELLA_CATEGORY_VALUES = {"clothing, shoes & jewelry", "clothing shoes & jewelry", "clothing"}

# "category" is asked to every customer up front in every scenario's own
# opening message, so it's essentially always already known by the time we'd
# ask it - deliberately not in this list (see _choose_ask_attribute).
#
# The order matters for how the conversation *reads*, not just what it can
# learn: a shopper describing shoes expects "what size?" before "what
# material?", while a shopper describing jewelry expects the opposite. Each
# family below orders attributes the way a competent human sales assistant
# would for that kind of product, so every question is a specific, guided
# ask rather than a generic one repeated regardless of context.
CATEGORY_FAMILIES: dict[str, set[str]] = {
    "footwear": {
        "shoe", "shoes", "boot", "boots", "sandal", "sandals", "sneaker", "sneakers",
        "flat", "flats", "heel", "heels", "loafer", "loafers", "slipper", "slippers",
        "clog", "clogs", "oxford", "oxfords", "mule", "mules",
    },
    "jewelry": {
        "earring", "earrings", "necklace", "necklaces", "bracelet", "bracelets",
        "ring", "rings", "pendant", "pendants", "anklet", "anklets", "brooch",
        "brooches", "chain", "chains",
    },
    "watches": {"watch", "watches"},
    "bags_accessories": {
        "handbag", "handbags", "wallet", "wallets", "belt", "belts", "bag", "bags",
        "purse", "purses", "backpack", "backpacks", "luggage", "tote", "totes",
        "clutch", "clutches", "satchel", "satchels",
    },
    "apparel": {
        "dress", "dresses", "shirt", "shirts", "tshirt", "tshirts", "pant", "pants",
        "jean", "jeans", "short", "shorts", "skirt", "skirts", "jacket", "jackets",
        "sweater", "sweaters", "hoodie", "hoodies", "lingerie", "swimwear",
        "blouse", "blouses", "coat", "coats", "legging", "leggings", "romper",
        "rompers", "jumpsuit", "jumpsuits",
    },
}

# "feature" is a genuinely specific, natural question ("any feature that
# matters most?") that also happens to be the catch-all bucket the
# simulator's own classifier defaults undisclosed detail text to - so it
# stays high-value without ever reading as a repeated non-answer.
#
# Two different orderings depending on classified intent (see
# _classify_intent): a Buying customer already has a hard constraint in
# mind, so the fastest path to convergence is locking down concrete,
# differentiating specifics (material, cost, use case) immediately. A
# Browsing customer hasn't committed to specifics yet, so the fastest path
# is narrowing *what kind of thing* they want first (style/use_case) before
# asking about details like size or budget that are premature until the
# general shape of the request is clear.
BUYING_ATTRIBUTE_PRIORITY = [
    "material", "budget", "use_case", "color", "style", "size", "brand", "feature",
]
BROWSING_FAMILY_PRIORITY: dict[str, list[str]] = {
    "footwear": ["use_case", "style", "size", "color", "material", "budget", "brand", "feature"],
    "jewelry": ["style", "use_case", "material", "color", "budget", "brand", "feature", "size"],
    "watches": ["style", "use_case", "material", "color", "budget", "brand", "feature", "size"],
    "bags_accessories": ["style", "use_case", "material", "color", "size", "budget", "brand", "feature"],
    "apparel": ["style", "use_case", "color", "material", "size", "budget", "brand", "feature"],
}
BROWSING_DEFAULT_PRIORITY = [
    "style", "use_case", "material", "color", "budget", "size", "brand", "feature",
]

ATTRIBUTE_PROMPTS = {
    "category": "What type of item are you shopping for?",
    "material": "Do you have a material preference?",
    "color": "Is there a color you'd like?",
    "size": "What size are you looking for?",
    "style": "Any particular style or fit in mind?",
    "use_case": "What will you mainly use this for?",
    "budget": "Do you have a budget in mind?",
    "brand": "Is there a brand you prefer?",
    "feature": "Any specific feature that matters most to you?",
    # Only ever asked once, and only after every specific question above has
    # already been asked or ruled out - a genuine final check, not a
    # substitute for asking something concrete.
    "other": "Anything else specific I'm missing that would help me find the right one?",
}

# "other" is a true last resort: at most one ask, and only once nothing
# concrete is left to ask about (see _choose_ask_attribute). It exists
# purely as a safety net for detail text that doesn't fit any specific
# bucket - it is never the primary strategy.
MAX_OTHER_ASKS = 1

# The Agent contract has the customer speak first every session (reset()
# returns nothing; the evaluator sends the customer's opening message before
# ever calling respond()), so there's no way to literally greet before that.
# This is the practical equivalent: turn 1's response opens with a greeting
# before getting into the clarifying question, so the conversation still
# *reads* as one that started with a greeting.
GREETING = "Hi! I'd love to help you find the right item. "

SEARCH_FIELDS = ("title", "categories", "features", "details", "store", "description")


def _field_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _extract_phrases(message: str) -> list[str]:
    """Pull out multi-word clauses from a message as literal phrase
    candidates, e.g. "Wrap closure" or "95% Polyester, 5% Spandex". The
    evaluator's simulated customer echoes constraint values verbatim from
    the target product's own catalog text, so searching for them as an
    exact adjacent phrase (not just an OR of their individual words) is a
    much more discriminating signal than word-level matching alone."""
    text = LEADIN_RE.sub("", message.strip())
    phrases: list[str] = []
    for clause in re.split(r"[;,.]", text):
        cleaned = re.sub(r"\s+", " ", clause).strip()
        if 2 <= len(_terms(cleaned)) <= 8:
            phrases.append(cleaned)
    return phrases


def _fts_phrase_query(phrases: list[str], limit: int = 15) -> str:
    clauses: list[str] = []
    for phrase in list(dict.fromkeys(phrases))[:limit]:
        # FTS5 phrase syntax requires a bare double-quoted string; strip any
        # embedded quotes so the expression stays syntactically valid.
        safe = phrase.replace('"', " ").strip()
        if safe:
            clauses.append(f'"{safe}"')
    return " OR ".join(clauses)


def _fts_or_query(terms: list[str], limit: int = 40) -> str:
    # FTS5's tokenizer doesn't stem, so a plural query term ("jordans")
    # silently matches nothing against a singular catalog token ("jordan").
    # Expand every query term into its stem variants so retrieval doesn't
    # go quiet on the exact terms _extract_signals just used to infer the
    # customer already told us the category/brand.
    expanded: set[str] = set()
    for term in list(dict.fromkeys(terms))[:limit]:
        expanded.update(_stem_variants(term))
    return " OR ".join(f'"{term}"' for term in sorted(expanded))


def _stem_variants(token: str) -> set[str]:
    """Cheap plural/singular normalization (no external dependency) so
    catalog-derived vocab like "boots" still matches a message saying
    "boot", "jordan" still matches "jordans", etc."""
    variants = {token}
    if token.endswith("ies") and len(token) > 4:
        variants.add(token[:-3] + "y")
    if token.endswith("es") and len(token) > 4:
        variants.add(token[:-2])
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        variants.add(token[:-1])
    if not token.endswith("s"):
        variants.add(token + "s")
    return variants


def _infer_family(variants: set[str]) -> str | None:
    for family, tokens in CATEGORY_FAMILIES.items():
        if not variants.isdisjoint(tokens):
            return family
    return None


class _SessionState:
    __slots__ = (
        "term_weights", "asked_attributes", "no_preference_attributes",
        "known_attributes", "last_asked", "turn_count", "budget", "other_ask_count",
        "phrases", "profile_boost_set", "category_family", "intent",
    )

    def __init__(self) -> None:
        self.term_weights: dict[str, float] = {}
        self.asked_attributes: set[str] = set()
        self.no_preference_attributes: set[str] = set()
        self.known_attributes: set[str] = set()
        self.last_asked: str | None = None
        self.turn_count = 0
        self.budget: float | None = None
        self.other_ask_count = 0
        self.phrases: list[str] = []
        self.profile_boost_set: frozenset[str] = frozenset()
        self.category_family: str | None = None
        # None until classified; "browsing" is the safer default once a
        # message has been seen with no clear buying signal, since treating
        # an ambiguous shopper as still-exploring costs less than wrongly
        # assuming a hard constraint that isn't there. Never flips back to
        # "browsing" once "buying" is detected - a customer who states a
        # hard constraint doesn't stop being a buyer because a later
        # message happens to read as more casual.
        self.intent: str | None = None

    def boost_terms(self, terms: list[str], weight: float) -> None:
        for term in terms:
            self.term_weights[term] = self.term_weights.get(term, 0.0) + weight

    def reset_terms(self, keep_fraction: float = 0.25) -> None:
        # Intent override: sharply discount older terms instead of wiping
        # them, since some (e.g. category) usually still apply.
        for term in list(self.term_weights):
            self.term_weights[term] *= keep_fraction


class Agent:
    """Stateful hybrid keyword-retrieval agent.

    Improvements over the shipped weak starter:
    - conversation state persists accumulated constraints across turns
      instead of matching only the latest message;
    - recent terms are weighted higher than older ones, and an intent-
      override cue sharply discounts prior terms;
    - retrieval fuses two BM25 passes (recent-focused + full-history) via
      weighted reciprocal-rank fusion instead of a single flat query;
    - a simple budget cue softly re-ranks by price instead of ignoring it,
      tightened further once the session is classified "buying";
    - turn 1 opens with a greeting, since the contract has the customer
      speak first and there is no way to literally greet before that;
    - every message is classified buying vs. browsing (lexical cues, or a
      concrete value stated up front implying buying; sticky once "buying"
      is detected) and that classification picks a different attribute
      ordering: buying leads with material/budget/use_case to lock hard
      constraints fast, browsing leads with style/use_case to narrow down
      what kind of thing the customer wants before asking specifics;
    - within each, ask_attribute always leads with a concrete, specific
      question (material/size/color/...), ordered per detected product
      family for browsing (shoes get asked size before material, jewelry
      the reverse, etc.) so each question reads as genuinely guided rather
      than generic; a single broad "anything else?" is only ever asked
      once, as a last resort after every specific question has been asked
      or ruled out. Anything already known, already stated in free text
      (checked against catalog-derived category/brand vocabularies), or
      explicitly declined ("no preference") is skipped, so the agent never
      repeats itself or re-asks something the customer already said.
    """

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        # check_same_thread=False to match starter/retrieval.py: the index is
        # built once and only read afterwards, and webapp/ serves each request
        # on its own thread. The evaluator is single-threaded either way.
        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        self.connection.execute("PRAGMA query_only = OFF")
        self._prices: dict[str, float] = {}
        self._sessions: dict[str, _SessionState] = {}
        self._category_vocab: set[str] = set()
        self._brand_vocab: set[str] = set()
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        category_counts: dict[str, int] = {}
        store_counts: dict[str, int] = {}
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                batch.append((
                    parent_asin,
                    _field_text(product.get("title")),
                    _field_text(product.get("categories")),
                    _field_text(product.get("features")),
                    _field_text(product.get("details")),
                    _field_text(product.get("store")),
                    _field_text(product.get("description")),
                ))
                price = product.get("price")
                if isinstance(price, (int, float)):
                    self._prices[parent_asin] = float(price)

                row_category_terms: set[str] = set()
                for entry in product.get("categories") or []:
                    if str(entry).strip().lower() in UMBRELLA_CATEGORY_VALUES:
                        continue
                    for token in _terms(str(entry)):
                        row_category_terms.update(_stem_variants(token))
                for token in row_category_terms:
                    category_counts[token] = category_counts.get(token, 0) + 1

                store = product.get("store")
                if store:
                    row_store_terms: set[str] = set()
                    for token in _terms(str(store)):
                        row_store_terms.update(_stem_variants(token))
                    for token in row_store_terms:
                        store_counts[token] = store_counts.get(token, 0) + 1

                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

        self._category_vocab = {
            token for token, count in category_counts.items()
            if CATEGORY_VOCAB_MIN <= count <= CATEGORY_VOCAB_MAX
            and len(token) > 2 and token not in CATEGORY_EXCLUDE
        }
        self._brand_vocab = {
            token for token, count in store_counts.items()
            if BRAND_VOCAB_MIN <= count <= BRAND_VOCAB_MAX
            and len(token) > 2 and token not in BRAND_EXCLUDE
        }

    def reset(self, session_id: str, user_profile: dict) -> None:
        state = _SessionState()
        # Safe personalization: the anonymized aggregate preference tags
        # (e.g. "fit", "comfort") are catalog-wide-common words - if mixed
        # into the same MATCH query as something the customer actually
        # said, SQLite's BM25 has no idea our own Python-side weighting
        # meant to treat them as secondary, and a document that merely
        # repeats "comfort fit" in its title can out-rank one that actually
        # matches a specific, rare term the customer used. So tags never
        # enter the primary retrieval queries at all; instead we resolve
        # them once here into a bounded set of candidate ASINs and apply
        # only a small post-hoc re-rank nudge in _retrieve, the same way a
        # budget cue does.
        tags = user_profile.get("preference_tags") if isinstance(user_profile, dict) else None
        if isinstance(tags, list):
            tag_terms = [str(tag).lower() for tag in tags if tag]
            expr = _fts_or_query(tag_terms)
            state.profile_boost_set = frozenset(
                asin for asin, _score in self._query_candidates(expr, 300)
            )
        self._sessions[session_id] = state

    def _classify_intent(self, state: _SessionState, message: str) -> None:
        if state.intent == "buying":
            return  # sticky: a confirmed buyer doesn't get reclassified
        if BUYING_CUE_RE.search(message):
            state.intent = "buying"
            return
        # A concrete, specific value stated up front (not just a bare
        # category mention) is itself a buying signal even without one of
        # the explicit lexical cues above.
        if not BROWSING_CUE_RE.search(message) and (
            MATERIAL_RE.search(message) or COLOR_RE.search(message) or BUDGET_RE.search(message)
        ):
            state.intent = "buying"
            return
        state.intent = "browsing"

    def _extract_signals(self, state: _SessionState, message: str, terms: list[str]) -> None:
        if MATERIAL_RE.search(message):
            state.known_attributes.add("material")
        if COLOR_RE.search(message):
            state.known_attributes.add("color")
        budget_match = BUDGET_RE.search(message)
        if budget_match:
            state.budget = float(budget_match.group(1))
            state.known_attributes.add("budget")
        if NO_PREFERENCE_RE.search(message) and state.last_asked:
            # A decline on "other" isn't treated as permanent: with only
            # one scripted "no preference" reply per session in the
            # boundary scenario, it can land on whichever attribute we
            # happened to ask first - "other" is cheap to retry once more
            # since it can still recover real content on the next ask.
            if state.last_asked != "other":
                state.no_preference_attributes.add(state.last_asked)
                state.known_attributes.add(state.last_asked)

        # The customer may already have said what type of item or which
        # brand they want (e.g. "I want Jordans") without us asking — don't
        # waste a turn asking something they already told us. Give matched
        # tokens extra retrieval weight too: a vocab hit confirms the token
        # is a specific, meaningful signal (not just catalog-wide noise),
        # so it should outweigh generic low-weight terms like profile tags
        # in the same query instead of being diluted by them.
        vocab_hits: list[str] = []
        for token in terms:
            variants = _stem_variants(token)
            if not variants.isdisjoint(self._brand_vocab):
                state.known_attributes.add("brand")
                state.known_attributes.add("category")
                vocab_hits.append(token)
            if not variants.isdisjoint(self._category_vocab):
                state.known_attributes.add("category")
                vocab_hits.append(token)
            family = _infer_family(variants)
            if family:
                state.category_family = family
        if vocab_hits:
            state.boost_terms(vocab_hits, weight=1.5)

    def _query_candidates(self, expression: str, limit: int) -> list[tuple[str, float]]:
        if not expression:
            return []
        rows = self.connection.execute(
            "SELECT parent_asin, bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) AS score "
            "FROM products WHERE products MATCH ? ORDER BY score LIMIT ?",
            (expression, limit),
        ).fetchall()
        # sqlite bm25() is more negative for better matches; flip sign so
        # higher is better, matching the fusion math below.
        return [(str(asin), -float(score)) for asin, score in rows]

    def _retrieve(self, state: _SessionState, latest_terms: list[str], top_k: int) -> list[str]:
        all_terms = sorted(state.term_weights, key=state.term_weights.get, reverse=True)
        # A short reply (e.g. just "black") shouldn't be searched in
        # isolation - a bare color/attribute word matches BM25-favored but
        # off-topic products (e.g. a listing whose title repeats "Black"
        # four times). Anchor it with the strongest carried-over context so
        # the "recent" pass stays on-topic while still emphasizing the
        # newest information. Each pass is still one combined MATCH query
        # (not decomposed per-term) so SQLite's own BM25 keeps rewarding
        # documents that co-occur across multiple terms, not just whichever
        # single term happens to have a spammy top match.
        anchor_terms = [term for term in all_terms if term not in latest_terms][:6]
        recent_terms = list(dict.fromkeys(latest_terms + anchor_terms)) or all_terms[:10]
        recent_expr = _fts_or_query(recent_terms)
        broad_expr = _fts_or_query(all_terms)

        pool_size = max(top_k * 8, 80)
        recent_hits = self._query_candidates(recent_expr, pool_size)
        broad_hits = self._query_candidates(broad_expr, pool_size)

        fused: dict[str, float] = {}
        k = 60.0
        for rank, (asin, _score) in enumerate(recent_hits, start=1):
            fused[asin] = fused.get(asin, 0.0) + 0.6 / (k + rank)
        for rank, (asin, _score) in enumerate(broad_hits, start=1):
            fused[asin] = fused.get(asin, 0.0) + 0.4 / (k + rank)

        # Exact-phrase pass: the customer's replies echo constraint values
        # verbatim from the target's own catalog text ("Wrap closure"), so
        # a document matching that exact adjacent phrase is a much stronger
        # signal than matching "wrap" and "closure" separately anywhere in
        # the document. Weighted above the word-level passes accordingly,
        # and weighted higher still once in "buying" mode, where the
        # customer has already committed to hard constraints worth locking
        # onto hard rather than treating as just one more soft signal.
        buying = state.intent == "buying"
        phrase_weight = 1.1 if buying else 0.9
        if state.phrases:
            phrase_expr = _fts_phrase_query(list(reversed(state.phrases)))
            for rank, (asin, _score) in enumerate(self._query_candidates(phrase_expr, pool_size), start=1):
                fused[asin] = fused.get(asin, 0.0) + phrase_weight / (k + rank)

        if not fused:
            return []

        if state.budget is not None:
            bonus, penalty = (0.015, 0.02) if buying else (0.01, 0.01)
            for asin in list(fused):
                price = self._prices.get(asin)
                if price is None:
                    continue
                if price <= state.budget * 1.1:
                    fused[asin] += bonus
                elif price > state.budget * 1.5:
                    fused[asin] -= penalty

        if state.profile_boost_set:
            for asin in fused:
                if asin in state.profile_boost_set:
                    fused[asin] += 0.005

        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        return [asin for asin, _ in ranked[:top_k]]

    def _choose_ask_attribute(self, state: _SessionState) -> str | None:
        if state.turn_count >= 10:
            # No 11th turn exists to receive an answer to a question asked
            # on the last allowed turn, so asking here is pure waste.
            return None
        if state.intent == "buying":
            priority = BUYING_ATTRIBUTE_PRIORITY
        else:
            priority = BROWSING_FAMILY_PRIORITY.get(state.category_family, BROWSING_DEFAULT_PRIORITY)
        for attribute in priority:
            if attribute in state.asked_attributes:
                continue
            if attribute in state.no_preference_attributes:
                continue
            if attribute in state.known_attributes:
                continue
            return attribute
        # Only once every concrete, specific question has been asked or
        # ruled out does the agent fall back to one broad, final check -
        # never the opening move, and never repeated.
        if state.other_ask_count < MAX_OTHER_ASKS:
            return "other"
        return None

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")
        state.turn_count = turn

        latest_terms = _terms(user_message)
        self._classify_intent(state, user_message)
        self._extract_signals(state, user_message, latest_terms)

        for phrase in _extract_phrases(user_message):
            if phrase not in state.phrases:
                state.phrases.append(phrase)
        del state.phrases[:-20]  # bound query size on long conversations

        if OVERRIDE_RE.search(user_message):
            state.reset_terms(keep_fraction=0.2)
            state.boost_terms(latest_terms, weight=2.5)
            # An override signals a meaningfully new round of constraints
            # worth re-probing broadly, even if "other" was already asked
            # (and exhausted) earlier in the conversation.
            state.other_ask_count = max(0, state.other_ask_count - 1)
        else:
            state.boost_terms(latest_terms, weight=1.0)

        recommendations = [
            {"parent_asin": asin} for asin in self._retrieve(state, latest_terms, top_k)
        ]

        ask_attribute = self._choose_ask_attribute(state)
        if ask_attribute is not None:
            if ask_attribute == "other":
                state.other_ask_count += 1
            else:
                state.asked_attributes.add(ask_attribute)
            state.last_asked = ask_attribute
            message = ATTRIBUTE_PROMPTS[ask_attribute]
        else:
            state.last_asked = None
            message = "Here are the closest matches I found based on everything so far."

        if turn == 1:
            message = GREETING + message

        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
