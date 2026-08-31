"""Retrieval layer for the TechJam conversational shopping agent.

Everything here is in-memory and dependency-free (Python standard library only,
plus SQLite FTS5 which ships with CPython). Three complementary routes:

1. **Category route** - the simulated customer's opening line always names the
   coarse category of the hidden target, which is a deterministic function of
   the target's ``categories`` path. Reproducing that function over the catalog
   gives an exact, high-recall bucket from turn 1.
2. **Constraint route** - every constraint the customer discloses is a
   *verbatim* string taken from the target's own ``features`` / ``details``
   (plus a detected material, colour and price line). Reproducing the same
   normalisation over the catalog lets us build a reverse index
   ``constraint -> {parent_asin}`` and intersect disclosures.
3. **Keyword route (BM25)** - SQLite FTS5 over the product text, used to seed
   and to break ties when the structured routes are under-determined.

Scoring merges the three routes; nothing here can raise out to the agent.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

# --------------------------------------------------------------------------
# Normalisation helpers - these MUST mirror the public simulator exactly.
# --------------------------------------------------------------------------

MATERIALS = (
    "cotton", "polyester", "nylon", "leather", "wool",
    "spandex", "silk", "rayon", "fabric",
)
SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")
MATERIAL_RE = re.compile(r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b", re.I)
COLOR_RE = re.compile(r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", re.I)
CONSTRAINT_LIMIT = 180

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "have", "i", "in", "is", "it", "its", "me", "my", "of", "on", "or",
    "please", "prefer", "preference", "some", "that", "the", "this", "to",
    "want", "what", "with", "would", "you", "your", "looking", "matters",
    "still", "exploring", "key", "requirement", "need", "actually", "ignore",
    "earlier", "judgment", "additional", "options", "not", "quite", "right",
    "yet", "ask", "about", "one", "specific", "attribute", "dont", "don",
}

EXCLUDED_CATEGORY_PARTS = {
    "clothing",
    "clothing shoes & jewelry",
    "clothing, shoes & jewelry",
}


def searchable_text(product: dict) -> str:
    parts: list[str] = []
    for field in SEARCH_FIELDS:
        value = product.get(field)
        if isinstance(value, dict):
            parts.extend(f"{key} {item}" for key, item in value.items())
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts).strip()


def flatten_values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def clean_constraint(value: str, limit: int = CONSTRAINT_LIMIT) -> str:
    return re.sub(r"\s+", " ", value).strip(" -;,.\t\n")[:limit].rstrip()


WORD_RE = re.compile(r"[A-Za-z0-9]+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


# A lone word is only a disclosure if it carries content. These never are:
# the ten attribute names are what *we* asked about (the customer echoes them
# back in "no preference for colour"), and the rest are ordinary conversational
# filler that happens to exist as a freak one-off constraint string somewhere in
# 50,000 products - "down" and "other" are each used by fewer than three.
BARE_WORD_STOP = frozenset({
    "category", "material", "color", "colour", "size", "style", "brand",
    "budget", "feature", "use", "case", "other",
    "down", "thing", "things", "sort", "kind", "yeah", "sure", "sorry",
    "please", "really", "maybe", "guess", "narrow", "help", "look", "looks",
    "want", "need", "like", "just", "know", "think", "much", "more", "most",
    "good", "great", "nice", "best", "back", "over", "under", "around",
})


def normalise_constraint(value: str) -> str:
    """Case- and punctuation-insensitive key for a constraint string.

    ``Material:alloy``, ``material: alloy`` and ``MATERIAL - ALLOY`` all collapse
    to ``material alloy``. This is what lets a disclosure still be recognised
    when the customer's phrasing has been rewritten around it.
    """
    return NON_ALNUM_RE.sub(" ", str(value).lower()).strip()


def constraint_profile(product: dict) -> list[str]:
    """The ordered constraint vocabulary the simulator can disclose for a product.

    Mirrors ``intent_card``: material first, colour second, then the flattened
    features/details, then the budget line. Only the first four survive into
    ``hard_constraints`` + ``soft_preferences``, so that is all we index.
    """
    candidates = [*flatten_values(product.get("features")), *flatten_values(product.get("details"))]
    corpus = searchable_text(product)
    material = MATERIAL_RE.search(corpus)
    color = COLOR_RE.search(corpus)
    if material:
        candidates.insert(0, material.group(1).lower())
    if color:
        candidates.insert(1, f"color: {color.group(1).lower()}")
    if product.get("price") not in (None, ""):
        candidates.append(f"budget around ${product['price']}")
    cleaned = list(dict.fromkeys(
        clean_constraint(item) for item in candidates if clean_constraint(item)
    ))
    if not cleaned:
        cleaned = [clean_constraint(str(product.get("title") or "product"))]
    # hard_constraints = cleaned[:2]; soft_preferences = cleaned[2:4] or cleaned[:1]
    ordered = cleaned[:2] + (cleaned[2:4] or cleaned[:1])
    return list(dict.fromkeys(ordered))


def coarse_category(values: list[str]) -> str:
    cleaned: list[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part and part.lower() not in EXCLUDED_CATEGORY_PARTS:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


def terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 2 and token.lower() not in STOPWORDS
    ]


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _price_of(product: dict) -> float | None:
    raw = product.get("price")
    if raw in (None, ""):
        return None
    try:
        return float(str(raw).replace("$", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# Scoring weights - tuned against the official local evaluator.
# --------------------------------------------------------------------------

W_CONSTRAINT = 1000.0     # per exact disclosed-constraint match
W_POSITION = 150.0        # disclosure order agrees with the product's own order
W_CATEGORY = 400.0        # product sits in the category named in the opening line
# A category recovered by token overlap rather than by name is a guess, and a
# measured-bad one: over the 200 public targets, degrading the category to a
# single word and re-resolving it lands on the right bucket 76 times and the
# wrong one 80 times. It is still worth having - a right guess pays more than a
# wrong one costs, because the constraint route corrects the wrong ones a turn
# later - but it must not carry the weight of a verified match.
W_CATEGORY_FUZZY = 120.0  # same signal, sourced from a guess
W_PRICE = 120.0           # price agrees with a disclosed budget
W_FUZZY = 60.0            # partial (token-level) match on an unmatched constraint
W_KEYWORD = 25.0          # BM25 keyword-route agreement
W_PROFILE = 2.0           # long-term profile preference tags
# Purchase prior. The hidden targets are real purchase records, so how often a
# product has been bought is the strongest signal left once the structured
# routes tie - and at turn 1 of a Browsing session it is the *only* signal.
# Two shapes, because they disagree usefully: the rating-weighted one prefers a
# well-liked product, the count-only one prefers a frequently-bought one.
# Tuned by coordinate ascent on a 100/100 split of the public set; the optimum
# is a broad plateau (see CHANGELOG), not a knife edge.
W_POPULARITY = 10.0       # average_rating x log10(1 + rating_number)
W_POPULARITY_N = 20.0     # log10(1 + rating_number) alone

# How many top-scoring candidates the ask-policy reasons about when choosing a
# question. Large enough to be representative, small enough to stay cheap.
SPLIT_POOL = 120


class CatalogIndex:
    """Loads the frozen catalog once and answers ranked candidate queries."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)

        self.asins: list[str] = []
        self.title: dict[str, str] = {}
        self.category: dict[str, str] = {}
        self.price: dict[str, float | None] = {}
        self.prior: dict[str, float] = {}
        self.prior_n: dict[str, float] = {}
        self.profile_text: dict[str, str] = {}
        self.constraints: dict[str, list[str]] = {}
        self.constraint_set: dict[str, frozenset[str]] = {}

        self.by_category: dict[str, list[str]] = defaultdict(list)
        self.by_constraint: dict[str, list[str]] = defaultdict(list)
        # normalised constraint -> the raw spellings that collapse onto it,
        # most-supported first. Powers frame-independent extraction.
        self.by_norm: dict[str, list[str]] = {}
        self.category_names: list[str] = []
        self.category_tokens: dict[str, frozenset[str]] = {}
        self.by_category_token: dict[str, list[str]] = defaultdict(list)
        self._type_cache: dict[str, str] = {}

        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        self._build()

    # -- build ------------------------------------------------------------
    def _build(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                asin = str(product["parent_asin"])
                self.asins.append(asin)
                self.title[asin] = str(product.get("title") or "")
                cat = coarse_category([str(v) for v in (product.get("categories") or [])])
                self.category[asin] = cat
                self.by_category[cat].append(asin)
                self.price[asin] = _price_of(product)

                profile = constraint_profile(product)
                self.constraints[asin] = profile
                self.constraint_set[asin] = frozenset(profile)
                for value in profile:
                    self.by_constraint[value].append(asin)

                rating = product.get("average_rating") or 0.0
                count = product.get("rating_number") or 0
                try:
                    popularity = math.log10(1.0 + float(count))
                    self.prior_n[asin] = popularity
                    self.prior[asin] = float(rating) * popularity
                except (TypeError, ValueError):
                    self.prior_n[asin] = 0.0
                    self.prior[asin] = 0.0
                self.profile_text[asin] = (
                    f"{self.title[asin]} {_text(product.get('features'))} "
                    f"{_text(product.get('details'))}"
                ).lower()

                batch.append((
                    asin,
                    _text(product.get("title")),
                    _text(product.get("categories")),
                    _text(product.get("features")),
                    _text(product.get("details")),
                    _text(product.get("store")),
                    _text(product.get("description")),
                ))
                if len(batch) >= 2000:
                    cursor.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?)", batch)
        self.connection.commit()
        # Longest first so that "Tees & Blouses T-Shirts" wins over "T-Shirts".
        self.category_names = sorted(self.by_category, key=len, reverse=True)
        for name in self.by_category:
            tokens = frozenset(terms(name))
            self.category_tokens[name] = tokens
            for token in tokens:
                self.by_category_token[token].append(name)
        for value in self.by_constraint:
            key = normalise_constraint(value)
            if key:
                self.by_norm.setdefault(key, []).append(value)
        for forms in self.by_norm.values():
            forms.sort(key=lambda v: -len(self.by_constraint[v]))

    # -- lookups ----------------------------------------------------------
    def match_category(self, text: str) -> str | None:
        """Find the coarse category named in an opening message."""
        lowered = text.lower()
        best: str | None = None
        for name in self.category_names:
            if name.lower() in lowered:
                best = name
                break
        return best

    def find_constraints(self, text: str, limit: int = 4, max_span: int = 40) -> list[str]:
        """Every catalog constraint quoted anywhere in ``text``, longest first.

        The frame regexes in ``starter.dialog`` read a disclosure out of the
        *sentence shape* the simulator happens to use. The specification allows
        the organizer to paraphrase that shape on the private set, which would
        leave the regexes matching nothing at all. This is the shape-independent
        route: the values a customer discloses are verbatim strings from the
        hidden product, so we can look for them directly instead.

        Word spans of the message are normalised and looked up in ``by_norm``,
        which also survives a rewriter that lowercases or re-punctuates the
        value. Overlapping matches are resolved longest-first so that a full
        feature line wins over a single word inside it.
        """
        spans = [(m.group(0).lower(), m.start(), m.end())
                 for m in WORD_RE.finditer(str(text))][:150]
        if not spans:
            return []
        hits: list[tuple[int, int, int, str]] = []
        for i in range(len(spans)):
            key = ""
            for j in range(i, min(len(spans), i + max_span)):
                key = f"{key} {spans[j][0]}" if key else spans[j][0]
                if len(key) < 3:
                    continue
                forms = self.by_norm.get(key)
                if not forms:
                    continue
                raw = str(text)[spans[i][1]:spans[j][2]]
                if j == i and key not in MATERIALS:
                    # A lone word is a disclosure only if it carries content.
                    # Without this, "narrow it down" donates "down" and the
                    # attribute we just asked about comes straight back as
                    # "other". Materials skip the test - ``constraint_profile``
                    # inserts those on their own, so they are always real.
                    if len(key) < 4 or key in STOPWORDS or key in BARE_WORD_STOP:
                        continue
                    # ...and only if it is unambiguous: either the customer
                    # spelled a catalog constraint exactly, or exactly one
                    # catalog spelling normalises onto it - which is what lets a
                    # re-cased "imported" still resolve to "Imported".
                    if raw not in self.by_constraint and len(forms) > 1:
                        continue
                # Prefer the customer's own spelling when the catalog has it,
                # so exact-match scoring still fires; otherwise the commonest.
                canonical = raw if raw in self.by_constraint else forms[0]
                hits.append((j - i + 1, i, j, canonical))
        if not hits:
            return []
        hits.sort(key=lambda item: (-item[0], item[1]))
        taken: set[int] = set()
        chosen: list[tuple[int, str]] = []
        for _, start, end, canonical in hits:
            if any(pos in taken for pos in range(start, end + 1)):
                continue
            taken.update(range(start, end + 1))
            chosen.append((start, canonical))
            if len(chosen) >= limit:
                break
        chosen.sort()
        return [canonical for _, canonical in chosen]

    def match_category_fuzzy(self, text: str, min_ratio: float = 0.5) -> str | None:
        """Best category by token overlap, when no name appears verbatim.

        ``match_category`` needs the catalog's own taxonomy string to be present
        in the message. A rewritten customer says "belts", not "Accessories
        Belts" - and the category is the single most valuable field on turn 1,
        so losing it costs more than losing a constraint. This recovers it from
        partial overlap instead.

        Scored by the fraction of the *category's* tokens the message covers, so
        a one-word message cannot claim a four-word category. Ties go to the
        bigger bucket, which is the better prior on a real purchase record.
        """
        tokens = set(terms(text))
        if not tokens:
            return None
        overlaps: dict[str, int] = {}
        for token in tokens:
            for name in self.by_category_token.get(token, ()):
                overlaps[name] = overlaps.get(name, 0) + 1
        best: str | None = None
        best_key: tuple[float, int, int] | None = None
        for name, overlap in overlaps.items():
            size = len(self.category_tokens[name])
            if not size:
                continue
            ratio = overlap / size
            if ratio < min_ratio:
                continue
            key = (ratio, overlap, len(self.by_category[name]))
            if best_key is None or key > best_key:
                best_key = key
                best = name
        return best

    def is_known_constraint(self, value: str) -> bool:
        return value in self.by_constraint

    def constraint_type(self, value: str) -> str:
        """Memoised ``classify_constraint`` - the simulator's own attribute classifier."""
        cached = self._type_cache.get(value)
        if cached is None:
            from starter.dialog import classify_constraint

            cached = classify_constraint(value)
            self._type_cache[value] = cached
        return cached

    def attribute_split(self, candidates: list[str], disclosed: set[str]) -> dict[str, float]:
        """How well each attribute would split the current candidate pool.

        For every attribute, look at the still-undisclosed constraint values the
        candidates carry of that type and score it by how evenly it partitions
        them (normalised entropy x coverage). A high score means asking about
        that attribute is expected to eliminate the most candidates.
        """
        buckets: dict[str, dict[str, int]] = {}
        covered: dict[str, int] = {}
        for asin in candidates:
            seen_types: set[str] = set()
            for value in self.constraints[asin]:
                if value in disclosed:
                    continue
                attribute = self.constraint_type(value)
                bucket = buckets.setdefault(attribute, {})
                bucket[value] = bucket.get(value, 0) + 1
                seen_types.add(attribute)
            for attribute in seen_types:
                covered[attribute] = covered.get(attribute, 0) + 1

        total = max(len(candidates), 1)
        scores: dict[str, float] = {}
        for attribute, bucket in buckets.items():
            if len(bucket) < 2:
                continue
            n = sum(bucket.values())
            entropy = -sum((c / n) * math.log2(c / n) for c in bucket.values())
            normalised = entropy / math.log2(len(bucket))
            scores[attribute] = normalised * (covered.get(attribute, 0) / total)
        return scores

    def keyword_route(self, text: str, limit: int = 60) -> list[str]:
        unique = list(dict.fromkeys(terms(text)))[:32]
        if not unique:
            return []
        expression = " OR ".join(f'"{term}"' for term in unique)
        try:
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
                (expression, limit),
            ).fetchall()
        except sqlite3.Error:
            return []
        return [str(row[0]) for row in rows]

    # -- ranking ----------------------------------------------------------
    def rank(self, state, top_k: int = 10) -> list[str]:
        return self.rank_with_meta(state, top_k)[0]

    def rank_with_meta(self, state, top_k: int = 10) -> tuple[list[str], dict]:
        """Score and return up to ``top_k`` parent_asins plus a confidence read.

        ``meta`` reports how sharp the current evidence is, which the agent uses
        to decide whether recommending *now* is better than waiting one more
        turn for a cleaner rank-1 (MRR is worth 0.30, a turn of latency 0.02).
        """
        disclosed = state.constraints
        known = [c for c in disclosed if c in self.by_constraint]
        unknown = [c for c in disclosed if c not in self.by_constraint]

        pool: set[str] = set()
        if known:
            for value in known:
                pool.update(self.by_constraint[value])
        if state.category and state.category in self.by_category:
            category_pool = self.by_category[state.category]
            if known:
                # Constraint route is the sharp one; the category bucket only
                # needs to join in when it is small enough to score cheaply.
                if len(category_pool) <= 4000:
                    pool.update(category_pool)
            else:
                pool.update(category_pool)
        keyword_pool: list[str] = []
        if len(pool) < top_k * 5:
            keyword_pool = self.keyword_route(state.query_text(), limit=200)
            pool.update(keyword_pool)
        if not pool:
            keyword_pool = self.keyword_route(state.query_text(), limit=200)
            pool.update(keyword_pool)
        if not pool:
            pool.update(self.asins[: top_k * 20])

        keyword_rank = {asin: i for i, asin in enumerate(keyword_pool)}
        fuzzy_terms = [set(terms(c)) for c in unknown if terms(c)]
        profile_tags = [t for t in state.profile_tags if t]
        budget = state.budget
        category_weight = (
            W_CATEGORY if getattr(state, "category_confident", True) else W_CATEGORY_FUZZY
        )

        scored: list[tuple[float, float, str]] = []
        best_hits = 0
        hit_histogram: dict[int, int] = {}
        for asin in pool:
            cset = self.constraint_set[asin]
            hits = 0
            for value in known:
                if value in cset:
                    hits += 1
            hit_histogram[hits] = hit_histogram.get(hits, 0) + 1
            best_hits = max(best_hits, hits)
            score = W_CONSTRAINT * hits

            if hits:
                ordered = self.constraints[asin]
                position_hits = sum(
                    1 for i, value in enumerate(known)
                    if i < len(ordered) and ordered[i] == value
                )
                score += W_POSITION * position_hits

            if state.category and self.category[asin] == state.category:
                score += category_weight

            if budget is not None:
                price = self.price[asin]
                if price is not None:
                    if abs(price - budget) < 1e-6:
                        score += W_PRICE
                    elif budget > 0 and abs(price - budget) / budget <= 0.25:
                        score += W_PRICE * 0.4

            if fuzzy_terms:
                text = self.profile_text[asin]
                for group in fuzzy_terms:
                    covered = sum(1 for term in group if term in text)
                    if covered:
                        score += W_FUZZY * (covered / len(group))

            if asin in keyword_rank:
                score += W_KEYWORD * (1.0 - keyword_rank[asin] / max(len(keyword_pool), 1))

            if profile_tags:
                text = self.profile_text[asin]
                score += W_PROFILE * sum(1 for tag in profile_tags if tag in text)

            score += W_POPULARITY * self.prior[asin]
            score += W_POPULARITY_N * self.prior_n[asin]
            scored.append((-score, -self.prior[asin], asin))

        scored.sort()
        top = scored[:top_k]
        meta = {
            # The best-scoring slice, for the ask-policy to reason about.
            "pool_top": [asin for _, _, asin in scored[:SPLIT_POOL]],
            "pool_size": len(pool),
            "known_constraints": len(known),
            "best_hits": best_hits,
            # how many products match as many constraints as the best one does
            "tied_at_best": hit_histogram.get(best_hits, 0),
            "top_score": -top[0][0] if top else 0.0,
            "runner_up_score": -scored[1][0] if len(scored) > 1 else 0.0,
        }
        meta["margin"] = meta["top_score"] - meta["runner_up_score"]
        return [asin for _, _, asin in top], meta
