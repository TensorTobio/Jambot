"""LLM-in-the-loop agent: rephrase -> retrieve -> Haiku picks the top 10.

Per turn:

    customer text
        -> deterministic frame parser (starter.dialog.SessionState)   [backbone]
        -> Haiku rephrase: verbatim constraints + expanded keywords    [stage 1]
        -> hybrid retrieval over the 50k catalogue -> N candidates     [stage 2]
        -> Haiku rerank: the ordered top 10                            [stage 3]

The deterministic parser stays in front of the model on purpose. It is exact and
free, and the LLM's contribution is what it is good at: reading intent, carrying
context across turns, expanding vocabulary, and judging semantic fit inside a
shortlist. If the API key is missing or every call fails, this class degrades to
exactly the rule-based agent - same contract, same guarantees, no exceptions.

Interface is identical to ``starter.agent.Agent`` so the official evaluator can
drive it unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starter.dialog import SessionState  # noqa: E402
from starter.retrieval import CatalogIndex  # noqa: E402

from .llm_client import DEFAULT_MODEL, ClaudeClient  # noqa: E402
from .rephrase import rephrase  # noqa: E402
from .rerank import rerank  # noqa: E402

# Same recommend-now policy as the deterministic track: the session ends at the
# first hit and freezes MRR at that rank, so a scattershot early list is worse
# than one more turn of information gathering.
FORCE_RECOMMEND_TURN = 4
CONFIDENT_TIE = 3
MIN_CONFIDENT_HITS = 1

# How many candidates the model is asked to judge. More context costs tokens and
# latency; 30 keeps a rerank call around 2k input tokens.
CANDIDATE_POOL = 30


class LLMSessionState(SessionState):
    """Session state plus the LLM's vocabulary expansion for this session."""

    def __init__(self, session_id: str, user_profile: dict, index) -> None:
        super().__init__(session_id, user_profile, index)
        self.llm_keywords: list[str] = []
        self.llm_search_query: str = ""

    def query_text(self) -> str:
        base = super().query_text()
        extra = " ".join(self.llm_keywords[:15])
        return f"{base} {self.llm_search_query} {extra}".strip()

    def absorb(self, parsed: dict) -> None:
        """Fold a stage-1 result into the slot store."""
        if not parsed:
            return
        for value in parsed.get("constraints", []):
            # Only exact catalogue constraints join the high-weight route; the
            # rest still helps as fuzzy text inside the scorer.
            self.add_constraint(value)
        for keyword in parsed.get("keywords", []):
            lowered = keyword.lower().strip()
            if lowered and lowered not in self.llm_keywords:
                self.llm_keywords.append(lowered)
        if parsed.get("search_query"):
            self.llm_search_query = parsed["search_query"]
        if parsed.get("budget") is not None and self.budget is None:
            self.budget = parsed["budget"]
        if not self.category and parsed.get("category"):
            matched = self.index.match_category(parsed["category"])
            if matched:
                self.category = matched
                self.raw_category = matched


class Agent:
    """Conversational recommender, Claude Haiku 4.5 in the loop."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        *,
        index: CatalogIndex | None = None,
        client: ClaudeClient | None = None,
        model: str = DEFAULT_MODEL,
        use_cache: bool = True,
        use_rephrase: bool = True,
        use_rerank: bool = True,
        candidate_pool: int = CANDIDATE_POOL,
        verbose: bool = False,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.index = index if index is not None else CatalogIndex(self.catalog_path)
        self.client = client if client is not None else ClaudeClient(
            model=model, use_cache=use_cache, verbose=verbose
        )
        self.use_rephrase = use_rephrase
        self.use_rerank = use_rerank
        self.candidate_pool = max(int(candidate_pool), 10)
        self._sessions: dict[str, LLMSessionState] = {}

    # ------------------------------------------------------------------
    def reset(
        self,
        session_id: str,
        user_profile: dict,
        *,
        ask_policy: str | None = None,
        always_recommend: bool | None = None,
    ) -> None:
        state = LLMSessionState(session_id, user_profile, self.index)
        state.ask_policy = ask_policy or "other"
        state.always_recommend = bool(always_recommend)
        self._sessions[session_id] = state

    # ------------------------------------------------------------------
    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        before_in = self.client.input_tokens
        before_out = self.client.output_tokens
        try:
            state = self._sessions.get(session_id)
            if state is None:
                state = LLMSessionState(session_id, {}, self.index)
                self._sessions[session_id] = state

            limit = max(int(top_k or 10), 1)

            # -- backbone: deterministic frame parsing ------------------
            state.observe(user_message, turn)

            # -- stage 1: rephrase into a structured query --------------
            if self.use_rephrase:
                parsed = rephrase(
                    self.client,
                    state.messages,
                    state.profile,
                    known_constraints=state.constraints,
                )
                if parsed:
                    state.absorb(parsed)

            # -- stage 2: hybrid retrieval ------------------------------
            pool_size = max(self.candidate_pool, limit)
            candidates, meta = self.index.rank_with_meta(state, top_k=pool_size)

            confident = (
                meta["best_hits"] >= MIN_CONFIDENT_HITS
                and meta["tied_at_best"] <= CONFIDENT_TIE
            )
            withhold = (
                not getattr(state, "always_recommend", False)
                and turn < FORCE_RECOMMEND_TURN
                and not confident
                and not state.exhausted
            )

            if withhold:
                final: list[str] = []
            elif self.use_rerank:
                # -- stage 3: the model picks and orders the top 10 -----
                final = rerank(self.client, self.index, candidates, state, top_k=limit)
            else:
                final = candidates[:limit]

            attribute = state.next_ask(
                meta.get("pool_top"), policy=getattr(state, "ask_policy", "other")
            )
            return {
                "message": state.clarification(attribute, len(final)),
                "ask_attribute": attribute,
                "recommendations": [{"parent_asin": asin} for asin in final],
                "usage": {
                    "prompt_tokens": max(self.client.input_tokens - before_in, 0),
                    "completion_tokens": max(self.client.output_tokens - before_out, 0),
                },
            }
        except Exception:
            # Never raise: the evaluator zeroes any turn that throws.
            return {
                "message": "Let me refine that - what else matters to you?",
                "ask_attribute": "other",
                "recommendations": [],
                "usage": {
                    "prompt_tokens": max(self.client.input_tokens - before_in, 0),
                    "completion_tokens": max(self.client.output_tokens - before_out, 0),
                },
            }

    # ------------------------------------------------------------------
    def usage_report(self) -> dict:
        return self.client.usage_report()
