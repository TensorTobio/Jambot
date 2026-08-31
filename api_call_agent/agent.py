"""LLM-in-the-loop agent: rephrase -> retrieve -> Haiku picks the top 10.

Per turn:

    customer text
        -> deterministic frame parser (starter.dialog.SessionState)   [backbone]
        -> Haiku rephrase: verbatim constraints + expanded keywords    [stage 1]
        -> hybrid retrieval over the 50k catalogue -> N candidates     [stage 2]
        -> Haiku rerank: order what the evidence left tied             [stage 3]
        -> Haiku reply: the sentence the customer reads                [stage 4]

The deterministic parser stays in front of the model on purpose. It is exact and
free, and the LLM's contribution is what it is good at: reading intent, carrying
context across turns, expanding vocabulary, judging semantic fit inside a
shortlist, and writing like a person. If the API key is missing or every call
fails, this class degrades to exactly the rule-based agent - same contract, same
guarantees, no exceptions.

Each stage is guarded by the layer below it, and the guards all point the same
way - the model may add information, never overwrite it:

* stage 1 output must be a verbatim quote of the customer to reach the
  exact-match store (:class:`LLMSessionState`); otherwise it is demoted to a
  keyword;
* stage 3 may only reorder candidates the evidence has tied, and is skipped
  entirely when the evidence already determines the answer;
* stage 4 chooses no scored field at all, and falls back to the rule-based
  template whenever the sentence fails validation.

Interface is identical to ``starter.agent.Agent`` so the official evaluator can
drive it unchanged.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starter.dialog import SessionState  # noqa: E402
from starter.retrieval import CatalogIndex  # noqa: E402

from .llm_client import DEFAULT_MODEL, ClaudeClient  # noqa: E402
from .rephrase import rephrase  # noqa: E402
from .rerank import MODEL_WEIGHT, rerank, should_rerank  # noqa: E402
from .reply import compose_reply  # noqa: E402

# Same recommend-now policy as the deterministic track: the session ends at the
# first hit and freezes MRR at that rank, so answering with a short, sharp list
# beats answering with a wide one. ``SHOW_SCHEDULE`` is imported rather than
# re-declared so the two tracks cannot drift apart.
from starter.agent import SHOW_SCHEDULE  # noqa: E402

# How many candidates the model is asked to judge. More context costs tokens and
# latency; 30 keeps a rerank call around 2k input tokens.
CANDIDATE_POOL = 30

# Output cap for the customer-facing sentence. Two sentences is roughly 45
# tokens; 70 leaves headroom without letting the model start a list.
REPLY_TOKENS = 70


class LLMSessionState(SessionState):
    """Session state plus the LLM's vocabulary expansion, behind a verbatim gate.

    The gate is the precision mechanism of this track. A constraint string is
    worth ``W_CONSTRAINT`` (1000) in the scorer *only* because it is an exact
    quote of the hidden product's own metadata; a helpful paraphrase that
    happens to collide with some other product's constraint would put 1000
    points on the wrong rows and there is no signal downstream that could undo
    it. So nothing the model writes enters the exact-match store unless it also
    appears, verbatim, in something the customer actually typed.

    Rejected strings are not thrown away - they become keywords, which feed the
    fuzzy and BM25 routes at ``W_FUZZY`` (60). Interpretation still helps; it
    just may not impersonate evidence.
    """

    def __init__(self, session_id: str, user_profile: dict, index) -> None:
        super().__init__(session_id, user_profile, index)
        self.llm_keywords: list[str] = []
        self.llm_search_query: str = ""
        self.accepted = 0
        self.rejected = 0
        # What we have already said, so stage 4 can avoid opening the same
        # way twice - repetition is the loudest tell that a bot is talking.
        self.replies: list[str] = []

    def query_text(self) -> str:
        base = super().query_text()
        extra = " ".join(self.llm_keywords[:15])
        return f"{base} {self.llm_search_query} {extra}".strip()

    # -- the gate ---------------------------------------------------------
    @staticmethod
    def _normalise(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()

    def _transcript(self) -> str:
        return self._normalise(" ".join(self.messages))

    def _is_verbatim(self, value: str) -> bool:
        """Did the customer actually say this, ignoring case and punctuation?"""
        normalised = self._normalise(value)
        # Two characters of signal is not a quote, it is a coincidence.
        return len(normalised) >= 3 and normalised in self._transcript()

    def _add_keyword(self, text: str) -> None:
        lowered = str(text).lower().strip()
        if lowered and lowered not in self.llm_keywords:
            self.llm_keywords.append(lowered)

    def absorb(self, parsed: dict) -> None:
        """Fold a stage-1 result into the slot store, gated on verbatim quoting."""
        if not parsed:
            return

        for value in parsed.get("constraints", []):
            if self._is_verbatim(value):
                before = len(self.constraints)
                self.add_constraint(value)
                self.accepted += len(self.constraints) - before
            else:
                # Demoted, not discarded: it still helps the fuzzy route.
                self.rejected += 1
                self._add_keyword(value)

        for keyword in parsed.get("keywords", []):
            self._add_keyword(keyword)

        if parsed.get("search_query"):
            self.llm_search_query = parsed["search_query"]

        # A budget is a number the customer said. If it is not in the
        # transcript, the model inferred it, and an inferred budget silently
        # re-scores every priced product in the pool.
        budget = parsed.get("budget")
        if budget is not None and self.budget is None:
            for form in (f"{budget:g}", f"{budget:.2f}"):
                if self._is_verbatim(form):
                    self.budget = budget
                    break

        if not self.category and parsed.get("category"):
            if self._is_verbatim(parsed["category"]):
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
        use_reply: bool = True,
        candidate_pool: int = CANDIDATE_POOL,
        reply_tokens: int = REPLY_TOKENS,
        model_weight: float = MODEL_WEIGHT,
        verbose: bool = False,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.index = index if index is not None else CatalogIndex(self.catalog_path)
        self.client = client if client is not None else ClaudeClient(
            model=model, use_cache=use_cache, verbose=verbose
        )
        self.use_rephrase = use_rephrase
        self.use_rerank = use_rerank
        self.use_reply = use_reply
        self.candidate_pool = max(int(candidate_pool), 10)
        self.reply_tokens = max(int(reply_tokens), 16)
        # How loud stage 3's vote is against the retrieval order. 0 disables the
        # reranker's influence entirely without skipping the call; 1.0 makes the
        # two orderings equal partners. See rerank._fuse_within_tiers.
        self.model_weight = max(float(model_weight), 0.0)
        self._sessions: dict[str, LLMSessionState] = {}
        # Observability for the disclosure the brief asks for: how often each
        # stage actually did something, and how often a guard caught the model.
        self.stats = {
            "turns": 0,
            "constraints_added_by_model": 0,
            "constraints_rejected_not_verbatim": 0,
            "rerank_called": 0,
            "rerank_skipped_determined": 0,
            "reply_from_model": 0,
            "reply_from_template": 0,
        }

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

            # How many products this turn is allowed to show. Early turns bet
            # on a single best guess: a miss only costs the turn, while a hit
            # banks rank 1. See ``starter.agent.SHOW_SCHEDULE``.
            if getattr(state, "always_recommend", False):
                shown = limit
            elif state.exhausted:
                shown = limit
            else:
                shown = SHOW_SCHEDULE[min(max(int(turn), 1), len(SHOW_SCHEDULE)) - 1]
            shown = max(min(shown, limit), 1)

            # The window the model is asked about is the window we will show -
            # judging ten positions when only the first is visible is pure spend.
            if self.use_rerank and should_rerank(self.index, candidates, state, shown):
                # -- stage 3: the model orders what the evidence tied ---
                final = rerank(
                    self.client, self.index, candidates, state,
                    top_k=limit, model_weight=self.model_weight,
                )
                self.stats["rerank_called"] += 1
            else:
                # Either stage 3 is off, or the evidence already fixes both the
                # membership and the order of the shown window - the call could
                # not change the answer, so it is not made.
                final = candidates[:limit]
                if self.use_rerank:
                    self.stats["rerank_skipped_determined"] += 1
            final = final[:shown]

            attribute = state.next_ask(
                meta.get("pool_top"), policy=getattr(state, "ask_policy", "other")
            )

            # -- stage 4: the sentence the customer actually reads ------
            # Scored fields are already decided above; this only chooses words,
            # so a failure here costs tone and never a point of MRR.
            message = None
            if self.use_reply:
                message = compose_reply(
                    self.client,
                    state,
                    attribute,
                    shown=len(final),
                    lead_title=self.index.title.get(final[0], "") if final else "",
                    max_tokens=self.reply_tokens,
                )
            if message:
                self.stats["reply_from_model"] += 1
            else:
                message = state.clarification(attribute, len(final))
                self.stats["reply_from_template"] += 1

            state.replies.append(message)
            self.stats["turns"] += 1
            self.stats["constraints_added_by_model"] = state.accepted
            self.stats["constraints_rejected_not_verbatim"] = state.rejected
            return {
                "message": message,
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
        return {**self.client.usage_report(), "stages": dict(self.stats)}
