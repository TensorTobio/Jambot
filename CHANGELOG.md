# CHANGELOG — TechJam conversational shopping agent

All work is in `starter/`. The evaluator, the public labels, the catalog and the
docs were **not** modified. Every number below comes from the untouched official
scorer: `python3 -m evaluator.local_evaluator`.

---

## 2026-08-31 — v3.4: the agent could not survive being spoken to differently

**No change to the public score (0.9738).** This entry is entirely about a
failure mode that the public set cannot show you.

### The hole

`docs/competition_specification.md` is explicit that the simulator's wording is
not frozen:

> The simulator policy decides what information to reveal. If natural-language
> paraphrasing is added by the organizer, it cannot decide correctness.

Everything the dialog layer knew came from eight regexes matching the shipped
sentence frames exactly. Nothing had ever tested what happens when they do not
match. `paraphrase_eval.py` (new) answers that: it calls the untouched
evaluator's own `initial_message` / `customer_reply`, then rewrites only the
sentence they return. The disclosure logic, the hidden card, the override
schedule and the scoring are all bit-identical to an official run - the only
thing that changes is how the customer phrases itself. The evaluator file is
never modified; the wrappers are installed on the imported module inside the
harness and removed in a `finally`.

The result was not a degradation. It was a cliff:

| level | HR@10 | MRR | MTTC | TS |
|---|---|---|---|---|
| L0 original frames | 1.000 | 0.9800 | 2.010 | **0.9738** |
| L1 *light* rewording | 0.035 | 0.0107 | 10.740 | **0.0259** |
| L2 heavy rewording | 0.035 | 0.0107 | 10.740 | 0.0259 |
| L3 + values re-cased | 0.035 | 0.0107 | 10.740 | 0.0259 |

A 97 % agent becomes a 2.6 % agent because "I'm looking for X. A key
requirement is: Y." became "Hi, I need X. One thing that matters: Y.". The bug
was structural rather than subtle: if no frame matched, `observe()` fell off the
end and set **nothing at all** - not the category, not a constraint, not the
scenario - so the ranker spent all ten turns scoring an empty query.

### The fix: read the message for content, not for shape

The same property the retrieval layer already exploits fixes this. Everything
the customer discloses is a *verbatim string from the hidden product*, so those
values can be looked for directly instead of being parsed out of a sentence.

* **`normalise_constraint()` + `CatalogIndex.by_norm`** (58,800 keys) - case and
  punctuation are collapsed, so `Material:alloy`, `material: alloy` and
  `MATERIAL - ALLOY` all resolve to the same catalog entry. 98 % of keys are
  unambiguous, and for public targets the most-supported spelling is the
  target's own for 776 of 800 constraints.
* **`CatalogIndex.find_constraints()`** - scans the word spans of any message,
  longest-match-first and non-overlapping, so a full feature line beats a single
  word inside it.
* **`CatalogIndex.match_category_fuzzy()`** - token overlap against the
  taxonomy, scored as the fraction of the *category's* tokens the message
  covers, so a one-word message cannot claim a four-word category. A reworded
  customer says "belts", not "Accessories Belts", and on turn 1 the category is
  worth more than any constraint. Because it is a guess and not a reading, a
  category recovered this way scores `W_CATEGORY_FUZZY` (120) instead of
  `W_CATEGORY` (400).
* **`SessionState._observe_freeform()`** - recovers category, override cue,
  exhaustion cue and constraints. It runs **only when no frame matched**, which
  is what keeps L0 at exactly 0.9738 - verified against the official scorer, not
  assumed.

### Three rounds of false positives

The scanner is only useful if it is quiet. Each of these was caught by
inspecting extractions on hand-written paraphrases, and each is now a test:

1. `"Jewelry Necklaces"` donated `jewelry` as a *constraint*. The category name
   is masked out of the text before scanning.
2. `"narrow it down"` donated `down`, and `"nothing more on other"` donated back
   the very attribute we had just asked about. Both are freak one-off constraint
   strings - fewer than three products in 50,000 use them - so bare single words
   now need to carry content (`BARE_WORD_STOP`) as well as exist.
3. A re-cased `imported` was being *dropped*, because it failed an exact-key
   test that `Imported` would have passed. Bare words now also resolve through
   an unambiguous normalised lookup.

### Things this deliberately does not do

* **It does not touch the scoring path.** Making constraint matching itself
  normalisation-aware would recover more of L3, but it also dilutes exact
  matching on L0 - two different raw constraints that normalise together would
  both start counting as hits. 0.9738 is the number that gets scored; it was not
  worth risking for a hypothetical.
* **It does not replace the frames.** They are exact, they are free, and they
  are still tried first. The fallback is a safety net.

### After

| level | what the rewriter does | HR@10 | MRR | MTTC | TS |
|---|---|---|---|---|---|
| L0 | nothing (control) | 1.000 | 0.9800 | 2.010 | **0.9738** |
| L1 | reworded carrier, values quoted | 1.000 | 0.9750 | 2.015 | **0.9722** |
| L2 | heavily reworded carrier | 1.000 | 0.9750 | 2.015 | **0.9722** |
| L3 | + values lower-cased and de-punctuated | 0.945 | 0.8811 | 2.600 | 0.9048 |
| L4 | + category cut to one lower-case word | 0.845 | 0.7734 | 3.790 | 0.7987 |

**L0 is unchanged to the digit**, which is the point: the fallback is inert
whenever a frame matches, so nothing about the scored configuration moved.

L1 and L2 are the realistic cases - a paraphraser rewrites the sentence but goes
on quoting the product's own attributes, because that is where the attributes
come from. Those cost 0.0016.

L3 and L4 are adversarial cases invented here, and they should be read as a
floor rather than a forecast. L4 in particular compounds two independent
degradations; it is where the residual risk now lives, and the diagnosis is
specific: recovering "belts" back to "Accessories Belts" is a *guess*, and over
the 200 public targets it is the wrong guess 80 times against 76 right ones.
Downweighting that guess to `W_CATEGORY_FUZZY` measured neutral (0.7988 →
0.7987) and is kept as a hedge, not claimed as a gain. The honest next step for
L4 is soft evidence over several candidate categories rather than committing to
one, which is a change to the scoring path and was not worth making against a
synthetic adversary.

### Verification

`tests/test_dialog_robustness.py` (new, 10 tests) pins both halves: that the
original frames still parse exactly as before, that reworded and
case-damaged input still resolves, and that the scanner does not invent
constraints out of category names or filler words. Full suite: 13 tests.
`api_call_agent` still passes its selftest in all three stub modes, including
the dead-API path.

---

## 2026-08-30 — v3.3: the list length is the decision, not whether to answer

**TechnicalScore 0.9532 → 0.9738** on the full 200-session official scorer.
Hit Rate stays 1.0, MRR 0.9699 → 0.9800, MTTC 2.890 → 2.010.

| | HR@10 | MRR | MTTC | Efficiency | TechnicalScore |
|---|---|---|---|---|---|
| v3.2 | 1.0 | 0.9699 | 2.890 | 0.811 | 0.9532 |
| **v3.3** | **1.0** | **0.9800** | **2.010** | **0.899** | **0.9738** |

### Where the remaining score actually was

v3.2 was already at Hit Rate 1.0 with 191 of 200 targets landing at rank 1, so
the only two things left to buy were the last 0.009 of MRR and the 0.038 of
Efficiency being lost to MTTC. Efficiency was four times the bigger prize, and
v3.2's recommend-now policy was spending turns to protect an MRR that was
already nearly maxed.

To size the prize before writing any code, every public session was replayed for
all 10 turns with the recommendation list forced empty, so the session never
ends and the target's rank at *every* turn is recorded. Because the simulated
customer's reply depends only on `ask_attribute` and never on the products we
show, that trace is exact, not an approximation. It gives two ceilings:

* **0.9705** — the best any stopping rule could score on top of the v3.2 ranker.
* **0.9922** — the ceiling if every session hit at rank 1 on its earliest legal
  turn (intent-override sessions cannot convert before turn 3 or 4, which alone
  puts a floor of 1.39 under MTTC).

So v3.2 was leaving 0.017 on the table in *stopping* and another 0.022 in
*ranking*. The stopping half is the one with a clean answer.

### 1. `SHOW_SCHEDULE` — a short list is a free bet

The session ends at the first hit, so what a turn really decides is not
*whether* to answer but **how many products to show**. Accepting rank `r` on
turn `t` is worth `0.30/r − 0.02*t`. Two consequences:

* Accepting rank 2 instead of rank 1 costs 0.15, which is **seven turns** of
  Efficiency. Early hits at rank 2+ are almost never worth taking.
* Showing a *short* list costs nothing when it misses. The target is simply not
  in it, the session continues, and the bill is one turn at 0.02.

So the agent opens with its single best guess and widens only when the
conversation stops producing constraints:

```python
SHOW_SCHEDULE = (1, 1, 1, 10)   # products shown on turns 1, 2, 3, 4+
```

This replaces v3.2's confidence heuristic (`best_hits`/`tied_at_best`
thresholds) entirely — and it needs no confidence estimate at all, which is why
it beats one. Measured against the trace, `(1, 1, 1, 10)` scores 0.9741 where
the *oracle* stopping rule for the same ranker scores 0.9748: within 0.0007 of
perfect play, from four integers.

| schedule (products shown per turn) | HR | MRR | MTTC | TS |
|---|---|---|---|---|
| `10,10,10,10` (always show full) | 1.0 | 0.7183 | 1.515 | 0.9052 |
| `0,0,0,10` (v3.2 hold-back, idealised) | 1.0 | 0.9808 | 4.000 | 0.9343 |
| `1,1,10,10` | 1.0 | 0.9634 | 1.960 | 0.9698 |
| `1,1,1,1,10` | 1.0 | 0.9808 | 2.040 | 0.9735 |
| **`1,1,1,10`** | **1.0** | **0.9808** | **2.010** | **0.9741** ← shipped |
| `2,2,2,10` | 1.0 | 0.9058 | 1.835 | 0.9550 |

Those rows are replayed off the 10-turn trace so that all six schedules can be compared without six evaluator runs; the shipped row reproduces as **0.9738** under `python3 -m evaluator.local_evaluator` itself (the 0.0003 is tie-break ordering between equally-scored products).

Widening to the full ten by turn 4 is the safety net, not the strategy: it is
what keeps Hit Rate at 1.0 if the short bets all miss. `state.exhausted` widens
early for the same reason — once the customer has nothing left to disclose there
is no later turn worth waiting for.

### 2. The purchase prior was under-weighted

The hidden targets are real purchase records, so how often a product has been
bought is the strongest signal left once the structured routes tie — and on turn
1 of a Browsing session it is the *only* signal. Ranking each target's own
category bucket by popularity alone already puts 70 of 200 targets at rank 1.
That prior was carrying a weight of 3.

| constant | v3.2 | v3.3 |
|---|---|---|
| `W_POPULARITY` (`average_rating × log10(1+n)`) | 3.0 | **10.0** |
| `W_POPULARITY_N` (`log10(1+n)` alone) | — | **20.0** |
| `W_PROFILE` (profile `preference_tags` hits) | 8.0 | **2.0** |

The two prior shapes are kept because they disagree usefully: the
rating-weighted one prefers a well-liked product, the count-only one prefers a
frequently-bought one. Targets at rank 1 on turn 1 go from 70 to 76, and by turn
4 from 191 to 194.

`W_PROFILE` moved the *wrong* way at 8.0 — the aggregate profile's
`preference_tags` are generic ("fit", "comfort", "style") and match most of the
catalog, so an 8-point boost was mostly adding noise to the tie-break the
popularity prior was trying to win. It is kept at 2.0 rather than 0.0: the two
score identically and a small non-zero weight keeps the personalisation signal
live for a private set whose tags may be sharper.

### 3. Tuned on a split, not on the answer

Weights were fitted by coordinate ascent on **half** the public set (even
sample indices) and checked on the other half. That check earned its keep: full
unrestricted ascent found `W_CONSTRAINT 1000 → 250` and `W_KEYWORD 25 → 400`
worth +0.006 on the training half and **−0.004** on the held-out half — textbook
overfitting to 100 sessions, and both were discarded. Only changes that improved
*both* halves were shipped, and each sits on a broad plateau
(`W_POPULARITY` anywhere in 6–30 scores within 0.0005) rather than a knife edge.

### Things that were tried and did not work

Recorded because a negative result measured on the real scorer is worth more
than an untested idea:

* **User-profile rating affinity.** `average_prior_rating` correlates 0.18 with
  the target's `average_rating`, which sounded usable. Scoring products by
  proximity to it changed the score by 0.0000 at every weight from 0.5 to 50.
  Same for tilting by `rating_style` ("critical" users buying lower-rated
  items). The correlation is real and too weak to rank on.
* **IDF-weighted constraint matching.** Weighting each matched constraint by
  `log(1 + N/df)` so that a rare feature line outranks "cotton": −0.001. The
  constraint routes are already decisive when they fire; the ties they leave are
  between products that match the *same* constraint, where IDF says nothing.
* **Hard category filtering.** The coarse category parsed from the opening line
  matches the target's own bucket for 200 of 200 public sessions, so restricting
  the pool to it looked free. It changed nothing — at `W_CATEGORY` 400 the
  category term already dominates — and it converts a perfect-recall soft signal
  into a single point of failure on the private set. Not shipped.
* **Pure popularity as the only prior** (dropping the rating-weighted shape):
  −0.019. It wins in isolation and loses in combination.

### Files touched

* `starter/agent.py` — `SHOW_SCHEDULE` replaces `FORCE_RECOMMEND_TURN` /
  `CONFIDENT_TIE` / `MIN_CONFIDENT_HITS`.
* `starter/retrieval.py` — `W_POPULARITY`, new `W_POPULARITY_N`, `W_PROFILE`;
  `CatalogIndex.prior_n`.
* `api_call_agent/agent.py` — imports `SHOW_SCHEDULE` from the deterministic
  track rather than re-declaring the policy, so the two cannot drift. The rerank
  gate is now asked about the window that will actually be *shown*, not a
  ten-deep window of which only the first position is visible.
* `sweep.py` — sweeps `SHOW_SCHEDULE` and the prior weights; the old grid swept
  constants that no longer exist.

---

## 2026-08-29 — v3.2: varied questions, and the ask-policy ablation

Two proposals: (a) ask a **different, distinguishing** question each turn rather
than the same one, and (b) **print the top 10 after every question**. Both are
right about a real shopper and both cost score against this simulator. Rather
than argue, `sweep_policy.py` measures all six combinations with the official
evaluator.

### The ablation (full 200 sessions, official scorer)

| Question strategy | Recommend | HR@10 | MRR | MTTC | TechnicalScore |
|---|---|---|---|---|---|
| `other` every turn | hold back | 1.0 | **0.9699** | 2.890 | **0.9532** ← shipped |
| `other` every turn | always show | 1.0 | 0.7163 | 1.595 | 0.9030 |
| `split` (information gain) | hold back | 1.0 | 0.9428 | 3.105 | 0.9407 |
| `split` (information gain) | always show | 1.0 | 0.7232 | 1.640 | 0.9041 |
| `rotate` attributes | hold back | 1.0 | 0.8612 | 3.545 | 0.9075 |
| `rotate` attributes | always show | 1.0 | 0.6997 | 1.780 | 0.8943 |

**Hit Rate is 1.0 in every row.** Nothing here breaks the agent — the entire
cost is MRR, because the session ends at the first hit and freezes the rank.

**Why varied attributes lose (−0.013).** In the simulator's `customer_reply` a
constraint is disclosed when
`attribute == "other" or classify_constraint(value) == attribute`. `"other"`
matches *every* undisclosed constraint, so it returns the cap of two per turn;
a specific attribute returns at most two *of that one type*, usually zero or
one. It is a strict superset — no clever choice of attribute can beat it on
disclosure rate. Asking about the best-splitting attribute is the textbook
answer for a real shopper and the wrong answer here.

**Why showing the top 10 every turn loses (−0.050).** Hitting at rank 8 on turn
1 scores RR 0.125; waiting two turns and hitting at rank 1 scores RR 1.0 and
costs only 0.04 of Efficiency. Always showing trades 0.25 of MRR for 0.13 of
Efficiency — a bad trade at weights 0.30 vs 0.20.

### What actually was repetitive, and is now fixed for free

The *wording* was hard-coded to one sentence and repeated verbatim every turn.
That is presentation, not scoring. `SessionState.clarification()` now rotates
through several phrasings per attribute, varies the opener, and names the most
recent constraint back to the customer ("I have \"3 Year Battery\" down. Here
are 10 that fit so far. Is there another must-have I should factor in?").
Deterministic — indexed by turn, not random, so runs stay reproducible. Zero
score change: still 0.953175 on the nose.

### Files

| File | Status | What changed |
|---|---|---|
| `starter/dialog.py` | modified | Rotating clarification wording; `next_ask(candidates, policy=…)` with `other`/`split`/`rotate`; tracks per-attribute `unhelpful` so an emptied attribute is never re-asked |
| `starter/retrieval.py` | modified | `attribute_split()` — normalised entropy × coverage per attribute over the candidate pool; memoised `constraint_type()`; `meta["pool_top"]` (top 120) for the ask-policy |
| `starter/agent.py` | modified | `ASK_POLICY` / `ALWAYS_RECOMMEND` module settings, constructor arguments, and keyword-only `reset()` overrides |
| `api_call_agent/agent.py` | modified | Same per-session overrides, same defaults |
| `webapp/server.py` | modified | `_reset_options()` whitelists demo overrides from the browser; `/api/session` and `/api/autoplay` accept `options` |
| `webapp/index.html` | modified | "Question strategy" picker + "Show top 10 after every question" toggle, with the measured TechnicalScore for the current combination shown live |
| `sweep_policy.py` | **new** | The ablation harness; writes `results_policy_ablation.json` |

`next_ask` also got a real fix along the way: `"I don't have an additional
preference for X"` previously set `exhausted` for *any* X. It now only means the
whole card is drained when X is `"other"`; otherwise just that attribute is
marked spent. Under the `split` policy this matters, and it is why `split`
improved from 0.9368 to 0.9407 once a 0.5^n decay on already-asked attributes
was added.

### How to use it

The evaluator calls `reset(session_id, profile)` with two positional arguments
and never reaches the overrides, so **the scored configuration cannot drift** —
it stays `other` + hold back. The webapp sets them per session, so you can demo
varied questions with a top 10 every turn and still submit the optimal agent.
The sidebar prints the measured TechnicalScore of whatever combination is
selected, which makes the trade-off visible on camera instead of hidden.

**Recommendation:** demo with `split` + always-show, submit with `other` +
hold-back, and put this ablation table in the Devpost writeup — "we tried the
intuitive thing and measured it" is a stronger story than either result alone.

### Verification

- Default config re-scored after every change: **0.953175**, unchanged to six
  decimal places.
- All six ablation rows are real evaluator runs, one shared index.
- Webapp driven headless with `split` + always-show: varied attributes and
  varied wording confirmed in the transcript, zero console errors.
- `python -m unittest discover -s tests` — 3 tests, OK.

---

## 2026-08-29 — v3.1: `.env` support for the API key

The key can now live in a `.env` file instead of a shell variable, which is what
teammates actually want. Nothing about key handling got looser: the file is
gitignored in two places, the key is still never logged, written to disk by the
code, or used in a cache key.

| File | Status | What changed |
|---|---|---|
| `api_call_agent/llm_client.py` | modified | Added `load_dotenv()` / `dotenv_source()`; `api_key()` falls back to the file; `ANTHROPIC_MODEL` override |
| `api_call_agent/check_key.py` | **new** | Diagnoses the setup and proves the key with one live call |
| `api_call_agent/.env.example` | rewritten | Real copy-paste instructions instead of "the code does not read this" |
| `api_call_agent/README.md` | modified | New "Setting the API key" section |
| `api_call_agent/run_eval.py` | modified | Header now prints the *resolved* model, not the argparse default |
| `.gitignore` (repo root) | **new** | Adopted from the webapp folder's, plus `.env`, `api_call_agent/.env`, `api_call_agent/.cache/` |

**Resolution order:** shell environment → `api_call_agent/.env` → `<repo root>/.env`.
A variable already exported in the shell is never overwritten, so you can point
one run at a different key without editing the file.

The parser is ~25 lines of standard library — no `python-dotenv` dependency —
and tolerates `export KEY=value`, spaces around `=`, single or double quotes,
`#` comments, trailing whitespace and a UTF-8 BOM (Windows Notepad writes one).
Anything it cannot parse is skipped rather than raising.

`.gitignore` deliberately does **not** ignore `results.json`, unlike the copy in
the webapp folder — the submission checklist asks for it from the final clean
evaluator run.

**Verification**

| Case | Result |
|---|---|
| No key anywhere | `check_key` prints the exact path and line to create, exits 1 |
| `.env` with BOM + `export` + spaces + quotes + comment | Parsed; key reaches the API (401 on the deliberately fake key, proving the request was made) |
| Shell variable set *and* `.env` present | Shell wins, `dotenv_source()` reports `None` |
| `ANTHROPIC_MODEL` in `.env` | Picked up; an explicit `--model` argument still overrides it |
| `run_eval` with a `.env` present | Runs without the "key not set" banner |

---

## 2026-08-29 — v3: `webapp/` integrated — browser demo + 3-way agent comparison

Merged the `Webapp_Trying_To_Integrate/` folder into the main repo. That folder
was a full second copy of the kit; only the parts that did not already exist
here were taken, and nothing in `evaluator/`, `docs/` or `data/` was touched.

**Kit-drift check first.** `Webapp_Trying_To_Integrate/evaluator/local_evaluator.py`
differs from ours by 312 bytes — exactly the 312 line endings (CRLF vs LF).
Normalised, the two files are byte-identical. Same for `public_set.jsonl`
(200-byte delta = 200 lines) and the catalog (50,000-byte delta = 50,000 lines).
No evaluator or data drift; the duplicated `data/`, `docs/` and `evaluator/`
trees were not copied.

| File | Status | What it is |
|---|---|---|
| `webapp/server.py` | **new** (rewritten from theirs) | Stdlib JSON API + static server, now multi-agent, threaded, with auto-play |
| `webapp/index.html` | **new** (extended from theirs) | Chat UI + agent picker + auto-play + usage readout |
| `webapp/README.md` | **new** | Setup, routes, what the two "looks-like-a-bug" behaviours mean |
| `starter/agent_keyword.py` | **new** | Their `retrieval_agent.py`, kept as a selectable third agent |
| `tests/test_evaluator.py`, `tests/__init__.py` | **new** | Their evaluator unit tests, adopted as-is — 3 tests, passing |
| `.claude/launch.json` | **new** | Their launch config for the webapp port |

### Benchmarked their agent

`starter/agent_keyword.py` scored through our untouched evaluator, full 200
sessions:

| Agent | HR@10 | MRR | MTTC | TechnicalScore |
|---|---|---|---|---|
| Weak BM25 baseline | 0.125 | 0.068 | 9.81 | 0.1108 |
| `agent_keyword` (theirs) | 0.8350 | 0.5527 | 6.17 | **0.6799** |
| `starter/agent.py` (ours) | 1.0000 | 0.9699 | 2.89 | **0.9532** |

Per scenario theirs is 0.81–0.90 Hit Rate with MTTC around 6. It is a solid
generic NLP/FTS5 agent — the gap is the exact-constraint reverse index, which it
does not have. Kept in the repo because a live A/B in the demo is more
persuasive than a table, and it is a genuine fallback if the exact-match
approach is ever ruled out.

### What changed in the webapp

- **Agent picker.** `rules` / `llm` / `keyword` selectable per session, built
  lazily on first use. `rules` and `llm` share one `CatalogIndex`, so switching
  between them is instant after the first build. `llm` is greyed out with an
  explanation when `ANTHROPIC_API_KEY` is unset.
- **Auto-play.** New `/api/autoplay` runs a whole session server-side with the
  evaluator's own `initial_message` / `customer_reply` / override logic and
  returns the transcript, so the browser shows exactly what the scorer saw —
  same messages, same hit rule, same reciprocal rank. This is the demo-video
  asset.
- **Two behaviours that look like bugs are now explained in the UI.** A
  withheld turn renders "No list this turn — on purpose" with the MRR-vs-latency
  reason; in `intent_override` sessions where the target is already ranked, a
  note says the evaluator will not count a hit until the override fires.
- **Live token/cost readout** in the sidebar when the `llm` agent is selected.
- `ThreadingHTTPServer` instead of `HTTPServer`, so a slow LLM auto-play does
  not block the rest of the UI.
- Single-pass catalog load shared by the display cards, the category map and the
  raw products auto-play needs, instead of reading the 60 MB file twice.

### Verification

- Server driven end to end over HTTP: `/api/agents`, `/api/samples`,
  `/api/session`, `/api/message`, `/api/autoplay` for all three agents, plus
  unknown `sample_id` and unknown `session_id` (both return a JSON `error`, no
  crash).
- Auto-play of `public_0001`: `rules` hits turn 2 rank 1 (RR 1.0); `keyword`
  hits turn 8 rank 1; `llm` with no key falls back and hits turn 4 rank 1.
- UI driven in a headless browser through both flows (auto-play and manual
  chat), screenshots inspected, **zero console errors**.
- `python -m unittest discover -s tests` — 3 tests, OK.

`Webapp_Trying_To_Integrate/` is now redundant and can be deleted.

---

## 2026-08-29 — v2: `api_call_agent/` — Claude Haiku 4.5 in the loop

A **second, parallel agent** in a new `api_call_agent/` folder. `starter/` is
untouched and remains the default submission candidate. Full detail in
`api_call_agent/README.md`.

Per turn: deterministic frame parser (backbone) → **Haiku rephrases** the
customer turn into a structured query → hybrid retrieval returns 30 candidates →
**Haiku reranks** them into the ordered top 10.

| File | Status | What it is |
|---|---|---|
| `api_call_agent/llm_client.py` | **new** | `urllib` Claude client — retries with backoff, disk cache, token accounting, never raises |
| `api_call_agent/rephrase.py` | **new** | Stage 1: customer turn → `{category, constraints, keywords, budget, scenario, search_query}` |
| `api_call_agent/rerank.py` | **new** | Stage 3: shortlist → ordered top 10, output hard-validated |
| `api_call_agent/agent.py` | **new** | `Agent` class composing the stages; identical contract to `starter/agent.py` |
| `api_call_agent/run_eval.py` | **new** | Runner: 20-session default, `--limit`, `--compare`, ablations, cost report |
| `api_call_agent/selftest.py` | **new** | Offline wiring test with a stubbed model — no key, no spend |
| `api_call_agent/README.md`, `.env.example`, `.gitignore` | **new** | Setup, cost, failure behaviour; keeps `.env` and `.cache/` out of git |

Model `claude-haiku-4-5-20251001` ($1 / $5 per MTok, 200K context). Key comes
from the `ANTHROPIC_API_KEY` environment variable only — never on disk, never in
a log, never in a cache key. No `pip install` needed.

**Design decisions worth flagging**

- *The LLM is not the retriever.* No prompt holds 50,000 products, so retrieval
  shortlists and the model judges. It does the two things it is good at:
  normalising a partial conversational turn, and semantic fit inside a
  shortlist it can actually read.
- *Constraints must be quoted verbatim.* The retrieval index is keyed on the
  customer's exact wording, so a helpful paraphrase silently destroys an exact
  match. The prompt forbids paraphrasing in `constraints` and gives
  interpretation its own `keywords` field feeding the fuzzy/BM25 route.
- *Stage 3 is validated, not trusted.* Ids outside the shortlist are dropped,
  duplicates removed, short lists topped up from the deterministic ordering.
  The model can improve the ranking; it cannot corrupt it below the rule-based
  answer.
- *Every failure path returns `None`, not an exception* — missing key, HTTP
  error, timeout, malformed JSON — and the turn continues deterministically.

**Verification** (`python -m api_call_agent.selftest`, model stubbed, 12 sessions):

| stub mode | HR | MRR | MTTC | TS |
|---|---|---|---|---|
| `normal` (stub shuffles the rerank) | 1.0 | 0.3232 | 2.833 | 0.7603 |
| `garbage` (invalid JSON, unknown ids) | 1.0 | 0.9028 | 2.833 | 0.9342 |
| `dead` (every call fails) | 1.0 | 0.9028 | 2.833 | 0.9342 |
| rule-based reference | — | — | — | **0.9342** |

`dead` and `garbage` match the rule-based reference to the digit, which is the
guarantee. `normal` degrading proves stage 3 genuinely drives the final order
rather than being decorative. A live `--compare` run with no key set also
returns a delta of exactly `+0.0`.

**Honest expectation:** on the public set this track will most likely score
*below* v1. The simulated customer speaks in verbatim catalogue strings and
exact matching already reaches Hit Rate 1.0 / MRR 0.970 — there is almost no
headroom to rerank into. Its value is the "LLM semantic ranking" pillar,
demo-quality transcripts, and a measured A/B via `--compare`. Decide with
numbers from a subset run before choosing which agent to submit.

**Cost:** two calls per turn, ~2k input tokens for a 30-product rerank prompt.
Roughly $0.5–1.5 for a full 200-session run, cents for the default 20. Disk
cache makes re-runs free. `run_eval` prints measured tokens and estimated cost —
use those for the competition's cost disclosure.

---

## 2026-08-29 — v1: rule-based constraint agent (no LLM)

### Headline result

| Metric | Weak BM25 baseline | This agent | Δ |
|---|---|---|---|
| Hit Rate@10 | 0.125 | **1.000** | +0.875 |
| MRR | 0.068034 | **0.969917** | +0.902 |
| MTTC | 9.81 | **2.89** | −6.92 |
| Efficiency | 0.119 | **0.811** | +0.692 |
| **TechnicalScore** | **0.1108** | **0.9532** | **+0.842** |

Per scenario (all Hit Rate 1.0):

| Scenario | n | MRR | MTTC |
|---|---|---|---|
| buying | 80 | 0.98125 | 2.60 |
| browsing | 80 | 0.951875 | 2.775 |
| intent_override | 30 | 0.977778 | 3.70 |
| boundary | 10 | 1.000 | 3.70 |

Reported token usage: 0 / 0 (no model calls). Index build ≈ 20 s, full 200-session
run ≈ 44 s on a laptop-class CPU. Python standard library only.

---

## Files

| File | Status | What it is |
|---|---|---|
| `starter/agent.py` | **rewritten** | Composition root: session table, recommend-now policy, guaranteed-valid response |
| `starter/retrieval.py` | **new** | Catalog indexes + candidate scoring (Pillar I) |
| `starter/dialog.py` | **new** | Intent routing, slot store, ask-policy (Pillar II) |
| `starter/agent_baseline.py` | **new** | Verbatim copy of the shipped weak BM25 starter, kept for A/B reference |
| `try_agent.py` | **new** | Hands-on driver: chat with the agent, or replay a labelled session as a printed transcript |
| `sweep.py` | **new** | Threshold sweep harness; builds the index once, re-runs the real evaluator per config |
| `results.json` | regenerated | Output of the final clean evaluator run |
| `CHANGELOG.md` | **new** | This file |
| `evaluator/`, `docs/`, `data/` | untouched | — |

---

## How to run it

From the repo root (`Jambot/`), Python 3.10+, no `pip install` needed:

```bat
cd C:\Users\snattawat\Documents\Jambot

:: 1. the official score -> prints metrics, rewrites results.json  (~45s)
python -m evaluator.local_evaluator

:: 2. watch one labelled session as a transcript (good for the demo video)
python try_agent.py --sample public_0001
python try_agent.py --sample random --n 3 --scenario intent_override

:: 3. talk to it yourself
python try_agent.py

:: 4. re-run the threshold sweep  (~4.5 min, builds the index once)
python sweep.py
```

The LLM track lives in `api_call_agent/` and has its own README:

```bat
set ANTHROPIC_API_KEY=sk-ant-...

python -m api_call_agent.selftest            :: offline wiring test, spends nothing
python -m api_call_agent.run_eval            :: 20 sessions, live API calls
python -m api_call_agent.run_eval --compare  :: head-to-head vs the rule-based agent
python -m api_call_agent.run_eval --limit 200
```

Browser demo (`webapp/README.md` for the detail):

```bat
python webapp\server.py          :: then open http://127.0.0.1:8787
python -m unittest discover -s tests
```

Run from the repo root, not from inside `starter/` — `evaluator` and `starter`
are imported as packages. Index build is ~20 s and happens once per process.
Requires SQLite with FTS5, which stock CPython ships on Windows; check with
`python -c "import sqlite3;sqlite3.connect(':memory:').execute('CREATE VIRTUAL TABLE t USING fts5(a)')"`.

To revert to the shipped weak starter for an A/B, point the evaluator at
`starter/agent_baseline.py` (copy it over `starter/agent.py`).

---

## What changed, and why

### 1. `starter/retrieval.py` (new) — three retrieval routes over one in-memory index

Built once at construction over the 50 000-product catalog:

- `by_category: coarse_category -> [parent_asin]`
- `by_constraint: constraint_string -> [parent_asin]` (reverse index)
- `constraints[asin]` — the **ordered** constraint vocabulary of each product
- `price`, `prior` (rating × log₁₀(1+rating_count)), lower-cased product text
- SQLite FTS5 table for the BM25 keyword route (same schema as the baseline)

Three reasons this works, all derived from reading `evaluator/local_evaluator.py`:

1. **Category route.** The opening line is always
   `"I'm looking for {coarse_category(target.categories)}…"`, and
   `coarse_category` is a pure function of the product's category path. Running
   the same function over the catalog reproduces the exact bucket the target
   sits in. `match_category()` recovers it from the message by longest-name
   match (so `Tees & Blouses T-Shirts` wins over `T-Shirts`). Median bucket:
   ~180 products, from 50 000 — a high-recall filter available from turn 1.
2. **Constraint route.** Every value the simulated customer discloses is a
   *verbatim* string produced by `intent_card()` from the target's own
   `features`/`details`, plus a regex-detected material, a `color: x` line and a
   `budget around $price` line — normalised by `_clean_constraint` (whitespace
   collapsed, `-;,.` stripped, truncated to 180 chars). `constraint_profile()`
   reproduces that pipeline exactly, including the `hard[:2] + (soft[2:4] or
   [0])` slice, so a disclosed string can be looked up as an exact key.
   Measured on the catalog: `(category, first hard constraint)` is already
   unique for **78 %** of products; with all four constraints known the
   candidate pool is ≤10 for **99 %** of the public targets.
3. **Keyword route.** BM25 over FTS5 seeds the pool when the structured routes
   are under-determined, and breaks ties inside it.

Scoring merges the routes (`W_*` constants at the top of the module):
exact constraint match (1000 each) ≫ disclosure-order agreement (150) ≫
category membership (400) > price agreement with a disclosed budget (120) >
token-level fuzzy match for constraints that failed exact lookup (60) >
BM25 agreement (25) > profile `preference_tags` hits (8) > popularity prior (3).

`rank_with_meta()` also returns a confidence read (`best_hits`, `tied_at_best`,
`margin`, `pool_size`) used by the recommend-now policy below.

### 2. `starter/dialog.py` (new) — slot store and ask-policy

Recognises the simulator's sentence frames and extracts constraints from each:

| Frame | Carries |
|---|---|
| `I'm looking for {cat}. A key requirement is: {v}.` | category + `hard[0]` → **buying** |
| `I'm looking for {cat}, but I'm still exploring.` | category → **browsing** |
| `I'm looking for {cat}. {v}` | category + `soft[-1]` → **intent_override** |
| `For that, what matters is: X; Y.` | up to 2 constraints |
| `Actually, ignore my earlier preference. What I need is: {v}.` | `hard[0]` |
| `I don't have a preference for {a}; please use your judgment.` | boundary, no info |
| `I don't have an additional preference for {a}.` | card exhausted |

Three decisions worth flagging:

- **The override does not erase anything.** The weekend plan called for wiping
  the earlier soft preference when `"ignore my earlier preference"` fires. That
  turns out to be wrong: `old_value` is `soft_preferences[-1]`, i.e. *also* a
  verbatim string from the same hidden target. Erasing it throws away a valid
  clue about a product that never changed. The slot store is append-only.
  (`intent_override` MRR is 0.978, the second-best scenario.)
- **`ask_attribute` is `"other"` every turn.** In `customer_reply`, `"other"`
  matches *any* undisclosed constraint type, so it drains 2 values per turn
  where a specific attribute drains at most 2 *of that type*. Once the customer
  says "no additional preference", the card is empty and we rotate through
  specific attributes instead (free, occasionally shakes something loose).
- **Semicolon-safe splitting.** Disclosures are joined with `"; "` but a value
  may contain its own semicolon. `_split_payload` verifies each fragment against
  the catalog constraint index and re-joins greedily rather than shredding a
  real constraint; anything still unresolved is kept as fuzzy text.

### 3. `starter/agent.py` (rewritten) — composition + recommend-now policy

- Composes `CatalogIndex` + `SessionState`; keeps one state per `session_id`.
- **Never raises.** `respond()` is wrapped end to end; on any failure it returns
  a schema-valid empty response, because the evaluator zeroes a throwing turn.
  Also handles `respond()` before `reset()` instead of raising (the shipped
  starter raised there).
- **Recommend-now policy** — the one non-obvious lever. The session *ends* the
  moment the target enters the list, which freezes MRR at that turn's rank.
  MRR is worth 0.30 of the score; one extra turn of latency costs only
  0.02 of Efficiency. So hitting at rank 8 on turn 1 (RR 0.125) is far worse
  than hitting at rank 1 on turn 3 (RR 1.0, −0.04 Efficiency). The agent
  therefore returns an **empty** list on turns 1–3 unless the evidence is
  already sharp (`best_hits ≥ 1` and `≤ 3` products tied at that match count),
  and always recommends from turn 4 on so Hit Rate is never at risk.
  Measured effect: MRR 0.716 → 0.970, TechnicalScore 0.903 → 0.953.
- `LLM_RERANKER` is a module-level hook (`callable(state, candidates, top_k) ->
  [asin]`), `None` by default, wrapped in its own try/except. This is where
  person C's paid-API re-ranker plugs in without touching anything else — and
  the deterministic ordering is the fallback if the API errors or times out.

### 4. `sweep.py` (new)

Builds the index once and re-runs the *real* evaluator for each configuration
in a grid, so no tuning number is estimated. Used for the table below.

---

## Tuning log

Grid over the recommend-now thresholds, full 200-session official evaluator:

| FORCE_RECOMMEND_TURN | CONFIDENT_TIE | HR | MRR | MTTC | TS |
|---|---|---|---|---|---|
| 1 | any | 1.0 | 0.7163 | 1.595 | 0.9030 |
| 2 | 1 / 3 | 1.0 | 0.9053 | 2.230 | 0.9470 |
| 2 | 10 | 1.0 | 0.9053 | 2.225 | 0.9471 |
| 3 | 1 | 1.0 | 0.9527 | 2.740 | 0.9510 |
| 3 | 3 | 1.0 | 0.9527 | 2.680 | 0.9522 |
| 3 | 10 | 1.0 | 0.9502 | 2.620 | 0.9527 |
| 4 | 1 | 1.0 | 0.9699 | 3.010 | 0.9508 |
| **4** | **3** | **1.0** | **0.9699** | **2.890** | **0.9532** ← shipped |
| 4 | 10 | 1.0 | 0.9609 | 2.760 | 0.9531 |

The top five configurations sit within 0.003 of each other, so the choice is not
knife-edge.

## Verification

- **Official scorer only.** Every metric above is from
  `python3 -m evaluator.local_evaluator`; the evaluator and public labels are
  byte-identical to what shipped.
- **Overfitting check.** The 200 sessions were shuffled (seed 7) and split in
  half: TS **0.9555** vs **0.9509** (MRR 0.9715 / 0.9683). The two halves agree,
  and nothing in the agent is keyed to a sample id or a specific ASIN — the
  exploited structure is the *simulator's published logic*, which the brief says
  is identical on the private set.
- **Robustness suite.** `respond()` was fed empty strings, `None`, an integer, a
  truncated opening line, a disclosure of only punctuation, a null byte, a
  5 000-character message, and a call before `reset()`. No exception; every
  response was schema-valid, ≤10 recommendations, no duplicates, and every
  `parent_asin` present in the catalog.

## Known limitations / next steps

- **No LLM in the loop yet.** Score comes entirely from deterministic rules.
  The `LLM_RERANKER` hook is the intended home for the semantic re-ranker; on
  the public set headroom is small (MRR is already 0.970) — the honest place for
  it is the private set and the `message` copy in the demo.
- **Popularity prior is dataset-shaped.** Targets are drawn from the Clothing
  5-core split, so they skew toward well-reviewed products; the popularity term
  is a genuine prior for this sampling but would be weaker on a uniform catalog.
- **Category matching is substring-based.** A pathological category name that is
  a substring of a message for unrelated reasons could mis-bucket; the
  constraint route dominates the score, so the failure mode is soft.
- **`message` text is templated.** Fine for scoring; worth an LLM pass before
  the demo video so the transcripts read naturally.
