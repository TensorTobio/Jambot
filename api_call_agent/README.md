# `api_call_agent/` — Claude Haiku 4.5 in the loop

A second, self-contained agent that puts an LLM at the centre of the turn. Same
`Agent` interface as `starter/`, so the official evaluator drives it unchanged.
The rule-based agent in `starter/` is untouched and still the default.

## The pipeline

```
customer turn
   │
   ├─ deterministic frame parser (starter.dialog.SessionState)      backbone, free
   │
   ├─ STAGE 1  Haiku rephrase  ──►  {category, constraints, keywords,
   │                                 budget, scenario, search_query}
   │                                       │
   │                                 verbatim gate ──► paraphrases demoted to keywords
   │
   ├─ STAGE 2  hybrid retrieval over the 50k catalogue  ──►  30 candidates
   │
   ├─ STAGE 3  Haiku rerank    ──►  evidence tiers + weighted rank fusion  ──► top 10
   │
   └─ STAGE 4  Haiku reply     ──►  the sentence the customer actually reads
```

**Why the LLM is not the retriever.** No prompt holds 50,000 products, so
something has to shortlist first. Stage 2 reuses the indexes from
`starter/retrieval.py` (category bucket, exact-constraint reverse index, BM25),
and the model does the things it is actually good at: turning a partial,
conversational turn into a clean structured query, judging semantic fit inside a
shortlist it can read, and writing like a person.

### The design rule: the model may add information, never overwrite it

Every stage is guarded by the layer beneath it, and all three guards point the
same way. This is what makes the track *precise* rather than merely fluent — the
measured result is that the agent now scores **identically** to the rule-based
agent on the public set while every customer-facing sentence is model-written.

**Stage 1 — the verbatim gate.** A constraint string is worth `W_CONSTRAINT`
(1000) in the scorer *only* because it is an exact quote of the hidden product's
own metadata. A helpful paraphrase that happens to collide with some other
product's constraint would put 1000 points on the wrong rows, and nothing
downstream could undo it. So nothing the model writes reaches the exact-match
store unless it also appears, verbatim, in something the customer actually
typed — checked case- and punctuation-insensitively in
`LLMSessionState._is_verbatim`. The same gate covers an inferred budget (which
would silently re-score every priced product) and an inferred category.

Rejected strings are demoted, not discarded: they become keywords and feed the
fuzzy/BM25 route at `W_FUZZY` (60). Interpretation still helps; it just may not
impersonate evidence.

**Stage 3 — tiers, then a weighted vote.** The shortlist the model sees now
carries the *evidence*, not just the products: each row states how many of the
customer's stated requirements that product verifiably matches, which ones, and
how its price sits against a stated budget. Two guards then apply:

* **Evidence tiers (hard).** Candidates are grouped by verified match count and
  the model's opinion is applied only *inside* a group. A product matching two
  requirements can never fall below one matching one, whatever comes back.
* **Rank fusion (soft).** Inside a tier the model does not replace the retrieval
  order, it votes against it — reciprocal-rank scores from both orderings are
  summed, the model's weighted at `MODEL_WEIGHT`. The deterministic order
  carries real signal the tier count throws away (disclosure position, price
  agreement, category, popularity), and discarding that wholesale on the model's
  say-so is exactly how a reranker loses points.

The weight is measured, not guessed. Driving the whole evaluator with a
deliberately **random** reranker — the worst model there is — over 40 public
sessions:

| fusion weight | 0.0 | 0.2 | 0.4 | 0.6 | 1.0 |
|---|---|---|---|---|---|
| MRR | 0.9583 | **0.9583** | 0.9271 | 0.9104 | 0.8938 |

0.2 is the largest weight at which an adversarial model does no damage at all,
so it is the most influence the model can be handed for free, and it is the
default. Because reciprocal rank is steep at the head and flat in the tail, a
0.2 vote can barely move rank 1 but can move something several places at rank
10+ — it bites exactly where the deterministic score has stopped
discriminating, which is where semantics are all that is left. Raise it with
`--model-weight` only on a measured win from the real model.

`should_rerank` also skips the call outright when the evidence already fixes
both the membership and the order of the top 10. It is correct and free, but be
honest about it: with only two or three constraints disclosed there are rarely
ten singleton tiers, so in practice it almost never fires. `run_eval` reports
how often it did.

**Stage 4 — words only.** `message` is the only part of the response a human
reads and the only part the scorer ignores. That asymmetry is the whole design:
the model chooses the *words* and nothing else, so a bad reply costs tone and
never a point of MRR.

* **Bounded** — `max_tokens` defaults to 70, about two sentences (`--reply-tokens`).
* **Grounded** — the prompt carries the customer's own last requirement and the
  title of the product about to be shown, so the sentence refers to real things.
  It also carries *the agent's own last two replies*, with an instruction never
  to open the same way twice; without that the model opens every single turn
  with "So you're after…", which is the loudest tell that a machine is talking.
* **Validated** — `reply._clean` rejects markdown, JSON, product ids, links,
  emoji, boilerplate ("as an AI"), anything over 240 characters, and anything
  that forgot to ask the question. A rejected reply falls back to the rule-based
  template, so the worst case is the old behaviour.

What it sounds like, from `python try_agent.py --api --sample random --n 3`:

```
customer: I'm looking for Medical Scrub Bottoms. A key requirement is: polyester.
agent   : You want something durable that'll hold up through washing - polyester's
          the right call for that. Do you need pockets, or is that not something
          you're fussed about?
customer: For that, what matters is: Pull On closure.
agent   : Pull-on's going to be way easier for getting changed in a rush. Is there
          anything else like fit or pocket count that'd make a difference to you?
customer: I don't have an additional preference for other.
agent   : The Iflex ones here have that pull-on waistband you're after and they've
          got plenty of give in the fabric. What size or fit are you looking for?
```

## Setting the API key

Two ways; either works, and a shell variable always wins over the file.

**A `.env` file (easiest, and what teammates should use):**

```bat
copy api_call_agent\.env.example api_call_agent\.env
:: open api_call_agent\.env and replace the placeholder with your real key
python -m api_call_agent.check_key
```

The file must be named exactly `.env` — Windows Explorer likes to save
`.env.txt`, which will not be found. It holds one line:

```
ANTHROPIC_API_KEY=sk-ant-...
```

`.env` is gitignored at both the repo root and in this folder. `.env.example`
is the committed placeholder and holds no key.

Two locations are checked, in order: `api_call_agent/.env`, then a `.env` at the
repo root. `export KEY=value`, quotes, `#` comments and a UTF-8 BOM are all
handled, and nothing is overwritten if the variable is already set in your
shell. `ANTHROPIC_MODEL` can be set the same way to override the model.

**Or a shell variable, if you prefer not to have the key on disk:**

```bat
set ANTHROPIC_API_KEY=sk-ant-...        :: this terminal only
setx ANTHROPIC_API_KEY "sk-ant-..."     :: persisted; open a new terminal after
```

```bash
export ANTHROPIC_API_KEY=sk-ant-...     # bash / zsh
```

**Check it worked:**

```bat
python -m api_call_agent.check_key
```

It reports where the key came from, whether `.env` is gitignored, and makes one
tiny live call to confirm the key is accepted — a 401 here means the key is
wrong, and the key itself is never printed.

The key is never written to disk by the code, never logged, and never used in a
cache key. No `pip install` — the client is `urllib` from the standard library.

## Running

From the repo root (`Jambot/`):

```bat
:: is the key set up correctly?
python -m api_call_agent.check_key

:: offline wiring test - stubs the model, spends nothing
python -m api_call_agent.selftest

:: 20-session evaluation with live API calls (default)
python -m api_call_agent.run_eval

:: the full public set
python -m api_call_agent.run_eval --limit 200

:: head-to-head against the rule-based agent on the same sessions
python -m api_call_agent.run_eval --compare

:: ablations
python -m api_call_agent.run_eval --no-rerank        :: drop stage 3
python -m api_call_agent.run_eval --no-rephrase      :: drop stage 1
python -m api_call_agent.run_eval --no-reply         :: drop stage 4, template wording
python -m api_call_agent.run_eval --model-weight 0   :: stage 3 called but ignored
python -m api_call_agent.run_eval --model-weight 0.6 :: louder rerank vote
python -m api_call_agent.run_eval --reply-tokens 40  :: terser replies, cheaper
python -m api_call_agent.run_eval --candidates 50    :: bigger shortlist
```

Useful flags: `--scenario buying|browsing|intent_override|boundary`,
`--limit N`, `--seed N`, `--model`, `--no-cache`, `--verbose`.

`run_eval` prints a stage-activity block after every run: how many rerank calls
were made, how many were skipped as already-determined, how many replies came
from the model versus the template, and how many model constraints the verbatim
gate demoted. Those are the numbers to quote in the disclosure.

To see the conversation rather than the score:

```bat
python try_agent.py --api --sample random --n 3
python try_agent.py --api --sample public_0181
```

## Cost and latency

Model: `claude-haiku-4-5-20251001` — $1 / MTok input, $5 / MTok output, 200K context.

Up to three calls per turn (rephrase + rerank + reply). A rerank prompt with a
30-product shortlist is roughly 2k input tokens; rephrase is a few hundred; the
reply call is the cheapest of the three, capped at 70 output tokens.

Measured on a 20-session run (56 turns): **82 calls, 143k input / 15k output
tokens, $0.22**. That extrapolates to roughly **$2 for a full 200-session run**.
Stage 4 is a third of the calls and a small fraction of the cost — `--no-reply`
is the lever if that matters, and `--reply-tokens 40` is the cheaper middle
ground. `run_eval` prints measured token counts and an estimated cost after
every run — use those numbers for the disclosure the brief asks for, not this
estimate.

Responses are cached on disk under `api_call_agent/.cache/`, keyed by model,
prompts and temperature. Re-running after an unrelated change is free. Delete
the folder or pass `--no-cache` to force fresh calls.

Latency is the real constraint: ~3 sequential calls × ~1 s × ~3 turns × N
sessions. Expect a few minutes for 20 sessions and roughly an hour for the full
200. Stage 4 feeds nothing that is scored, so it is the obvious candidate if
this ever needs to be parallelised.

## Failure behaviour

Every model call returns `None` rather than raising: missing key, HTTP error,
timeout, malformed JSON. When that happens the turn proceeds on the
deterministic path, so the worst case is exactly the rule-based agent. This is
verified — `selftest.py` runs the whole evaluator with the model stubbed dead
and asserts the score matches `starter/` to the digit.

All three stubbed failure modes now land on the identical score:

| stubbed model | TechnicalScore |
|---|---|
| `normal` (plausible output, but a **randomly shuffled** rerank) | 0.934167 |
| `garbage` (invalid JSON, unknown ids, a markdown reply) | 0.934167 |
| `dead` (every call returns `None`) | 0.934167 |
| `starter/` rule-based reference | 0.934167 |

The `normal` row is the one worth staring at. Before the tier guard and rank
fusion, an adversarially random reranker dragged MRR from 0.90 to 0.79; it now
cannot move the score at all. That is the precision property stated as a test
rather than as a claim.

`selftest.py` also unit-tests the guards directly: the tier boundary holds
against a model that tries to cross it, the fusion vote is bounded at the
shipped weight but still bites deep in the list, and `reply._clean` rejects
markdown, leaked ids, overlong text, boilerplate and replies that forgot to ask
the question.

429/5xx responses are retried three times with exponential backoff.

## Honest expectation

On the **public** set this track scores *level with* the rule-based agent, not
above it. Measured, 20 sessions, `--compare`:

| | Hit Rate@10 | MRR | MTTC | TechnicalScore |
|---|---|---|---|---|
| `api_call_agent` | 1.0 | 1.0 | 2.8 | 0.964 |
| `starter/` rule-based | 1.0 | 1.0 | 2.8 | 0.964 |
| **delta** | - | - | - | **+0.000** |

There is almost no headroom left to win: the simulated customer speaks in
verbatim catalogue strings, and exact matching already puts the target at rank 1
by turn 3. An LLM can only reorder a shortlist that is usually already correct.

That makes "+0.000" the right result rather than a disappointing one, and it is
a change from where this track started: it previously scored *below* the
rule-based agent. The guards are what closed that gap. The model now contributes
the fluent conversation and the semantic ordering, and provably cannot spend
accuracy to do it.

Where this track earns its place:

- it is the honest implementation of the brief's "LLM semantic ranking" and
  "adaptive clarification" pillars;
- the transcripts read like a real assistant, which is what the demo video and
  the "one demonstrated multi-turn session" deliverable need;
- it degrades gracefully in three separately tested failure modes, which is the
  interesting engineering story;
- `--compare` gives a measured A/B rather than an assertion, and the ablation
  flags give a measured contribution per stage.

One claim here is *not* measured, and is flagged as such: the guards should
matter more on the private set than on the public one. The public sessions are
where exact matching already wins, so there is little for the reranker to fix;
the value of a bounded vote and a verbatim gate shows up when retrieval is less
certain, which is precisely the case this repo cannot measure. Run `--compare`
on a subset before deciding which agent to submit, and report the ablation
numbers either way.

## Files

| File | What it is |
|---|---|
| `llm_client.py` | `urllib` Claude client: retries, disk cache, token accounting, never raises |
| `rephrase.py` | Stage 1 prompt + parsing — customer turn → structured query |
| `rerank.py` | Stage 3 prompt + evidence tiers + weighted rank fusion — shortlist → ordered top 10 |
| `reply.py` | Stage 4 prompt + validation — the customer-facing sentence |
| `agent.py` | `Agent` class composing the stages; same contract as `starter/agent.py` |
| `run_eval.py` | Evaluation runner: subset by default, `--compare`, cost report |
| `selftest.py` | Offline wiring test with a stubbed model — no key, no spend |
| `check_key.py` | Diagnoses the key setup and verifies it with one live call |
| `.env.example` | Committed placeholder — copy to `.env` and put the real key there |
| `.gitignore` | Keeps `.env` and `.cache/` out of git |
