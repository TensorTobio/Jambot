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
   │
   ├─ STAGE 2  hybrid retrieval over the 50k catalogue  ──►  30 candidates
   │
   └─ STAGE 3  Haiku rerank    ──►  ordered top 10
```

**Why the LLM is not the retriever.** No prompt holds 50,000 products, so
something has to shortlist first. Stage 2 reuses the indexes from
`starter/retrieval.py` (category bucket, exact-constraint reverse index, BM25),
and the model does the two things it is actually good at: turning a partial,
conversational turn into a clean structured query, and judging semantic fit
inside a shortlist it can read.

**The one rule that carries stage 1.** The prompt insists that every string in
`constraints` be copied *verbatim* from what the customer typed. The retrieval
index is keyed on the customer's exact wording, so a helpful paraphrase silently
destroys an exact match. Interpretation, synonyms and inferred attributes go in
a separate `keywords` field that feeds the fuzzy/BM25 route instead. This is the
single most important line in the prompt.

**Stage 3 output is validated, not trusted.** Ids outside the shortlist are
dropped, duplicates removed, and a short list is topped up from the
deterministic ordering. The model can improve the ranking; it cannot corrupt it
below the rule-based answer.

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

From the repo root (`amazon_data/`):

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
python -m api_call_agent.run_eval --no-rerank      :: stage 1 only
python -m api_call_agent.run_eval --no-rephrase    :: stage 3 only
python -m api_call_agent.run_eval --candidates 50  :: bigger shortlist
```

Useful flags: `--scenario buying|browsing|intent_override|boundary`,
`--limit N`, `--seed N`, `--model`, `--no-cache`, `--verbose`.

## Cost and latency

Model: `claude-haiku-4-5-20251001` — $1 / MTok input, $5 / MTok output, 200K context.

Two calls per turn (rephrase + rerank). A rerank prompt with a 30-product
shortlist is roughly 2k input tokens; a rephrase call is a few hundred. Budget
around **$0.5–1.5 for a full 200-session run**, a few cents for the default 20.
`run_eval` prints measured token counts and an estimated cost after every run —
use those numbers for the disclosure the brief asks for, not this estimate.

Responses are cached on disk under `api_call_agent/.cache/`, keyed by model,
prompts and temperature. Re-running after an unrelated change is free. Delete
the folder or pass `--no-cache` to force fresh calls.

Latency is the real constraint: ~2 sequential calls × ~1 s × ~3 turns × N
sessions. Expect a couple of minutes for 20 sessions and roughly 40 minutes for
the full 200.

## Failure behaviour

Every model call returns `None` rather than raising: missing key, HTTP error,
timeout, malformed JSON. When that happens the turn proceeds on the
deterministic path, so the worst case is exactly the rule-based agent. This is
verified — `selftest.py` runs the whole evaluator with the model stubbed dead
and asserts the score matches `starter/` to the digit.

429/5xx responses are retried three times with exponential backoff.

## Honest expectation

On the **public** set this track will most likely score *below* the rule-based
agent (TechnicalScore 0.9532, Hit Rate 1.0, MRR 0.970). There is almost no
headroom left: the simulated customer speaks in verbatim catalogue strings, and
exact matching already puts the target at rank 1 by turn 3. An LLM can only
reorder a shortlist that is usually already correct.

Where this track earns its place:

- it is the honest implementation of the brief's "LLM semantic ranking" pillar;
- the transcripts read like a real assistant, which is what the demo video needs;
- it degrades gracefully, which is the interesting engineering story;
- `--compare` gives a measured A/B rather than an assertion.

Run `--compare` on a subset before deciding which agent to submit, and report
the ablation numbers either way.

## Files

| File | What it is |
|---|---|
| `llm_client.py` | `urllib` Claude client: retries, disk cache, token accounting, never raises |
| `rephrase.py` | Stage 1 prompt + parsing — customer turn → structured query |
| `rerank.py` | Stage 3 prompt + hard validation — shortlist → ordered top 10 |
| `agent.py` | `Agent` class composing the stages; same contract as `starter/agent.py` |
| `run_eval.py` | Evaluation runner: subset by default, `--compare`, cost report |
| `selftest.py` | Offline wiring test with a stubbed model — no key, no spend |
| `check_key.py` | Diagnoses the key setup and verifies it with one live call |
| `.env.example` | Committed placeholder — copy to `.env` and put the real key there |
| `.gitignore` | Keeps `.env` and `.cache/` out of git |
