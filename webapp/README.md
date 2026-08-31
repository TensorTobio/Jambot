# `webapp/` — local browser demo

A local chat UI for trying the agents by hand and for recording the demo video.
Not part of the evaluator or the submission scoring path. Python standard
library only; no build step, no npm.

```bat
cd C:\Users\snattawat\Documents\Jambot
python webapp\server.py
:: then open http://127.0.0.1:8787
```

Flags: `--port 8787`, `--agent rules|llm|keyword` (which one to warm up at
startup — all stay selectable in the UI), `--catalog`, `--dataset`.

Startup loads the catalog and builds one agent (~40 s total). Other agents are
built on first selection.

## Three agents, one contract

| id | Module | What it is | Public-set TechnicalScore |
|---|---|---|---|
| `rules` | `starter/agent.py` | Constraint reverse-index + category filter + BM25, deterministic | **0.9738** |
| `llm` | `api_call_agent/agent.py` | Haiku 4.5 rephrase → retrieve → rerank → reply | 0.964, needs `ANTHROPIC_API_KEY` |
| `keyword` | `starter/agent_keyword.py` | Generic phrase extraction over FTS5 (earlier variant) | 0.6799 |

`rules` and `llm` share one `CatalogIndex`, so switching between them is
instant after the first build. The agent picker in the sidebar shows each one's
description and greys out `llm` when no API key is set.

### What `llm` sounds like

`rules` writes its `message` from a rotating template. `llm` adds a fourth stage
that writes the sentence with the model instead, so the transcript reads like a
person rather than a form:

```
customer: I'm looking for Sets Tracksuits, but I'm still exploring.
agent   : You're just getting a feel for what's out there, makes sense. Are you
          after something you can wear for workouts, or more for lounging around?
customer: For that, what matters is: polyester; Drawstring closure.
agent   : Polyester and a drawstring works for us - that narrows things down
          nicely. Does the fit matter to you, like whether you want something
          more fitted or relaxed?
```

That is a real `/api/autoplay` transcript for `public_0181` (hit at rank 1 on
turn 3, ~$0.01). The scored fields are unchanged by that stage - only the words
are model-written - and a reply that fails validation silently falls back to the
`rules` template. See `api_call_agent/README.md` for the guards.

`llm` costs roughly $0.01 per autoplayed session and takes a second or two per
turn, so a live demo of it is noticeably slower than `rules`. Warm it up before
recording: `python webapp\server.py --agent llm`.

## Question strategy / always-show toggle

Two sidebar controls change how the agent behaves **in the demo only**. The
evaluator calls `reset(session_id, profile)` with two positional arguments and
never reaches these overrides, so the scored configuration cannot drift.

| Question strategy | Recommend | TechnicalScore |
|---|---|---|
| Same attribute (`other`) | short list | **0.9738** (shipped) |
| Same attribute (`other`) | show every turn | 0.9050 |
| Distinguishing (`split`) | short list | 0.9688 |
| Distinguishing (`split`) | show every turn | 0.9046 |
| Rotate | short list | 0.9367 |
| Rotate | show every turn | 0.8990 |

Hit Rate is 1.0 in every combination — the cost is all MRR. The sidebar prints
the measured score for whatever is selected, so the trade-off is visible on
camera. Full reasoning and the ablation harness: `CHANGELOG.md` (v3.2) and
`sweep_policy.py`.

For the demo video: `split` + show-every-turn reads best. For submission:
`other` + short list.

### Robustness to a reworded customer

The scores above all assume the organizer's customer keeps its shipped sentence
frames. The specification does not promise that, so `paraphrase_eval.py`
measures what happens when it does not - it rewrites the wording of the
evaluator's own customer output and leaves the information content untouched:

| customer wording | TechnicalScore |
|---|---|
| shipped frames | **0.9738** |
| lightly reworded | 0.9722 |
| heavily reworded | 0.9722 |
| + attribute values re-cased | 0.9048 |
| + category loosely named | 0.7987 |

Before `starter/dialog.py` grew its free-form fallback, every reworded row
scored **0.0259**. See the v3.4 CHANGELOG entry.


## Auto-play

The **▶ Auto-play as simulated customer** button (enabled once you pick a
public-set profile) runs the entire session server-side using the evaluator's
own `initial_message` / `customer_reply` / override logic, then renders the
transcript. What you see is exactly what the scorer saw — same messages, same
hit rule, same reciprocal rank. This is the thing to record for the demo video.

It also surfaces two behaviours that look like bugs and are not:

- **"No list this turn — on purpose."** The agent withholds recommendations on
  turns 1–3 unless the evidence is already sharp. The session ends at the first
  hit and freezes MRR at that rank, so one more turn of information beats a
  scattershot guess. It always recommends from turn 4.
- **"Target is already in the list, but…"** In `intent_override` sessions the
  evaluator refuses to count a hit until the override message fires on turn 3
  or 4, however good the ranking is before then.

## API

| Route | Method | Body / result |
|---|---|---|
| `/` | GET | the chat page |
| `/api/agents` | GET | selectable agents, their descriptions, availability |
| `/api/samples` | GET | the 200 public sessions for the profile picker |
| `/api/usage` | GET | Haiku token counts and estimated cost so far |
| `/api/session` | POST | `{sample_id, agent}` → session id, profile, revealed target |
| `/api/message` | POST | `{session_id, message}` → one agent turn |
| `/api/autoplay` | POST | `{sample_id, agent}` → the full scored transcript |

The server is a `ThreadingHTTPServer` so a slow LLM auto-play does not block the
rest of the UI. Bad input returns a JSON `error` rather than crashing the demo.

## Note on ground truth

`public_set.jsonl` ships `ground_truth.parent_asin` openly — only the
organizer's 800 private sessions are hidden — so revealing the target in the
sidebar is using development data as intended, not a leak. Hit detection here
mirrors the evaluator exactly: first turn the target id appears anywhere in that
turn's top 10. Clicking a product card is a demo convenience only; real scoring
has no "select" action.
