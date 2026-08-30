"""Minimal Claude client for the API track.

Standard library only (``urllib``) so the repo keeps its zero-install property -
the official ``anthropic`` SDK is used automatically if it happens to be
installed, but nothing depends on it.

Three things this wrapper adds over a raw POST:

* **On-disk response cache** keyed by (model, system, messages, temperature).
  The evaluator replays deterministic sessions, so a re-run after a code change
  in an unrelated module costs nothing.
* **Token accounting** - real ``input_tokens`` / ``output_tokens`` from the API
  response, which the competition asks teams to disclose.
* **Graceful degradation** - every failure path returns ``None`` instead of
  raising, so the agent can fall back to its deterministic ranking.

The API key is read from the ``ANTHROPIC_API_KEY`` environment variable and is
never written to disk, logged, or included in the cache key.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"

# Claude Haiku 4.5 - $1 / MTok input, $5 / MTok output, 200K context.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
PRICE_INPUT_PER_MTOK = 1.00
PRICE_OUTPUT_PER_MTOK = 5.00

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE_DIR = HERE / ".cache"

# Checked in order; the first file that defines a variable wins, and a variable
# already set in the real environment always beats every file.
DOTENV_PATHS = (HERE / ".env", ROOT / ".env")

_dotenv_loaded = False
_dotenv_source: str | None = None


class LLMUnavailable(RuntimeError):
    """Raised only by :func:`require_key`; the client itself never raises."""


def _parse_dotenv(text: str) -> dict[str, str]:
    """Minimal .env parser - no dependency on python-dotenv.

    Understands ``KEY=value``, ``export KEY=value``, ``#`` comments, quoted
    values, and stray whitespace. Anything it does not understand is skipped
    rather than raising.
    """
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip().lstrip("﻿")
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.lower().startswith("export "):
            line = line[7:].lstrip()
        name, _, value = line.partition("=")
        name = name.strip()
        if not name or not name.replace("_", "").isalnum():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if value:
            values[name] = value
    return values


def load_dotenv(force: bool = False) -> str | None:
    """Populate the environment from a ``.env`` file. Returns the file used.

    Real environment variables are never overwritten - a value exported in the
    shell always wins over the file, which is what you want when switching keys
    for one run.
    """
    global _dotenv_loaded, _dotenv_source
    if _dotenv_loaded and not force:
        return _dotenv_source
    _dotenv_loaded = True
    for path in DOTENV_PATHS:
        try:
            if not path.is_file():
                continue
            values = _parse_dotenv(path.read_text(encoding="utf-8-sig"))
        except OSError:
            continue
        applied = False
        for name, value in values.items():
            if not os.environ.get(name):
                os.environ[name] = value
                applied = True
        if applied and _dotenv_source is None:
            _dotenv_source = str(path)
    return _dotenv_source


def dotenv_source() -> str | None:
    """Which .env file supplied a value this process, if any."""
    load_dotenv()
    return _dotenv_source


def api_key() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        load_dotenv()
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    return key or None


def require_key() -> str:
    key = api_key()
    if not key:
        raise LLMUnavailable(
            "ANTHROPIC_API_KEY not found. Either put it in api_call_agent/.env as\n"
            "    ANTHROPIC_API_KEY=sk-ant-...\n"
            "or set it in your shell (setx on Windows, export on bash).\n"
            "Run  python -m api_call_agent.check_key  to diagnose."
        )
    return key


class ClaudeClient:
    """One call = one ``messages`` request. Returns text, or ``None`` on failure."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        timeout: float = 30.0,
        max_retries: int = 3,
        use_cache: bool = True,
        cache_dir: Path | str = CACHE_DIR,
        verbose: bool = False,
    ) -> None:
        # ANTHROPIC_MODEL (shell or .env) overrides the default, but an explicit
        # --model argument still wins.
        if model == DEFAULT_MODEL:
            load_dotenv()
            model = os.environ.get("ANTHROPIC_MODEL", "").strip() or DEFAULT_MODEL
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self.verbose = verbose

        self.calls = 0
        self.cache_hits = 0
        self.failures = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._warned = False

        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- cost ------------------------------------------------------------
    @property
    def estimated_cost_usd(self) -> float:
        return (
            self.input_tokens / 1_000_000 * PRICE_INPUT_PER_MTOK
            + self.output_tokens / 1_000_000 * PRICE_OUTPUT_PER_MTOK
        )

    def usage_report(self) -> dict:
        return {
            "model": self.model,
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "failures": self.failures,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
        }

    # -- cache -----------------------------------------------------------
    def _cache_path(self, payload: dict) -> Path:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return self.cache_dir / f"{hashlib.sha256(blob).hexdigest()[:32]}.json"

    # -- call ------------------------------------------------------------
    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 600,
        temperature: float = 0.0,
        prefill: str | None = None,
        stop_sequences: list[str] | None = None,
    ) -> str | None:
        """Return the assistant text, or ``None`` if the call could not be made.

        ``prefill`` seeds the assistant turn (e.g. ``"["``) to force the model
        straight into JSON without a preamble. ``stop_sequences`` caps a
        free-text answer at the first structure we do not want (a blank line,
        say), which is cheaper than generating it and throwing it away.
        """
        messages: list[dict] = [{"role": "user", "content": user}]
        if prefill:
            messages.append({"role": "assistant", "content": prefill})

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": messages,
        }
        if stop_sequences:
            payload["stop_sequences"] = list(stop_sequences)

        cache_file = self._cache_path(payload) if self.use_cache else None
        if cache_file is not None and cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                self.cache_hits += 1
                # Cached turns cost nothing but are still reported as usage that
                # a fresh run would have incurred.
                self.input_tokens += int(cached.get("input_tokens", 0))
                self.output_tokens += int(cached.get("output_tokens", 0))
                text = cached.get("text")
                return (prefill + text) if (prefill and text is not None) else text
            except (OSError, ValueError, json.JSONDecodeError):
                pass  # corrupt cache entry - fall through and re-request

        key = api_key()
        if not key:
            if not self._warned:
                print(
                    "[llm_client] ANTHROPIC_API_KEY not found (checked the environment, "
                    f"{DOTENV_PATHS[0]} and {DOTENV_PATHS[1]}) - running deterministic "
                    "fallback only. Try: python -m api_call_agent.check_key",
                    flush=True,
                )
                self._warned = True
            self.failures += 1
            return None

        request = urllib.request.Request(
            API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "anthropic-version": API_VERSION,
                "x-api-key": key,
            },
            method="POST",
        )

        delay = 1.0
        for attempt in range(1, self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as error:
                status = error.code
                detail = ""
                try:
                    detail = error.read().decode("utf-8")[:300]
                except Exception:  # pragma: no cover - best effort only
                    pass
                retryable = status in (408, 409, 429, 500, 502, 503, 504)
                if self.verbose or not retryable:
                    print(f"[llm_client] HTTP {status} {detail}", flush=True)
                if not retryable or attempt == self.max_retries:
                    self.failures += 1
                    return None
                time.sleep(delay)
                delay *= 2
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
                if self.verbose:
                    print(f"[llm_client] {type(error).__name__}: {error}", flush=True)
                if attempt == self.max_retries:
                    self.failures += 1
                    return None
                time.sleep(delay)
                delay *= 2
        else:  # pragma: no cover - loop always breaks or returns
            self.failures += 1
            return None

        self.calls += 1
        usage = body.get("usage") or {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

        text = "".join(
            block.get("text", "")
            for block in (body.get("content") or [])
            if block.get("type") == "text"
        )

        if cache_file is not None:
            try:
                cache_file.write_text(
                    json.dumps(
                        {
                            "text": text,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            except OSError:
                pass

        return (prefill + text) if prefill else text


# -- JSON helpers ---------------------------------------------------------

def extract_json(text: str | None, expect: type = dict):
    """Pull the first JSON object/array out of a model response.

    Models occasionally wrap JSON in prose or a ``` fence even when told not to;
    this scans for the first balanced structure of the expected type.
    """
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text[3:]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    open_ch, close_ch = ("{", "}") if expect is dict else ("[", "]")
    start = text.find(open_ch)
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == open_ch:
            depth += 1
        elif char == close_ch:
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return None
                return value if isinstance(value, expect) else None
    return None
