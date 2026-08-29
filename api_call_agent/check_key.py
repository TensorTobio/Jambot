"""Diagnose the API key setup and make one tiny live call to prove it works.

    python -m api_call_agent.check_key

Reports where the key came from (shell environment or which .env file), whether
the .env file is gitignored, and then spends ~1 cent of a cent on a one-token
request to confirm the key is actually accepted. The key itself is never
printed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_call_agent.llm_client import (  # noqa: E402
    DEFAULT_MODEL,
    DOTENV_PATHS,
    ClaudeClient,
    api_key,
    dotenv_source,
)


def main() -> int:
    print("Anthropic API key check\n" + "-" * 46)

    shell_key = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
    key = api_key()
    source = dotenv_source()

    for path in DOTENV_PATHS:
        state = "found" if path.is_file() else "not present"
        print(f"  .env candidate : {path}  [{state}]")

    if not key:
        print("\n  RESULT: no key found.\n")
        print("  Create a file named exactly  .env  (not .env.txt) at:")
        print(f"      {DOTENV_PATHS[0]}")
        print("  containing one line:")
        print("      ANTHROPIC_API_KEY=sk-ant-...")
        print("\n  Then re-run this command.")
        return 1

    where = "shell environment" if shell_key else f".env file ({source})"
    print(f"\n  key found      : yes, from the {where}")
    print(f"  key length     : {len(key)} characters, starts {key[:7]!r}")
    if not key.startswith("sk-ant-"):
        print("  WARNING        : keys normally start with 'sk-ant-'. Check for a stray"
              " quote, space, or a truncated paste.")

    gitignore = ROOT / ".gitignore"
    ignored = gitignore.is_file() and ".env" in gitignore.read_text(encoding="utf-8", errors="ignore")
    print(f"  .env gitignored: {'yes' if ignored else 'NO - add .env to .gitignore before pushing!'}")

    print(f"\n  calling {DEFAULT_MODEL} once to verify...", flush=True)
    client = ClaudeClient(use_cache=False, verbose=True)
    reply = client.complete(
        "Reply with the single word: ok",
        "ping",
        max_tokens=5,
    )
    if reply is None:
        print("\n  RESULT: the key was found but the call failed. The error above says why"
              " (401 = bad key, 429 = rate limited, 400 = bad request).")
        return 1

    usage = client.usage_report()
    print(f"  model replied  : {reply.strip()[:40]!r}")
    print(f"  tokens         : {usage['input_tokens']} in / {usage['output_tokens']} out")
    print("\n  RESULT: working. You can now run:")
    print("      python -m api_call_agent.run_eval")
    print("      python webapp\\server.py --agent llm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
