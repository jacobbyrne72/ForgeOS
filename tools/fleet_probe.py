"""Probe free-tier models and report which are alive THIS WEEK.

This is the data behind the weekly Fleet Report. It sends a 3-token
prompt ("Reply ok") to every free model in the catalog and records
whether it answered, how fast, and what it cost ($0.00 for all of them).

    python tools/fleet_probe.py            # human-readable
    python tools/fleet_probe.py --json     # machine-readable for CI

Spends nothing — every model probed is $0.00/1M input and output.
Requires OPENROUTER_API_KEY in env (free tier, no billing needed).

The output is the "Free Fleet Report" — the recurring, useful, citable
artifact that makes the repo's growth loop run itself. Nobody else
publishes which free models actually work this week.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROBE = "Reply ok"
TIMEOUT = 15  # seconds per model; dead ones hang, so cap it

# Free models known to exist on OpenRouter. The probe discovers which
# are alive TODAY — this list is the seed, not the truth.
FREE_MODELS = [
    "openrouter/nousresearch/hermes-3-llama-3.1-405b:free",
    "openrouter/meta-llama/llama-3.3-70b-instruct:free",
    "openrouter/google/gemma-2-9b-it:free",
    "openrouter/mistralai/mistral-small-3.1-24b-instruct:free",
    "openrouter/qwen/qwen-2.5-72b-instruct:free",
    "openrouter/deepseek/deepseek-chat-v3-0324:free",
    "openrouter/deepseek/deepseek-r1:free",
    "openrouter/nvidia/llama-3.1-nemotron-70b-instruct:free",
]


def probe_model(model_ref: str, api_key: str) -> dict:
    """Send a probe to one model. Returns alive/dead + latency."""
    import httpx

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_ref.replace("openrouter/", "", 1),
        "messages": [{"role": "user", "content": PROBE}],
        "max_tokens": 5,
    }

    t0 = time.time()
    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=TIMEOUT)
        latency = int((time.time() - t0) * 1000)
        if resp.status_code == 200:
            return {"model": model_ref, "alive": True, "latency_ms": latency,
                    "status": 200, "error": ""}
        return {"model": model_ref, "alive": False, "latency_ms": latency,
                "status": resp.status_code, "error": resp.text[:200]}
    except Exception as e:
        latency = int((time.time() - t0) * 1000)
        return {"model": model_ref, "alive": False, "latency_ms": latency,
                "status": 0, "error": str(e)[:200]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--models", nargs="*", default=None,
                    help="override the model list")
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: set OPENROUTER_API_KEY (free tier, no billing)", file=sys.stderr)
        return 1

    models = args.models or FREE_MODELS
    results = []
    for ref in models:
        provider = ref.split("/")[0] if "/" in ref else "unknown"
        r = probe_model(ref, api_key)
        r["provider"] = provider
        results.append(r)
        if not args.json:
            icon = "✅" if r["alive"] else "❌"
            lat = f"{r['latency_ms']}ms" if r["alive"] else ""
            print(f"  {icon} {ref:<55} {lat}")

    alive = sum(1 for r in results if r["alive"])

    if args.json:
        print(json.dumps({
            "date": time.strftime("%Y-%m-%d"),
            "alive": alive,
            "total": len(results),
            "results": results,
        }, indent=2))
    else:
        print(f"\n  {alive}/{len(results)} free tiers alive.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
