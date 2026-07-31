"""Run a deterministic, no-provider smoke test against the ForgeOS dashboard.

With no arguments this starts an isolated local dashboard, writes two synthetic
measured receipts, opens the page in Chromium, and verifies both the JSON API
and rendered leaderboard rollup. Pass ``--url`` to inspect an already-running
dashboard instead. The command never submits work or contacts a provider.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

REPORT_SCHEMA = "forgeos.dashboard_smoke.v1"


def _receipt(model_ref: str, forgeos_usd_micros: int) -> dict[str, object]:
    return {
        "schema": "forgeos.forgebench.v1",
        "mode": "live",
        "provenance": "measured",
        "model_ref": model_ref,
        "suite": {"name": "dashboard-smoke", "savings_class": "A", "tasks": [{"id": "t"}]},
        "totals": {
            "baseline": {"accepted_count": 1, "attempted_count": 1, "usd_micros": 500_000},
            "forgeos": {"accepted_count": 1, "attempted_count": 1, "usd_micros": forgeos_usd_micros},
        },
        "comparison_voided": False,
        "exit_gate_passed": True,
        "proof": {"mission_id": model_ref, "contract_hash": "dashboard-smoke"},
    }


def _write_fixture(receipt_dir: Path) -> None:
    receipt_dir.mkdir(parents=True, exist_ok=True)
    for name, model, cost in (
        ("a.json", "smoke/model-a", 125_000),
        ("b.json", "smoke/model-b", 275_000),
    ):
        (receipt_dir / name).write_text(
            json.dumps(_receipt(model, cost)), encoding="utf-8"
        )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _emit_failure(kind: str, error: BaseException) -> None:
    print(json.dumps({
        "report_schema": REPORT_SCHEMA,
        "ok": False,
        "kind": kind,
        "error": str(error),
    }, indent=2, sort_keys=True), file=sys.stderr)


def _wait_for_server(base_url: str, process: subprocess.Popen[str], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read()[-2_000:] if process.stdout else ""
            raise RuntimeError(f"dashboard exited with {process.returncode}: {output}")
        try:
            with urlopen(f"{base_url}/api/leaderboard", timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError(f"dashboard did not become ready at {base_url}")


def _validate_payload(payload: object, *, expect_fixture: bool) -> dict[str, object]:
    if not isinstance(payload, dict) or payload.get("schema") != "forgeos.leaderboard.v1":
        schema = payload.get("schema") if isinstance(payload, dict) else None
        raise AssertionError(f"unexpected leaderboard schema: {schema!r}")
    rollup = payload.get("fleet_rollup", {})
    if not isinstance(rollup, dict):
        raise AssertionError(f"fleet_rollup is not an object: {rollup!r}")
    if expect_fixture:
        if rollup.get("accepted_count") != 2:
            raise AssertionError(f"unexpected fleet accepted count: {rollup}")
        if abs(float(rollup.get("cost_per_accepted_usd", -1)) - 0.2) > 1e-9:
            raise AssertionError(f"unexpected fleet unit cost: {rollup}")
    return {"schema": payload["schema"], "fleet_rollup": rollup}


def _inspect_api(base_url: str, timeout: float, *, expect_fixture: bool) -> dict[str, object]:
    try:
        with urlopen(f"{base_url}/api/leaderboard", timeout=timeout) as response:
            if response.status != 200:
                raise AssertionError(f"/api/leaderboard returned HTTP {response.status}")
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise AssertionError(f"/api/leaderboard returned HTTP {exc.code}") from exc
    return _validate_payload(payload, expect_fixture=expect_fixture)


async def _inspect(
    base_url: str,
    screenshot: Path | None,
    timeout: float,
    *,
    expect_fixture: bool,
) -> dict[str, object]:
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("Playwright is required; install it and its Chromium browser") from exc

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        try:
            page = await browser.new_page(viewport={"width": 1440, "height": 1000})
            page.set_default_timeout(timeout * 1000)
            api_response = await page.request.get(f"{base_url}/api/leaderboard")
            if api_response.status != 200:
                raise AssertionError(f"/api/leaderboard returned HTTP {api_response.status}")
            payload = await api_response.json()
            validated = _validate_payload(payload, expect_fixture=expect_fixture)
            rollup = validated["fleet_rollup"]

            await page.goto(f"{base_url}/", wait_until="domcontentloaded")
            await page.locator("#leaderboard-body tr").first.wait_for(state="visible")
            rows = await page.locator("#leaderboard-body").inner_text()
            meta = await page.locator("#leaderboard-meta").inner_text()
            if expect_fixture and (
                "fleet 2 accepted" not in meta or "$0.2000/accepted" not in meta
            ):
                raise AssertionError(f"fleet rollup missing from dashboard: {meta!r}")
            if expect_fixture and ("smoke/model-a" not in rows or "smoke/model-b" not in rows):
                raise AssertionError(f"ranked rows missing from dashboard: {rows!r}")
            if screenshot is not None:
                screenshot.parent.mkdir(parents=True, exist_ok=True)
                await page.screenshot(path=str(screenshot), full_page=True)
            return {
                "schema": validated["schema"],
                "fleet_rollup": rollup,
                "rows": rows.splitlines(),
                "meta": meta,
                "screenshot": str(screenshot) if screenshot else None,
            }
        finally:
            await browser.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="inspect an existing dashboard instead of starting a fixture")
    parser.add_argument(
        "--port", type=int, default=0,
        help="local fixture port (default: 0, select a free port; CI/Make pin theirs)",
    )
    parser.add_argument("--timeout", type=float, default=45.0, help="startup/browser timeout in seconds")
    parser.add_argument("--screenshot", type=Path, help="write a full-page verification screenshot")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="run the API-only contract check (used by CI without Playwright)",
    )
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.no_browser and args.screenshot:
        parser.error("--screenshot requires browser mode")

    process: subprocess.Popen[str] | None = None
    try:
        if args.url:
            base_url = args.url.rstrip("/")
        else:
            port = args.port or _free_port()
            root = Path(tempfile.mkdtemp(prefix="forgeos-dashboard-smoke-"))
            state_dir = root / "state"
            receipt_dir = root / "receipts"
            _write_fixture(receipt_dir)
            env = os.environ.copy()
            env["FORGEOS_STATE_DIR"] = str(state_dir)
            env["FORGEOS_LEADERBOARD_DIR"] = str(receipt_dir)
            repo_root = Path(__file__).resolve().parents[1]
            env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
            server_code = (
                "import os, uvicorn; "
                "from forgeos.dashboard.app import create_app; "
                f"uvicorn.run(create_app(os.environ['FORGEOS_STATE_DIR']), "
                f"host='127.0.0.1', port={port})"
            )
            process = subprocess.Popen(
                [sys.executable, "-c", server_code],
                cwd=repo_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            base_url = f"http://127.0.0.1:{port}"
            _wait_for_server(base_url, process, args.timeout)

        if args.no_browser:
            result = _inspect_api(base_url, args.timeout, expect_fixture=not bool(args.url))
        else:
            result = asyncio.run(
                _inspect(base_url, args.screenshot, args.timeout, expect_fixture=not bool(args.url))
            )
        result["mode"] = "existing" if args.url else "fixture"
        result["report_schema"] = REPORT_SCHEMA
        result["ok"] = True
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except AssertionError as exc:
        _emit_failure("assertion", exc)
        return 1
    except (OSError, RuntimeError, TimeoutError) as exc:
        _emit_failure("runtime", exc)
        return 2
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
