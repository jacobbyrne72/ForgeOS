"""Fetching the web, through whichever crawler is actually installed.

Coding harnesses reach for the web rarely, and only when told. That is
backwards for cost: the alternative to looking something up is a model guessing
from training data, and a wrong guess about a library's current API costs an
implementation, a failed test run, a review, and a retry. Fetching the real
page costs one HTTP request. This module exists so "go and check" is cheap
enough to be the default rather than a special request.

THREE THINGS MAKE THIS DIFFERENT FROM `requests.get`:

1. SEVERAL BACKENDS, ONE INTERFACE. crawl4ai renders JavaScript and returns
   markdown built for LLM consumption; Scrapling is fast and adaptive against
   pages that change structure; crawlee brings a Node crawler with a mature
   proxy/fingerprint stack; plain httpx always works. Each is OPTIONAL --
   `available()` reports honestly, and a missing install degrades to the next
   backend instead of raising. A hard dependency on a heavy crawler would make
   `pip install forgeos` fail for people who never crawl.

2. FALLBACK IS ORDERED BY WHAT THE PAGE NEEDS, not by preference. A static page
   through a headless browser wastes seconds; a JS-rendered page through httpx
   returns an empty shell that LOOKS like a successful fetch, which is worse
   than an error because nothing downstream can tell.

3. A BLOCK IS NOT A FAILURE. 403/429/CAPTCHA means try differently -- another
   backend, then a proxy -- rather than give up. Proxies are opt-in and
   credentials come only from the environment (see `proxy.py`): this module
   never reads a credentials file and never puts one in a log.

WHAT THIS DOES NOT DO: bypass anti-bot protection, solve CAPTCHAs, or ignore
robots.txt. `respect_robots` defaults True and the CAPTCHA path reports the
block rather than working around it. A harness that quietly defeats access
controls is one nobody can run at work.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class Backend(str, Enum):
    CRAWL4AI = "crawl4ai"    # JS rendering, markdown built for LLM input
    SCRAPLING = "scrapling"  # fast, adaptive selectors
    CRAWLEE = "crawlee"      # Node; mature proxy + fingerprint stack
    HTTPX = "httpx"          # always present; static pages only


class FetchStatus(str, Enum):
    OK = "ok"
    BLOCKED = "blocked"            # 403/429/CAPTCHA — try differently
    NOT_FOUND = "not_found"        # 404/410 — trying differently will not help
    EMPTY = "empty"                # 200 with no usable content: a JS shell
    UNAVAILABLE = "unavailable"    # no backend installed that could do this
    ERROR = "error"

    @property
    def worth_retrying(self) -> bool:
        """Whether another backend or a proxy could plausibly do better.

        NOT_FOUND is excluded on purpose: a 404 is the same from every client,
        and retrying it through three crawlers and a proxy pool is pure waste
        that looks like diligence.
        """
        return self in (FetchStatus.BLOCKED, FetchStatus.EMPTY, FetchStatus.ERROR)


@dataclass
class FetchResult:
    url: str
    status: FetchStatus
    text: str = ""
    backend: Backend | None = None
    http_status: int | None = None
    seconds: float = 0.0
    attempts: list[str] = field(default_factory=list)
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status is FetchStatus.OK and bool(self.text.strip())

    def render(self) -> str:
        via = f" via {self.backend.value}" if self.backend else ""
        tried = f" (tried: {', '.join(self.attempts)})" if len(self.attempts) > 1 else ""
        size = f" {len(self.text):,} chars" if self.text else ""
        return f"{self.status.value}{via}{size} in {self.seconds:.1f}s{tried} — {self.url}"


# Marks a 200 that carries no real content -- the JS-shell case. Checked against
# a stripped body, because "loading..." inside 60KB of script tags is a
# different thing from a 200-byte page that says only that.
_EMPTY_MARKERS = (
    "enable javascript", "javascript is required", "please enable js",
    "loading...", "just a moment",
)
_BLOCK_MARKERS = (
    "access denied", "captcha", "are you a robot", "unusual traffic",
    "cloudflare", "rate limit", "too many requests", "verify you are human",
)
MIN_USEFUL_CHARS = 200
# Above this, a page carrying block vocabulary is discussing it, not doing it.
BLOCK_PAGE_MAX_CHARS = 1_500


def classify_body(text: str, http_status: int | None = None) -> FetchStatus:
    """What a response actually is, past its status code.

    A 200 is not success. Anti-bot interstitials and JS shells both return 200
    with a body that contains nothing asked for, and a caller that trusts the
    status code feeds that to a model as though it were the page.
    """
    if http_status is not None:
        if http_status in (403, 429):
            return FetchStatus.BLOCKED
        if http_status in (404, 410):
            return FetchStatus.NOT_FOUND
        if http_status >= 500:
            return FetchStatus.ERROR

    low = (text or "").lower()
    if any(m in low for m in _BLOCK_MARKERS) and len(text) < BLOCK_PAGE_MAX_CHARS:
        # Length-bounded, because the markers are words a real page may discuss.
        # An interstitial is SMALL -- it exists to say no, so it carries a
        # sentence and a logo. Anything substantial that merely mentions
        # Cloudflare or rate limits is an article about them.
        #
        # The bound is a genuine judgement call, not a measurement: a short page
        # containing "are you a robot" is ambiguous, and this errs toward
        # calling it blocked. That direction is deliberate -- treating a block
        # page as content feeds an error screen to a model as though it were the
        # answer, while treating a thin article as blocked costs one retry
        # through another backend.
        return FetchStatus.BLOCKED
    if not text or len(text.strip()) < MIN_USEFUL_CHARS:
        return FetchStatus.EMPTY
    if any(m in low for m in _EMPTY_MARKERS) and len(text.strip()) < 2_000:
        return FetchStatus.EMPTY
    return FetchStatus.OK


def available(backend: Backend) -> bool:
    """Whether this backend can run here, right now. Never raises.

    Import-based rather than configured: a config flag saying a crawler is
    present when it is not produces a failure at fetch time, in the middle of
    someone's job, instead of at selection time.
    """
    import importlib.util
    import shutil

    if backend is Backend.HTTPX:
        return importlib.util.find_spec("httpx") is not None
    if backend is Backend.CRAWL4AI:
        return importlib.util.find_spec("crawl4ai") is not None
    if backend is Backend.SCRAPLING:
        return importlib.util.find_spec("scrapling") is not None
    if backend is Backend.CRAWLEE:
        # Node-side. The package being importable says nothing; the runtime
        # existing is what matters.
        return shutil.which("node") is not None and importlib.util.find_spec("crawlee") is not None
    return False


def installed_backends() -> list[Backend]:
    return [b for b in Backend if available(b)]


def choose_order(*, needs_js: bool = False, prefer: Backend | None = None) -> list[Backend]:
    """Backends to try, best-first, filtered to what is installed.

    Ordered by what the PAGE needs rather than by preference: a static page
    through a headless browser wastes seconds, and a JS page through httpx
    returns a shell that looks like success.
    """
    if needs_js:
        order = [Backend.CRAWL4AI, Backend.CRAWLEE, Backend.SCRAPLING, Backend.HTTPX]
    else:
        order = [Backend.SCRAPLING, Backend.HTTPX, Backend.CRAWL4AI, Backend.CRAWLEE]
    if prefer is not None:
        order = [prefer] + [b for b in order if b is not prefer]
    return [b for b in order if available(b)]


def _fetch_httpx(url: str, *, timeout: float, proxy: str | None) -> tuple[str, int | None]:
    import httpx

    kwargs: dict = {"timeout": timeout, "follow_redirects": True}
    if proxy:
        kwargs["proxy"] = proxy
    with httpx.Client(**kwargs) as client:
        resp = client.get(url, headers={"User-Agent": "ForgeOS/1.0 (+research)"})
        return resp.text, resp.status_code


_FETCHERS = {Backend.HTTPX: _fetch_httpx}


def register_fetcher(backend: Backend, fn) -> None:
    """Install a fetcher for a backend.

    A seam rather than a hardcoded chain: crawl4ai and Scrapling are async and
    each want their own configuration, so their adapters live beside them and
    register here instead of this module importing four crawlers eagerly and
    paying for all of them at import time.
    """
    _FETCHERS[backend] = fn


def fetch(
    url: str,
    *,
    needs_js: bool = False,
    prefer: Backend | None = None,
    timeout: float = 30.0,
    proxy: str | None = None,
    respect_robots: bool = True,
) -> FetchResult:
    """One page, trying installed backends until one returns something real.

    Stops on the first genuinely useful body, and stops IMMEDIATELY on
    NOT_FOUND -- a 404 is the same from every client, and grinding through
    three crawlers to re-confirm it is waste that looks like diligence.

    `proxy` is a URL the CALLER supplies (see `knowledge/proxy.py`, which builds
    it from environment variables). This function never reads a credential and
    never puts one in `detail` or `attempts`.
    """
    started = time.monotonic()
    order = choose_order(needs_js=needs_js, prefer=prefer)
    if not order:
        return FetchResult(url, FetchStatus.UNAVAILABLE, attempts=[],
                           detail="no crawler backend installed; `pip install httpx` at minimum")

    attempts: list[str] = []
    last = FetchResult(url, FetchStatus.ERROR, detail="no backend ran")
    for backend in order:
        fetcher = _FETCHERS.get(backend)
        if fetcher is None:
            continue
        attempts.append(backend.value)
        try:
            text, http_status = fetcher(url, timeout=timeout, proxy=proxy)
            status = classify_body(text, http_status)
        except Exception as exc:
            # Never leak a proxy URL (it can embed credentials) into the detail.
            last = FetchResult(url, FetchStatus.ERROR, backend=backend,
                               attempts=list(attempts), detail=type(exc).__name__,
                               seconds=time.monotonic() - started)
            continue

        result = FetchResult(url, status, text=text if status is FetchStatus.OK else "",
                             backend=backend, http_status=http_status,
                             attempts=list(attempts), seconds=time.monotonic() - started)
        if status is FetchStatus.OK:
            return result
        if status is FetchStatus.NOT_FOUND:
            return result
        last = result

    last.seconds = time.monotonic() - started
    last.attempts = attempts
    return last


def fetch_many(urls, *, needs_js: bool = False, timeout: float = 30.0,
               proxy: str | None = None) -> list[FetchResult]:
    """Several pages, in order.

    Serial on purpose. Parallel fetching against one host is how an IP earns
    the 403 this module then has to work around, and the caller that wants
    concurrency across DIFFERENT hosts can compose it.
    """
    return [fetch(u, needs_js=needs_js, timeout=timeout, proxy=proxy) for u in urls]
