"""Web fetching: several backends, honest failure, and no leaked credential.

Two properties carry the weight here.

A 200 IS NOT SUCCESS. Anti-bot interstitials and JavaScript shells both return
200 with a body containing nothing that was asked for. A caller trusting the
status code hands that to a model as though it were the page, and the model
answers confidently from an error screen.

A PROXY URL CAN CONTAIN A PASSWORD. It must never reach a log, a receipt, or a
traceback, and that has to be a property of the types rather than of everyone
remembering to be careful.
"""

from __future__ import annotations

import pytest

from forgeos.knowledge.crawl import (
    BLOCK_PAGE_MAX_CHARS,
    MIN_USEFUL_CHARS,
    Backend,
    FetchStatus,
    choose_order,
    classify_body,
    fetch,
    installed_backends,
    register_fetcher,
)
from forgeos.knowledge.proxy import (
    ENV_PROXY_HOST,
    ENV_PROXY_PASS,
    ENV_PROXY_PORT,
    ENV_PROXY_URL,
    ENV_PROXY_USER,
    ProxyEndpoint,
    ProxyKind,
    ProxyPool,
    shuffled_free,
)

GOOD = "x" * (MIN_USEFUL_CHARS + 500)


# ------------------------------------------------- a 200 is not success


def test_a_block_page_is_not_a_successful_fetch():
    assert classify_body("Access denied. Are you a robot?", 200) is FetchStatus.BLOCKED


def test_a_javascript_shell_is_not_a_successful_fetch():
    """The failure that hurts most: 200, no error, and nothing in it."""
    assert classify_body("<html><body>Please enable JavaScript</body></html>", 200) \
        is FetchStatus.EMPTY


def test_a_short_ambiguous_page_errs_toward_blocked():
    """Stated because it is a judgement call, not a measurement. Treating a
    block page as content feeds an error screen to a model as the answer;
    treating a thin article as blocked costs one retry."""
    short = "Cloudflare. " + ("x" * 400)
    assert len(short) < BLOCK_PAGE_MAX_CHARS
    assert classify_body(short, 200) is FetchStatus.BLOCKED


def test_a_thin_body_is_empty_however_healthy_the_status():
    assert classify_body("ok", 200) is FetchStatus.EMPTY


def test_a_real_article_about_cloudflare_is_not_read_as_a_block():
    """The block markers are length-bounded for exactly this. A page discussing
    an anti-bot vendor must not be discarded as one."""
    # Realistically sized. A block interstitial is a sentence and a logo; an
    # article is not. The heuristic errs toward "blocked" on genuinely short,
    # genuinely ambiguous pages, which is the safer direction -- see
    # BLOCK_PAGE_MAX_CHARS.
    article = "Cloudflare is a CDN. " + ("Discussion of how it works. " * 200)
    assert len(article) > 1_500
    assert classify_body(article, 200) is FetchStatus.OK


@pytest.mark.parametrize("code,expected", [
    (403, FetchStatus.BLOCKED),
    (429, FetchStatus.BLOCKED),
    (404, FetchStatus.NOT_FOUND),
    (410, FetchStatus.NOT_FOUND),
    (500, FetchStatus.ERROR),
    (503, FetchStatus.ERROR),
])
def test_status_codes_map_to_meanings(code, expected):
    assert classify_body(GOOD, code) is expected


def test_a_good_body_with_a_good_status_is_ok():
    assert classify_body(GOOD, 200) is FetchStatus.OK


# ---------------------------------------------------------- retry policy


def test_a_block_is_worth_trying_differently():
    assert FetchStatus.BLOCKED.worth_retrying
    assert FetchStatus.EMPTY.worth_retrying


def test_a_404_is_never_worth_retrying():
    """Grinding a 404 through three crawlers and a proxy pool is waste that
    looks like diligence."""
    assert not FetchStatus.NOT_FOUND.worth_retrying


# ------------------------------------------------------------- fallback


def test_fetch_falls_through_to_a_backend_that_works():
    calls: list[str] = []

    def bad(url, *, timeout, proxy):
        calls.append("bad")
        raise OSError("connection reset")

    def good(url, *, timeout, proxy):
        calls.append("good")
        return GOOD, 200

    register_fetcher(Backend.SCRAPLING, bad)
    register_fetcher(Backend.HTTPX, good)
    try:
        result = fetch("https://example.com/x")
    finally:
        register_fetcher(Backend.SCRAPLING, None)
    assert result.ok and result.backend is Backend.HTTPX
    assert len(result.attempts) >= 1


def test_fetch_stops_immediately_on_not_found():
    calls: list[str] = []

    def counting(url, *, timeout, proxy):
        calls.append("hit")
        return "", 404

    register_fetcher(Backend.HTTPX, counting)
    result = fetch("https://example.com/missing", prefer=Backend.HTTPX)
    assert result.status is FetchStatus.NOT_FOUND
    assert len(calls) == 1, "kept trying a 404 through other backends"


def test_no_backend_installed_reports_unavailable_rather_than_raising(monkeypatch):
    import forgeos.knowledge.crawl as c

    monkeypatch.setattr(c, "choose_order", lambda **kw: [])
    result = c.fetch("https://example.com")
    assert result.status is FetchStatus.UNAVAILABLE
    assert "install" in result.detail


def test_an_exception_never_leaks_the_proxy_url_into_the_detail():
    """A proxy URL can embed `user:password@`. A traceback string is exactly
    the sort of place it must not appear."""
    secret_proxy = "http://alice:hunter2@proxy.example.com:8080"

    def raiser(url, *, timeout, proxy):
        raise RuntimeError(f"failed talking to {proxy}")

    register_fetcher(Backend.HTTPX, raiser)
    result = fetch("https://example.com", prefer=Backend.HTTPX, proxy=secret_proxy)
    blob = result.detail + result.render() + " ".join(result.attempts)
    assert "hunter2" not in blob and "alice" not in blob


# ----------------------------------------------------------- ordering


def test_a_js_page_prefers_a_rendering_backend(monkeypatch):
    import forgeos.knowledge.crawl as c

    monkeypatch.setattr(c, "available", lambda b: True)
    order = c.choose_order(needs_js=True)
    assert order[0] is Backend.CRAWL4AI
    assert order.index(Backend.CRAWL4AI) < order.index(Backend.HTTPX)


def test_a_static_page_does_not_pay_for_a_browser(monkeypatch):
    import forgeos.knowledge.crawl as c

    monkeypatch.setattr(c, "available", lambda b: True)
    order = c.choose_order(needs_js=False)
    assert order.index(Backend.HTTPX) < order.index(Backend.CRAWL4AI)


def test_ordering_only_offers_installed_backends():
    for backend in choose_order():
        assert backend in installed_backends()


def test_httpx_is_available_because_it_is_a_hard_dependency():
    assert Backend.HTTPX in installed_backends()


# ================================================================ proxies


def test_a_proxy_endpoint_cannot_hold_a_password():
    """Structural, not careful. There is no field to put one in."""
    fields = set(ProxyEndpoint.__dataclass_fields__)
    for banned in ("password", "passwd", "secret", "token", "auth", "credential"):
        assert banned not in fields


def test_the_password_never_appears_in_a_rendered_endpoint(monkeypatch):
    monkeypatch.setenv(ENV_PROXY_USER, "alice")
    monkeypatch.setenv(ENV_PROXY_PASS, "hunter2")
    ep = ProxyEndpoint(ProxyKind.PAID, "p", host="h", port=1,
                       user_env=ENV_PROXY_USER, pass_env=ENV_PROXY_PASS)
    assert "hunter2" not in ep.render() and "hunter2" not in repr(ep)


def test_the_url_is_assembled_from_the_environment_at_use(monkeypatch):
    monkeypatch.setenv(ENV_PROXY_USER, "alice")
    monkeypatch.setenv(ENV_PROXY_PASS, "hunter2")
    ep = ProxyEndpoint(ProxyKind.PAID, "p", host="proxy.example.com", port=8080,
                       user_env=ENV_PROXY_USER, pass_env=ENV_PROXY_PASS)
    assert ep.connection_url() == "http://alice:hunter2@proxy.example.com:8080"


def test_a_configured_proxy_with_no_credential_set_fails_closed(monkeypatch):
    """Silently connecting unauthenticated would look like it worked and send
    traffic somewhere nobody chose."""
    monkeypatch.delenv(ENV_PROXY_PASS, raising=False)
    ep = ProxyEndpoint(ProxyKind.PAID, "p", host="h", port=1,
                       user_env=ENV_PROXY_USER, pass_env=ENV_PROXY_PASS)
    assert ep.connection_url() is None


def test_direct_has_no_connection_url():
    assert ProxyEndpoint(ProxyKind.DIRECT, "direct").connection_url() is None


# ------------------------------------------------------------ pool order


def test_direct_is_always_tried_first():
    """A pool starting at a proxy has already lost the argument."""
    pool = ProxyPool.from_environment()
    assert pool.next_endpoint().kind is ProxyKind.DIRECT


def test_paid_outranks_free():
    pool = ProxyPool(endpoints=[])
    pool.add_free(["1.2.3.4:8080"])
    pool.endpoints.append(ProxyEndpoint(ProxyKind.PAID, "paid", url_env=ENV_PROXY_URL))
    assert pool.next_endpoint().kind is ProxyKind.PAID


def test_a_failing_proxy_cools_off():
    pool = ProxyPool(endpoints=[ProxyEndpoint(ProxyKind.PAID, "p", host="h", port=1)])
    ep = pool.next_endpoint()
    pool.record_failure(ep)
    assert not ep.available()
    assert pool.next_endpoint() is None


def test_direct_never_cools_off():
    """Losing DIRECT would force every later fetch through a proxy because one
    page was blocked."""
    pool = ProxyPool.from_environment()
    direct = pool.next_endpoint()
    pool.record_failure(direct)
    assert direct.available()


def test_success_clears_the_cooldown():
    pool = ProxyPool(endpoints=[ProxyEndpoint(ProxyKind.PAID, "p", host="h", port=1)])
    ep = pool.endpoints[0]
    pool.record_failure(ep)
    pool.record_success(ep)
    assert ep.available() and ep.failures == 0


def test_free_proxies_are_capped():
    """A thousand dead endpoints is a thousand timeouts, not resilience."""
    pool = ProxyPool(endpoints=[])
    added = pool.add_free([f"10.0.0.{i}:8080" for i in range(500)], limit=20)
    assert added == 20


def test_malformed_free_entries_are_skipped():
    pool = ProxyPool(endpoints=[])
    assert pool.add_free(["not-a-proxy", "1.2.3.4:notaport", "", "5.6.7.8:3128"]) == 1


def test_selection_is_deterministic():
    pool = ProxyPool(endpoints=[
        ProxyEndpoint(ProxyKind.PAID, "b", host="h2", port=2),
        ProxyEndpoint(ProxyKind.PAID, "a", host="h1", port=1),
    ])
    first = pool.next_endpoint().label
    for _ in range(5):
        assert pool.next_endpoint().label == first


def test_the_free_shuffle_is_reproducible():
    """Everyone hammers the top of a free list, so spreading out helps -- but an
    unseeded shuffle makes a failed crawl impossible to replay."""
    entries = [f"10.0.0.{i}:80" for i in range(20)]
    assert shuffled_free(entries, seed=7) == shuffled_free(entries, seed=7)
    assert shuffled_free(entries, seed=7) != shuffled_free(entries, seed=8)


def test_the_pool_reads_only_environment_variable_names(monkeypatch):
    monkeypatch.setenv(ENV_PROXY_HOST, "proxy.example.com")
    monkeypatch.setenv(ENV_PROXY_PORT, "8080")
    monkeypatch.setenv(ENV_PROXY_USER, "alice")
    monkeypatch.setenv(ENV_PROXY_PASS, "hunter2")
    pool = ProxyPool.from_environment()
    blob = pool.render()
    assert "hunter2" not in blob, "a rendered pool carried a password"
    assert any(e.kind is ProxyKind.PAID for e in pool.endpoints)
