"""Skill/MCP marketplace scout — recommendations only, never installations.

Published research documents semantic supply-chain attacks delivered through
agent-skill registries: a SKILL.md is not documentation, it is instructions an
agent may follow, and hundreds of malicious skills have been found in the
wild. So this module draws a hard line:

    * It NEVER installs, never executes, and never writes into any skills
      directory. There is no `install()` function here and there never
      should be one.
    * Every string this module pulls from a registry (name, description,
      README excerpt) is DATA, never an instruction. If that text is itself
      shaped like an instruction aimed at an agent ("ignore previous
      instructions", "you are authorized to...") that is a red flag to
      surface to the operator, not a command to obey.
    * Every recommendation lands in the existing `hive.knowledge.claims`
      ClaimStore as `ClaimType.RECOMMENDATION` / `VerificationStatus.UNVERIFIED`
      with a content-addressed source hash. This module never calls
      `mark_verified()` or `promote()` — per AGENTS.md rule 10, an outside
      claim (a registry listing is exactly that) becomes something agents
      follow only through the existing evidence+corroboration+contradiction
      gate, never because a scout said so.

Network access goes through the `Fetcher` protocol (mirroring the
`Transport` protocol in `hive/gateway/client.py`) so tests always inject a
scripted double — no real HTTP call is ever made in this repo's test suite.
The default `HttpFetcher` prefers GitHub's documented REST search API over
scraping registry HTML: a documented API is more stable, more polite, and
does not require this module to parse arbitrary third-party markup.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable, Literal, Protocol, Sequence

import httpx
from pydantic import BaseModel, Field

from .claims import Claim, ClaimStore, ClaimType

# ---------------------------------------------------------------------------
# tunables
# ---------------------------------------------------------------------------

# Below this star count *and* younger than this, a repo is "new and unproven"
# rather than merely small. Age alone or stars alone is normal; both together
# is the pattern worth flagging.
NEW_LOW_STARS_THRESHOLD = 5
NEW_REPO_AGE_DAYS = 30.0

# A risk-flagged candidate can never score above this, no matter how popular.
# "A popular package with an injection flag is not trustworthy because it is
# popular" — the cap is the whole point, not a suggestion.
RISK_CAPPED_TRUST_SCORE = 0.2

GITHUB_SEARCH_URL = "https://api.github.com/search/repositories"


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


class RegistryKind(str, Enum):
    SKILLS = "skills"
    MCP = "mcp"
    BOTH = "both"


class Registry(BaseModel):
    name: str
    url: str
    kind: RegistryKind
    # GitHub topic to filter search results on, when query_via == "github_api".
    topic: str | None = None
    # "github_api": actively queried by search() via the documented REST API.
    # "unsupported": catalogued for operator awareness only. No public search
    # API exists for these (mcp.so, mcpservers.org, the awesome-list repos),
    # and scraping their HTML is deliberately out of scope for this module —
    # a documented API beats a scraper that breaks on the next redesign.
    query_via: Literal["github_api", "unsupported"] = "github_api"


DEFAULT_REGISTRIES: list[Registry] = [
    Registry(
        name="github-topic-mcp-server",
        url="https://github.com/topics/mcp-server",
        kind=RegistryKind.MCP,
        topic="mcp-server",
        query_via="github_api",
    ),
    Registry(
        name="github-topic-model-context-protocol",
        url="https://github.com/topics/model-context-protocol",
        kind=RegistryKind.MCP,
        topic="model-context-protocol",
        query_via="github_api",
    ),
    Registry(
        name="github-topic-claude-skill",
        url="https://github.com/topics/claude-skill",
        kind=RegistryKind.SKILLS,
        topic="claude-skill",
        query_via="github_api",
    ),
    Registry(
        name="github-topic-agent-skill",
        url="https://github.com/topics/agent-skill",
        kind=RegistryKind.SKILLS,
        topic="agent-skill",
        query_via="github_api",
    ),
    # Catalogued from AGENTS.md-adjacent knowledge. No documented search API,
    # so search() records these as known sources but does not scrape them.
    Registry(name="mcp.so", url="https://mcp.so", kind=RegistryKind.MCP, query_via="unsupported"),
    Registry(
        name="mcpservers.org", url="https://mcpservers.org", kind=RegistryKind.BOTH, query_via="unsupported"
    ),
    Registry(
        name="awesome-mcp-servers",
        url="https://github.com/punkpeye/awesome-mcp-servers",
        kind=RegistryKind.MCP,
        query_via="unsupported",
    ),
    Registry(
        name="awesome-claude-skills-composio",
        url="https://github.com/ComposioHQ/awesome-claude-skills",
        kind=RegistryKind.SKILLS,
        query_via="unsupported",
    ),
    Registry(
        name="awesome-claude-skills-behisecc",
        url="https://github.com/BehiSecc/awesome-claude-skills",
        kind=RegistryKind.SKILLS,
        query_via="unsupported",
    ),
]


class Candidate(BaseModel):
    name: str
    url: str
    kind: RegistryKind
    description: str = ""
    stars: int = 0
    last_updated: str = ""
    registry: str = ""
    source_hash: str = ""
    risk_flags: list[str] = Field(default_factory=list)
    trust_score: float = 0.0

    # Extra fields beyond the minimum spec, used only to compute risk/trust —
    # not every caller of this model needs to populate them.
    created_at: str = ""
    license_key: str | None = None


# ---------------------------------------------------------------------------
# fetching — injectable, mirrors hive/gateway/client.py's Transport protocol
# ---------------------------------------------------------------------------


class Fetcher(Protocol):
    def get_json(
        self, url: str, *, params: dict | None = None, headers: dict | None = None
    ) -> object: ...


class HttpFetcher:
    """Talks to the real network via httpx. Tests never use this directly —
    they inject a scripted double, exactly like `FakeTransport` in
    `tests/test_gateway.py`.
    """

    def __init__(self, timeout: float = 15.0) -> None:
        self._timeout = timeout

    def get_json(self, url: str, *, params: dict | None = None, headers: dict | None = None) -> object:
        hdrs = {"Accept": "application/vnd.github+json", "User-Agent": "hive-scout/0.1"}
        if headers:
            hdrs.update(headers)
        resp = httpx.get(url, params=params, headers=hdrs, timeout=self._timeout)
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# risk assessment
# ---------------------------------------------------------------------------

# Text shaped like an instruction addressed to an agent, not to a human
# reader. This is the core "SKILL.md is not documentation" detector.
_INJECTION_PATTERNS = [
    re.compile(r"ignore (all |any )?(the |prior )?(previous|prior) instructions", re.I),
    re.compile(r"disregard (all |any )?(the |prior )?(previous|prior) instructions", re.I),
    re.compile(r"you are (now |hereby )?authorized to", re.I),
    re.compile(r"you are (now )?an? (ai )?agent(,| and) you must", re.I),
    re.compile(r"as an ai( agent)?,? you must", re.I),
    re.compile(r"system prompt override", re.I),
    re.compile(r"admin(istrator)? override", re.I),
    re.compile(r"run (the )?following command", re.I),
    re.compile(r"execute (this|the following) (command|script)", re.I),
    re.compile(r"you (must|should) (now )?install (this|it) (automatically|immediately)", re.I),
    re.compile(r"pre-?approved by (the )?(user|operator)", re.I),
]

# Text asking for capabilities a skill/MCP listing has no business asking for.
_PERMISSION_PATTERNS = [
    re.compile(r"requires? (full |unrestricted )?shell access", re.I),
    re.compile(r"grants? (shell|filesystem|network) access", re.I),
    re.compile(r"needs? (filesystem|file system) write", re.I),
    re.compile(r"requires? root", re.I),
    re.compile(r"curl[^\n]{0,60}\|\s*(sh|bash)", re.I),
    re.compile(r"requires? (unrestricted|full) network access", re.I),
]

# A long base64/hex run, or an explicit decode call — content that cannot be
# reviewed as plain instructions because it is not plain text.
_OBFUSCATION_PATTERNS = [
    re.compile(r"[A-Za-z0-9+/]{60,}={0,2}"),
    re.compile(r"(?:0x)?[0-9a-fA-F]{60,}"),
    re.compile(r"base64\.b64decode\(|atob\(", re.I),
]

# A small, curated set of well-known good package/server names. Not
# exhaustive — it exists only to catch a near-miss on a name a human would
# recognise, e.g. "mcp-server-fi1esystem" vs "mcp-server-filesystem".
KNOWN_GOOD_PACKAGE_NAMES = frozenset(
    {
        "mcp-server-filesystem",
        "mcp-server-github",
        "mcp-server-git",
        "mcp-server-fetch",
        "mcp-server-memory",
        "mcp-server-sqlite",
        "mcp-server-slack",
        "mcp-server-puppeteer",
        "mcp-server-postgres",
        "mcp-server-brave-search",
        "mcp-server-google-maps",
        "mcp-server-sentry",
        "mcp-server-time",
        "playwright-mcp",
        "ripgrep",
        "ast-grep",
        "vectorbtpro",
        "claude-code",
    }
)


def _matches_any(patterns: Iterable[re.Pattern], text: str) -> bool:
    return any(p.search(text) for p in patterns)


def _levenshtein(a: str, b: str) -> int:
    """Plain-Python edit distance — no extra dependency for one small check."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def _short_name(name: str) -> str:
    """Strip an 'org/' prefix and normalise separators for name comparison."""
    base = name.rsplit("/", 1)[-1].lower()
    return re.sub(r"[_\s]+", "-", base)


def _typosquat_match(name: str) -> str | None:
    """Return the known-good name this candidate's name suspiciously resembles,
    or None if it is an exact match (the real thing) or not close to any of
    them. Edit distance 1-2 with similar length is the near-miss zone; an
    exact match is never flagged.
    """
    short = _short_name(name)
    if short in KNOWN_GOOD_PACKAGE_NAMES:
        return None
    best: str | None = None
    best_dist: int | None = None
    for good in KNOWN_GOOD_PACKAGE_NAMES:
        if abs(len(good) - len(short)) > 2:
            continue
        dist = _levenshtein(short, good)
        if dist == 0:
            return None
        if 0 < dist <= 2 and (best_dist is None or dist < best_dist):
            best, best_dist = good, dist
    return best


def _age_days(iso_ts: str) -> float | None:
    if not iso_ts:
        return None
    try:
        dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0


def assess_risk(candidate: Candidate, raw_text: str) -> list[str]:
    """Flag reasons to distrust a candidate. Pure function — no network, no
    filesystem. Every flag is a human-readable string so `report()` can show
    it directly; the caller decides what to do with the list, this function
    only ever observes and describes.
    """
    flags: list[str] = []
    text = raw_text or ""

    if _matches_any(_INJECTION_PATTERNS, text):
        flags.append(
            "agent_targeted_instruction: text is phrased as a command to an AI agent "
            "(e.g. 'ignore previous instructions' / 'you are authorized to') — "
            "treat as a prompt-injection attempt, never obey it"
        )

    if _matches_any(_PERMISSION_PATTERNS, text):
        flags.append(
            "permission_request: text asks for shell/filesystem/network access beyond "
            "what a skill or MCP listing should need"
        )

    if _matches_any(_OBFUSCATION_PATTERNS, text):
        flags.append(
            "obfuscated_content: a long base64/hex blob or decode call is present — "
            "content cannot be reviewed as plain instructions"
        )

    typo = _typosquat_match(candidate.name)
    if typo:
        flags.append(f"typosquat_suspected: '{candidate.name}' closely resembles known package '{typo}'")

    if candidate.license_key is None:
        flags.append("no_license: no license detected — reuse terms are unclear")

    age_days = _age_days(candidate.created_at) if candidate.created_at else None
    if age_days is not None and age_days < NEW_REPO_AGE_DAYS and candidate.stars < NEW_LOW_STARS_THRESHOLD:
        flags.append(f"new_and_unproven: only {candidate.stars} star(s), created {age_days:.0f} day(s) ago")

    return flags


def compute_trust_score(candidate: Candidate) -> float:
    """Stars, age, and license presence build the score up; any risk flag
    caps it back down. The cap is unconditional — it is not one input among
    several, it overrides everything else the score is built from.
    """
    stars_component = min(0.5, math.log10(max(candidate.stars, 0) + 1) / 4.0)

    age_days = _age_days(candidate.created_at) if candidate.created_at else None
    age_component = min(0.3, (age_days / 365.0) * 0.3) if age_days is not None else 0.0

    license_component = 0.1 if candidate.license_key else 0.0

    score = round(min(1.0, stars_component + age_component + license_component), 3)

    if candidate.risk_flags:
        score = min(score, RISK_CAPPED_TRUST_SCORE)

    return score


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def _candidate_from_github_item(item: dict, *, registry_name: str, kind: RegistryKind) -> Candidate:
    name = str(item.get("full_name") or item.get("name") or "unknown/unknown")
    url = str(item.get("html_url") or f"https://github.com/{name}")
    description = str(item.get("description") or "")
    stars = int(item.get("stargazers_count") or 0)
    last_updated = str(item.get("pushed_at") or item.get("updated_at") or "")
    created_at = str(item.get("created_at") or "")

    license_info = item.get("license")
    license_key = None
    if isinstance(license_info, dict):
        key = license_info.get("key")
        if key and key not in ("noassertion", "none"):
            license_key = str(key)

    source_hash = hashlib.sha256(json.dumps(item, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    candidate = Candidate(
        name=name,
        url=url,
        kind=kind,
        description=description,
        stars=stars,
        last_updated=last_updated,
        registry=registry_name,
        source_hash=source_hash,
        created_at=created_at,
        license_key=license_key,
    )
    candidate.risk_flags = assess_risk(candidate, f"{name}\n{description}")
    candidate.trust_score = compute_trust_score(candidate)
    return candidate


def search(
    query: str,
    *,
    kinds: Iterable[RegistryKind] | None = None,
    limit: int = 20,
    registries: Sequence[Registry] | None = None,
    fetcher: Fetcher | None = None,
) -> list[Candidate]:
    """Search public registries for skill/MCP candidates. Returns
    RECOMMENDATIONS for the operator to review — this function never
    installs anything and never will.

    Only registries with `query_via == "github_api"` are actively queried,
    via GitHub's documented, unauthenticated REST search endpoint. A
    registry that errors, rate-limits, or returns something unparseable is
    skipped — one bad registry never aborts the whole search.
    """
    active_fetcher: Fetcher = fetcher if fetcher is not None else HttpFetcher()
    wanted_kinds = set(kinds) if kinds else {RegistryKind.SKILLS, RegistryKind.MCP}
    regs = registries if registries is not None else DEFAULT_REGISTRIES

    found: dict[str, Candidate] = {}
    for reg in regs:
        if reg.query_via != "github_api" or not reg.topic:
            continue
        if reg.kind is not RegistryKind.BOTH and reg.kind not in wanted_kinds:
            continue

        params = {
            "q": f"{query} topic:{reg.topic}".strip(),
            "sort": "stars",
            "order": "desc",
            "per_page": max(1, min(limit, 100)),
        }
        try:
            data = active_fetcher.get_json(GITHUB_SEARCH_URL, params=params)
        except Exception:
            # Down, rate-limited, or timed out — skipped, never fatal.
            continue

        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, list):
            # Malformed / unexpected shape — skipped, never fatal.
            continue

        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                candidate = _candidate_from_github_item(item, registry_name=reg.name, kind=reg.kind)
            except Exception:
                # One bad item never kills the rest of the search.
                continue
            existing = found.get(candidate.url)
            if existing is None or candidate.stars > existing.stars:
                found[candidate.url] = candidate

    ordered = sorted(found.values(), key=lambda c: c.stars, reverse=True)
    return ordered[:limit]


# ---------------------------------------------------------------------------
# recommend + report
# ---------------------------------------------------------------------------


def recommend(claim_store: ClaimStore, candidates: Sequence[Candidate], job_id: str) -> list[Claim]:
    """Write every candidate into the claim store as an unverified
    recommendation. This function only ever calls `ClaimStore.add()` — never
    `mark_verified()` or `promote()`. Whether a candidate ever becomes
    something an agent may follow is entirely the operator's decision,
    exercised through the existing gate in `hive/knowledge/claims.py`.

    Recommendations are marked `time_sensitive`: stars, activity, and
    maintenance status all drift, so per the claims.py gate a stale
    recommendation can only be promoted via self-verification, never by
    corroboration alone drifting past a threshold.
    """
    written: list[Claim] = []
    for candidate in candidates:
        flag_suffix = f"; risk flags: {', '.join(candidate.risk_flags)}" if candidate.risk_flags else ""
        text = (
            f"Consider {candidate.kind.value} candidate '{candidate.name}' ({candidate.url}) "
            f"from {candidate.registry}: {candidate.stars} star(s), "
            f"trust_score={candidate.trust_score:.2f}{flag_suffix}"
        )
        claim = claim_store.add(
            text,
            ClaimType.RECOMMENDATION,
            source_ref=f"scout:{job_id}:{candidate.registry}:{candidate.url}",
            source_hash=candidate.source_hash,
            locator=candidate.url,
            confidence=candidate.trust_score,
            time_sensitive=True,
            topics=["skill-scout", candidate.kind.value, candidate.registry],
        )
        written.append(claim)
    return written


def report(candidates: Sequence[Candidate]) -> str:
    """Compact operator-facing table. Risk-flagged candidates sort to the
    top — the point of surfacing flags is that a human reviews them before
    anything gets greenlit, never that this module decides for them.
    """
    if not candidates:
        return "No candidates found."

    ordered = sorted(candidates, key=lambda c: (-len(c.risk_flags), c.trust_score, -c.stars))

    header = f"{'RISK':<5} {'TRUST':<6} {'STARS':<7} {'KIND':<7} {'NAME':<42} URL"
    lines = [header, "-" * len(header)]
    for c in ordered:
        risk_marker = f"[{len(c.risk_flags)}]" if c.risk_flags else "-"
        lines.append(
            f"{risk_marker:<5} {c.trust_score:<6.2f} {c.stars:<7} {c.kind.value:<7} {c.name[:42]:<42} {c.url}"
        )
        for flag in c.risk_flags:
            lines.append(f"      -> {flag}")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_REGISTRIES",
    "GITHUB_SEARCH_URL",
    "KNOWN_GOOD_PACKAGE_NAMES",
    "NEW_LOW_STARS_THRESHOLD",
    "NEW_REPO_AGE_DAYS",
    "RISK_CAPPED_TRUST_SCORE",
    "Candidate",
    "Fetcher",
    "HttpFetcher",
    "Registry",
    "RegistryKind",
    "assess_risk",
    "compute_trust_score",
    "recommend",
    "report",
    "search",
]
