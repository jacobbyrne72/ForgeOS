"""Retrieve broad, then rerank narrow.

Single-stage retrieval has one scorer doing two incompatible jobs. To find
candidates at all it has to be permissive -- any chunk sharing vocabulary with
the question is worth a look. To fill a small budget it has to be strict --
only the chunks that actually answer. One function tuned for both is tuned for
neither, and it fails in a specific direction: it fills the budget with the
chunks that look most like the question rather than the ones that answer it.

Measured on ForgeBench before this existed: asking which function records spend
and what stops a double charge, the top slots went to module docstrings
*discussing* double-charging, while the chunk defining `record_spend` ranked
#53 of 302. Every word of the question appeared in those docstrings. None of the
answer did.

Two stages, both deterministic, no model call in either:

  STAGE 1 (recall). Keep everything with any signal, cheaply, up to
  `candidate_limit`. Being wrong here is unrecoverable -- a chunk not retrieved
  cannot be reranked back in -- so this stage is deliberately generous.

  STAGE 2 (precision). Rescore the survivors with signals too expensive or too
  sharp to run over the whole corpus: does the chunk DEFINE something the
  question names, does it satisfy several distinct parts of the question rather
  than one part repeatedly, is it dense in question terms per line, and is it a
  definition rather than prose about one.

The point is that stage 2 sees a small set, so it can afford to ask better
questions than stage 1 could. Cost stays bounded: stage 1 is linear and cheap,
stage 2 runs over at most `candidate_limit` chunks.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# Enough candidates that the answer is almost certainly inside, few enough that
# stage 2 stays cheap. A chunk missed here can never be recovered.
DEFAULT_CANDIDATE_LIMIT = 60

_DEF_RE = re.compile(r"^[ \t]*(?:async[ \t]+)?(?:def|class)[ \t]+([A-Za-z_]\w*)", re.M)


@dataclass(frozen=True)
class Candidate:
    """One chunk with both stages' scores, so a receipt can show the movement."""

    ref: str
    body: str
    recall_score: float
    precision_score: float = 0.0
    rank_before: int = 0
    rank_after: int = 0

    @property
    def moved(self) -> int:
        """Positions gained by reranking. Negative means it fell."""
        return self.rank_before - self.rank_after


def idf(terms, bodies) -> dict[str, float]:
    """Inverse document frequency over the corpus actually being searched.

    Computed per query rather than globally: "ledger" is uninformative inside
    ledger.py and highly informative outside it, and a global table cannot say
    both.
    """
    n = len(bodies) or 1
    out: dict[str, float] = {}
    for t in terms:
        df = sum(1 for b in bodies if t in b)
        if df:
            out[t] = math.log(1.0 + n / df)
    return out


def recall_stage(chunks, terms, *, candidate_limit: int = DEFAULT_CANDIDATE_LIMIT):
    """Stage 1: everything with any signal, cheapest possible scoring.

    `chunks` is an iterable of `(ref, body)`. Ties break on `ref` so the
    candidate set is identical run to run -- a benchmark whose retrieval varies
    between runs cannot attribute a cost difference to anything.
    """
    chunks = list(chunks)
    lows = [b.lower() for _, b in chunks]
    weights = idf(terms, lows)

    scored: list[tuple[float, str, str]] = []
    for (ref, body), low in zip(chunks, lows):
        score = sum(w for t, w in weights.items() if t in low)
        if score > 0:
            scored.append((score, ref, body))
    scored.sort(key=lambda c: (-c[0], c[1]))
    return [
        Candidate(ref=ref, body=body, recall_score=score, rank_before=i)
        for i, (score, ref, body) in enumerate(scored[:candidate_limit])
    ], weights


def precision_stage(candidates, terms, weights, *, def_weight: float = 4.0):
    """Stage 2: sharper signals over the small surviving set.

    Three things stage 1 could not afford to ask:

    - DEFINITION SITES. A term matching a `def`/`class` NAME is the title field
      of a code chunk; the same term in prose is the body. This alone is what
      moved `record_spend` from #53 into the sent set.
    - COVERAGE. How many DISTINCT parts of the question a chunk touches, rather
      than one part repeated. A docstring saying "spend" nine times answers less
      than a function touching spend, ledger and guard once each.
    - DENSITY. Signal per line, so a long chunk cannot win by dilution alone.
    """
    out: list[Candidate] = []
    for cand in candidates:
        low = cand.body.lower()
        names = " ".join(_DEF_RE.findall(cand.body)).lower()

        base = sum(w for t, w in weights.items() if t in low)
        defines = sum(w for t, w in weights.items() if t in names) if names else 0.0
        hit_terms = [t for t in weights if t in low]
        coverage = len(hit_terms) / max(len(weights), 1)
        lines = max(cand.body.count("\n") + 1, 1)
        density = base / lines

        score = base + def_weight * defines + (base * coverage) + density
        out.append(Candidate(
            ref=cand.ref, body=cand.body, recall_score=cand.recall_score,
            precision_score=score, rank_before=cand.rank_before,
        ))

    out.sort(key=lambda c: (-c.precision_score, c.ref))
    return [
        Candidate(ref=c.ref, body=c.body, recall_score=c.recall_score,
                  precision_score=c.precision_score, rank_before=c.rank_before,
                  rank_after=i)
        for i, c in enumerate(out)
    ]


def two_stage_rank(chunks, terms, *, candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
                   def_weight: float = 4.0) -> list[Candidate]:
    """Both stages. Returns candidates ordered by the precision score.

    Deterministic end to end: same corpus and same terms give the same order,
    every time, on every machine.
    """
    candidates, weights = recall_stage(chunks, terms, candidate_limit=candidate_limit)
    if not candidates:
        return []
    return precision_stage(candidates, terms, weights, def_weight=def_weight)


def rerank_report(ranked: list[Candidate], *, top: int = 5) -> str:
    """What reranking actually changed, for a receipt.

    Printed rather than asserted: if stage 2 never moves anything, it is dead
    weight worth deleting, and that should be visible rather than inferred.
    """
    if not ranked:
        return "no candidates"
    lines = [f"reranked {len(ranked)} candidate(s); top {min(top, len(ranked))}:"]
    for c in ranked[:top]:
        arrow = "=" if c.moved == 0 else (f"+{c.moved}" if c.moved > 0 else str(c.moved))
        lines.append(f"  {c.ref:<44} recall#{c.rank_before:<4} -> #{c.rank_after:<4} ({arrow})")
    return "\n".join(lines)
