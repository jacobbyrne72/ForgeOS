"""The Forge Lowerer: ask, before any model call, whether this is model work at all.

Reading 20,000 files to find imports is a tree-sitter query. Extracting seven
fields from 50,000 records is a parser. Counting failures by root cause is a
log parser. Paying a model per item for work that code does once is the largest
avoidable cost in an agent system — not a 20% saving, but the difference
between O(n) model calls and O(1). The architecture shift:

    from:  a model processes every item
    to:    code processes the normal 95-99%; the model handles the ambiguous tail

Everything here is deterministic — `classify` is keyword/structure analysis and
never calls a model, because a lowerer that spends tokens deciding how to save
tokens has already lost.

Safety properties, in order of importance:

1. **Sample validation before the full run.** A lowered path is a synthesised
   script, and a wrong script silently corrupts 50,000 records — far worse than
   the tokens it saved. So whenever lowering touches more than a handful of
   items, the verdict demands validation on a stratified sample first. The
   workflow is: synthesise/select the script → validate on a SAMPLE → only then
   run the corpus.
2. **The exception-rate tripwire.** If the deterministic path throws on more
   than ~5% of items, the uniformity assumption was wrong — the corpus is not
   what the script was written for. The operation goes back to a model; it does
   not limp item by item.
3. **Ties break toward REASON.** A wrong lowering corrupts data; a wrong
   escalation only costs money. Unrecognised shapes default to model work.

The cost model is conservative by construction: the deterministic path is
priced at full script-synthesis overhead plus full per-item cost for every
sample item, so small batches (roughly under ~25 items when validation is
required) come back "not worth lowering" — correctly, since below that the
model just doing the work is cheaper than writing and proving a script.

All numbers here are ESTIMATES and say so. Measured truth lives in the
avoidance ledger, and every row written there names how its baseline was
counted.
"""

from __future__ import annotations

import json
import re
from enum import Enum

from pydantic import BaseModel, Field

from .avoidance import AvoidanceLog, AvoidanceMethod

# ------------------------------------------------------------------ constants

#: Below this many items, script synthesis overhead exceeds the model calls it
#: avoids. A single-item operation is never worth lowering.
MIN_ITEMS_TO_LOWER = 3

#: "A handful". Above this, a proposed lowering MUST be validated on a sample
#: before the full run — this is the module's most important safety property.
HANDFUL = 5

#: Stratified sample size checked before trusting the script on the corpus.
SAMPLE_SIZE = 20

#: Estimated one-off model cost to synthesise or select the script (the O(1)
#: call that replaces the O(n) ones).
SCRIPT_SYNTHESIS_TOKENS = 1_000

#: Above this exception rate, the lowering assumption is wrong: revert.
EXCEPTION_RATE_MAX = 0.05

#: Return values of exception_rate_policy.
PROCEED = "proceed_deterministic"
REVERT = "revert_to_model"


class OperationKind(str, Enum):
    """The algorithmic shapes worth recognising. REASON means: genuinely not lowerable."""

    SEARCH = "search"                            # find occurrences/patterns across a corpus
    EXTRACT = "extract"                          # pull fixed fields out of structured records
    CLASSIFY_RULES = "classify_rules"            # bucket items by stated, mechanical rules
    COUNT_AGGREGATE = "count_aggregate"          # tally, group, sum, histogram
    DIFF_COMPARE = "diff_compare"                # what changed between two versions
    DEDUPE = "dedupe"                            # identical-content detection
    TRANSFORM_MECHANICAL = "transform_mechanical"  # rename, reformat, case-convert, sort
    REASON = "reason"                            # judgment, design, ambiguity — model work


#: Concrete tooling per kind — real tools present on most machines, so a
#: verdict's strategy is a command someone can actually run, not an invention.
STRATEGIES: dict[OperationKind, str] = {
    OperationKind.SEARCH: "ripgrep (rg) for text patterns; ast-grep / tree-sitter query for syntax-aware search",
    OperationKind.EXTRACT: "python json/csv parse + schema validate",
    OperationKind.CLASSIFY_RULES: "python rule table (regex / dict dispatch), one pass over the corpus",
    OperationKind.COUNT_AGGREGATE: "ripgrep + count, or sqlite GROUP BY for structured rows",
    OperationKind.DIFF_COMPARE: "git diff for tracked files; python difflib otherwise",
    OperationKind.DEDUPE: "hashlib content digest + set membership (or sqlite UNIQUE)",
    OperationKind.TRANSFORM_MECHANICAL: "python str/regex transform; ast-grep rewrite for code",
    OperationKind.REASON: "no deterministic strategy — this is model work",
}


class Operation(BaseModel):
    """One prospective bulk operation, described before any model touches it."""

    description: str
    item_count: int = Field(ge=0)
    per_item_tokens_estimate: int = Field(ge=0)
    has_deterministic_structure: bool = False
    sample_items: list[str] = Field(default_factory=list)


class LoweringVerdict(BaseModel):
    """Whether — and how — an operation can be handled by code instead of a model."""

    lowerable: bool
    kind: OperationKind
    strategy: str
    confidence: float = Field(ge=0.0, le=1.0)
    estimated_model_tokens: int = Field(ge=0)
    estimated_deterministic_tokens: int = Field(ge=0)
    reason: str
    requires_sample_validation: bool


# ----------------------------------------------------------- shape detection

_KIND_PATTERNS: dict[OperationKind, tuple[str, ...]] = {
    OperationKind.SEARCH: (
        r"\bfind\b", r"\bsearch\b", r"\bscan(?:ning)?\b", r"\blocate\b", r"\bgrep\b",
        r"\bimports?\b", r"\breferences?\b", r"\boccurrences?\b", r"\busages?\b",
        r"\bmatch(?:es|ing)?\b", r"\bpatterns?\b", r"\bwhich files\b",
    ),
    OperationKind.EXTRACT: (
        r"\bextract(?:ing)?\b", r"\bpars(?:e|ing)\b", r"\bfields?\b", r"\bjson\b",
        r"\bcsv\b", r"\bxml\b", r"\brecords?\b", r"\bcolumns?\b", r"\bpull(?:ing)? out\b",
    ),
    OperationKind.CLASSIFY_RULES: (
        r"\bclassif(?:y|ying|ication)\b", r"\bcategori[sz]e\b", r"\blabel(?:l?ing)?\b",
        r"\bbuckets?\b", r"\btag(?:ging)?\b", r"\btriage\b",
    ),
    OperationKind.COUNT_AGGREGATE: (
        r"\bcount(?:ing)?\b", r"\btally\b", r"\baggregate\b", r"\bsum\b", r"\bhow many\b",
        r"\bgroup(?:ed)? by\b", r"\bfrequenc(?:y|ies)\b", r"\bhistogram\b",
        r"\bbreakdown\b", r"\btotals?\b",
    ),
    OperationKind.DIFF_COMPARE: (
        r"\bdiff(?:s|ing)?\b", r"\bcompar(?:e|ing|ison)\b", r"\bdifferences?\b",
        r"\bchanged\b", r"\bdelta\b", r"\bversus\b", r"\bvs\b",
    ),
    OperationKind.DEDUPE: (
        r"\bdedup(?:e|licate|lication)?\b", r"\bduplicates?\b", r"\bunique\b",
        r"\bdistinct\b", r"\bidentical\b",
    ),
    OperationKind.TRANSFORM_MECHANICAL: (
        r"\brenam(?:e|ing)\b", r"\bconvert(?:ing)?\b", r"\breformat\b", r"\bnormali[sz]e\b",
        r"\bsnake_case\b", r"\bcamel ?case\b", r"\blowercas(?:e|ing)\b",
        r"\buppercas(?:e|ing)\b", r"\bsort(?:ing)?\b", r"\breplac(?:e|ing)\b",
        r"\bstrip(?:ping)?\b", r"\bescap(?:e|ing)\b",
    ),
}

_REASON_PATTERNS: tuple[str, ...] = (
    r"\bdecide\b", r"\bwhether\b", r"\bjudg(?:e|ment|ement)\b", r"\bopinions?\b",
    r"\barchitecture\b", r"\bdesign\b", r"\bsound(?:ness)?\b", r"\btrade-?offs?\b",
    r"\bshould (?:we|i)\b", r"\brecommend\b", r"\bcritique\b", r"\bevaluate\b",
    r"\bassess\b", r"\bwhy\b", r"\bintent\b", r"\bmeaning\b", r"\bambiguous\b",
    r"\bsubjective\b",
)

#: Cues that a classification is rule-based (mechanical) rather than semantic.
_RULE_CUES: tuple[str, ...] = (
    r"\bregex\b", r"\brules?\b", r"\bextensions?\b", r"\bsuffix\b", r"\bprefix\b",
    r"\berror codes?\b", r"\bstatus codes?\b", r"\bexit codes?\b", r"\bfixed set\b",
    r"\bkeywords?\b", r"\broot causes?\b", r"\bknown\b",
)


def _score(text: str, patterns: tuple[str, ...]) -> int:
    return sum(1 for p in patterns if re.search(p, text))


def samples_look_structured(samples: list[str]) -> bool:
    """True when every sample parses as JSON, or all share a uniform delimiter count.

    Cheap structural evidence from the data itself — stronger than a claim in
    the description, weaker than validation on a real sample run.
    """
    if not samples:
        return False
    try:
        for s in samples:
            json.loads(s)
        return True
    except (json.JSONDecodeError, TypeError):
        pass
    for delim in (",", "\t", "|"):
        counts = {s.count(delim) for s in samples}
        if len(counts) == 1 and counts.pop() > 0:
            return True
    return False


def _reason_verdict(model_tokens: int, confidence: float, reason: str) -> LoweringVerdict:
    """Not lowerable: the deterministic path does not exist, so it costs the same."""
    return LoweringVerdict(
        lowerable=False,
        kind=OperationKind.REASON,
        strategy=STRATEGIES[OperationKind.REASON],
        confidence=confidence,
        estimated_model_tokens=model_tokens,
        estimated_deterministic_tokens=model_tokens,
        reason=reason,
        requires_sample_validation=False,
    )


def classify(op: Operation) -> LoweringVerdict:
    """Deterministic keyword/structure analysis. Never calls a model.

    A high item_count over a uniform, structured operation is the strongest
    lowering signal. Ties and unrecognised shapes break toward REASON: a wrong
    lowering corrupts data, a wrong escalation only costs money.
    """
    text = op.description.lower()
    model_tokens = op.item_count * op.per_item_tokens_estimate

    scores = {kind: _score(text, pats) for kind, pats in _KIND_PATTERNS.items()}
    best_kind = max(scores, key=scores.__getitem__)  # first max in stable enum order
    best_score = scores[best_kind]
    reason_score = _score(text, _REASON_PATTERNS)
    structured = op.has_deterministic_structure or samples_look_structured(op.sample_items)

    if best_score == 0 and reason_score == 0:
        return _reason_verdict(
            model_tokens, 0.3,
            "no algorithmic shape recognised in the description — defaulting to model work",
        )
    if reason_score > 0 and reason_score >= best_score:
        return _reason_verdict(
            model_tokens, min(0.9, 0.5 + 0.1 * reason_score),
            f"judgment-shaped operation ({reason_score} reasoning cue(s) outweigh "
            f"{best_score} algorithmic cue(s)) — genuinely model work",
        )
    if (
        best_kind is OperationKind.CLASSIFY_RULES
        and not structured
        and _score(text, _RULE_CUES) == 0
    ):
        return _reason_verdict(
            model_tokens, 0.6,
            "classification with no stated rules and no structure is semantic work — "
            "a model call, not a rule table",
        )

    strategy = STRATEGIES[best_kind]
    confidence = min(
        0.95,  # never certain without sample validation
        0.45
        + 0.1 * min(best_score, 3)
        + (0.15 if structured else 0.0)
        + (0.1 if op.item_count >= 1_000 else 0.0),
    )
    # Mandatory whenever more than a handful of items would be touched — set even
    # on a not-lowerable verdict, so lowering against advice still gets validated.
    requires_validation = op.item_count > HANDFUL

    if op.item_count == 0:
        return LoweringVerdict(
            lowerable=False, kind=best_kind, strategy=strategy, confidence=confidence,
            estimated_model_tokens=0, estimated_deterministic_tokens=0,
            reason="zero items — nothing to lower, nothing to save",
            requires_sample_validation=False,
        )
    if op.item_count < MIN_ITEMS_TO_LOWER:
        return LoweringVerdict(
            lowerable=False, kind=best_kind, strategy=strategy, confidence=confidence,
            estimated_model_tokens=model_tokens,
            estimated_deterministic_tokens=model_tokens,
            reason=(
                f"only {op.item_count} item(s): script synthesis overhead exceeds the "
                f"model call(s) it would avoid"
            ),
            requires_sample_validation=requires_validation,
        )

    sample_n = min(op.item_count, SAMPLE_SIZE)
    deterministic_tokens = SCRIPT_SYNTHESIS_TOKENS + (
        sample_n * op.per_item_tokens_estimate if requires_validation else 0
    )
    if deterministic_tokens >= model_tokens:
        return LoweringVerdict(
            lowerable=False, kind=best_kind, strategy=strategy, confidence=confidence,
            estimated_model_tokens=model_tokens,
            estimated_deterministic_tokens=deterministic_tokens,
            reason=(
                f"deterministic path (~{deterministic_tokens} tokens incl. synthesis and "
                f"sample validation) does not undercut the model baseline "
                f"(~{model_tokens} tokens) — let the model do it"
            ),
            requires_sample_validation=requires_validation,
        )

    return LoweringVerdict(
        lowerable=True, kind=best_kind, strategy=strategy, confidence=confidence,
        estimated_model_tokens=model_tokens,
        estimated_deterministic_tokens=deterministic_tokens,
        reason=(
            f"{op.item_count} uniform item(s) with an algorithmic shape ({best_kind.value}): "
            f"O(1) script replaces O(n) model calls via {strategy}"
            + ("; validate on a stratified sample before the full run" if requires_validation else "")
        ),
        requires_sample_validation=requires_validation,
    )


# ------------------------------------------------------------------- savings

def savings_estimate(op: Operation, verdict: LoweringVerdict) -> dict:
    """Projected saving — ESTIMATED, never measured.

    Measured truth belongs in the avoidance ledger via record_lowering. An
    unlowerable verdict saves nothing by definition. Zero baselines yield zero
    percentages, never a division error.
    """
    model = verdict.estimated_model_tokens
    deterministic = verdict.estimated_deterministic_tokens
    saved = max(0, model - deterministic) if verdict.lowerable else 0
    return {
        "model_tokens": model,
        "deterministic_tokens": deterministic,
        "saved": saved,
        "saved_pct": round(100.0 * saved / model, 2) if model else 0.0,
        "estimated": True,
        "basis": (
            f"ESTIMATED — {op.item_count} items x {op.per_item_tokens_estimate} "
            f"tokens/item baseline vs synthesis + sample overhead; not measured"
        ),
    }


def record_lowering(
    avoidance_log: AvoidanceLog,
    job_id: str,
    task_id: str | None,
    op: Operation,
    verdict: LoweringVerdict,
    actual_tokens: int,
) -> str:
    """Write the realised lowering to the avoidance ledger. Returns the row id.

    `actual_tokens` is what the lowered path really spent (synthesis + sample
    validation). The baseline is the per-item estimate, and baseline_source
    says exactly that — the ledger rejects a blank source, and a guessed
    baseline dressed up as a measured one would be worse than blank.
    """
    if not verdict.lowerable:
        raise ValueError(
            "refusing to record a saving for an operation that was not lowered — "
            "an unlowered operation avoided nothing"
        )
    baseline_source = (
        f"estimated per-item model baseline: {op.item_count} items x "
        f"{op.per_item_tokens_estimate} tokens/item = {verdict.estimated_model_tokens} "
        f"tokens (preflight-style estimate, not a tokeniser count of real prompts)"
    )
    return avoidance_log.record(
        job_id,
        AvoidanceMethod.DETERMINISTIC,
        baseline_tokens=verdict.estimated_model_tokens,
        actual_tokens=actual_tokens,
        baseline_source=baseline_source,
        task_id=task_id,
    )


# ------------------------------------------------------------------ tripwire

def exception_rate_policy(processed: int, exceptions: int) -> str:
    """PROCEED or REVERT based on how often the deterministic path threw.

    More than ~5% exceptions means the uniformity assumption behind the
    lowering was wrong — the corpus is not what the script was written for.
    The whole operation goes back to a model; it does not limp item by item,
    because every limped item is a silent-corruption risk. Exactly at the
    threshold still proceeds ("more than", not "at least"). Zero processed is
    zero evidence of failure, not a division error.
    """
    if processed < 0 or exceptions < 0:
        raise ValueError("counts cannot be negative")
    if exceptions > processed:
        raise ValueError("exceptions cannot exceed processed")
    if processed == 0:
        return PROCEED
    if exceptions / processed > EXCEPTION_RATE_MAX:
        return REVERT
    return PROCEED


__all__ = [
    "EXCEPTION_RATE_MAX",
    "HANDFUL",
    "MIN_ITEMS_TO_LOWER",
    "PROCEED",
    "REVERT",
    "SAMPLE_SIZE",
    "SCRIPT_SYNTHESIS_TOKENS",
    "STRATEGIES",
    "LoweringVerdict",
    "Operation",
    "OperationKind",
    "classify",
    "exception_rate_policy",
    "record_lowering",
    "samples_look_structured",
    "savings_estimate",
]
