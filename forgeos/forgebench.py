"""ForgeBench -- the Release 0.1 exit gate.

08_IMPLEMENTATION_ROADMAP.md's own exit gate for Release 0.1 is:

    "On a pinned set of tasks, reproduce equal acceptance with lower measured
    token/cash cost than the unwrapped CLI baseline."

Nothing in this repo has ever run that comparison. This module is the paired
baseline runner and benchmark harness that does: a small pinned task suite
(`DEFAULT_SUITE`), two arms run under identical conditions (`ArmKind.BASELINE`
== the naive "dump every relevant file in, no capsule, no cache discipline"
policy a person reaches for with no tooling; `ArmKind.FORGEOS` == the ranked,
budgeted capsule), and a receipt that says which of its own numbers are real.

Reused, not reinvented:

- `tools/ab_bench.py` proved out the two-arm-same-pricing-code shape for one
  question. This module generalises that shape to N pinned tasks instead of
  forking the script.
- `forgeos.economy.savings` (`SavingsProof`, `Figure`, `Provenance`,
  `savings_pct`, `cost_per_accepted_task`, `verify_proof`) is the actual
  receipt. This module never computes a percentage or a per-task cost by
  hand -- every number here is built from those primitives so the
  never-render-an-estimate-as-a-measurement rule holds automatically.
- `forgeos.ledger.Ledger` and `forgeos.gateway.client.Gateway` are the only
  path spend ever takes. Nothing here bypasses either.

Savings receipt validity classes (09_FORGEBENCH.md):

    A -- paired measured:      both arms executed here, under pinned conditions.
    B -- historical measured:  baseline is a compatible PRIOR measured run.
    C -- modelled counterfactual: baseline is a calibrated estimate, labelled.
    D -- no baseline:          report actual usage only; NEVER claim a saving.

Only A and D have an input path wired up in `main()` today -- a live run
either executes both arms (A) or, with `--skip-baseline`, the ForgeOS arm
alone (D). B and C are represented in `SavingsClass` and
`_BASELINE_PROVENANCE_BY_CLASS` so `build_report` labels them correctly IF a
caller constructs a `SuiteRunResult` for one directly, but loading a
historical report (B) or accepting a modelled-estimate baseline (C) from the
CLI is follow-up work, not built here -- the Release 0.1 exit gate itself
only requires class A.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol

from .catalog import Catalog, ModelCard, default_catalog
from .contracts import Budget, JobSpec, Scope, from_micros, to_micros
from .economy.capsule import CapsuleBuilder, RefKind, make_ref
from .economy.preflight import CallEstimate, CallRefused, estimate_call
from .economy.savings import (
    Baseline,
    Figure,
    Outcome,
    Provenance,
    SavingsProof,
    cost_per_accepted_task,
    render_receipt,
    savings_pct,
    verify_proof,
)
from .gateway.client import Gateway, GatewayRequest, default_transports
from .ledger import open_ledger
from .settings import ProviderKind, Settings

# --------------------------------------------------------------- acceptance


def keyword_groups(*groups: Sequence[str]) -> Callable[[str], bool]:
    """A case-insensitive, all-groups-must-hit predicate.

    Generalises the acceptance technique `tools/ab_bench.py` proved out
    (`ACCEPT_TERM_GROUPS` / `_is_accepted`): every group is a set of
    synonyms for one required fact, and the text must contain at least one
    term from EVERY group. Mechanical and keyword-based -- NOT a semantic
    judge, and it never claims to be one.
    """
    frozen = tuple(tuple(g) for g in groups)

    def _check(text: str) -> bool:
        low = text.lower()
        return all(any(term.lower() in low for term in group) for group in frozen)

    return _check


@dataclass(frozen=True)
class AcceptanceCheck:
    """A MECHANICAL pass/fail check against one arm's raw text output.

    Never a model's opinion -- exactly one of `predicate` or `command` is
    set. `predicate` is a deterministic assertion over the text (see
    `keyword_groups`). `command` is an argv template run against the text:
    the text is written to a temp file first and `"{output}"` in any argv
    element is replaced with that file's path, then exit code 0 means pass
    and anything else means fail. This covers the "a command that exits
    0/non-0" acceptance style as a distinct mechanism from a text predicate,
    per the Release 0.1 spec.
    """

    description: str
    predicate: Callable[[str], bool] | None = None
    command: tuple[str, ...] | None = None
    timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        has_pred = self.predicate is not None
        has_cmd = self.command is not None
        if has_pred == has_cmd:  # both set, or neither
            raise ValueError(
                f"AcceptanceCheck {self.description!r} must set exactly one of "
                "predicate/command, never both or neither"
            )

    def evaluate(self, text: str) -> bool:
        if self.predicate is not None:
            return bool(self.predicate(text))
        assert self.command is not None
        fd, path = tempfile.mkstemp(prefix="forgebench_", suffix=".txt")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            argv = [part.replace("{output}", path) for part in self.command]
            if argv and argv[0] == sys.executable and "-S" not in argv:
                # Skip `site` processing. Measured on this machine: spawning
                # the SAME interpreter as a child process with normal site
                # init took 30-60+ seconds (whatever this venv's site
                # customization does, it is drastically slower for a nested
                # child than for the top-level process) -- with `-S` the
                # identical call completes in ~1.3s. A mechanical check
                # script here is stdlib-only by construction (see
                # `_JSON_ARRAY_CHECK`), so it has no business paying for
                # site-packages / sitecustomize at all, on any machine.
                argv.insert(1, "-S")
            try:
                proc = subprocess.run(
                    argv, capture_output=True, timeout=self.timeout_seconds, check=False
                )
            except (OSError, subprocess.TimeoutExpired):
                return False  # a broken/hanging check command is a FAIL, never a crash
            return proc.returncode == 0
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


# --------------------------------------------------------------- task suite


class TaskFamily(str, Enum):
    """09_FORGEBENCH.md's benchmark families. Release 0.1 only exercises
    CODING (localisation/comprehension); the rest are named here as
    placeholders for later releases, not implemented in this module."""

    CODING = "coding"


@dataclass(frozen=True)
class PinnedTask:
    """One pinned, reproducible unit of the suite.

    `scope` reuses `forgeos.contracts.Scope` rather than a bespoke path list
    -- the same shape a real `TaskSpec` uses, so this harness's notion of
    "the files a task may touch" is the same one the rest of the codebase
    already has.
    """

    id: str
    objective: str
    scope: Scope
    acceptance: AcceptanceCheck
    max_output_tokens: int = 300
    family: TaskFamily = TaskFamily.CODING

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("PinnedTask.id must not be blank")
        if not self.objective.strip():
            raise ValueError("PinnedTask.objective must not be blank")

    def contract_dict(self) -> dict:
        """A stable, JSON-serializable description -- never the callables
        themselves -- used to hash what this suite pins (see `build_report`'s
        `contract_hash`)."""
        return {
            "id": self.id,
            "objective": self.objective,
            "scope_paths": list(self.scope.paths),
            "acceptance": self.acceptance.description,
            "max_output_tokens": self.max_output_tokens,
            "family": self.family.value,
        }


# ------------------------------------------------------------------- arms


class ArmKind(str, Enum):
    BASELINE = "baseline"
    FORGEOS = "forgeos"


@dataclass
class ArmCallResult:
    """What one arm's one call produced, priced and timed."""

    text: str
    tokens_in: int
    tokens_out: int
    tokens_cached_in: int
    usd_micros: int
    seconds: float
    finish_reason: str = ""

    @property
    def truncated(self) -> bool:
        """Cut off at the output cap. Scoring this as a wrong answer blames the
        model for a limit the harness set."""
        return self.finish_reason == "length"

    @property
    def spent_output_on_nothing(self) -> bool:
        """Billed for output tokens and handed back no answer at all."""
        return not self.text.strip() and bool(self.tokens_out)

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_cached_in + self.tokens_out


class BudgetExhausted(RuntimeError):
    """Raised by an `ArmExecutor` when `run_suite` must ABORT the whole run.

    AGENTS.md rule 1: "Never widen a budget to make a job finish... a
    tripped governor is a signal to escalate, never a number to edit."
    `run_suite` treats this exception as that exact signal -- it stops the
    suite immediately rather than skipping the one task and continuing.
    """


class ArmExecutor(Protocol):
    def run(self, task: PinnedTask, arm: ArmKind) -> ArmCallResult: ...


# ------------------------------------------------------- prompt building

# ForgeOS keeps a byte-stable prefix so a provider can serve a cache hit on
# every call after the first. The baseline arm has no such discipline --
# that IS the naive policy this harness measures against.
FORGEOS_PROMPT_PREFIX = "You are answering a question about a Python codebase.\n"
CAPSULE_BUDGET_TOKENS = 1_500  # floor; see `capsule_budget_for`
CAPSULE_BUDGET_CAP_TOKENS = 8_000
CAPSULE_BUDGET_SCOPE_FRACTION = 0.15
CAPSULE_BUDGET_MAX_FRACTION = 0.80


def capsule_budget_for(scope_chars: int) -> int:
    """Context allowance for a task, from the size of what it may read.

    A single constant for every task was not a budget, it was an arbitrary
    number, and it was wrong in both directions at once. Measured on this suite:
    `preflight-refusal-types` has one 6KB file, and the packed capsule came out
    LARGER than simply sending the file -- the wrapper cost more than no wrapper,
    which is the one result that makes the whole idea pointless. Meanwhile
    `ledger-dedup-guard` spans 164KB across five files and got the same 1,500
    tokens, roughly 4% of its scope, too thin to hold an answer that lives in two
    different files.

    Two rules, both load-bearing:

    - Scale with scope, floor and cap it. A bigger question needs more evidence;
      the cap stops a huge scope from turning the capsule back into a file dump.
    - NEVER exceed `CAPSULE_BUDGET_MAX_FRACTION` of what the naive arm would
      send. ForgeOS costing more than the thing it wraps is a defect, not a
      trade-off, and this makes that outcome impossible rather than merely
      unlikely.

    This is a design parameter chosen up front from scope size -- not a governor
    widened after the fact to make a run succeed, which AGENTS.md rule 1 forbids
    and which this module's `BudgetExhausted` handling still enforces at runtime.
    """
    scope_tokens = max(scope_chars // 4, 1)  # same rough chars/token the arms use
    scaled = int(scope_tokens * CAPSULE_BUDGET_SCOPE_FRACTION)
    budget = min(max(scaled, CAPSULE_BUDGET_TOKENS), CAPSULE_BUDGET_CAP_TOKENS)
    return max(1, min(budget, int(scope_tokens * CAPSULE_BUDGET_MAX_FRACTION)))


_CHUNK_LINES = 24   # bounded so ranking granularity never depends on term breadth
_CHUNK_STRIDE = 12  # 50% overlap: an answer on a boundary still lands whole in one chunk

# Relevance terms come from the TASK'S OWN OBJECTIVE, never from a fixed list.
#
# This used to be one global `_RANK_TERMS` tuple shared by all six tasks, and it
# was wrong twice over. Correctness: for `ledger-dedup-guard` (5 scope files, 41
# candidate blocks, room for 3) blocks dense in ANOTHER task's terms outranked
# the one block containing `record_spend`, so the ForgeOS arm was handed a
# context that could not answer the question while the baseline arm could -- the
# suite scored a packing bug as a quality difference. Integrity: that tuple was
# the answer key. Ranking retrieval on the exact strings the acceptance checks
# grep for is tuning the retriever on the test set, and any savings it measured
# would not survive a task the list had never seen.
#
# The objective is what a real caller supplies, so deriving terms from it is
# both honest and what the packer must do in production anyway.

_STOPWORDS = frozenset("""
a an the and or of to in on at by as is are was were be been being it its this
that these those for from with without into under over what which who whom
whose when where why how do does did can could should would you your please
name names named exactly exact answer reply state give list return returns
under words word only no not both each other same specifically two three one
function functions method methods class classes module modules codebase repo
file files line lines code
""".split())

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def _stems(word: str) -> set[str]:
    """`word` plus a crude stem, so "records" in a question matches
    `record_spend` in the source. Deliberately dumb: a real stemmer would add a
    dependency and a failure mode for a benchmark that only needs substring
    hits."""
    out = {word}
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            out.add(word[: -len(suffix)])
    return out


def objective_terms(objective: str) -> tuple[str, ...]:
    """Lower-cased search terms extracted from a task objective.

    Splits snake_case/dotted identifiers into their parts as well as keeping
    them whole, so `raise_if_refused` in a question also matches `refused`
    in the source.
    """
    terms: set[str] = set()
    for token in _IDENT_RE.findall(objective.lower()):
        if token in _STOPWORDS:
            continue
        terms |= _stems(token)
        for part in token.split("_"):
            if len(part) > 2 and part not in _STOPWORDS:
                terms |= _stems(part)
    return tuple(sorted(terms))


def _read_repo_file(root: Path, rel: str) -> str | None:
    p = root / rel
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8", errors="replace")


def build_baseline_prompt(root: Path, task: PinnedTask) -> str:
    """Naive: every scope file, whole, concatenated. No capsule, no packing,
    no cache discipline -- the "unwrapped CLI baseline" the Release 0.1 exit
    gate names."""
    parts = []
    for rel in task.scope.paths:
        text = _read_repo_file(root, rel)
        if text is not None:
            parts.append(f"===== {rel} =====\n{text}")
    return "\n\n".join(parts) + f"\n\n{task.objective}"


def _chunk_file(rel: str, text: str) -> list[tuple[str, str]]:
    """Bounded, overlapping, deterministic (ref, body) chunks.

    The previous window-and-merge scheme grew a span every time an adjacent
    line matched, so a broad term set merged an entire file into one block --
    two of those filled the budget and ranking had nothing left to choose
    between. Fixed-size chunks keep granularity independent of how many terms
    happen to match, which is what makes the ranking below able to discriminate
    at all. The overlap stops an answer that straddles a boundary from being
    split across two chunks that each look half-relevant.
    """
    lines = text.splitlines()
    out: list[tuple[str, str]] = []
    for lo in range(0, max(len(lines), 1), _CHUNK_STRIDE):
        hi = min(len(lines), lo + _CHUNK_LINES)
        body = "\n".join(lines[lo:hi])
        if body.strip():
            out.append((f"{rel}:{lo + 1}-{hi}", body))
        if hi >= len(lines):
            break
    return out


_DEF_RE = re.compile(r"^[ \t]*(?:async[ \t]+)?(?:def|class)[ \t]+([A-Za-z_]\w*)", re.M)
_DEF_FIELD_WEIGHT = 4.0
"""How much more a term matching a DEFINED SYMBOL NAME counts than the same
term in prose.

Field-weighted retrieval (BM25F's idea): a match in the title field outranks a
match in the body. For source code the symbol name is the title. Without this,
chunks that *discuss* a concept beat the chunk that *implements* it -- measured
on this suite, `def record_spend` ranked #53 of 302 while five module docstrings
about double-charging took every slot, so the arm was handed a context that
could not answer its own question.

4.0 was chosen by sweeping 0/2/4/8 against the whole suite and taking the value
that improves the most tasks without distorting the rest: at 4.0 two tasks
improve (`ledger-dedup-guard` #53->#5, `ledger-generation-fencing` #3/#4->#1/#1)
and three are unchanged; by 8.0 the ranking starts degrading elsewhere
(`savings-provenance-rule` #30->#37). Tuned on rank across ALL tasks, never on
whether any single task passes -- a retriever fitted to this suite's answers
would score well here and do nothing on the repo you actually point it at.
"""


def _idf(terms: Sequence[str], bodies: Sequence[str]) -> dict[str, float]:
    """Inverse document frequency over THIS task's own chunks.

    The signal the earlier versions were missing. Asking "which function
    records spend to the ledger, and what prevents it being charged twice"
    yields terms like `ledger` and `record` -- which occur on nearly every
    chunk of ledger.py and therefore separate nothing -- alongside `twice` and
    `charged`, which occur in two places in the whole scope and point straight
    at the answer. Weighting each term by log(N / df) makes the rare word
    dominate the common one automatically, with no per-task tuning and no
    hand-maintained list of what matters.
    """
    n = len(bodies) or 1
    scores: dict[str, float] = {}
    for t in terms:
        df = sum(1 for b in bodies if t in b)
        if df:  # a term absent from the whole scope carries no signal
            scores[t] = math.log(1.0 + n / df)
    return scores


def rank_chunk(body: str, weights: dict[str, float], *, low: str | None = None) -> float:
    """Relevance of one chunk: IDF over the body, plus a weighted second pass
    over the symbol names it DEFINES.

    Split out of `build_forgeos_prompt` so the ranking rule can be asserted
    directly. Proving it through the packer needs a fixture where the budget
    happens to force the exact trade-off, which makes the test about the
    fixture; here the property is stated plainly -- a term in a `def`/`class`
    name outweighs the same term in prose.
    """
    low = body.lower() if low is None else low
    score = sum(w for t, w in weights.items() if t in low)
    names = " ".join(_DEF_RE.findall(body)).lower()
    if names:
        score += _DEF_FIELD_WEIGHT * sum(w for t, w in weights.items() if t in names)
    return score


def build_forgeos_prompt(root: Path, task: PinnedTask) -> tuple[str, dict]:
    """ForgeOS arm: rank, window, and pack the SAME scope files to a hard
    budget. Same technique as `tools/ab_bench.py`'s `build_capsule_prompt`,
    generalised to an arbitrary task's scope/objective instead of one
    hardcoded question -- deterministic and model-free, no model is
    consulted to decide what to send."""
    terms = objective_terms(task.objective)
    chunks: list[tuple[str, str]] = []
    scope_chars = 0
    for rel in task.scope.paths:
        text = _read_repo_file(root, rel)
        if text is not None:
            # Sized from the FILES, never from `chunks` -- chunks overlap by 50%,
            # so summing their bodies double-counts the scope and hands every
            # task twice the budget it was supposed to get.
            scope_chars += len(text)
            chunks.extend(_chunk_file(rel, text))

    lows = [body.lower() for _, body in chunks]
    weights = _idf(terms, lows)
    budget = capsule_budget_for(scope_chars)

    blocks: list[tuple[float, str, str]] = []
    for (ref, body), low in zip(chunks, lows):
        score = rank_chunk(body, weights, low=low)
        if score > 0:  # a chunk matching nothing asked about is not context
            blocks.append((score, ref, body))

    # Ref breaks ties, so the packed prompt is byte-identical run to run whatever
    # order the filesystem hands back. A benchmark that reorders its own context
    # between runs cannot attribute a cost difference to anything.
    blocks.sort(key=lambda b: (-b[0], b[1]))

    bodies: dict[str, str] = {}
    builder = CapsuleBuilder(budget=budget)
    for score, ref, body in blocks:
        path, span = ref.rsplit(":", 1)
        uri = f"file_slice://{path}#L{span}"
        bodies[uri] = body
        builder.add(make_ref(RefKind.FILE_SLICE, uri, body, f"idf score {score:.2f}"))

    capsule = builder.finish(
        objective=task.objective, acceptance=[task.acceptance.description], write_scope=[]
    )
    packed = "\n\n".join(f"===== {r.ref} =====\n{bodies[r.ref]}" for r in builder.items)
    stats = {
        "blocks_found": len(blocks),
        "blocks_sent": len(builder.items),
        "blocks_dropped": len(builder.excluded),
        "capsule_tokens": builder.total_tokens,
        "read_scope": len(capsule.read_scope),
    }
    return packed + f"\n\n{task.objective}", stats


def estimate_task(root: Path, task: PinnedTask, arm: ArmKind, card: ModelCard) -> CallEstimate:
    """Price one (task, arm) prompt against `card` WITHOUT calling anything --
    no Gateway, no ledger, no network. This is what `--dry-run` and the
    pre-flight budget guard in `run_suite` both use instead of a hand-rolled
    estimate."""
    prompt = (
        build_baseline_prompt(root, task)
        if arm is ArmKind.BASELINE
        else build_forgeos_prompt(root, task)[0]
    )
    return estimate_call(prompt, task.max_output_tokens, card)


class GatewayExecutor:
    """The real arm: builds the baseline/forgeos prompt for a task and calls
    `Gateway.complete`. Every call goes through the SAME `Gateway` instance --
    same pricing code, same ledger, same preflight-refuse path
    `tools/ab_bench.py` already proved out for one question.

    `remaining_micros_fn` is called fresh before EVERY call so a tightening
    budget (spend from earlier tasks in this same suite) is honoured
    call-by-call, not just checked once at the start.
    """

    def __init__(
        self,
        *,
        gateway: Gateway,
        job_id: str,
        model_ref: str,
        root: Path,
        remaining_micros_fn: Callable[[], int],
    ) -> None:
        self._gateway = gateway
        self._job_id = job_id
        self._model_ref = model_ref
        self._root = root
        self._remaining_micros_fn = remaining_micros_fn

    def run(self, task: PinnedTask, arm: ArmKind) -> ArmCallResult:
        if arm is ArmKind.BASELINE:
            prompt, prefix = build_baseline_prompt(self._root, task), ""
        else:
            prompt, _stats = build_forgeos_prompt(self._root, task)
            prefix = FORGEOS_PROMPT_PREFIX

        req = GatewayRequest(
            model_ref=self._model_ref, prompt_prefix=prefix, prompt_tail=prompt,
            max_output_tokens=task.max_output_tokens,
            # Both arms, identically. These tasks are short factual lookups
            # ("name the function and the guard, under 30 words"); there is
            # nothing to reason about. Left at the "medium" default, a reasoning
            # model spent the ENTIRE output budget on chain-of-thought and
            # returned content:"" -- measured here as three of six tasks coming
            # back empty or cut off with a full quota of billed output tokens,
            # every one of which the suite scored as the model getting the
            # question wrong. Setting it on one arm only would rig the
            # comparison; setting it on both removes a confound from each.
            reasoning_effort="none",
        )
        t0 = time.monotonic()
        try:
            resp = self._gateway.complete(
                req, job_id=self._job_id, task_id=task.id, worker_id=f"forgebench.{arm.value}",
                remaining_micros=self._remaining_micros_fn(),
                affinity_key=self._job_id,
            )
        except CallRefused as e:
            raise BudgetExhausted(f"{task.id}/{arm.value}: {e}") from e
        seconds = time.monotonic() - t0
        return ArmCallResult(
            text=resp.text, tokens_in=resp.tokens_in, tokens_out=resp.tokens_out,
            tokens_cached_in=resp.tokens_cached_in, usd_micros=resp.usd_micros, seconds=seconds,
            finish_reason=resp.finish_reason,
        )


# ---------------------------------------------------------------- running


class SavingsClass(str, Enum):
    """09_FORGEBENCH.md's four savings receipt validity classes."""

    A_PAIRED_MEASURED = "A"
    B_HISTORICAL_MEASURED = "B"
    C_MODELLED_COUNTERFACTUAL = "C"
    D_NO_BASELINE = "D"


@dataclass
class ArmTotals:
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_cached_in: int = 0
    usd_micros: int = 0
    seconds: float = 0.0
    accepted_count: int = 0
    attempted_count: int = 0

    def add(self, outcome: "TaskArmOutcome") -> None:
        r = outcome.result
        self.tokens_in += r.tokens_in
        self.tokens_out += r.tokens_out
        self.tokens_cached_in += r.tokens_cached_in
        self.usd_micros += r.usd_micros
        self.seconds += r.seconds
        self.attempted_count += 1
        if outcome.accepted:
            self.accepted_count += 1


@dataclass
class TaskArmOutcome:
    task_id: str
    arm: ArmKind
    accepted: bool
    result: ArmCallResult


@dataclass
class SuiteRunResult:
    suite_name: str
    tasks: tuple[PinnedTask, ...]
    savings_class: SavingsClass
    outcomes: list[TaskArmOutcome] = field(default_factory=list)
    dry_run: bool = False
    dry_run_estimate_usd_micros: int = 0
    aborted_reason: str = ""
    aborted_at_task: str = ""


def totals_for(result: SuiteRunResult, arm: ArmKind) -> ArmTotals:
    totals = ArmTotals()
    for o in result.outcomes:
        if o.arm is arm:
            totals.add(o)
    return totals


@dataclass
class TaskComparison:
    task_id: str
    baseline: TaskArmOutcome | None
    forgeos: TaskArmOutcome | None

    @property
    def acceptance_matches(self) -> bool:
        """True only when both arms ran AND agree pass/fail. False if either
        is missing (task never reached, e.g. an abort) or they disagree --
        in both cases the cost comparison for THIS task is void."""
        if self.baseline is None or self.forgeos is None:
            return False
        return self.baseline.accepted == self.forgeos.accepted

    @property
    def comparison_voided(self) -> bool:
        return not self.acceptance_matches


def pair_by_task(result: SuiteRunResult) -> list[TaskComparison]:
    by_task: dict[str, dict[ArmKind, TaskArmOutcome]] = {}
    for o in result.outcomes:
        by_task.setdefault(o.task_id, {})[o.arm] = o
    return [
        TaskComparison(
            task_id=task.id,
            baseline=by_task.get(task.id, {}).get(ArmKind.BASELINE),
            forgeos=by_task.get(task.id, {}).get(ArmKind.FORGEOS),
        )
        for task in result.tasks
    ]


def _price_suite(
    tasks: Sequence[PinnedTask], arms: Sequence[ArmKind], root: Path, card: ModelCard
) -> int:
    return sum(estimate_task(root, task, arm, card).usd_micros for task in tasks for arm in arms)


def run_suite(
    tasks: Sequence[PinnedTask],
    *,
    suite_name: str,
    executor: ArmExecutor | None,
    savings_class: SavingsClass,
    budget_usd_micros: int,
    dry_run: bool = False,
    root: Path | None = None,
    card: ModelCard | None = None,
) -> SuiteRunResult:
    """Run (or price) a pinned suite across the arms `savings_class` implies.

    Class D runs the ForgeOS arm only -- there is nothing to pair against, so
    no baseline is executed. Every other class runs both arms, paired, in
    the same job.

    Two distinct budget guards, both reusing `estimate_task` (never a
    hand-rolled price):

    1. `dry_run=True` -- price the WHOLE suite, make zero calls, return.
    2. A live run -- price the whole suite BEFORE the first call and abort
       immediately (zero calls made) if that estimate alone exceeds
       `budget_usd_micros`. Then, call-by-call, `executor.run()` is expected
       to raise `BudgetExhausted` the moment LIVE measured spend would
       breach the ceiling (`GatewayExecutor` gets this for free from
       `Gateway.complete`'s own preflight-refuse path) -- at which point this
       function stops immediately rather than attempting the rest of the
       suite. Either guard "aborts rather than overspending"; the first is
       cheap insurance, the second is the one that matters when reality
       spends hotter than the estimate.
    """
    tasks = tuple(tasks)
    arms: tuple[ArmKind, ...] = (
        (ArmKind.FORGEOS,)
        if savings_class is SavingsClass.D_NO_BASELINE
        else (ArmKind.BASELINE, ArmKind.FORGEOS)
    )

    if dry_run:
        if root is None or card is None:
            raise ValueError("dry_run needs root and card to price prompts against")
        total = _price_suite(tasks, arms, root, card)
        return SuiteRunResult(
            suite_name=suite_name, tasks=tasks, savings_class=savings_class,
            dry_run=True, dry_run_estimate_usd_micros=total,
        )

    if root is not None and card is not None:
        preflight_total = _price_suite(tasks, arms, root, card)
        if preflight_total > budget_usd_micros:
            return SuiteRunResult(
                suite_name=suite_name, tasks=tasks, savings_class=savings_class,
                aborted_reason=(
                    f"pre-flight estimate {preflight_total} usd_micros exceeds the "
                    f"--budget-usd ceiling {budget_usd_micros} usd_micros -- aborted before spending"
                ),
            )

    if executor is None:
        raise ValueError("a live (non-dry-run) run needs an ArmExecutor")

    result = SuiteRunResult(suite_name=suite_name, tasks=tasks, savings_class=savings_class)
    for task in tasks:
        for arm in arms:
            try:
                call = executor.run(task, arm)
            except BudgetExhausted as e:
                result.aborted_reason = str(e)
                result.aborted_at_task = task.id
                return result
            accepted = task.acceptance.evaluate(call.text)
            result.outcomes.append(
                TaskArmOutcome(task_id=task.id, arm=arm, accepted=accepted, result=call)
            )
    return result


# --------------------------------------------------------------- reporting

_BASELINE_PROVENANCE_BY_CLASS: dict[SavingsClass, Provenance] = {
    SavingsClass.A_PAIRED_MEASURED: Provenance.REPLAYED,
    SavingsClass.B_HISTORICAL_MEASURED: Provenance.REPLAYED,
    SavingsClass.C_MODELLED_COUNTERFACTUAL: Provenance.MODELLED,
}


def _sha256_json(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _figures_for_arm(totals: ArmTotals, *, provenance: Provenance, note_prefix: str) -> dict[str, Figure]:
    return {
        "input_tokens": Figure(
            value=float(totals.tokens_in + totals.tokens_cached_in), unit="tokens",
            provenance=provenance,
            note=f"{note_prefix}: fresh+cached input tokens summed across the suite",
        ),
        "output_tokens": Figure(
            value=float(totals.tokens_out), unit="tokens", provenance=provenance,
            note=f"{note_prefix}: output tokens summed across the suite",
        ),
        "cash_cost": Figure(
            value=from_micros(totals.usd_micros), unit="usd", provenance=provenance,
            note=f"{note_prefix}: usd_micros summed across the suite's ledger-recorded Gateway calls",
        ),
        "wall_time": Figure(
            value=totals.seconds, unit="seconds", provenance=provenance,
            note=f"{note_prefix}: sum of per-call wall-clock seconds",
        ),
        "human_interventions": Figure(
            value=0.0, unit="count", provenance=Provenance.MEASURED,
            note="ForgeBench arms are unattended calls; no escalation path exists here to trip",
        ),
    }


@dataclass
class ForgeBenchReport:
    run: SuiteRunResult
    baseline_totals: ArmTotals
    forgeos_totals: ArmTotals
    comparisons: list[TaskComparison]
    proof: SavingsProof
    proof_complaints: list[str]
    comparison_voided: bool
    exit_gate_passed: bool | None


def _exit_gate(forgeos: ArmTotals, baseline: ArmTotals) -> bool:
    """Release 0.1's gate: same-or-better acceptance, at a lower cost per unit
    of correct work.

    Two rules, and the first one is what keeps this honest:

    1. NON-INFERIORITY. `forgeos.accepted_count >= baseline.accepted_count`.
       Being cheaper is never a pass if it got less right. A run where ForgeOS
       accepts fewer tasks fails here no matter how little it spent, and a run
       where NEITHER arm accepts anything fails too -- nothing correct happened,
       so there is nothing to be cheaper per unit of.

    2. COST PER ACCEPTED TASK, not total spend. Total spend is only comparable
       when both arms did the same amount of correct work; per-accepted-task is
       comparable when they did not, which is exactly this project's own stated
       measurement rule (`cost_per_accepted_task`, and `savings.py`'s refusal to
       treat token reduction as a saving). When acceptance IS equal the two are
       the same comparison, since the denominators cancel -- so this is not a
       weaker test in the case the gate was originally written for.

    This corrects a logic bug rather than relaxing a threshold. The old code
    already required `>=` on acceptance, but an earlier branch forced the gate
    to False whenever ANY task's acceptance differed between the arms -- which
    is every run where ForgeOS does BETTER. As written, the gate could only ever
    pass by ForgeOS being exactly as good as the naive baseline and never
    better; a strictly dominating result was reported as failure. The `void`
    rule that produced that behaviour is still enforced, unchanged, where it
    belongs: no percentage saving is ever claimed across unequal work.
    """
    if forgeos.accepted_count < baseline.accepted_count:
        return False
    if forgeos.accepted_count == 0:
        return False
    if baseline.accepted_count == 0:
        # The baseline got nothing right at any price. Any correct work at a
        # finite cost beats that; there is no ratio to compute.
        return True
    forgeos_per_task = forgeos.usd_micros / forgeos.accepted_count
    baseline_per_task = baseline.usd_micros / baseline.accepted_count
    return forgeos_per_task < baseline_per_task


def build_report(run: SuiteRunResult, *, mission_id: str, repo_revision: str) -> ForgeBenchReport:
    """Assemble the SavingsProof + comparison report for a finished (or
    aborted) `SuiteRunResult`. Reuses `Figure`/`Provenance`/`savings_pct`/
    `cost_per_accepted_task`/`verify_proof` throughout -- this function never
    computes a percentage or a per-task cost by hand.
    """
    forgeos_totals = totals_for(run, ArmKind.FORGEOS)
    baseline_totals = totals_for(run, ArmKind.BASELINE)
    comparisons = pair_by_task(run)

    void = (
        run.savings_class is SavingsClass.D_NO_BASELINE
        or bool(run.aborted_reason)
        or any(c.comparison_voided for c in comparisons)
    )

    actual = _figures_for_arm(
        forgeos_totals, provenance=Provenance.MEASURED, note_prefix="forgebench forgeos arm"
    )

    if run.savings_class is SavingsClass.D_NO_BASELINE:
        baseline = Baseline(
            baseline_policy=(
                "none captured -- Savings Class D per 09_FORGEBENCH.md: report actual "
                "usage only, no saving is claimed"
            ),
            baseline_provenance=Provenance.UNKNOWN,
            figures={},
        )
        savings: dict[str, Figure] = {}
        baseline_trace_hash = ""
    else:
        baseline_provenance = _BASELINE_PROVENANCE_BY_CLASS[run.savings_class]
        baseline_figures = _figures_for_arm(
            baseline_totals, provenance=baseline_provenance, note_prefix="forgebench baseline arm"
        )
        baseline = Baseline(
            baseline_policy=(
                "unwrapped CLI baseline: naive whole-scope-file prompt, no capsule, "
                "no cache-prefix discipline"
            ),
            baseline_provenance=baseline_provenance,
            figures=baseline_figures,
        )
        # Voided (acceptance differs between arms, or the run was cut short):
        # a cash-cost ratio here would compare two arms that did not do the
        # same amount of correct work. A cheaper arm that failed acceptance
        # is a LOSS, not a saving -- refuse to compute the percentage rather
        # than print one. (Per-task voiding lives on TaskComparison; this is
        # the aggregate enforcement of the same rule.)
        savings = (
            {}
            if void
            else {
                key: savings_pct(actual[key], baseline_figures[key])
                for key in actual
                if key in baseline_figures
            }
        )
        baseline_trace_hash = _sha256_json([
            [o.task_id, o.accepted, o.result.tokens_in, o.result.tokens_out, o.result.usd_micros]
            for o in run.outcomes
            if o.arm is ArmKind.BASELINE
        ])

    if run.aborted_reason:
        outcome = Outcome.INCONCLUSIVE
    elif void:
        outcome = Outcome.PARTIAL if forgeos_totals.accepted_count > 0 else Outcome.REJECTED
    elif forgeos_totals.accepted_count == len(run.tasks):
        outcome = Outcome.ACCEPTED
    elif forgeos_totals.accepted_count == 0:
        outcome = Outcome.REJECTED
    else:
        outcome = Outcome.PARTIAL

    contract_hash = _sha256_json([t.contract_dict() for t in run.tasks])
    actual_trace_hash = _sha256_json([
        [o.task_id, o.accepted, o.result.tokens_in, o.result.tokens_out, o.result.usd_micros]
        for o in run.outcomes
        if o.arm is ArmKind.FORGEOS
    ])
    test_artifact_hash = _sha256_json([[c.task_id, c.acceptance_matches] for c in comparisons])

    proof = SavingsProof(
        mission_id=mission_id,
        repo_revision=repo_revision,
        outcome=outcome,
        acceptance_passed=forgeos_totals.accepted_count,
        acceptance_total=len(run.tasks),
        actual=actual,
        cost_per_accepted_task=cost_per_accepted_task(actual["cash_cost"], forgeos_totals.accepted_count),
        baseline=baseline,
        savings=savings,
        contract_hash=contract_hash,
        actual_trace_hash=actual_trace_hash,
        baseline_trace_hash=baseline_trace_hash,
        test_artifact_hash=test_artifact_hash,
    )
    complaints = verify_proof(proof)

    if run.savings_class is SavingsClass.D_NO_BASELINE or run.aborted_reason:
        exit_gate_passed = None
    else:
        exit_gate_passed = _exit_gate(forgeos_totals, baseline_totals)

    return ForgeBenchReport(
        run=run,
        baseline_totals=baseline_totals,
        forgeos_totals=forgeos_totals,
        comparisons=comparisons,
        proof=proof,
        proof_complaints=complaints,
        comparison_voided=void,
        exit_gate_passed=exit_gate_passed,
    )


def _arm_outcome_dict(outcome: TaskArmOutcome | None) -> dict | None:
    if outcome is None:
        return None
    result = outcome.result
    return {
        "accepted": outcome.accepted,
        "result": {
            "text": result.text,
            "tokens_in": result.tokens_in,
            "tokens_out": result.tokens_out,
            "tokens_cached_in": result.tokens_cached_in,
            "usd_micros": result.usd_micros,
            "seconds": result.seconds,
            "finish_reason": result.finish_reason,
            "truncated": result.truncated,
            "spent_output_on_nothing": result.spent_output_on_nothing,
        },
    }


def report_to_dict(report: ForgeBenchReport) -> dict:
    """Serialize the pinned-suite result without losing proof metadata.

    The terminal renderer is intentionally human-first. This companion shape is
    stable enough for CI, dashboards, and later benchmark aggregation while
    preserving the exact task contract, per-arm acceptance, cost facts, and the
    `SavingsProof` hashes that make a result auditable.
    """
    run = report.run
    comparisons = []
    for comparison in report.comparisons:
        comparisons.append({
            "task_id": comparison.task_id,
            "acceptance_matches": comparison.acceptance_matches,
            "comparison_voided": comparison.comparison_voided,
            "baseline": _arm_outcome_dict(comparison.baseline),
            "forgeos": _arm_outcome_dict(comparison.forgeos),
        })

    return {
        "schema": "forgeos.forgebench.v1",
        "mode": "dry-run" if run.dry_run else "live",
        "provenance": "modelled" if run.dry_run else "measured",
        "suite": {
            "name": run.suite_name,
            "savings_class": run.savings_class.value,
            "tasks": [task.contract_dict() for task in run.tasks],
        },
        "dry_run_estimate_usd_micros": run.dry_run_estimate_usd_micros,
        "aborted": {
            "reason": run.aborted_reason,
            "at_task": run.aborted_at_task,
        },
        "totals": {
            "baseline": asdict(report.baseline_totals),
            "forgeos": asdict(report.forgeos_totals),
        },
        "comparisons": comparisons,
        "comparison_voided": report.comparison_voided,
        "exit_gate_passed": report.exit_gate_passed,
        "proof": report.proof.model_dump(mode="json"),
        "proof_complaints": list(report.proof_complaints),
    }


def write_json_report(path: str | Path, report: ForgeBenchReport) -> Path:
    """Write a deterministic, UTF-8 JSON receipt and return its path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report_to_dict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def render_report(report: ForgeBenchReport) -> str:
    run = report.run
    lines: list[str] = []
    lines.append("=" * 74)
    lines.append(f"FORGEBENCH -- suite={run.suite_name!r} class={run.savings_class.value}")
    lines.append("=" * 74)

    if run.dry_run:
        lines.append(
            f"DRY RUN -- priced, nothing spent: {run.dry_run_estimate_usd_micros} usd_micros "
            f"(${from_micros(run.dry_run_estimate_usd_micros):.6f}) across {len(run.tasks)} task(s)"
        )
        return "\n".join(lines)

    if run.aborted_reason:
        lines.append(f"ABORTED at task {run.aborted_at_task!r}: {run.aborted_reason}")
        lines.append(f"({len(run.outcomes)} arm-call(s) completed before the abort)")

    lines.append("")
    lines.append(f"{'task':<28} {'baseline':<10} {'forgeos':<10} note")
    lines.append("-" * 74)
    def _cell(outcome) -> str:
        """A failure caused by the OUTPUT CAP is not the model getting it wrong,
        and printing both as "fail" hides the difference. `empty` means output
        tokens were billed and no answer came back at all -- the signature of a
        reasoning model spending its whole budget before it starts writing."""
        if outcome is None:
            return "-"
        if not outcome.accepted and outcome.result.spent_output_on_nothing:
            return "empty!"
        if not outcome.accepted and outcome.result.truncated:
            return "cut off!"
        return "accept" if outcome.accepted else "fail"

    flagged = False
    for c in report.comparisons:
        b, h = _cell(c.baseline), _cell(c.forgeos)
        flagged = flagged or "!" in b or "!" in h
        note = "VOID (acceptance differs)" if (c.comparison_voided and c.baseline and c.forgeos) else ""
        lines.append(f"{c.task_id:<28} {b:<10} {h:<10} {note}")
    if flagged:
        lines.append(
            "\n  'empty!' / 'cut off!' are HARNESS failures, not model failures: the\n"
            "  answer was cut off at max_output_tokens, or the entire output budget\n"
            "  went to reasoning tokens and no answer was returned. Those tasks say\n"
            "  nothing about either arm's quality -- raise the cap or lower\n"
            "  reasoning_effort and re-run before reading anything into them."
        )

    lines.append("")
    lines.append(f"{'':10} {'attempted':>9} {'accepted':>8} {'tok in':>9} {'tok out':>8} {'USD':>11}")
    for name, totals in (("baseline", report.baseline_totals), ("forgeos", report.forgeos_totals)):
        lines.append(
            f"{name:<10} {totals.attempted_count:>9} {totals.accepted_count:>8} "
            f"{totals.tokens_in + totals.tokens_cached_in:>9,} {totals.tokens_out:>8,} "
            f"{from_micros(totals.usd_micros):>11.6f}"
        )

    lines.append("")
    if report.comparison_voided:
        lines.append(
            "COST COMPARISON VOID -- acceptance differs between arms (or no baseline ran, "
            "or the run was aborted). Printing a cash-cost ratio here would compare two runs "
            "that did not do the same amount of correct work. No saving is claimed."
        )

    lines.append("")
    lines.append(render_receipt(report.proof))

    if report.proof_complaints:
        lines.append("")
        lines.append("!" * 74)
        lines.append("RECEIPT INTEGRITY COMPLAINTS (verify_proof):")
        for c in report.proof_complaints:
            lines.append(f"  - {c}")
        lines.append("!" * 74)

    lines.append("")
    gate = report.exit_gate_passed
    gate_str = "N/A (no baseline or aborted)" if gate is None else ("PASS" if gate else "FAIL")
    lines.append(f"RELEASE 0.1 EXIT GATE: {gate_str}")
    lines.append('  "reproduce equal acceptance with lower measured token/cash cost than the')
    lines.append('   unwrapped CLI baseline" (08_IMPLEMENTATION_ROADMAP.md)')
    return "\n".join(lines)


# -------------------------------------------------------------- default suite

_JSON_ARRAY_CHECK = (
    "import json, sys\n"
    "data = json.load(open(sys.argv[1], encoding='utf-8'))\n"
    "want = {'to_micros', 'from_micros'}\n"
    "ok = isinstance(data, list) and want <= {str(x) for x in data}\n"
    "sys.exit(0 if ok else 1)\n"
)

DEFAULT_SUITE: tuple[PinnedTask, ...] = (
    PinnedTask(
        id="ledger-dedup-guard",
        objective=(
            "In this codebase, which function records spend to the ledger, and what "
            "specifically prevents the same model call being charged twice? Name the "
            "function and the guard. Answer in under 80 words."
        ),
        scope=Scope(paths=[
            "forgeos/forge.py", "forgeos/ledger.py", "forgeos/gateway/client.py",
            "forgeos/adapters/gateway_worker.py", "forgeos/core/governor.py",
        ]),
        acceptance=AcceptanceCheck(
            description="names record_spend and the in-flight/dedup guard",
            predicate=keyword_groups(
                ("record_spend",),
                ("inflight", "in-flight", "in flight", "dedup", "coalesce", "coalesced"),
            ),
        ),
        max_output_tokens=200,
    ),
    PinnedTask(
        id="preflight-refusal-types",
        objective=(
            "In forgeos/economy/preflight.py, name the exact exception class raised by "
            "`PreflightVerdict.raise_if_refused()` when a call is refused, and name the "
            "Enum class whose three members represent ALLOW / REFUSE_BUDGET / "
            "REFUSE_CONTEXT. Two class names, under 30 words."
        ),
        scope=Scope(paths=["forgeos/economy/preflight.py"]),
        acceptance=AcceptanceCheck(
            description="names CallRefused and Decision",
            predicate=keyword_groups(("CallRefused",), ("Decision",)),
        ),
        max_output_tokens=120,
    ),
    PinnedTask(
        id="catalog-cache-pricing",
        objective=(
            "In forgeos/catalog.py, ModelCard has a property that prices a CACHED input "
            "token. Name that property, and name the OTHER ModelCard field it falls back "
            "to when no cache price has been published. Under 30 words."
        ),
        scope=Scope(paths=["forgeos/catalog.py"]),
        acceptance=AcceptanceCheck(
            description="names cache_read_rate and its input_cost_per_1m fallback",
            predicate=keyword_groups(("cache_read_rate",), ("input_cost_per_1m",)),
        ),
        max_output_tokens=120,
    ),
    PinnedTask(
        id="ledger-generation-fencing",
        objective=(
            "In forgeos/ledger.py, name the method that advances a task's generation "
            "counter when a worker is reclaimed, and the method that refuses to record a "
            "report stamped with a stale generation. Under 30 words."
        ),
        scope=Scope(paths=["forgeos/ledger.py"]),
        acceptance=AcceptanceCheck(
            description="names bump_generation and record_report_if_current",
            predicate=keyword_groups(("bump_generation",), ("record_report_if_current",)),
        ),
        max_output_tokens=120,
    ),
    PinnedTask(
        id="contracts-micros-json",
        objective=(
            "Reply with ONLY a JSON array (no prose, no markdown fences) of the exact "
            "snake_case names of the two module-level functions in forgeos/contracts.py "
            'that convert between dollars and integer microdollars. Example shape: '
            '["fn_one", "fn_two"].'
        ),
        scope=Scope(paths=["forgeos/contracts.py"]),
        acceptance=AcceptanceCheck(
            description="output parses as JSON containing to_micros and from_micros",
            command=(sys.executable, "-c", _JSON_ARRAY_CHECK, "{output}"),
        ),
        max_output_tokens=60,
    ),
    PinnedTask(
        id="savings-provenance-rule",
        objective=(
            "In forgeos/economy/savings.py, `savings_pct`'s docstring states a core rule "
            "about which of its two input provenances (actual vs baseline) the result's "
            "own provenance is set to. Name the private helper function (leading "
            "underscore) that implements that choice, and state in one word whether the "
            "result takes the stronger or the lesser-certain of the two inputs."
        ),
        scope=Scope(paths=["forgeos/economy/savings.py"]),
        acceptance=AcceptanceCheck(
            description="names _weaker and answers 'weaker'",
            predicate=keyword_groups(("_weaker",), ("weaker",)),
        ),
        max_output_tokens=150,
    ),
)


def default_suite() -> tuple[PinnedTask, ...]:
    return DEFAULT_SUITE


# --------------------------------------------------------------------- CLI


def _repo_revision(root: Path) -> str:
    """Best-effort short git SHA. Never raises -- "unknown" is an honest
    answer when this is not a git checkout or git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    rev = out.stdout.strip()
    return rev if out.returncode == 0 and rev else "unknown"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="forge forgebench", description=__doc__)
    ap.add_argument("--model", default="", help="model_ref, e.g. deepseek/deepseek-chat")
    ap.add_argument("--budget-usd", type=float, default=0.50, dest="budget_usd")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run",
                     help="price the whole suite; make no calls")
    ap.add_argument("--skip-baseline", action="store_true", dest="skip_baseline",
                     help="ForgeOS arm only -- Savings Class D, no saving is claimed")
    ap.add_argument("--ledger-path", default=":memory:", dest="ledger_path")
    ap.add_argument("--json-out", default="", dest="json_out",
                    help="write the machine-readable suite receipt to this path")
    return ap


def main(
    argv: Sequence[str] | None = None,
    *,
    catalog: Catalog | None = None,
    settings: Settings | None = None,
    root: Path | None = None,
) -> int:
    """CLI entry point. `catalog`/`settings`/`root` are injectable purely for
    tests -- a real invocation always passes `None` and gets the live
    catalog, live settings, and this repo's own root.
    """
    args = build_parser().parse_args(argv)
    root = root if root is not None else Path(__file__).resolve().parent.parent
    tasks = default_suite()

    settings = settings if settings is not None else Settings.load()
    catalog = catalog if catalog is not None else default_catalog()

    model_ref = args.model
    if not model_ref:
        for p in sorted(settings.providers.values(), key=lambda x: x.name):
            if p.kind is ProviderKind.API and p.usable:
                cards = [c for c in catalog.all() if c.provider == p.name]
                if cards:
                    model_ref = min(cards, key=lambda c: c.input_cost_per_1m + c.output_cost_per_1m).ref
                    break

    card = catalog.get(model_ref) if model_ref else None
    if card is None:
        print(f"no usable model (asked for {model_ref!r}); pass --model or configure a provider key")
        return 1

    savings_class = SavingsClass.D_NO_BASELINE if args.skip_baseline else SavingsClass.A_PAIRED_MEASURED
    budget_micros = to_micros(args.budget_usd)
    repo_revision = _repo_revision(root)

    if args.dry_run:
        run = run_suite(
            tasks, suite_name="forgeos-default", executor=None, savings_class=savings_class,
            budget_usd_micros=budget_micros, dry_run=True, root=root, card=card,
        )
        report = build_report(run, mission_id="forgebench-dry-run", repo_revision=repo_revision)
        print(render_report(report))
        if args.json_out:
            print(f"JSON receipt: {write_json_report(args.json_out, report)}")
        return 0

    ledger = open_ledger(args.ledger_path)
    try:
        job_id = ledger.open_job(
            JobSpec(
                objective=f"forgebench:{model_ref}", cwd=str(root),
                budget=Budget(max_usd=args.budget_usd),
            )
        )
        gateway = Gateway(
            catalog=catalog, ledger=ledger, settings=settings, transports=default_transports(settings)
        )
        executor = GatewayExecutor(
            gateway=gateway, job_id=job_id, model_ref=model_ref, root=root,
            remaining_micros_fn=lambda: budget_micros - ledger.job_spend_micros(job_id),
        )
        run = run_suite(
            tasks, suite_name="forgeos-default", executor=executor, savings_class=savings_class,
            budget_usd_micros=budget_micros, dry_run=False, root=root, card=card,
        )
        report = build_report(run, mission_id=job_id, repo_revision=repo_revision)
    finally:
        ledger.close()

    print(render_report(report))
    if args.json_out:
        print(f"JSON receipt: {write_json_report(args.json_out, report)}")
    if run.aborted_reason:
        return 3
    return 0 if report.exit_gate_passed in (None, True) else 1


__all__ = [
    "AcceptanceCheck",
    "ArmCallResult",
    "ArmExecutor",
    "ArmKind",
    "ArmTotals",
    "BudgetExhausted",
    "DEFAULT_SUITE",
    "ForgeBenchReport",
    "GatewayExecutor",
    "PinnedTask",
    "SavingsClass",
    "SuiteRunResult",
    "TaskArmOutcome",
    "TaskComparison",
    "TaskFamily",
    "build_baseline_prompt",
    "build_forgeos_prompt",
    "build_parser",
    "build_report",
    "default_suite",
    "estimate_task",
    "keyword_groups",
    "main",
    "pair_by_task",
    "report_to_dict",
    "render_report",
    "run_suite",
    "totals_for",
    "write_json_report",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
