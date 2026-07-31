"""The README's own code example is a contract, and this holds it to one.

A README is the first thing a reader runs and the last thing anyone thinks
to test. Rename a method, move a class out of the package root, change a
constructor's required arguments — every unit test still passes and the
front page of the repo quietly starts lying. For a project whose whole pitch
is "measured, never asserted", a documented example that no longer imports
is the cheapest possible way to lose a reader's trust.

What this file deliberately does NOT do: execute `forge.run(...)` from the
example. That call routes to a real worker and spends real money; a test
suite that bills the operator to prove a docstring is a worse bug than the
one it catches. So the guard is exact but bounded — every name the example
imports must import, every attribute it reaches must exist, and every
keyword it passes must be accepted by the real signature.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parent.parent / "README.md"


def _python_blocks(text: str) -> list[str]:
    return re.findall(r"```python\n(.*?)```", text, re.DOTALL)


@pytest.fixture(scope="module")
def readme_text() -> str:
    if not README.exists():  # pragma: no cover - the repo always ships one
        pytest.skip("README.md not found")
    return README.read_text(encoding="utf-8")


def test_the_readme_has_a_python_example_at_all(readme_text):
    """If this fails, the front page stopped showing anyone how to use it."""
    assert _python_blocks(readme_text), "README.md has no ```python example block"


def test_every_name_the_readme_imports_from_forgeos_actually_imports(readme_text):
    """`from forgeos import X` in the README must work for every X.

    The package moved to lazy PEP-562 attribute loading for startup cost, so
    a name can drop out of `_LAZY` and only fail at first access — precisely
    the failure a reader hits and no other test does.
    """
    import forgeos

    imported: set[str] = set()
    for block in _python_blocks(readme_text):
        for match in re.finditer(r"^from forgeos import (.+)$", block, re.MULTILINE):
            imported.update(name.strip() for name in match.group(1).split(","))

    assert imported, "no `from forgeos import ...` line found in the README example"
    missing = [name for name in sorted(imported) if not hasattr(forgeos, name)]
    assert not missing, f"README imports names forgeos no longer exports: {missing}"


def test_the_example_only_reaches_attributes_that_exist(readme_text):
    """`forge.doctor()` and `result.cost_per_accepted` are load-bearing promises."""
    from forgeos.forge import Forge, ForgeResult

    assert callable(getattr(Forge, "doctor", None)), "README shows forge.doctor()"
    assert isinstance(getattr(ForgeResult, "cost_per_accepted", None), property), (
        "README prints result.cost_per_accepted"
    )


def test_every_keyword_the_example_passes_is_accepted_by_the_real_signature(readme_text):
    """The example constructs a TaskSpec by keyword. A renamed or removed
    field would leave the README showing a TypeError as documentation."""
    from forgeos.contracts import TaskSpec

    blocks = "\n".join(_python_blocks(readme_text))
    shown = set(re.findall(r"\b(\w+)=", blocks))
    fields = set(TaskSpec.model_fields)
    # Only judge the words that are plausibly TaskSpec fields: the block also
    # contains Budget(...) and Scope(...) keywords, and inventing a rule that
    # every keyword anywhere must be a TaskSpec field would fail on those.
    taskspec_like = {"job_id", "subject", "description", "capabilities",
                     "scope", "acceptance", "budget"}
    for name in sorted(shown & taskspec_like):
        assert name in fields, f"README passes TaskSpec({name}=...) but the field is gone"


def test_forge_is_still_constructible_with_no_arguments(readme_text):
    """`Forge()` with no arguments is the example's first line. If a required
    argument appears, the documented quickstart stops being a quickstart."""
    from forgeos.forge import Forge

    params = inspect.signature(Forge.__init__).parameters
    required = [
        name for name, p in params.items()
        if name != "self"
        and p.default is inspect.Parameter.empty
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    assert not required, f"Forge() now requires arguments the README does not pass: {required}"
