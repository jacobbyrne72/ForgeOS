"""Packaging metadata is a contract with anyone who installs this.

Written after a real near-miss: adding `[project.urls]` above the
`dependencies` array silently swallowed it. TOML table headers capture every
key until the next header, so the file still parsed, every test still passed,
and the package would have shipped declaring NO dependencies — pydantic,
fastapi, httpx, psutil and tiktoken all silently absent from a fresh install.
Nothing in a test suite that imports from the source tree can notice that;
the source tree has them installed already. Only reading the parsed metadata
back catches it.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


@pytest.fixture(scope="module")
def project() -> dict:
    return tomllib.load(PYPROJECT.open("rb"))["project"]


def test_runtime_dependencies_survived_the_toml_layout(project):
    """The near-miss this file exists for: dependencies absorbed by a table."""
    deps = project.get("dependencies", [])
    assert deps, "pyproject declares no runtime dependencies — a table header likely swallowed them"
    names = {d.split(">")[0].split("=")[0].split("[")[0].strip().lower() for d in deps}
    # Each of these is on the DEFAULT code path, not an extra: the resource
    # governor and token preflight import psutil/tiktoken unconditionally.
    for required in ("pydantic", "psutil", "tiktoken", "httpx"):
        assert required in names, f"{required} missing from runtime dependencies"


def test_urls_table_holds_only_urls(project):
    """If a non-URL key appears here, an array above it was captured."""
    urls = project.get("urls", {})
    assert urls, "no [project.urls]: a PyPI listing with no repo link reads as abandoned"
    for key, value in urls.items():
        assert isinstance(value, str) and value.startswith("http"), (
            f"[project.urls].{key} is not a URL ({value!r}) — a stray key landed in this table"
        )


def test_pypi_listing_metadata_is_present(project):
    """Without classifiers and urls, the listing is unfindable and unlinked."""
    assert project.get("classifiers"), "no trove classifiers: PyPI search cannot find this"
    assert project.get("keywords"), "no keywords"
    assert project.get("authors"), "no authors"


def test_no_license_classifier_alongside_a_license_expression(project):
    """PEP 639: a `license = "MIT"` expression and a `License ::` classifier
    cannot coexist — setuptools refuses to build.

    Written after exactly that broke every CI leg at `pip install`, before a
    single test ran. Parsing the TOML said nothing was wrong; only a real
    build does. This test is the cheap standing check that a parse gives you.
    """
    if not isinstance(project.get("license"), str):
        pytest.skip("no PEP 639 license expression in use")
    offenders = [c for c in project.get("classifiers", []) if c.startswith("License ::")]
    assert not offenders, (
        f"license expression {project['license']!r} cannot coexist with {offenders}"
    )


def test_the_package_actually_builds(tmp_path):
    """The check the TOML parse could not make.

    Metadata that parses can still be unbuildable — a rejected classifier
    combination, a bad dependency spec, a packages-find that resolves to
    nothing. This runs the real backend and fails the way CI fails.
    """
    import subprocess
    import sys

    try:
        import build  # noqa: F401
    except ImportError:
        pytest.skip("the `build` package is not installed; CI's pip install covers this path")

    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(tmp_path),
         str(PYPROJECT.parent)],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, f"wheel build failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    assert list(tmp_path.glob("*.whl")), "build reported success but produced no wheel"


def test_classifiers_only_claim_python_versions_ci_actually_proves():
    """A classifier is a promise. CI runs 3.11 and 3.12; claiming 3.13 here
    would be an untested assertion in the one place users read as fact."""
    project = tomllib.load(PYPROJECT.open("rb"))["project"]
    claimed = {
        c.rsplit("::", 1)[-1].strip()
        for c in project["classifiers"]
        if c.startswith("Programming Language :: Python :: 3.")
    }
    workflow = (PYPROJECT.parent / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for version in sorted(claimed):
        assert f'"{version}"' in workflow or f"'{version}'" in workflow, (
            f"classifiers claim Python {version} but CI never runs it"
        )


def test_requires_python_matches_the_lowest_claimed_classifier(project):
    """`requires-python` and the classifiers must not disagree about the floor."""
    claimed = sorted(
        c.rsplit("::", 1)[-1].strip()
        for c in project["classifiers"]
        if c.startswith("Programming Language :: Python :: 3.")
    )
    assert claimed, "no versioned Python classifiers"
    floor = claimed[0]
    assert floor in project["requires-python"], (
        f"requires-python={project['requires-python']!r} disagrees with lowest classifier {floor}"
    )
