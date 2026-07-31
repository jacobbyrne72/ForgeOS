<!-- See CONTRIBUTING.md's "PR expectations" section -- this is the checklist version of it,
     not a replacement for it. A box left unchecked, or a section left blank, reads as
     "not done," not as "not applicable" -- say so explicitly if something truly doesn't apply. -->

## What changed and why

## Scope

Files touched, and why this set is sufficient (a file in the diff that isn't explained here
is scope creep -- justify it or take it out):

## Verification

Paste the real pytest summary line -- not "tests pass":

```
$ python -m pytest tests -q -m "not slow"
<paste it>
```

- [ ] Every change has a test: new behavior gets a new test, a bug fix gets a test that
      reproduces the bug and passes after the fix.
- [ ] `ruff check forgeos/ tests/ tools/` is clean.
- [ ] No claim in this PR (a saving, a fix, a speedup) is made without the receipt above it.
- [ ] Any budget, ledger, or merge-gate semantics touched by this change are called out here:
