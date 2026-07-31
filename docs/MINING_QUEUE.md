# Mining queue — repos to port features from

Working list. A repo leaves this file only when its features are either ported
or explicitly marked SKIP with a reason. "Read it" is not done; "ported or
justified" is done.

## Ground truth about what is already on disk

- `C:\Users\byrne\Downloads\forgeos_github_mega_catalog_2026-07-30\` is a
  **catalogue, not code**: `clone_urls.txt` (713 URLs), `repos.json`,
  `repos.md`, `top_100.md`. Nothing to read for implementation.
- The clones of that catalogue DO exist, 713 across five directories:
  `ForgeOS-catalog-p0` (25), `-p1-partial` (235), `-p2-partial` (239),
  `-other-partial` (195), `-replacements` (19).
- `vendor/` holds 34 repos already vendored into this repo.
- So: **713 cloned, 34 vendored.** The gap between those two numbers is the
  backlog this file tracks.

## Explicitly requested (highest priority, in the order asked for)

| # | Repo | Why it was sent | Status |
|---|---|---|---|
| 1 | Hermes (`~/.hermes`, `AppData/Local/hermes`) | "we basically need all of its code cause we want to have every feature it has" | inventory in progress |
| 2 | `openai/codex` | the Codex CLI — feature parity target | inventory in progress |
| 3 | `Yeachan-Heo/oh-my-codex` | | inventory in progress |
| 4 | `1jehuang/jcode` | source reference | inventory in progress |
| 5 | `jgravelle/jcodemunch-mcp` | | inventory in progress |
| 6 | `diegosouzapw/OmniRoute` | routing; already partly vendored | inventory in progress |
| 7 | `justlovemaki/AIClient2API` | | inventory in progress |
| 8 | `MemPalace/mempalace` | memory | inventory in progress |
| 9 | `EvanZhouDev/openai-oauth` | see note below | queued |
| 10 | `code-yeongyu/oh-my-openagent` | | queued |
| 11 | `openclaw/openclaw` | | queued |
| 12 | `NousResearch/hermes-agent` | the *other* Hermes — distinct from the local `~/.hermes`; inventory both, do not conflate | queued |
| 13 | `garrytan/gbrain` | memory/brain | queued |
| 14 | `nesquena/hermes-webui` | web UI — compare against our dashboard | queued |
| 15 | `assafelovic/gpt-researcher` | research pipeline; multi-source retrieval | queued |
| 16 | `decolua/9router` | **already in `vendor/9router`** — confirm it is the same project, then mine it properly rather than re-cloning | queued |
| 17 | `ruvnet/ruflo` | multi-agent swarm harness, 100+ agent definitions | queued |

### Note on #9, openai-oauth

Read it for what it actually implements before porting anything. There is a
real distinction: a **device-code / official public-client flow** a vendor
publishes for third-party CLIs is legitimate; **presenting another product's
first-party client id** to capture a consumer subscription seat is client
impersonation, breaks vendor ToS, and is the single most likely thing to get a
public repo taken down. ForgeOS's whole position is that its claims survive
scrutiny — it cannot ship the second kind. Port the first if that is what this
is; document the refusal if it is not.

## Rules for porting

1. **Check `forgeos/` first.** ~300 modules already exist. Re-implementing
   something we have is worse than not porting at all.
2. **Check the LICENCE** before copying a line. Record it in the port commit.
3. **Feature-by-feature, not whole-harness.** Pasting an entire codebase brings
   its architecture, its dependencies and its dead code, and none of that
   survives this repo's own gates.
4. **Every port gets tests**, in this repo's style, or it is not ported.
5. **SKIP is a legitimate outcome** and must carry a reason — dead code, a
   thinner version of what we have, or a licence that forbids reuse.
