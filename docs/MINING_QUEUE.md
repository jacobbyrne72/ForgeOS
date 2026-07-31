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
| 18 | `HKUDS/ClawTeam` | | queued |
| 19 | `aiming-lab/AutoResearchClaw` | | queued |
| 20 | `MervinPraison/PraisonAI` | | queued |
| 21 | `Significant-Gravitas/AutoGPT` | | queued |
| 22 | `f/prompts.chat` | prompt library | queued |
| 23 | `langflow-ai/langflow` | | queued |
| 24 | `JuliusBrussee/caveman` | token compression | queued |
| 25 | `langgenius/dify` | | queued |
| 26 | `x1xhlol/system-prompts-and-models-of-ai-tools` | prompt archaeology | queued |
| 27 | `msitarzewski/agency-agents` | | queued |
| 28 | `Comfy-Org/ComfyUI` | | queued |
| 29 | `nextlevelbuilder/ui-ux-pro-max-skill` | dashboard design | queued |
| 30 | `punkpeye/awesome-mcp-servers` | MCP index | queued |
| 31 | `PaddlePaddle/PaddleOCR` | | queued |
| 32 | `CursorTouch/Windows-MCP` | | queued |
| 33 | `infiniflow/ragflow` | | queued |
| 34 | `unclecode/crawl4ai` | **REQUESTED AS A DEPENDENCY, not just source** | building now |
| 35 | `D4Vinci/Scrapling` | crawler #2 | building now |
| 36 | `apify/crawlee` | crawler #3 (Node) | building now |
| 37 | `iplocate/free-proxy-list` | free proxy source | building now |
| 38 | `jhao104/proxy_pool` | proxy pool | building now |
| 39 | `CloakHQ/CloakBrowser` | | queued |
| 40 | `CloakHQ/CloakBrowser-Manager` | | queued |
| 41 | `Johell1NS/browser-search` | | queued |
| 42 | `colbymchenry/codegraph` | **install-and-use, not port** | queued |
| 43 | `CodeGraphContext/CodeGraphContext` | **install-and-use, not port** | queued |
| 44 | `FoundationAgents/MetaGPT` | "this repo is a winner" | MINING NOW |
| 45 | `microsoft/autogen` | | MINING NOW |
| 46 | `conductor-oss/conductor` | "WOAH GOLDMINE" — durable workflow engine | MINING NOW |
| 47 | `apache/airflow` | "OP" — DAG scheduling, retries, SLAs | MINING NOW |
| 48 | `docling-project/docling` | document extraction | queued |
| 49 | `headroomlabs-ai/headroom` | token compression — pair, not port | queued |
| 50 | `Panniantong/Agent-Reach` | "this ones good" | queued |
| 51 | `mvanhorn/last30days-skill` | | queued |
| 52 | `BerriAI/litellm` | already the price-table source; mine the router | queued |
| 53 | `CherryHQ/cherry-studio` | | queued |
| 54 | `Aider-AI/aider` | already vendored | queued |
| 55 | `HKUDS/nanobot` | | queued |
| 56 | `rohitg00/ai-engineering-from-scratch` | | queued |
| 57 | `patchy631/ai-engineering-hub` | "NEED to read all of this" | queued |
| 58 | `tinyhumansai/openhuman` | "op" | queued |
| 59 | `continuedev/continue` | | queued |
| 60 | `VectifyAI/PageIndex` | document indexing | queued |
| 61 | `qdrant/qdrant` | vector DB — dependency, not port | queued |
| 62 | `lutzroeder/netron` | | queued |
| 63 | `lightpanda-io/browser` | fast headless browser — crawler backend candidate | queued |
| 64 | `linshenkx/prompt-optimizer` | | queued |
| 65 | `zeroclaw-labs/zeroclaw` | | queued |
| 66 | `langfuse/langfuse` | LLM observability — pair, not port | queued |
| 67 | `SillyTavern/SillyTavern` | | queued |
| 68 | `onyx-dot-app/onyx` | | queued |
| 69 | `iOfficeAI/AionUi` | "bingo" — 24/7 cowork UI | queued |
| 70 | `labring/FastGPT` | | queued |
| 71 | `e2b-dev/awesome-ai-agents` | index | queued |
| 72 | `Jakedismo/codegraph-rust` | | queued |
| 73 | `GlitterKill/sdl-mcp` | | queued |
| 74 | `affaan-m/ECC` | | queued |
| 75 | `n8n-io/n8n` | ranked #8 by tree triage | queued |
| 76 | `multica-ai/andrej-karpathy-skills` | already an overlay in global CLAUDE.md | queued |
| 77 | `garrytan/gstack` | | queued |
| 78 | `Graphify-Labs/graphify` | | queued |
| 79 | `zed-industries/zed` | | queued |
| 80 | `earendil-works/pi` | | queued |
| 81 | `Egonex-AI/Understand-Anything` | "bingo" — already the codebase-map tool per global CLAUDE.md | queued |
| 82 | `cline/cline` | | queued |
| 83 | `shanraisshan/claude-code-best-practice` | "gold" | queued |
| 84 | `upstash/context7` | live docs lookup — DEPENDENCY candidate | queued |
| 85 | `astral-sh/ruff` | already a dev dependency | SKIP-as-port, in use |
| 86 | `tmux/tmux` | runtime for cli_team; not a port | SKIP-as-port, in use |
| 87 | `Alishahryar1/free-claude-code` | **NOT on disk.** Name implies subscription-seat routing — read before any port; if it presents another product's first-party OAuth client id, that is client impersonation and ForgeOS will not ship it (see note on #9) | queued, read-first |
| 88 | `DeusData/codebase-memory-mcp` | already an MCP in use per global CLAUDE.md | queued |

## Use directly vs port

Not every repo should become ForgeOS source. Three outcomes:

- **DEPENDENCY** — installed and driven through an adapter, optional so a
  missing install degrades rather than breaks. crawl4ai, Scrapling, crawlee,
  CodeGraph belong here: they are maintained projects doing a job well, and
  vendoring a snapshot of them buys a fork to maintain and nothing else.
- **PORT** — a self-contained mechanism copied in with tests, because it is
  small, has no runtime we want, or must obey our gates.
- **SKIP** — with a reason.

## Credentials — hard line

Proxy credentials, API keys and account logins are read from ENVIRONMENT
VARIABLES the operator sets. They are never copied into this repo, never
written to config committed here, never logged, and never echoed into a
receipt. The proxy layer below follows `gateway/keyring.py`: the record type
structurally cannot hold a secret value, and a test asserts a known secret
cannot appear in any rendered output.


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
