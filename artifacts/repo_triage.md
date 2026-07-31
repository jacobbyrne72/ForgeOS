# Repo triage — ranked from file trees alone

The catalogue clones are `--filter=blob:none`: every FILENAME is local, no
file CONTENT is. `git show HEAD:README.md` returns `bad object` on all of
them. Reading them means 713 network fetches, so this ranks them first and
hydrates only what earns it.

- repos with a readable tree: **713**
- repos whose tree could not be read: 0
- scoring above zero and larger than 12 files: **647**

Scored on capability signals in path names, weighted by where ForgeOS is
actually weak — sandboxing highest (ForgeOS has none and ships a shell tool),
retrieval lowest (the capsule and two-stage reranker are already ahead).

## How common is each capability across the corpus

| signal | repos showing it |
|---|---|
| eval_harness | 464 |
| hooks | 434 |
| memory | 401 |
| observability | 401 |
| mcp | 323 |
| routing | 298 |
| compression | 289 |
| scheduling | 275 |
| sub_agents | 275 |
| durable_workflow | 249 |
| retry_policy | 231 |
| crawling | 224 |
| sandboxing | 210 |
| security | 204 |
| cost_tracking | 189 |
| prompt_cache | 45 |

## Hydration order — top 40

Fetch these sequentially (never in parallel; this machine has been crashed
by parallel cloning). Each is `git -C <path> fetch --refetch` or a checkout
of the paths that matter.

| # | repo | score | files | py | signals |
|---|---|---|---|---|---|
| 1 | `theredsix--agent-browser-protocol` | 360 | 482708 | 7226 | observability×8228, sandboxing×3847, memory×2065, eval_harness×1889, hooks×1725 |
| 2 | `NousResearch--hermes-agent` | 348 | 8075 | 3660 | hooks×1040, mcp×167, memory×113, compression×99, sub_agents×66 |
| 3 | `BerriAI--litellm` | 344 | 9244 | 5082 | hooks×680, mcp×396, routing×365, cost_tracking×253, memory×248 |
| 4 | `comet-ml--opik` | 344 | 11963 | 2835 | observability×1247, eval_harness×1131, hooks×309, mcp×86, compression×43 |
| 5 | `elizaOS--eliza` | 342 | 43046 | 2799 | hooks×10067, eval_harness×7415, sub_agents×744, memory×319, routing×273 |
| 6 | `ai-dynamo--dynamo` | 340 | 5080 | 1355 | routing×410, eval_harness×215, memory×172, durable_workflow×164, hooks×163 |
| 7 | `langgenius--dify` | 336 | 13386 | 3681 | hooks×1473, observability×155, eval_harness×137, mcp×90, sub_agents×71 |
| 8 | `n8n-io--n8n` | 336 | 26046 | 69 | crawling×1233, eval_harness×1170, mcp×655, hooks×596, memory×323 |
| 9 | `google-gemini--gemini-cli` | 336 | 2939 | 17 | hooks×229, mcp×83, observability×82, eval_harness×73, sandboxing×56 |
| 10 | `activepieces--activepieces` | 336 | 24345 | 2 | hooks×257, sub_agents×213, mcp×199, crawling×98, eval_harness×94 |
| 11 | `QwenLM--qwen-code` | 336 | 7566 | 0 | hooks×411, mcp×238, memory×132, eval_harness×113, sub_agents×108 |
| 12 | `lobehub--lobe-chat` | 336 | 13729 | 0 | hooks×606, memory×365, eval_harness×325, routing×276, mcp×150 |
| 13 | `mastra-ai--mastra` | 336 | 13123 | 0 | hooks×582, memory×526, observability×411, eval_harness×380, mcp×320 |
| 14 | `openclaw--openclaw` | 336 | 30060 | 0 | hooks×3258, memory×739, sub_agents×545, sandboxing×318, scheduling×262 |
| 15 | `Arize-ai--phoenix` | 335 | 6884 | 1389 | eval_harness×937, observability×800, sandboxing×95, mcp×75, routing×72 |
| 16 | `OpenInterpreter--open-interpreter` | 330 | 5910 | 137 | hooks×356, mcp×244, sandboxing×220, eval_harness×57, observability×56 |
| 17 | `openai--codex` | 330 | 5854 | 137 | hooks×361, mcp×275, sandboxing×194, observability×60, durable_workflow×53 |
| 18 | `coder--mux` | 328 | 2651 | 15 | hooks×108, mcp×32, memory×29, eval_harness×24, sub_agents×21 |
| 19 | `can1357--oh-my-pi` | 326 | 6017 | 167 | eval_harness×165, crawling×102, hooks×98, mcp×92, memory×59 |
| 20 | `stablyai--orca` | 326 | 11563 | 2 | hooks×611, sub_agents×221, observability×126, scheduling×111, durable_workflow×89 |
| 21 | `CopilotKit--CopilotKit` | 321 | 18961 | 658 | eval_harness×963, sub_agents×953, hooks×617, mcp×520, memory×87 |
| 22 | `elastic--elasticsearch` | 318 | 46375 | 0 | hooks×24299, eval_harness×1611, observability×958, memory×543, routing×412 |
| 23 | `microsoft--agent-governance-toolkit` | 317 | 4758 | 1879 | mcp×210, eval_harness×179, hooks×84, observability×79, sandboxing×74 |
| 24 | `windmill-labs--windmill` | 315 | 9031 | 46 | eval_harness×264, observability×186, mcp×47, durable_workflow×29, hooks×29 |
| 25 | `langfuse--langfuse` | 312 | 4600 | 5 | eval_harness×258, hooks×225, mcp×127, observability×82, routing×76 |
| 26 | `ruvnet--ruflo` | 312 | 5491 | 0 | hooks×1371, mcp×553, eval_harness×407, memory×320, sub_agents×309 |
| 27 | `Kilo-Org--kilocode` | 311 | 8666 | 16 | hooks×310, memory×113, mcp×78, sandboxing×74, sub_agents×41 |
| 28 | `inngest--inngest` | 310 | 12888 | 0 | observability×813, hooks×194, compression×192, retry_policy×67, mcp×40 |
| 29 | `bytedance--deer-flow` | 308 | 2094 | 1079 | eval_harness×501, hooks×121, memory×96, sandboxing×70, routing×55 |
| 30 | `davila7--claude-code-templates` | 308 | 9178 | 895 | eval_harness×1078, mcp×325, sub_agents×282, hooks×250, sandboxing×62 |
| 31 | `code-yeongyu--oh-my-openagent` | 308 | 6449 | 55 | hooks×1675, sub_agents×477, mcp×284, eval_harness×142, observability×66 |
| 32 | `grafana--grafana` | 308 | 22101 | 0 | hooks×3079, crawling×383, observability×364, sub_agents×301, eval_harness×110 |
| 33 | `Significant-Gravitas--AutoGPT` | 307 | 4648 | 1552 | eval_harness×132, hooks×96, observability×67, crawling×61, sub_agents×47 |
| 34 | `asheshgoplani--agent-deck` | 307 | 1894 | 25 | hooks×70, mcp×51, eval_harness×38, scheduling×12, sandboxing×9 |
| 35 | `vllm-project--semantic-router` | 306 | 5091 | 430 | routing×1722, hooks×469, memory×343, eval_harness×146, mcp×76 |
| 36 | `triggerdotdev--trigger.dev` | 306 | 5003 | 0 | observability×154, hooks×127, durable_workflow×100, mcp×39, scheduling×38 |
| 37 | `XiaomiMiMo--MiMo-Code` | 303 | 5256 | 68 | hooks×126, eval_harness×45, mcp×44, durable_workflow×38, sub_agents×33 |
| 38 | `UKGovernmentBEIS--inspect_ai` | 302 | 2055 | 1541 | sandboxing×178, eval_harness×177, durable_workflow×64, mcp×30, hooks×19 |
| 39 | `swarmclawai--swarmclaw` | 302 | 1808 | 4 | sub_agents×71, memory×57, hooks×36, mcp×26, sandboxing×25 |
| 40 | `cline--cline` | 301 | 3513 | 15 | hooks×241, mcp×101, observability×77, sub_agents×54, eval_harness×51 |
