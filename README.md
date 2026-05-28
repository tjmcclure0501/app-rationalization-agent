# App Rationalization Agent

## Architectural Philosophy

The rubric in `config/scoring_rubric.json` is the determinism layer of this
system. It pins down the parts of a rationalization decision that need to be
defensible to a stakeholder — category weights, recommendation thresholds,
confidence floors, override rules — and freezes them as configuration an
architect can review and tune without touching code. Everything else — which
tools Claude invokes, in what order, how it interprets ambiguous signals, the
prose of its rationale — is intentionally left to Claude.

This places the system at a deliberate midpoint on a three-point spectrum. At
one end, the deterministic pipeline (`agents/orchestrator.py`) runs every
analyzer in fixed order and applies the rubric mechanically: maximally
auditable, but unable to recognize that an EOL Java pin matters more than
dependency staleness on a specific repo. At the other end, a pure agentic
flow with no rubric reasons end-to-end by judgment alone: maximally flexible,
but the decision boundary moves run to run and the rationale is harder to
defend to a governance committee.

The guided agentic flow (`agents/agentic_orchestrator.py` plus the rubric)
sits between them. Claude decides how to investigate — which analyzer to call
first based on what it sees, when it has enough signal to stop — but the
final recommendation must clear the rubric's thresholds, the confidence floor
must be met or the report is flagged for human review, and override rules can
demote a recommendation regardless of what Claude concludes. The rubric
provides governance and consistency; Claude provides judgment on the edge
cases that rules alone cannot resolve.

The rubric constrains *what* Claude decides while leaving Claude free to
determine *how* it gets there.

A multi-agent pipeline that evaluates public GitHub repositories and produces an
application-rationalization recommendation against the 5-R framework: **Retain,
Rehost, Refactor, Rewrite, Retire**. Output is a structured JSON / markdown
report with a recommendation, confidence score, signal-level rationale, and
a human-review flag for ambiguous cases.

Built as a working reference for the agentic patterns covered in the Claude
Certified Architect — Foundations curriculum: tool use, multi-step
orchestration, structured output, human-in-the-loop review, and shared state
across cooperating agents.

## What it solves

Portfolio rationalization at scale needs a defensible, repeatable signal layer
under the human judgment call. This agent collapses public repo metadata,
dependency manifests, and commit history into a single weighted score plus the
evidence behind it — so an architect can triage hundreds of candidate apps
without manually opening each one.

### The 5-R framework

| Recommendation | Portfolio meaning |
|---|---|
| **RETAIN** | Healthy, modern, actively maintained. Leave alone. |
| **REHOST** | Stable but on aging infra. Lift-and-shift to cloud / containers; no code change. |
| **REFACTOR** | Working but structurally weak (no tests, monolith, stale deps). Business logic is sound — invest selectively. |
| **REWRITE** | Too degraded to salvage incrementally. EOL runtime, no tests, large surface area. |
| **RETIRE** | Low activity, superseded, or no longer needed. Decommission. |

## Architecture

The project ships **two orchestrators** on top of the same analyzer modules:

| Orchestrator | Drives the pipeline | When to use |
|---|---|---|
| [agents/agentic_orchestrator.py](agents/agentic_orchestrator.py) | **Claude**, via tool use against the Anthropic API | Primary entry point. Each analyzer is exposed as a tool; Claude reasons between calls. |
| [agents/orchestrator.py](agents/orchestrator.py) | Python, deterministic order | Fallback / reference. Runs every analyzer in fixed sequence; no LLM. |

Both produce the same report schema and share `print_report` / `save_report`.
The agentic flow is documented in [#The agentic loop](#the-agentic-loop) below.

The analyzer modules are standalone Python: each exposes a single
`analyze(repo_data)` (or `fetch_repo_data` / `score`) function. State is passed
forward through a shared `repo_data` dict; the scorer is the only consumer that
sees all stage outputs at once.

```
                    GitHub REST API
                          │
                          ▼
                 ┌──────────────────┐
                 │  github_fetcher  │  metadata, file_tree,
                 │                  │  manifests, README
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │  code_analyzer   │  language, frameworks, LOC,
                 │                  │  tests, CI, architecture
                 └────────┬─────────┘  → quality + complexity signals
                          │
                          ▼
                 ┌──────────────────┐
                 │   dependency_    │  manifest parse, runtime EOL,
                 │   scanner        │  outdated-dep heuristics
                 └────────┬─────────┘  → health signals
                          │
                          ▼
                 ┌──────────────────┐
                 │   activity_      │  last commit, commit frequency,
                 │   analyzer       │  contributor count
                 └────────┬─────────┘  → activity signals
                          │
                          ▼
                 ┌──────────────────┐
                 │      scorer      │  weighted score → 5-R,
                 │                  │  overrides, confidence
                 └────────┬─────────┘
                          │
                          ▼
              recommendation + confidence
                + rationale + flag
```

| Module | Responsibility | Owns |
|---|---|---|
| [agents/orchestrator.py](agents/orchestrator.py) | CLI entry, stage dispatch, report serialization | — |
| [agents/github_fetcher.py](agents/github_fetcher.py) | All GitHub HTTP traffic + retry/rate-limit | shared `_build_client`, `_request` |
| [agents/code_analyzer.py](agents/code_analyzer.py) | Static analysis from the fetch payload | `quality_signals`, `complexity_signals` |
| [agents/dependency_scanner.py](agents/dependency_scanner.py) | Per-manifest dep parsing + runtime EOL classification | `health_signals` |
| [agents/activity_analyzer.py](agents/activity_analyzer.py) | Commit + contributor stats | `activity_signals` |
| [agents/scorer.py](agents/scorer.py) | Final 5-R decision + overrides + confidence | aggregate |

## Prerequisites

- **Python 3.11+** (pinned via `.python-version`)
- **[uv](https://docs.astral.sh/uv/)** — the package manager and virtual
  environment tool. Replaces both `pip` and `virtualenv` in this workflow.
  Install with:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Node.js** (for Claude Code, if you're using this project as a Claude
  Code workspace; not required to run the pipeline itself)
- **GitHub personal access token** with `public_repo` scope — without one
  you'll hit the 60 req/hr unauthenticated limit.
- **Anthropic API key** — required by the agentic orchestrator. Not needed
  if you only run the deterministic `agents/orchestrator.py`.

## Installation

```bash
# From the project root:
uv venv                       # create .venv using the pinned Python
uv sync                       # install dependencies from pyproject.toml

# Activate the venv (Mac/Linux)
source .venv/bin/activate
# ...or on Windows
.venv\Scripts\activate

# Copy the env template and fill in your tokens
cp .env.template .env
# then edit .env:
#   GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
#   ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxx
```

`uv sync` is idempotent — re-run it any time `pyproject.toml` changes.
You can also skip activation entirely by prefixing commands with `uv run`,
e.g. `uv run python agents/agentic_orchestrator.py --repo facebook/react`.

Dependencies are minimal — `httpx`, `python-dotenv`, `rich`, `anthropic`
(see [pyproject.toml](pyproject.toml)).

## Configuration

All scoring behaviour lives in [config/scoring_rubric.json](config/scoring_rubric.json).
You can re-tune the model without touching code.

| Section | Controls |
|---|---|
| `category_weights` | Relative weight of activity / health / quality / complexity in the final score. Must sum to 1.0. |
| `thresholds` | `min_score` per recommendation tier, plus `min_confidence` for flagging. |
| `confidence.flag_for_review_below` | Confidence threshold under which a result is flagged for human review (default 0.55). |
| `confidence.low_signal_penalty` | Subtracted from confidence when fewer than 60% of expected signals are present (default 0.15). |
| `<category>_signals` | Per-signal `weight` and `score_map` buckets. Buckets support `"0-30"` (range), `"20+"` (open upper bound), and exact values. |
| `recommendation_logic.overrides` | `force_retire_if`, `force_rewrite_if`, `force_review_if` rules applied *after* the threshold mapping. |

### Tuning examples

```jsonc
// Make activity recency dominant for a brownfield audit
"category_weights": {
  "activity":   0.45,
  "health":     0.30,
  "quality":    0.15,
  "complexity": 0.10
}

// Demand higher confidence before recommending RETAIN
"thresholds": {
  "retain": { "min_score": 0.75, "min_confidence": 0.85 }
}
```

The rubric is loaded at scorer import; restart the orchestrator after editing.

## Usage

### Agentic mode (primary — Claude drives the pipeline)

```bash
# Single repo, pretty terminal output
python agents/agentic_orchestrator.py --repo facebook/react

# Observe each tool call and Claude's reasoning between them
python agents/agentic_orchestrator.py --repo facebook/react --trace

# Single repo, JSON to stdout + persisted report under outputs/reports/
python agents/agentic_orchestrator.py --repo expressjs/express --output json --save

# Batch mode — one repo per line in the file, # for comments
python agents/agentic_orchestrator.py --batch config/sample_repos.txt --save
```

Requires `ANTHROPIC_API_KEY` in `.env`.

### Deterministic mode (fallback — no LLM)

```bash
python agents/orchestrator.py --repo facebook/react
python agents/orchestrator.py --batch config/sample_repos.txt
```

Same flags, same output schema. Use this when you don't have an Anthropic
key, want bit-exact reproducibility, or are debugging an analyzer in isolation.

`--save` writes `outputs/reports/<owner>__<name>__<UTC-timestamp>.json`.
`--verbose` prints full tracebacks for any failing stage.

## The agentic loop

The agentic orchestrator does not run analyzers in a fixed order. Instead, it
exposes each module to Claude as an [Anthropic tool](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
and runs a manual tool-use loop. Claude reads the 5-R framework and the
scoring rubric in its system prompt, then decides — turn by turn — which
analyzer to call next.

### Tools exposed to Claude

| Tool | Wraps | Inputs |
|---|---|---|
| `fetch_repo_data` | [agents/github_fetcher.py](agents/github_fetcher.py) | `repo: "owner/name"` |
| `analyze_code` | [agents/code_analyzer.py](agents/code_analyzer.py) | none — uses accumulated state |
| `analyze_dependencies` | [agents/dependency_scanner.py](agents/dependency_scanner.py) | none |
| `analyze_activity` | [agents/activity_analyzer.py](agents/activity_analyzer.py) | none |
| `score_recommendation` | [agents/scorer.py](agents/scorer.py) | `rationale_notes` (optional) |

The orchestrator holds the cumulative `pipeline_stages` dict, so analyzer
tools don't have to round-trip large payloads through Claude's context — the
tool results Claude sees are *summaries* (signal values, scored fields,
errors), while the full data stays server-side.

### What Claude reasons over

The system prompt is byte-stable across every repo and is marked
`cache_control: ephemeral` so batch runs hit the Anthropic prompt cache. It
contains:

- The 5-R framework definitions
- Tool descriptions and recommended ordering (`fetch_repo_data` first, then
  pick analyzers based on what you see, then `score_recommendation`)
- The complete `config/scoring_rubric.json` payload as *reference guidance*
  ("not hard rules — you are the architect of record")
- The required final-output schema (`recommendation`, `confidence`,
  `rationale`, `signals_summary`, `flagged_for_review`)
- The `max_iterations` budget so Claude can pace itself

Between tool calls Claude states what it observed and what it intends to
investigate next — so an EOL-Java pom.xml drives it toward
`analyze_dependencies`, while a repo with no commits in three years drives it
toward `analyze_activity` first.

### Final report construction

After Claude stops calling tools (or hits `max_iterations`, default 10), the
orchestrator:

1. Runs `scorer.score(pipeline_stages)` over whatever signals were gathered,
   guaranteeing a valid `signals_summary` even if the loop bailed early.
2. Parses the fenced ```json``` block in Claude's final message.
3. Merges the two: Claude's `recommendation` / `confidence` / `rationale` win
   when present; the scorer's deterministic `signals_summary` is always used.

The full reasoning chain (every tool call, input, iteration number, plus
Claude's final text) is captured under `report.agentic.*` for audit.

### Observing the loop with `--trace`

`--trace` prints Claude's reasoning text and each tool call before it
executes, so you can watch the agent's decision-making in real time:

```text
─────────────────────── iteration 1 ───────────────────────
reasoning:
I'll start by fetching the repository data to see what language and
manifest types we're dealing with.

→ tool: fetch_repo_data({"repo": "expressjs/express"})
stop_reason=tool_use

─────────────────────── iteration 2 ───────────────────────
reasoning:
Active JS repo, MIT-licensed, 25 contributors visible in metadata,
package.json + package-lock.json present. Strong RETAIN candidate
on the face of it, but I should confirm activity and dep health
before locking that in.

→ tool: analyze_activity({})
→ tool: analyze_dependencies({})
stop_reason=tool_use
...
```

Use it for certification demos, debugging unexpected calls, or auditing
why a particular recommendation was reached.

### LLM cost logging

Every Anthropic API call's token usage and USD cost is logged. There's nothing
to enable — costs always land in three places:

- **Per-call in `--trace`** — token counts (`in / out / cache_w / cache_r`)
  and the per-call USD breakdown print after each iteration.
- **Per-repo summary** — a one-line LLM cost summary prints after the
  recommendation panel for every repo.
- **Batch total** — when running with `--batch` against multiple repos, an
  aggregate line prints at the end.
- **In the saved report JSON** — under `report.agentic.cost.total` (aggregate)
  and `report.agentic.cost.per_call` (one entry per API call). Useful for
  building dashboards or reconciling against billing exports.

Cost is computed from the response's `usage` object using rates in the
`_PRICING_PER_M_TOKENS` table at the top of
[agents/agentic_orchestrator.py](agents/agentic_orchestrator.py). Prompt-cache
writes are billed at 1.25× the input rate; cache reads at 0.1× — both
broken out separately so you can verify the system-prompt cache is actually
hitting. If the configured model isn't in the pricing table, the report
records `cost_usd: null` with `cost_unknown_reason` rather than silently
showing $0.

### Configuration

| Constant | Default | Defined in |
|---|---|---|
| Model | `claude-sonnet-4-20250514` | `MODEL` in [agents/agentic_orchestrator.py](agents/agentic_orchestrator.py) |
| Max tokens per turn | `4096` | `MAX_TOKENS` |
| Max tool-calling iterations | `10` | `MAX_ITERATIONS` |
| System prompt template | the 5-R framework + rubric | `_SYSTEM_PROMPT_TEMPLATE` |
| Pricing table | per-model $ / 1M tokens | `_PRICING_PER_M_TOKENS` |

Hitting `MAX_ITERATIONS` emits a warning and proceeds with the scorer's
fallback output; the report flags `agentic.max_iterations_hit: true`.

## Output

Each report is a single JSON object with the following top-level shape:

```json
{
  "repo": "facebook/react",
  "analyzed_at": "2026-05-28T14:32:11Z",
  "recommendation": "RETAIN",
  "confidence": 0.91,
  "rationale": "Recommendation: RETAIN (weighted score 0.86, confidence 91%). Strongest dimension: activity (0.95); weakest: complexity (0.70). Notable signals: actively maintained (last commit 2d ago); 1500+ contributors; runtime current; fresh dependencies (93%); test coverage present.",
  "flagged_for_review": false,
  "pipeline_stages": { "fetch": {...}, "code": {...}, "dependencies": {...}, "activity": {...} }
}
```

| Field | What to do with it |
|---|---|
| `recommendation` | Drives the portfolio decision. One of `RETAIN`, `REHOST`, `REFACTOR`, `REWRITE`, `RETIRE`, `UNKNOWN`. |
| `confidence` | 0.0–1.0. Below `flag_for_review_below` (default 0.55) the report is auto-flagged. Treat <0.7 as needing architect sign-off. |
| `rationale` | Single paragraph naming the strongest/weakest dimension and salient signals. Suitable to drop directly into a review write-up. |
| `flagged_for_review` | `true` if confidence is low, a `force_review_if` override fired, or category scores disagree sharply (spread > 0.5). |
| `pipeline_stages.*.signals` | Raw signal values. Use these when the rationale isn't enough and you need to defend a decision in a steering meeting. |

The scorer's `signals_summary` block (under `pipeline_stages` once persisted)
exposes `final_score`, the per-category breakdown, and the override list so you
can reconstruct exactly why the tool landed where it did.

## Sample repos

The repos in [config/sample_repos.txt](config/sample_repos.txt) are chosen to
exercise all five outcomes:

| Repo | Expected | Why |
|---|---|---|
| `facebook/react` | RETAIN | Daily commits, thousands of contributors, current runtime, deep test coverage. The "obvious keep" anchor. |
| `expressjs/express` | RETAIN / REHOST | Mature, low velocity but actively patched. Tests REHOST sensitivity to lower commit frequency on healthy code. |
| `bluesky-social/social-app` | REFACTOR | Active product, modern stack, but younger codebase with evolving structure — exercises the middle of the score band. |
| `php-fig/fig-standards` | RETIRE | Spec repo, low commit cadence, narrow contributor base. Tests retirement signals on a *legitimate* low-activity repo (not a dead one). |
| `kelseyhightower/nocode` | RETIRE (edge) | Near-zero code. Exercises the `lines_of_code < 50` `force_retire_if` override and the low-signal confidence penalty. |

Uncomment lines in `config/sample_repos.txt` to include them in batch runs.

## Limitations

Out of scope for the current pipeline — extensions are listed below.

- **No CVE / GHSA lookups.** Dependency freshness is heuristic only (wildcard
  patterns + >1 major behind known-current frameworks). It will not flag a
  specifically vulnerable version range.
- **No cloud cost or runtime telemetry.** REHOST decisions don't yet factor in
  hosting spend, traffic, or operational data.
- **No business context.** Criticality, customer impact, revenue tie, and
  regulatory exposure all need to come from a human reviewer or an external
  CMDB integration.
- **Single default-branch view.** Forks, release branches, and archived
  mirrors are not analyzed.
- **LOC is a byte-based estimate.** Source bytes / 32. Accurate to a rubric
  bucket, not to a line count.
- **GitHub conflates issues and PRs in `open_issues_count`.** The
  `open_issues_ratio` signal inherits that quirk.
- **Runtime EOL tables are point-in-time** (snapshot 2026-05-28). Update the
  classifiers in [agents/dependency_scanner.py](agents/dependency_scanner.py)
  as release calendars move.

## Extending the agent

To add a new signal source (e.g. a Snyk integration, a CMDB lookup, or a
cloud-cost analyzer):

1. **Write the agent module.** Drop `agents/<name>.py` exposing a single
   `analyze(repo_data: dict) -> dict` that returns

   ```python
   {
       "agent": "<name>",
       "signals": { "<signal_name>": <value>, ... },
       "scored":  { ... },          # optional — scorer re-scores from raw signals
       "errors":  [ ... ],
       "low_signal": False,         # set True if data was incomplete
   }
   ```

   Use the existing analyzers as templates. If you need GitHub data, reuse
   `_build_client` / `_request` from `github_fetcher` so all HTTP traffic stays
   under one rate-limiter.

2. **Declare the signals in the rubric.** Add the new signal names under the
   right category in `config/scoring_rubric.json` with a `weight` and a
   `score_map`. Keep weights summing to 1.0 within the category.

3. **Wire it into the orchestrator.** In [agents/orchestrator.py](agents/orchestrator.py):
   - Add an import block in `_import_agents()` mirroring the existing entries.
   - Append a tuple to the `stages` list in `run_pipeline()`:

     ```python
     stages = [
         # ...
         ("<stage_key>", "Running <name>", "<name>"),
     ]
     ```

4. **Teach the scorer where the signals live.** In
   [agents/scorer.py](agents/scorer.py), extend `_CATEGORY_TO_STAGE` if the new
   agent owns a new category, or do nothing if it just adds signals to an
   existing one — the scorer reads signal names dynamically from the rubric.

No changes are needed in any other module: `scorer.py` re-scores every signal
from the rubric on each run, so adding a signal cascades into the final score
automatically.
