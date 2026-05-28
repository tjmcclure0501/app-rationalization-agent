# App Rationalization Agent

## Project Purpose
This is an agentic framework that evaluates public GitHub repositories and produces
application rationalization recommendations using the "5 Rs" framework:
- **Retain** – healthy, modern, worth keeping as-is
- **Rehost** – stable but could move to modern infra (e.g. containerize, cloud lift-and-shift)
- **Refactor** – working but needs structural improvement; business logic is sound
- **Rewrite** – too degraded or outdated to salvage incrementally
- **Retire** – low activity, superseded, or no longer needed

## Architecture
Two orchestrators sit on top of the same analyzer modules. The agentic one is
the **primary entry point**; the deterministic one is kept as a reference and
a fallback when an Anthropic API key is unavailable.

```
agentic_orchestrator.py    ← PRIMARY entry point (Claude drives the pipeline via tool use)
  │
  │  Each analyzer below is exposed to Claude as an Anthropic tool.
  │  Claude reasons between tool calls and decides what to invoke.
  │
  ├── github_fetcher.py     → GitHub REST API: metadata, file tree, manifests, README
  ├── code_analyzer.py      → language, framework, LOC, test presence, architecture signals
  ├── dependency_scanner.py → dependency health, EOL runtimes, outdated packages
  ├── activity_analyzer.py  → commit frequency, contributor count, last active date
  └── scorer.py             → weighs signals against rubric → 5-R recommendation + confidence

orchestrator.py            ← reference deterministic pipeline (no LLM; runs every stage in order)
```

The agentic orchestrator implements a manual tool-use loop against the
Anthropic Messages API. The same `pipeline_stages` dict is built up, so the
final report shape is identical to the deterministic pipeline — print_report
and save_report from `orchestrator.py` are reused unchanged.

## Key Files
- `config/scoring_rubric.json` – weights and thresholds for each signal
- `outputs/reports/` – JSON and markdown reports per repo
- `.env` – GitHub PAT and any config (never commit this)
- `pyproject.toml` – project metadata + dependencies (managed by uv)
- `.python-version` – pinned to 3.11 (read by uv / pyenv)

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management
and virtual environments — it replaces both `pip` and `virtualenv`. Install
uv once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then from the project root:

```bash
uv venv                       # create the virtual environment in .venv/
uv sync                       # install dependencies from pyproject.toml
source .venv/bin/activate     # activate (Mac/Linux)
.venv\Scripts\activate        # activate (Windows)
```

Re-run `uv sync` after pulling changes that touch `pyproject.toml`.
You can also skip activation by prefixing commands with `uv run`.

## Environment Variables (.env)
```
GITHUB_TOKEN=your_pat_here
ANTHROPIC_API_KEY=your_key_here   # required by agentic_orchestrator.py
```

## Running the Agent

**Primary (agentic — Claude drives the pipeline):**
```bash
python agents/agentic_orchestrator.py --repo facebook/react
python agents/agentic_orchestrator.py --repo facebook/react --trace      # observe tool calls
python agents/agentic_orchestrator.py --batch config/sample_repos.txt --save
```

**Fallback (deterministic pipeline, no LLM):**
```bash
python agents/orchestrator.py --repo facebook/react
python agents/orchestrator.py --batch config/sample_repos.txt
```

## Sample Repos for Testing
These are chosen to produce varied recommendations across the 5 Rs:
- `facebook/react`             → expected: Retain
- `expressjs/express`          → expected: Retain / Rehost
- `bluesky-social/social-app`  → expected: Refactor
- `php-fig/fig-standards`      → expected: Retire
- `kelseyhightower/nocode`     → expected: Retire (edge case / low-signal)

## Scoring Rubric Overview
Signals are grouped into four categories, each weighted in `scoring_rubric.json`:
1. **Activity** (30%) – commit recency, frequency, contributor count
2. **Health** (30%) – dependency freshness, EOL runtimes, open issues ratio
3. **Quality** (20%) – test presence, documentation, architecture signals
4. **Complexity** (20%) – LOC, cyclomatic complexity proxies, monolith vs modular

## Design Principles
- Each agent is a standalone Python module with a single `analyze(repo)` function
- Agents return structured dicts; orchestrator aggregates and passes to scorer
- scorer.py produces a final JSON report with: recommendation, confidence (0-1), rationale, signals
- All GitHub API calls go through `github_fetcher.py` to centralize rate-limit handling
- Use exponential backoff for GitHub API rate limits (60 req/hr unauthenticated, 5000 with PAT)

## Agentic Patterns Demonstrated (for Claude Certified Architect - Foundations)
- **Tool use via the Anthropic API** – each analyzer exposed as a Claude tool; the agentic orchestrator runs a manual tool-use loop (see `agents/agentic_orchestrator.py`)
- **Model-driven planning** – Claude reads the rubric in its system prompt and decides which analyzers to invoke based on what it sees
- **Multi-step orchestration** – shared `pipeline_stages` state accumulates across tool calls; the scorer aggregates everything at the end
- **Structured output** – final report JSON with `recommendation`, `confidence`, `rationale`, `signals_summary`, `flagged_for_review`
- **Human-in-the-loop** – low-confidence recommendations (or override-triggered ones) are flagged for review
- **Observability** – `--trace` flag prints Claude's reasoning and each tool invocation so the loop is auditable
- **Prompt caching** – the system prompt (with the rubric inlined) is marked `cache_control: ephemeral` so batch runs hit the prompt cache

## Conventions
- Python 3.9+
- Use `httpx` for async HTTP (preferred over `requests` for future async support)
- Use `python-dotenv` for env vars
- Use `rich` for terminal output formatting
- Type hints on all public functions
- Each agent logs its signals to stdout in a structured format
