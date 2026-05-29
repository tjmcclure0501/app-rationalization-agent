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
`agents/agentic_orchestrator.py` is the sole entry point. Claude drives
the pipeline via the Anthropic Messages API: each analyzer below is
exposed as a Claude tool, and Claude reasons between tool calls to
decide which to invoke and how to interpret the results.

```
agentic_orchestrator.py   ← Claude-driven manual tool-use loop
  │
  ├── github_fetcher.py     → GitHub REST API: metadata, file tree, manifests, README
  ├── repo_classifier.py    → pre-scoring stage: APPLICATION / LIBRARY / DOCUMENTATION /
  │                            CONFIGURATION / DATA / UNKNOWN (runs automatically
  │                            inside fetch_repo_data; not exposed as a separate tool)
  ├── code_analyzer.py      → language, framework, LOC, test presence, architecture signals
  ├── dependency_scanner.py → dependency health, EOL runtimes, outdated packages
  ├── activity_analyzer.py  → commit frequency, contributor count, last active date
  └── scorer.py             → weighs signals against rubric → 5-R recommendation + confidence
                              (suppresses inapplicable override rules per repo_type)
```

### Repo-type taxonomy
The 5-R framework was designed for runtime software. For repositories
that aren't (specs, docs, datasets, dotfiles), the rubric's LOC / tests /
runtime / dependency signals are structurally inapplicable.
`repo_classifier` runs after `github_fetcher` and tags every repo with
one of six types so downstream stages can adapt:

| Type | Meaning | Rubric behavior |
|---|---|---|
| APPLICATION   | Deployable software | Full 5-R rubric |
| LIBRARY       | Reusable package / framework | Full 5-R rubric |
| DOCUMENTATION | Specs, awesome-lists, RFC collections | LOC/tests/runtime override rules **suppressed** (see `repo_type_overrides` in `scoring_rubric.json`); Claude is expected to base the recommendation on fitness-for-purpose signals (community adoption, maintenance cadence, institutional relevance) |
| CONFIGURATION | IaC, dotfiles, config-only repos | LOC/tests override rules suppressed |
| DATA          | Datasets, assets, content | LOC/tests/runtime override rules suppressed |
| UNKNOWN       | Signals insufficient to classify | Full rubric; report flagged for review |

The orchestrator implements a manual tool-use loop against the
Anthropic Messages API. State accumulates in a shared `pipeline_stages`
dict that each tool reads from / writes into; the scorer reads the
whole dict at the end. `print_report` and `save_report` live alongside
the orchestration code in the same module.

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

```bash
python agents/agentic_orchestrator.py --repo facebook/react
python agents/agentic_orchestrator.py --repo facebook/react --trace      # observe tool calls
python agents/agentic_orchestrator.py --batch config/sample_repos.txt --save
```

Prefix with `uv run` instead of activating the venv if you prefer.

## Sample Repos for Testing
See `config/sample_repos.txt` for the current batch of test repositories
covering different repo types and expected recommendation patterns. Use
with `--batch`:

```bash
python agents/agentic_orchestrator.py --batch config/sample_repos.txt --save
```

## Scoring Rubric Overview
Signals are grouped into four categories, each weighted in `scoring_rubric.json`:
1. **Activity** (30%) – commit recency, frequency, contributor count
2. **Health** (30%) – dependency freshness, EOL runtimes, open issues ratio
3. **Quality** (20%) – test presence, documentation, architecture signals
4. **Complexity** (20%) – LOC, cyclomatic complexity proxies, monolith vs modular

## Design Principles
- Each agent is a standalone Python module with a single `analyze(repo)` function
- Agents return structured dicts; the orchestrator accumulates them in `pipeline_stages` and passes the whole dict to scorer
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
