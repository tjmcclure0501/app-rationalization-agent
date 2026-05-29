"""
agentic_orchestrator.py — Claude-driven entry point
===================================================
The deterministic pipeline (agents/orchestrator.py) runs every analyzer in
a fixed order. This module replaces that scaffolding with an *agentic* loop:
Claude reasons about the repository, decides which analyzers to invoke and
in what order, and produces the final 5-R recommendation.

Each existing analyzer module is exposed to Claude as an Anthropic *tool*.
Claude calls them through the standard tool-use protocol; this orchestrator
holds the accumulated state and dispatches each tool call to the matching
Python function. The same `pipeline_stages` dict that the deterministic
orchestrator builds is constructed incrementally here, so all downstream
consumers (print_report, save_report) work unchanged.

Pipeline:
    1. Build a cached system prompt (5-R framework + scoring rubric).
    2. Open a manual agentic loop:
         - Send user request → repo name.
         - On `tool_use` stop, dispatch each tool call to the matching
           analyzer, return summarized results.
         - Loop until Claude stops calling tools or max_iterations is hit.
    3. Always run the deterministic scorer over the gathered signals so the
       final report has a guaranteed-valid `signals_summary`. Overlay
       Claude's parsed JSON for recommendation / confidence / rationale.

Usage mirrors agents/orchestrator.py:
    python agents/agentic_orchestrator.py --repo facebook/react
    python agents/agentic_orchestrator.py --repo owner/name --output json --save
    python agents/agentic_orchestrator.py --batch config/sample_repos.txt
    python agents/agentic_orchestrator.py --repo owner/name --trace
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import anthropic

# Existing analyzer modules — reused verbatim as tool implementations.
from github_fetcher import fetch_repo_data as _fetch_repo_data
from repo_classifier import classify as _classify_repo
from code_analyzer import analyze as _analyze_code
from dependency_scanner import analyze as _analyze_deps
from activity_analyzer import analyze as _analyze_activity
from scorer import score as _score

# Rich is optional — the formatting helpers below degrade gracefully when
# it's missing.
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich import print as rprint
    RICH_AVAILABLE = True
    _console = Console()
except ImportError:
    RICH_AVAILABLE = False
    _console = None


# ---------------------------------------------------------------------------
# Environment + report helpers (previously in orchestrator.py).
# ---------------------------------------------------------------------------

def load_env() -> None:
    """Load environment variables from .env file."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        # python-dotenv is optional; environment variables can be set
        # directly by the caller instead.
        pass


# Color used for each recommendation when rendering with Rich. REVIEW shares
# bold-yellow with REFACTOR but is disambiguated by the prominent
# _REVIEW_INCOMPLETE_NOTICE banner inside the panel.
RECOMMENDATION_COLORS = {
    "RETAIN":   "bold green",
    "REHOST":   "bold blue",
    "REFACTOR": "bold yellow",
    "REWRITE":  "bold magenta",
    "RETIRE":   "bold red",
    "REVIEW":   "bold yellow",   # incomplete-signal demotion — see scorer.py
    "UNKNOWN":  "bold white",
    "ERROR":    "bold red",
}

# Banner shown above the rationale when the scorer produced REVIEW because
# critical override signals were missing. The recommendation panel is
# already yellow but a REVIEW report can otherwise look superficially
# similar to a normal REFACTOR — this notice makes the difference loud.
_REVIEW_INCOMPLETE_NOTICE = (
    "⚠  REPORT INCOMPLETE — critical signals were missing during scoring. "
    "Human investigation is required before any portfolio decision is made."
)


def print_report(report: dict) -> None:
    """Pretty-print the final report to terminal."""
    rec = report.get("recommendation", "UNKNOWN")
    confidence = report.get("confidence", 0.0)
    confidence_pct = f"{confidence * 100:.0f}%" if confidence is not None else "N/A"
    flagged = report.get("flagged_for_review")
    reasons = report.get("flagged_for_review_reasons") or []
    is_review = rec == "REVIEW"

    if RICH_AVAILABLE:
        color = RECOMMENDATION_COLORS.get(rec, "white")
        panel_content = (
            f"[{color}]Recommendation: {rec}[/{color}]\n"
            f"Confidence: {confidence_pct}\n\n"
        )
        # REVIEW means we couldn't score with confidence — surface that
        # *before* the rationale so it's the first thing the architect reads.
        if is_review:
            panel_content += f"[bold yellow]{_REVIEW_INCOMPLETE_NOTICE}[/bold yellow]\n\n"
        panel_content += f"[italic]{report.get('rationale', 'No rationale provided')}[/italic]"

        if flagged:
            panel_content += "\n\n[bold yellow]⚠ Flagged for human review[/bold yellow]"
            if reasons:
                for reason in reasons:
                    panel_content += f"\n[yellow]    • {reason}[/yellow]"
            else:
                # No reasons attached — the flag was set without justification.
                # Surfacing the gap explicitly so it can be debugged.
                panel_content += "\n[yellow]    • (no reason recorded — check scorer output)[/yellow]"

        _console.print(Panel(
            panel_content,
            title=f"[bold]{report['repo']}[/bold]",
            subtitle=f"Analyzed {report['analyzed_at']}",
            border_style=color.replace("bold ", "")
        ))
    else:
        print(f"\n{'='*60}")
        print(f"  Repo:           {report['repo']}")
        print(f"  Recommendation: {rec}")
        print(f"  Confidence:     {confidence_pct}")
        if is_review:
            print(f"  {_REVIEW_INCOMPLETE_NOTICE}")
        print(f"  Rationale:      {report.get('rationale')}")
        if flagged:
            print("  ⚠ Flagged for human review")
            if reasons:
                for reason in reasons:
                    print(f"      • {reason}")
            else:
                print("      • (no reason recorded — check scorer output)")
        print(f"{'='*60}\n")


def save_report(report: dict, output_dir: str = "outputs/reports") -> Path:
    """Save the report as JSON under outputs/reports/."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    repo_safe = report["repo"].replace("/", "__")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = path / f"{repo_safe}__{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(report, f, indent=2)
    return filename


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

# Model: Claude Sonnet 4.6. Using the canonical alias rather than a
# date-pinned snapshot — auto-tracks the latest Sonnet 4.6 release.
MODEL = "claude-sonnet-4-6"

# Max tokens per assistant turn. Per spec — gives Claude room to reason
# between tool calls but stays under the streaming-required threshold.
MAX_TOKENS = 4096

# Guard rail: stop the loop after this many assistant turns even if Claude
# keeps calling tools. Prevents runaway loops on adversarial repos.
MAX_ITERATIONS = 10

RUBRIC_PATH = Path(__file__).resolve().parent.parent / "config" / "scoring_rubric.json"


# ---------------------------------------------------------------------------
# LLM cost tracking
# ---------------------------------------------------------------------------
# Rates are USD per 1M tokens. Snapshot date: 2026-04-29 (per the published
# Anthropic pricing table in the claude-api skill). Cross-check against
# https://www.anthropic.com/pricing when running large batches; the constants
# below are easy to update.

_PRICING_PER_M_TOKENS: dict[str, dict[str, float]] = {
    "claude-opus-4-7":           {"input": 5.00, "output": 25.00},
    "claude-opus-4-6":           {"input": 5.00, "output": 25.00},
    "claude-sonnet-4-6":         {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5":          {"input": 1.00, "output":  5.00},
}

# Prompt-cache multipliers, relative to the base input rate.
# 5-minute TTL cache writes cost 1.25× the input rate; reads cost 0.1×.
_CACHE_WRITE_MULTIPLIER = 1.25
_CACHE_READ_MULTIPLIER  = 0.10


def _compute_call_cost(usage, model: str) -> dict:
    """
    Project an Anthropic response's `usage` block into per-component USD cost.

    Handles missing fields defensively (older SDK versions don't expose the
    cache_* counters at all) and surfaces an explicit `cost_unknown_reason`
    when the model is missing from the pricing table — better than silently
    reporting $0.
    """
    input_tokens   = getattr(usage, "input_tokens", 0) or 0
    output_tokens  = getattr(usage, "output_tokens", 0) or 0
    cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read     = getattr(usage, "cache_read_input_tokens", 0) or 0

    tokens = {
        "input":          input_tokens,
        "output":         output_tokens,
        "cache_creation": cache_creation,
        "cache_read":     cache_read,
    }

    rates = _PRICING_PER_M_TOKENS.get(model)
    if rates is None:
        return {
            "model": model,
            "tokens": tokens,
            "cost_usd": None,
            "cost_unknown_reason": (
                f"no pricing entry for model {model!r}; update "
                "_PRICING_PER_M_TOKENS in agentic_orchestrator.py"
            ),
        }

    in_rate  = rates["input"]  / 1_000_000
    out_rate = rates["output"] / 1_000_000

    components = {
        "input":       input_tokens   * in_rate,
        "output":      output_tokens  * out_rate,
        "cache_write": cache_creation * in_rate * _CACHE_WRITE_MULTIPLIER,
        "cache_read":  cache_read     * in_rate * _CACHE_READ_MULTIPLIER,
    }
    components["total"] = sum(components.values())

    return {
        "model": model,
        "tokens": tokens,
        "cost_usd": {k: round(v, 6) for k, v in components.items()},
    }


def _accumulate_cost(running: dict, call_cost: dict) -> None:
    """In-place sum of per-call cost into a running per-repo total."""
    for tok_key in ("input", "output", "cache_creation", "cache_read"):
        running["tokens"][tok_key] += call_cost["tokens"][tok_key]
    call_usd = call_cost.get("cost_usd")
    if call_usd is None:
        # One unpriced call poisons the total — mark it so it isn't trusted.
        running["cost_usd"] = None
        running.setdefault("warnings", []).append(call_cost.get("cost_unknown_reason"))
        return
    if running["cost_usd"] is None:
        return  # already poisoned
    for k in ("input", "output", "cache_write", "cache_read", "total"):
        running["cost_usd"][k] = round(running["cost_usd"][k] + call_usd[k], 6)


def _new_cost_accumulator() -> dict:
    return {
        "model": MODEL,
        "calls": 0,
        "tokens": {"input": 0, "output": 0, "cache_creation": 0, "cache_read": 0},
        "cost_usd": {"input": 0.0, "output": 0.0, "cache_write": 0.0, "cache_read": 0.0, "total": 0.0},
    }


def _format_cost(cost_usd: Optional[dict]) -> str:
    """Compact one-liner for trace and summary output."""
    if cost_usd is None:
        return "cost: $unknown"
    return (
        f"cost: ${cost_usd['total']:.4f} "
        f"(in ${cost_usd['input']:.4f} + out ${cost_usd['output']:.4f} "
        f"+ cache_w ${cost_usd['cache_write']:.4f} + cache_r ${cost_usd['cache_read']:.4f})"
    )


# ---------------------------------------------------------------------------
# Tool definitions exposed to Claude
# ---------------------------------------------------------------------------
# Schemas are intentionally minimal: only `fetch_repo_data` takes a real
# argument (the repo slug). The analyzers run against the orchestrator's
# accumulated state, so Claude doesn't need to pass massive dicts back —
# that would be both wasteful and error-prone.

TOOLS: list[dict] = [
    {
        "name": "fetch_repo_data",
        "description": (
            "Fetch the GitHub repository's metadata, file tree, README, and "
            "dependency manifests (package.json, requirements.txt, pom.xml, "
            "go.mod, etc.). CALL THIS FIRST — every other tool requires this "
            "context. The tool also automatically runs the repo-type "
            "classifier and returns its verdict in the `classification` "
            "field (one of APPLICATION, LIBRARY, DOCUMENTATION, "
            "CONFIGURATION, DATA, UNKNOWN). Use that classification to "
            "shape your subsequent reasoning — see the system prompt."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {
                    "type": "string",
                    "description": (
                        "GitHub repository in 'owner/name' format, e.g. 'facebook/react'."
                    ),
                }
            },
            "required": ["repo"],
        },
    },
    {
        "name": "analyze_code",
        "description": (
            "Static-analyze the repository: primary language and runtime "
            "version, frameworks detected from dependency manifests, "
            "lines-of-code estimate, presence of tests, CI configuration, "
            "and modular-vs-monolith architecture signals. Requires "
            "fetch_repo_data to have been called first."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "analyze_dependencies",
        "description": (
            "Inspect every recognized dependency manifest. Returns "
            "dependency_freshness (ratio of up-to-date deps), runtime_eol "
            "(current | lts | maintenance | eol), and a best-effort "
            "open_issues_ratio. Useful when the repo's age, runtime "
            "currency, or dep staleness will drive the recommendation."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "analyze_activity",
        "description": (
            "Fetch commit recency, monthly commit frequency, and total "
            "contributor count. The right tool when you suspect a repo is "
            "dormant, single-maintainer, or otherwise low-activity."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "score_recommendation",
        "description": (
            "Aggregate every signal gathered so far against the 5-R scoring "
            "rubric and produce a deterministic structured recommendation. "
            "REQUIRES that fetch_repo_data, analyze_code, "
            "analyze_dependencies, AND analyze_activity have all already "
            "returned. Calling this tool before all four have run will be "
            "rejected with an error telling you which is missing — the "
            "scorer cannot trust an incomplete signal set and will demote "
            "the report to REVIEW if it tries. Returns one of RETAIN / "
            "REHOST / REFACTOR / REWRITE / RETIRE, or REVIEW when critical "
            "signals are missing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "rationale_notes": {
                    "type": "string",
                    "description": (
                        "Optional architect commentary. Will be surfaced "
                        "alongside the scorer's deterministic rationale."
                    ),
                }
            },
            "required": [],
        },
    },
]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

# Block 1 — role / framework / how-to-work / taxonomy / final-output.
# Static across runs; included in the cached prefix via the cache_control
# marker on Block 2 (the rubric) below.
_SYSTEM_PROMPT_PREAMBLE_TEMPLATE = """You are a senior solutions architect performing application rationalization on GitHub repositories. Your job is to decide whether each repo should be RETAINED, REHOSTED, REFACTORED, REWRITTEN, or RETIRED, and to produce a defensible recommendation with confidence and rationale.

# The 5-R framework
- RETAIN — healthy, modern, actively maintained; leave alone.
- REHOST — stable but on aging infra; lift-and-shift to cloud/containers, no code change.
- REFACTOR — working but structurally weak (no tests, monolith, stale deps); business logic is sound, invest selectively.
- REWRITE — too degraded to salvage incrementally; EOL runtime, no tests, broad surface area.
- RETIRE — low activity, superseded, or no longer needed; decommission.

There is also a sixth value the scorer can return — REVIEW — meaning the signal set was incomplete and the recommendation cannot be trusted. You should never *intend* to produce REVIEW; it is the scorer's safety net for when you skipped a required analyzer. Always gather a complete signal set so the scorer can make a real call.

# How to work — REQUIRED tool sequence
You have five tools. The first four gather signal; the fifth produces the recommendation. You MUST run every data-gathering tool before scoring.

1. **First**: call `fetch_repo_data`. This establishes context — language, manifests, file count, license, last-pushed date, archived status — AND automatically runs the repo-type classifier. Read the `classification` field in the response carefully before deciding what to do next (see "Repo-type taxonomy" below).

2. **Then call all three analyzers**, in any order, but every one of them:
   - `analyze_code` — language, frameworks, LOC, tests, CI, architecture
   - `analyze_dependencies` — runtime EOL, dep freshness
   - `analyze_activity` — last-commit recency, commit frequency, contributor count

   You MUST NOT skip any of these. The scorer's override rules (force_retire_if, force_rewrite_if) depend on signals from each — most critically, `analyze_activity` produces `last_commit_days`, which drives the dormant-repo retire override. Skipping an analyzer because the repo "looks obvious" produces a report with missing signals; the scorer will detect this and demote the recommendation to REVIEW. A 100K-star repo with daily commits still needs every analyzer to run — the cost is small and the safety is non-negotiable.

3. **Between tool calls**, briefly say what you observed and what you intend to check next. The reasoning is for the audit trail, not for skipping work.

4. **Only after fetch_repo_data + all three analyzers have returned**, call `score_recommendation`. The scorer applies the rubric mechanically to the full signal set. If you call it early, the orchestrator will reject the call with a tool_result error telling you which analyzer is still missing — at that point, run the missing analyzer and try again.

5. **Flag low confidence explicitly** when signals are sparse, conflicting, or an analyzer reported errors.

# Repo-type taxonomy — adjust your reasoning by type
The classifier's `repo_type` field tells you which kind of repository you're looking at. The 5-R framework was designed for runtime software; for everything else you must reason differently.

- **APPLICATION** — deployable software (web service, CLI, mobile app). The standard 5-R rubric applies cleanly. Use it as-is.
- **LIBRARY** — reusable package or framework consumed by other software. The standard rubric also applies, with one nuance: "no deployment" and "no CI for end-to-end tests" are normal — judge maintenance and dep-health signals against the library's role.
- **DOCUMENTATION** — pure docs, specs, standards, awesome-lists, RFC collections. **5-R analysis is structurally inapplicable in the rubric's usual sense.** Zero LOC, no tests, no CI, no runtime, and no dependencies are expected by design, not failures. The scorer suppresses the relevant override rules (per `repo_type_overrides` in the rubric), but the underlying category scores will still look weak. **You must override the scorer's threshold-based recommendation if necessary**, and base your call on fitness-for-purpose signals: community adoption (stars, forks, who depends on it institutionally), maintenance cadence (last commit date — even if frequency is low), and continued relevance (is the standard still cited / linked-to by active projects). Explain in the rationale that the standard rubric signals are inapplicable and which fitness-for-purpose signals drove your call.
- **CONFIGURATION** — IaC, dotfiles, configs. Similar treatment to DOCUMENTATION: LOC/test/CI rules are suppressed; recommendation should be based on whether the configuration is still in active use, still targets supported infra, and is being maintained.
- **DATA** — datasets, assets, content. Treat like DOCUMENTATION: rubric is structurally inapplicable; base call on currency, completeness, continued use.
- **UNKNOWN** — classifier could not determine the type. Treat with caution and flag for review.

For DOCUMENTATION, CONFIGURATION, and DATA repos: **still run all three analyzers** (you need the raw signals for the audit trail and so the scorer can compute what it can) — but in your final rationale, state explicitly that "the standard rubric signals are structurally inapplicable for this repo type," explain why, and ground the recommendation in fitness-for-purpose reasoning.

# Final output
After `score_recommendation`, your final assistant turn must end with a fenced JSON block matching this schema exactly:

```json
{{
  "recommendation": "RETAIN | REHOST | REFACTOR | REWRITE | RETIRE | REVIEW",
  "confidence": 0.0,
  "rationale": "Single architect-readable paragraph explaining the call: dominant signals, weak areas, anything that drove the confidence up or down.",
  "signals_summary": {{ "final_score": 0.0, "category_scores": {{ }} }},
  "flagged_for_review": false
}}
```

If the scorer returned REVIEW, your recommendation must be REVIEW — do not override that to one of the 5-R values. The rationale must explain which analyzer's signals were missing.

You may set `signals_summary` to the scorer's `signals_summary` verbatim. You may raise or lower `confidence` from the scorer's value if you have good reason — say so in the rationale. Outside the JSON, briefly state why you made any adjustments.
"""

# Block 2 — the scoring rubric. Its own content block so the cache_control
# marker that ends here defines the cached prefix (tools + system blocks 1
# and 2). The rubric is the largest single contributor to system-prompt
# tokens and is byte-stable across every repo in a batch run.
_SYSTEM_PROMPT_RUBRIC_HEADER = """# Scoring rubric (reference — not hard rules)
The rubric below shows the weights and thresholds a senior architect would typically apply. The scorer uses these mechanically; you should weigh them with judgment. Override the threshold-based call only when the signals genuinely warrant it.

```json
"""

_SYSTEM_PROMPT_RUBRIC_FOOTER = "\n```\n"

# Block 3 — constraints. Kept after the rubric (preserving the original
# narrative order so model behavior is unchanged) but OUTSIDE the cached
# prefix. Stays small to minimise the uncached-tail cost.
_SYSTEM_PROMPT_CONSTRAINTS = """# Constraints
- You have at most {max_iterations} tool-calling turns. Budget them.
- All four data-gathering tools (`fetch_repo_data`, `analyze_code`, `analyze_dependencies`, `analyze_activity`) must run before `score_recommendation`. The orchestrator enforces this — premature scorer calls are rejected with an error tool_result.
- If the scorer returns REVIEW, treat the report as incomplete: surface it as REVIEW in your final JSON and explain in the rationale which signals were missing.
"""


def _ephemeral_cache(cache_ttl: str) -> dict:
    """Build the cache_control dict, honoring the configured TTL."""
    control: dict = {"type": "ephemeral"}
    if cache_ttl == "1h":
        # 1-hour TTL costs 2× the input rate to write (vs 1.25× for 5min)
        # but keeps the cache alive across long batch runs.
        control["ttl"] = "1h"
    return control


def _build_system_prompt(cache_ttl: str = "5m") -> list[dict]:
    """
    Build the system prompt as 3 content blocks:
      Block 1 — role / framework / how-to-work / taxonomy / final-output
      Block 2 — scoring rubric (cache_control marker lands here)
      Block 3 — runtime constraints (small, uncached)

    The cache_control on Block 2 marks tools + Block 1 + Block 2 as the
    cached prefix. Block 3 stays small so the uncached tail is cheap.
    The narrative order is preserved exactly — only the transmission
    structure changes.
    """
    try:
        with open(RUBRIC_PATH) as f:
            rubric = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        # Fall back to an empty rubric; the scorer surfaces its own error.
        rubric = {"_error": f"failed to load rubric: {e}"}

    rubric_block_text = (
        _SYSTEM_PROMPT_RUBRIC_HEADER
        + json.dumps(rubric, indent=2)
        + _SYSTEM_PROMPT_RUBRIC_FOOTER
    )

    return [
        {
            "type": "text",
            "text": _SYSTEM_PROMPT_PREAMBLE_TEMPLATE,
        },
        {
            "type": "text",
            "text": rubric_block_text,
            "cache_control": _ephemeral_cache(cache_ttl),
        },
        {
            "type": "text",
            "text": _SYSTEM_PROMPT_CONSTRAINTS.format(max_iterations=MAX_ITERATIONS),
        },
    ]


def _build_tools(cache_ttl: str = "5m") -> list[dict]:
    """
    Return the TOOLS list with cache_control on the last entry, so the
    tool definitions are included in the cached prefix. The Anthropic
    API renders tools BEFORE system, so a marker at the end of tools
    extends the cache to cover the entire tool block.
    """
    if not TOOLS:
        return []
    # Shallow-copy so the module-level TOOLS list isn't mutated.
    cached_tools = [dict(t) for t in TOOLS]
    cached_tools[-1] = {**cached_tools[-1], "cache_control": _ephemeral_cache(cache_ttl)}
    return cached_tools


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

def _execute_tool(
    tool_name: str,
    tool_input: dict,
    state: dict,
    trace: bool,
) -> tuple[Any, bool]:
    """
    Run the tool locally and return (summary_payload, is_error).

    `state` is the orchestrator's accumulated repo_data dict; every analyzer
    receives this so per-tool calls don't have to round-trip large payloads
    through Claude's context.
    """
    try:
        if tool_name == "fetch_repo_data":
            repo = tool_input.get("repo")
            if not repo or "/" not in str(repo):
                return {"error": f"invalid repo: {repo!r}"}, True
            state["repo"] = repo
            result = _fetch_repo_data({"repo": repo})
            state["fetch"] = result
            state["pipeline_stages"]["fetch"] = result

            # Pre-scoring stage: classify the repository type. Runs
            # automatically here (not exposed as a separate tool) so
            # Claude always sees the classification alongside the fetch
            # summary, and the scorer can suppress rules for
            # non-APPLICATION repo types per the rubric.
            classification = _classify_repo({"fetch": result})
            state["classification"] = classification
            state["pipeline_stages"]["classification"] = classification

            return _summarize_fetch(result, classification), False

        # Every downstream analyzer needs the fetch payload.
        if "fetch" not in state:
            return {
                "error": (
                    "fetch_repo_data has not been called yet. Call it "
                    "first to establish repository context."
                )
            }, True

        if tool_name == "analyze_code":
            result = _analyze_code(state)
            state["code"] = result
            state["pipeline_stages"]["code"] = result
            return _summarize_analyzer(result, category_key="category_scores"), False

        if tool_name == "analyze_dependencies":
            result = _analyze_deps(state)
            state["dependencies"] = result
            state["pipeline_stages"]["dependencies"] = result
            return _summarize_analyzer(result), False

        if tool_name == "analyze_activity":
            result = _analyze_activity(state)
            state["activity"] = result
            state["pipeline_stages"]["activity"] = result
            return _summarize_analyzer(result), False

        if tool_name == "score_recommendation":
            # Belt-and-suspenders to the system-prompt rule: reject any
            # score call where one of the four data-gathering stages
            # hasn't populated state. The scorer would otherwise demote
            # to REVIEW via the missing-signal pathway, but failing fast
            # at the tool boundary gives Claude a clearer error to react
            # to and avoids a wasted score evaluation.
            required_stages = ("fetch", "code", "dependencies", "activity")
            missing = [s for s in required_stages if s not in state]
            if missing:
                tool_for_stage = {
                    "fetch":        "fetch_repo_data",
                    "code":         "analyze_code",
                    "dependencies": "analyze_dependencies",
                    "activity":     "analyze_activity",
                }
                missing_tools = [tool_for_stage[s] for s in missing]
                return {
                    "error": (
                        "score_recommendation called before all required "
                        "analyzers ran. Missing: "
                        + ", ".join(missing_tools)
                        + ". Run the missing tool(s) first, then retry "
                        "score_recommendation."
                    ),
                    "missing_tools": missing_tools,
                }, True

            score_result = _score(state["pipeline_stages"])
            # Attach any architect commentary Claude included.
            notes = (tool_input or {}).get("rationale_notes")
            if notes:
                score_result["architect_notes"] = notes
            state["score"] = score_result
            return score_result, False

        return {"error": f"unknown tool: {tool_name}"}, True

    except Exception as e:  # noqa: BLE001 — surface any analyzer failure to Claude
        return {"error": f"{type(e).__name__}: {e}"}, True


# ---------------------------------------------------------------------------
# Tool-result summarizers
# ---------------------------------------------------------------------------
# The analyzer modules return large dicts (full file trees, manifest content,
# per-signal scoring detail). Sending them back to Claude verbatim would
# burn context. These helpers return only the fields Claude needs to reason —
# the full data stays in `state` and feeds subsequent analyzers.

def _summarize_fetch(result: dict, classification: Optional[dict] = None) -> dict:
    metadata = result.get("metadata") or {}
    readme = result.get("readme") or {}
    summary = {
        "repo": result.get("repo"),
        "metadata": metadata,
        "file_count": len(result.get("file_tree") or []),
        "tree_truncated": result.get("tree_truncated", False),
        "manifests_found": list((result.get("manifests") or {}).keys()),
        "has_readme": bool(readme),
        "readme_size": readme.get("size"),
        "rate_limit": result.get("rate_limit"),
        "fetch_errors": result.get("errors") or [],
    }
    if classification is not None:
        # The classifier runs automatically right after fetch — surface
        # its verdict so Claude can adjust its reasoning for non-
        # APPLICATION repo types before deciding which analyzers to call.
        summary["classification"] = classification
    return summary


def _summarize_analyzer(result: dict, category_key: str = "category_score") -> dict:
    """Project an analyzer's output down to signals + scored summary."""
    summary = {
        "agent": result.get("agent"),
        "signals": result.get("signals"),
        "errors": result.get("errors") or [],
    }
    if category_key in result:
        summary[category_key] = result[category_key]
    if "low_signal" in result:
        summary["low_signal"] = result["low_signal"]
    # Pull in any raw breakdown (e.g. dependency_scanner.raw) so Claude can
    # cite specific outdated examples in its rationale.
    raw = result.get("raw")
    if isinstance(raw, dict):
        summary["raw"] = {
            k: raw[k]
            for k in (
                "manifests_analyzed",
                "manifests_unparseable",
                "total_dependencies",
                "classified",
                "up_to_date",
                "outdated",
                "runtime_findings",
            )
            if k in raw
        }
    return summary


# ---------------------------------------------------------------------------
# Final-JSON extraction from Claude's last message
# ---------------------------------------------------------------------------

_JSON_FENCE_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.S | re.I)


def _extract_final_json(text: str) -> Optional[dict]:
    """Pull the fenced ```json {...}``` block out of Claude's final reply."""
    if not text:
        return None
    m = _JSON_FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Fallback: try to parse the entire message as JSON.
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------

def run_agentic_pipeline(
    repo: str, trace: bool = False, cache_ttl: str = "5m"
) -> dict:
    """
    Run one full agentic pipeline for a single repo.

    Returns a report dict whose top-level fields (recommendation, confidence,
    rationale, flagged_for_review) match what print_report and save_report
    expect, plus an `agentic` block recording the full reasoning chain.

    `cache_ttl` is "5m" (default) or "1h" and controls the TTL of the
    prompt cache markers on the system prompt and tool definitions.
    """
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    state: dict[str, Any] = {
        "repo": repo,
        "pipeline_stages": {},
    }

    messages: list[dict] = [
        {
            "role": "user",
            "content": (
                f"Analyze the GitHub repository `{repo}` against the 5-R "
                "rationalization framework. Reason aloud between tool calls, "
                "then produce the final structured JSON recommendation."
            ),
        }
    ]

    tool_call_log: list[dict] = []
    per_call_cost: list[dict] = []
    cost_total = _new_cost_accumulator()
    iteration = 0
    final_response = None
    max_iterations_hit = False
    final_text = ""

    while iteration < MAX_ITERATIONS:
        iteration += 1

        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=_build_system_prompt(cache_ttl),
                tools=_build_tools(cache_ttl),
                messages=messages,
            )
        except anthropic.APIStatusError as e:
            # Surface the error in the report rather than crashing the batch.
            return _error_report(repo, f"Anthropic API error: {e.status_code} {e.message}")
        except anthropic.APIConnectionError as e:
            return _error_report(repo, f"Anthropic API connection error: {e}")

        # Per-call cost is computed for every successful response, regardless
        # of stop_reason — even an end_turn response was billed for its tokens.
        call_cost = _compute_call_cost(response.usage, MODEL)
        call_cost["iteration"] = iteration
        per_call_cost.append(call_cost)
        cost_total["calls"] += 1
        _accumulate_cost(cost_total, call_cost)

        # Trace: show reasoning text + tool calls + cost as they happen.
        if trace:
            _print_trace_turn(iteration, response, call_cost)

        final_response = response

        if response.stop_reason == "end_turn":
            # Claude is done — collect final text and exit the loop.
            final_text = _collect_text(response)
            break

        if response.stop_reason != "tool_use":
            # Unexpected stop reason (refusal, max_tokens, etc.) — capture
            # whatever text was produced and bail out.
            final_text = _collect_text(response)
            break

        # Append assistant's full content (text + tool_use blocks) so Claude
        # sees the same conversation state on the next turn.
        messages.append({"role": "assistant", "content": response.content})

        # Execute every tool call in this turn and collect tool_result blocks.
        tool_result_blocks: list[dict] = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            result, is_error = _execute_tool(block.name, block.input or {}, state, trace)
            tool_call_log.append(
                {
                    "iteration": iteration,
                    "name": block.name,
                    "input": block.input,
                    "is_error": is_error,
                }
            )
            tool_result_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, default=str),
                    "is_error": is_error,
                }
            )

        messages.append({"role": "user", "content": tool_result_blocks})

    else:
        # Loop exited via the while condition rather than `break` — Claude
        # kept calling tools and we hit the iteration cap.
        max_iterations_hit = True
        warnings.warn(
            f"[agentic_orchestrator] max_iterations ({MAX_ITERATIONS}) "
            f"reached for {repo}; truncating loop.",
            stacklevel=2,
        )

    # Always compute the deterministic score over whatever signals we gathered
    # so the final report has a valid signals_summary even if Claude bailed early.
    fallback_score = _score(state.get("pipeline_stages") or {})

    claude_json = _extract_final_json(final_text)
    report = _build_final_report(
        repo=repo,
        claude_json=claude_json,
        fallback_score=fallback_score,
        pipeline_stages=state.get("pipeline_stages") or {},
        tool_call_log=tool_call_log,
        iteration=iteration,
        max_iterations_hit=max_iterations_hit,
        final_text=final_text,
        per_call_cost=per_call_cost,
        cost_total=cost_total,
    )
    return report


# ---------------------------------------------------------------------------
# Report construction
# ---------------------------------------------------------------------------

def _build_final_report(
    *,
    repo: str,
    claude_json: Optional[dict],
    fallback_score: dict,
    pipeline_stages: dict,
    tool_call_log: list,
    iteration: int,
    max_iterations_hit: bool,
    final_text: str,
    per_call_cost: list[dict],
    cost_total: dict,
) -> dict:
    """
    Merge Claude's parsed JSON with the deterministic scorer's output.

    Strategy:
      - Use Claude's recommendation / confidence / rationale when present.
      - Always use the scorer's signals_summary and review_reasons.
      - **REVIEW is binding.** If the scorer returns REVIEW, that outcome
        cannot be overridden by Claude's final JSON. REVIEW means the
        input data is insufficient — that's a property of the inputs,
        not a judgment Claude can talk its way out of. Claude's
        contextual analysis is preserved as an audit note in the
        rationale, and the override attempt is captured in
        flagged_for_review_reasons.
    """
    claude_json = claude_json or {}

    scorer_rec = fallback_score.get("recommendation")
    scorer_review_locked = scorer_rec == "REVIEW"

    # Start from the scorer's deterministic review reasons; we may append.
    review_reasons: list = list(fallback_score.get("flagged_for_review_reasons", []))

    if scorer_review_locked:
        # Scorer's REVIEW is a hard ceiling — surface Claude's attempted
        # call (if any) and lock the outcome.
        claude_rec = claude_json.get("recommendation")
        final_recommendation = "REVIEW"
        final_confidence = fallback_score.get("confidence")
        final_rationale = fallback_score.get("rationale") or ""

        if claude_rec and claude_rec != "REVIEW":
            claude_rationale = claude_json.get("rationale") or ""
            final_rationale = (
                final_rationale.rstrip(". ") + ". "
                f"[LLM proposed {claude_rec} based on contextual analysis but "
                "the scorer's REVIEW outcome is binding — the underlying signal "
                "set was incomplete. LLM rationale preserved below for audit: "
                f"\"{claude_rationale}\"]"
            )
            review_reasons.append(
                f"LLM proposed {claude_rec} but was overridden — REVIEW is "
                "binding when set by the scorer (signal set is incomplete)"
            )
        final_flagged = True
    else:
        # Normal merge path — Claude may set recommendation/confidence/
        # rationale; scorer fills any blanks.
        def _pick(key: str, default: Any) -> Any:
            v = claude_json.get(key)
            return v if v not in (None, "") else default

        final_recommendation = _pick("recommendation", scorer_rec)
        final_confidence = _pick("confidence", fallback_score.get("confidence"))
        final_rationale = _pick("rationale", fallback_score.get("rationale"))
        final_flagged = bool(
            claude_json.get("flagged_for_review")
            if "flagged_for_review" in claude_json
            else fallback_score.get("flagged_for_review")
        )

    return {
        "repo": repo,
        "analyzed_at": datetime.utcnow().isoformat() + "Z",
        "model": MODEL,
        "pipeline_stages": pipeline_stages,
        "recommendation": final_recommendation,
        "confidence": final_confidence,
        "rationale": final_rationale,
        "signals_summary": fallback_score.get("signals_summary"),
        "flagged_for_review": final_flagged,
        "flagged_for_review_reasons": review_reasons,
        "agentic": {
            "model": MODEL,
            "iterations": iteration,
            "max_iterations_hit": max_iterations_hit,
            "tool_calls": tool_call_log,
            "claude_final_text": final_text,
            "claude_parsed_json_present": bool(claude_json),
            "scorer_review_locked": scorer_review_locked,
            "cost": {
                "total":     cost_total,
                "per_call":  per_call_cost,
            },
        },
    }


def _error_report(repo: str, message: str) -> dict:
    return {
        "repo": repo,
        "analyzed_at": datetime.utcnow().isoformat() + "Z",
        "model": MODEL,
        "pipeline_stages": {},
        "recommendation": "ERROR",
        "confidence": 0.0,
        "rationale": message,
        "signals_summary": {},
        "flagged_for_review": True,
        "flagged_for_review_reasons": [f"pipeline error: {message}"],
        "agentic": {
            "error": message,
            "cost": {"total": _new_cost_accumulator(), "per_call": []},
        },
    }


# ---------------------------------------------------------------------------
# Trace output
# ---------------------------------------------------------------------------

def _collect_text(response) -> str:
    return "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )


def _print_trace_turn(iteration: int, response, call_cost: dict) -> None:
    """Print Claude's reasoning text, tool calls, and per-call cost for this turn."""
    text = _collect_text(response).strip()
    tool_calls = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
    tokens = call_cost["tokens"]
    cost_line = _format_cost(call_cost.get("cost_usd"))
    token_line = (
        f"tokens: in={tokens['input']} out={tokens['output']} "
        f"cache_w={tokens['cache_creation']} cache_r={tokens['cache_read']}"
    )

    if RICH_AVAILABLE:
        _console.rule(f"[bold]iteration {iteration}[/bold]")
        if text:
            _console.print(f"[italic dim]reasoning:[/italic dim]\n{text}\n")
        for block in tool_calls:
            _console.print(
                f"[bold cyan]→ tool[/bold cyan] [yellow]{block.name}[/yellow]"
                f"({json.dumps(block.input, default=str)})"
            )
        _console.print(f"[dim]stop_reason={response.stop_reason}[/dim]")
        _console.print(f"[dim]{token_line}[/dim]")
        _console.print(f"[dim]{cost_line}[/dim]")
    else:
        print(f"\n--- iteration {iteration} ---")
        if text:
            print(f"reasoning:\n{text}\n")
        for block in tool_calls:
            print(f"→ tool: {block.name}({json.dumps(block.input, default=str)})")
        print(f"stop_reason={response.stop_reason}")
        print(token_line)
        print(cost_line)


# ---------------------------------------------------------------------------
# Cost summary printers
# ---------------------------------------------------------------------------

def _print_cost_summary(report: dict) -> None:
    """One-line LLM cost summary for the repo just analyzed."""
    cost = ((report.get("agentic") or {}).get("cost") or {}).get("total")
    if not cost:
        return
    calls = cost.get("calls", 0)
    tokens = cost.get("tokens") or {}
    usd = cost.get("cost_usd")
    token_line = (
        f"LLM: {calls} call(s) | "
        f"in={tokens.get('input',0):,} out={tokens.get('output',0):,} "
        f"cache_w={tokens.get('cache_creation',0):,} cache_r={tokens.get('cache_read',0):,}"
    )
    if usd is None:
        cost_line = "cost: $unknown (model not in pricing table)"
    else:
        cost_line = (
            f"cost: ${usd['total']:.4f} "
            f"(in ${usd['input']:.4f} + out ${usd['output']:.4f} "
            f"+ cache_w ${usd['cache_write']:.4f} + cache_r ${usd['cache_read']:.4f})"
        )
    if RICH_AVAILABLE:
        _console.print(f"[dim]{token_line}[/dim]")
        _console.print(f"[dim]{cost_line}[/dim]")
    else:
        print(token_line)
        print(cost_line)


def _print_cache_breakdown(report: dict) -> None:
    """
    Per-repo prompt-cache breakdown — only displayed when --show-costs
    or --verbose is set. Surfaces the three numbers the user typically
    cares about for diagnosing cache effectiveness within a single run.
    """
    cost = ((report.get("agentic") or {}).get("cost") or {}).get("total")
    if not cost:
        return
    tokens = cost.get("tokens") or {}
    cache_write = tokens.get("cache_creation", 0)
    cache_read  = tokens.get("cache_read", 0)
    full_price  = tokens.get("input", 0)

    lines = [
        "Cache usage (aggregate across this repo's API calls):",
        f"  cache_creation_input_tokens: {cache_write:>10,}  "
        f"(tokens written to cache — ~1.25× / ~2× write premium)",
        f"  cache_read_input_tokens:     {cache_read:>10,}  "
        f"(tokens read from cache — ~0.1× full price; this is the savings)",
        f"  input_tokens:                {full_price:>10,}  "
        f"(uncached, charged at full input rate)",
    ]
    if RICH_AVAILABLE:
        for line in lines:
            _console.print(f"[dim]{line}[/dim]")
    else:
        for line in lines:
            print(line)


def _print_batch_total(batch_total: dict, repo_count: int) -> None:
    """Aggregate LLM cost across every repo in a batch run."""
    calls = batch_total.get("calls", 0)
    tokens = batch_total.get("tokens") or {}
    usd = batch_total.get("cost_usd")
    header = f"Batch total across {repo_count} repos"
    token_line = (
        f"  {calls} LLM call(s) | "
        f"in={tokens.get('input',0):,} out={tokens.get('output',0):,} "
        f"cache_w={tokens.get('cache_creation',0):,} cache_r={tokens.get('cache_read',0):,}"
    )
    # Highlight cumulative cache_read separately as the savings indicator.
    cumulative_savings_line = (
        f"  cumulative cache_read_input_tokens: "
        f"{tokens.get('cache_read',0):,}  "
        f"(↓ total tokens served from the prompt cache across the batch)"
    )
    if usd is None:
        cost_line = "  cost: $unknown (one or more calls used an unpriced model)"
    else:
        cost_line = (
            f"  cost: ${usd['total']:.4f} "
            f"(in ${usd['input']:.4f} + out ${usd['output']:.4f} "
            f"+ cache_w ${usd['cache_write']:.4f} + cache_r ${usd['cache_read']:.4f})"
        )
    if RICH_AVAILABLE:
        _console.rule(f"[bold]{header}[/bold]")
        _console.print(token_line)
        _console.print(cumulative_savings_line)
        _console.print(cost_line)
    else:
        print(f"\n=== {header} ===")
        print(token_line)
        print(cumulative_savings_line)
        print(cost_line)


# ---------------------------------------------------------------------------
# CLI — preserves the orchestrator.py interface, adds --trace
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Agentic App Rationalization Agent — Claude drives the pipeline "
            "via tool use, calling each analyzer as a tool."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--repo",
        help="GitHub repository in owner/name format (e.g. facebook/react)",
    )
    group.add_argument(
        "--batch",
        help="Path to a text file with one repo per line (# for comments)",
    )
    parser.add_argument(
        "--output",
        choices=["pretty", "json"],
        default="pretty",
        help="Output format (default: pretty)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the full report JSON to outputs/reports/",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help=(
            "Print each tool call Claude makes and its reasoning before "
            "executing it. Required for certification demonstration."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Print detailed exception tracebacks on failure. Also implies "
            "--show-costs (prints the per-repo cache-token breakdown)."
        ),
    )
    parser.add_argument(
        "--cache-ttl",
        choices=["5m", "1h"],
        default="5m",
        help=(
            "Prompt cache TTL. '5m' (default) charges 1.25× the input rate "
            "to write the cache. '1h' charges 2× to write but keeps the "
            "cache warm across long batch runs that take longer than 5 "
            "minutes end-to-end."
        ),
    )
    parser.add_argument(
        "--show-costs",
        action="store_true",
        help=(
            "After each repo, print the per-repo cache-token breakdown: "
            "cache_creation_input_tokens (writes), cache_read_input_tokens "
            "(reads — the savings), and input_tokens (uncached, full price)."
        ),
    )
    return parser.parse_args()


def main():
    load_env()
    args = parse_args()

    if not __import__("os").environ.get("ANTHROPIC_API_KEY"):
        sys.stderr.write(
            "error: ANTHROPIC_API_KEY is not set. Copy .env.template to .env "
            "and populate it.\n"
        )
        sys.exit(1)

    repos: list[str] = []
    if args.repo:
        repos = [args.repo]
    elif args.batch:
        with open(args.batch) as f:
            repos = [
                line.strip()
                for line in f
                if line.strip() and not line.startswith("#")
            ]

    if RICH_AVAILABLE:
        _console.print(
            f"\n[bold cyan]Agentic App Rationalization[/bold cyan] — "
            f"model={MODEL}, repos={len(repos)}, trace={args.trace}, "
            f"cache_ttl={args.cache_ttl}\n"
        )

    batch_total = _new_cost_accumulator()
    show_cache_breakdown = args.show_costs or args.verbose

    for repo in repos:
        if RICH_AVAILABLE:
            _console.rule(f"[bold]{repo}[/bold]")
        else:
            print(f"\nAnalyzing: {repo}")

        try:
            report = run_agentic_pipeline(
                repo, trace=args.trace, cache_ttl=args.cache_ttl
            )
        except Exception as e:  # noqa: BLE001 — preserve batch progress on failure
            if args.verbose and RICH_AVAILABLE:
                _console.print_exception()
            report = _error_report(repo, f"{type(e).__name__}: {e}")

        if args.output == "json":
            print(json.dumps(report, indent=2, default=str))
        else:
            print_report(report)
            _print_cost_summary(report)
            if show_cache_breakdown:
                _print_cache_breakdown(report)

        # Roll this repo's cost into the batch total.
        repo_cost = (report.get("agentic") or {}).get("cost") or {}
        repo_total = repo_cost.get("total") or {}
        if repo_total:
            batch_total["calls"] += repo_total.get("calls", 0)
            tokens = repo_total.get("tokens") or {}
            for k in ("input", "output", "cache_creation", "cache_read"):
                batch_total["tokens"][k] += tokens.get(k, 0)
            repo_usd = repo_total.get("cost_usd")
            if repo_usd is None:
                batch_total["cost_usd"] = None
            elif batch_total["cost_usd"] is not None:
                for k in ("input", "output", "cache_write", "cache_read", "total"):
                    batch_total["cost_usd"][k] = round(
                        batch_total["cost_usd"][k] + repo_usd.get(k, 0), 6
                    )

        if args.save:
            saved = save_report(report)
            msg = f"Report saved → {saved}"
            if RICH_AVAILABLE:
                _console.print(f"[dim]{msg}[/dim]")
            else:
                print(msg)

    if len(repos) > 1 and args.output != "json":
        _print_batch_total(batch_total, len(repos))


if __name__ == "__main__":
    main()
