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
from code_analyzer import analyze as _analyze_code
from dependency_scanner import analyze as _analyze_deps
from activity_analyzer import analyze as _analyze_activity
from scorer import score as _score

# Reuse the deterministic orchestrator's report formatting/saving helpers.
from orchestrator import print_report, save_report, load_env

# Rich is optional — orchestrator's helpers degrade gracefully when missing.
try:
    from rich.console import Console
    from rich import print as rprint
    RICH_AVAILABLE = True
    _console = Console()
except ImportError:
    RICH_AVAILABLE = False
    _console = None


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

# User-specified model. NOTE: claude-sonnet-4-20250514 is Claude Sonnet 4.0,
# which is *deprecated* and slated for retirement on 2026-06-15. The current
# Sonnet (and documented migration target) is `claude-sonnet-4-6`. Update
# here if you migrate.
MODEL = "claude-sonnet-4-20250514"

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
    "claude-sonnet-4-20250514":  {"input": 3.00, "output": 15.00},  # Sonnet 4.0
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
            "context. Returns a summary; the full payload is cached for the "
            "downstream analyzers."
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
            "rubric and produce a deterministic structured recommendation "
            "(RETAIN / REHOST / REFACTOR / REWRITE / RETIRE) with "
            "confidence and per-category scores. Call this AFTER you have "
            "enough signal — it is the closing tool. You may still adjust "
            "the call in your final message."
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

_SYSTEM_PROMPT_TEMPLATE = """You are a senior solutions architect performing application rationalization on GitHub repositories. Your job is to decide whether each repo should be RETAINED, REHOSTED, REFACTORED, REWRITTEN, or RETIRED, and to produce a defensible recommendation with confidence and rationale.

# The 5-R framework
- RETAIN — healthy, modern, actively maintained; leave alone.
- REHOST — stable but on aging infra; lift-and-shift to cloud/containers, no code change.
- REFACTOR — working but structurally weak (no tests, monolith, stale deps); business logic is sound, invest selectively.
- REWRITE — too degraded to salvage incrementally; EOL runtime, no tests, broad surface area.
- RETIRE — low activity, superseded, or no longer needed; decommission.

# How to work
You have five tools. Use them deliberately — not every analyzer is relevant for every repo.

1. Call `fetch_repo_data` first. The summary tells you the language, manifests present, file count, license, last-pushed date, and whether the repo is archived. Reason about that before calling more tools.
2. Decide which analyzers will most cheaply move you toward a confident call:
   - Dormant-looking repo? → `analyze_activity` is the high-signal next step.
   - Java pom.xml with old Java pin? → `analyze_dependencies` will tell you if the runtime is EOL.
   - Many source files, unclear test coverage? → `analyze_code` for tests/CI/architecture.
3. Between tool calls, briefly say what you observed and what you intend to check next. Skip an analyzer when its result is unlikely to change your call — a 100K-star repo with daily commits doesn't need a deep activity scan.
4. When you have enough signal, call `score_recommendation`. The scorer produces a deterministic 5-R recommendation using the rubric below; you can use it verbatim or adjust the confidence/rationale based on your judgment.
5. Flag low confidence explicitly when signals are sparse, conflicting, or you couldn't gather one of the analyzers.

# Final output
After `score_recommendation`, your final assistant turn must end with a fenced JSON block matching this schema exactly:

```json
{{
  "recommendation": "RETAIN | REHOST | REFACTOR | REWRITE | RETIRE",
  "confidence": 0.0,
  "rationale": "Single architect-readable paragraph explaining the call: dominant signals, weak areas, anything that drove the confidence up or down.",
  "signals_summary": {{ "final_score": 0.0, "category_scores": {{ }} }},
  "flagged_for_review": false
}}
```

You may set `signals_summary` to the scorer's `signals_summary` verbatim. You may raise or lower `confidence` from the scorer's value if you have good reason — say so in the rationale. Outside the JSON, briefly state why you made any adjustments.

# Scoring rubric (reference — not hard rules)
The rubric below shows the weights and thresholds a senior architect would typically apply. The scorer uses these mechanically; you should weigh them with judgment. Override the threshold-based call only when the signals genuinely warrant it.

```json
{rubric_json}
```

# Constraints
- You have at most {max_iterations} tool-calling turns. Budget them.
- The scorer is the canonical scorer — always call it before producing your final JSON. If you cannot gather enough signal to call it confidently, still call it with whatever you have and set `flagged_for_review: true`.
"""


def _build_system_prompt() -> list[dict]:
    """Inline the rubric into the system prompt and mark it cacheable."""
    try:
        with open(RUBRIC_PATH) as f:
            rubric = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        # Fall back to an empty rubric; the scorer surfaces its own error.
        rubric = {"_error": f"failed to load rubric: {e}"}

    text = _SYSTEM_PROMPT_TEMPLATE.format(
        rubric_json=json.dumps(rubric, indent=2),
        max_iterations=MAX_ITERATIONS,
    )

    # Prompt caching: the system prompt is byte-stable across every repo in a
    # batch run, so we mark it ephemeral. The cache is keyed by the rendered
    # prefix; do NOT interpolate timestamps or per-run IDs above this point.
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


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
            return _summarize_fetch(result), False

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

def _summarize_fetch(result: dict) -> dict:
    metadata = result.get("metadata") or {}
    readme = result.get("readme") or {}
    return {
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

def run_agentic_pipeline(repo: str, trace: bool = False) -> dict:
    """
    Run one full agentic pipeline for a single repo.

    Returns a report dict whose top-level fields (recommendation, confidence,
    rationale, flagged_for_review) match what orchestrator.print_report and
    orchestrator.save_report expect, plus an `agentic` block recording the
    full reasoning chain.
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
                system=_build_system_prompt(),
                tools=TOOLS,
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
      - Use Claude's recommendation / confidence / rationale / flagged_for_review
        when present and well-typed; fall back to the scorer otherwise.
      - Always use the scorer's signals_summary (deterministic and complete).
      - Surface the full tool-call trace for inspection.
    """
    claude_json = claude_json or {}

    def _pick(key: str, default: Any) -> Any:
        v = claude_json.get(key)
        return v if v not in (None, "") else default

    return {
        "repo": repo,
        "analyzed_at": datetime.utcnow().isoformat() + "Z",
        "model": MODEL,
        "pipeline_stages": pipeline_stages,
        "recommendation": _pick("recommendation", fallback_score.get("recommendation")),
        "confidence": _pick("confidence", fallback_score.get("confidence")),
        "rationale": _pick("rationale", fallback_score.get("rationale")),
        "signals_summary": fallback_score.get("signals_summary"),
        "flagged_for_review": bool(
            claude_json.get("flagged_for_review")
            if "flagged_for_review" in claude_json
            else fallback_score.get("flagged_for_review")
        ),
        "agentic": {
            "model": MODEL,
            "iterations": iteration,
            "max_iterations_hit": max_iterations_hit,
            "tool_calls": tool_call_log,
            "claude_final_text": final_text,
            "claude_parsed_json_present": bool(claude_json),
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
        _console.print(cost_line)
    else:
        print(f"\n=== {header} ===")
        print(token_line)
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
        help="Print detailed exception tracebacks on failure",
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
            f"model={MODEL}, repos={len(repos)}, trace={args.trace}\n"
        )

    batch_total = _new_cost_accumulator()

    for repo in repos:
        if RICH_AVAILABLE:
            _console.rule(f"[bold]{repo}[/bold]")
        else:
            print(f"\nAnalyzing: {repo}")

        try:
            report = run_agentic_pipeline(repo, trace=args.trace)
        except Exception as e:  # noqa: BLE001 — preserve batch progress on failure
            if args.verbose and RICH_AVAILABLE:
                _console.print_exception()
            report = _error_report(repo, f"{type(e).__name__}: {e}")

        if args.output == "json":
            print(json.dumps(report, indent=2, default=str))
        else:
            print_report(report)
            _print_cost_summary(report)

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
