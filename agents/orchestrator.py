"""
orchestrator.py — App Rationalization Agent
============================================
Entry point for the multi-agent pipeline. Coordinates sub-agents,
aggregates signals, and produces a final 5-R recommendation report.

Usage:
    python agents/orchestrator.py --repo facebook/react
    python agents/orchestrator.py --repo owner/repo --output json
    python agents/orchestrator.py --batch config/sample_repos.txt
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Rich for pretty terminal output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None

# ---------------------------------------------------------------------------
# Sub-agent imports (stubs until each module is built out)
# ---------------------------------------------------------------------------
# These will be filled in as you build each agent. For now, the orchestrator
# calls them and handles ImportError gracefully so you can run it immediately.

def _import_agents():
    agents = {}
    try:
        from github_fetcher import fetch_repo_data
        agents["github_fetcher"] = fetch_repo_data
    except ImportError:
        agents["github_fetcher"] = _stub_agent("github_fetcher")

    try:
        from code_analyzer import analyze as analyze_code
        agents["code_analyzer"] = analyze_code
    except ImportError:
        agents["code_analyzer"] = _stub_agent("code_analyzer")

    try:
        from dependency_scanner import analyze as analyze_deps
        agents["dependency_scanner"] = analyze_deps
    except ImportError:
        agents["dependency_scanner"] = _stub_agent("dependency_scanner")

    try:
        from activity_analyzer import analyze as analyze_activity
        agents["activity_analyzer"] = analyze_activity
    except ImportError:
        agents["activity_analyzer"] = _stub_agent("activity_analyzer")

    try:
        from scorer import score
        agents["scorer"] = score
    except ImportError:
        agents["scorer"] = _stub_scorer

    return agents


def _stub_agent(name: str):
    """Returns a placeholder function for agents not yet implemented."""
    def stub(repo_data: dict) -> dict:
        return {
            "agent": name,
            "status": "not_implemented",
            "signals": {},
            "notes": f"{name} not yet built — returning empty signals"
        }
    stub.__name__ = name
    return stub


def _stub_scorer(aggregated: dict) -> dict:
    """Placeholder scorer until scorer.py is built."""
    return {
        "recommendation": "UNKNOWN",
        "confidence": 0.0,
        "rationale": "scorer.py not yet implemented",
        "signals_summary": aggregated,
        "flagged_for_review": True
    }


# ---------------------------------------------------------------------------
# Core orchestration logic
# ---------------------------------------------------------------------------

def load_env():
    """Load environment variables from .env file."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv optional; user can set env vars directly


def run_pipeline(repo: str, agents: dict, verbose: bool = False) -> dict:
    """
    Execute the full agent pipeline for a single repository.

    Pipeline stages:
        1. Fetch raw repo data from GitHub API
        2. Analyze code characteristics
        3. Scan dependencies
        4. Analyze activity signals
        5. Score and produce recommendation

    Args:
        repo: GitHub repo in 'owner/name' format
        agents: dict of loaded agent functions
        verbose: print detailed signal output

    Returns:
        Final report dict with recommendation, confidence, rationale
    """
    report = {
        "repo": repo,
        "analyzed_at": datetime.utcnow().isoformat() + "Z",
        "pipeline_stages": {},
        "recommendation": None,
        "confidence": None,
        "rationale": None,
        "flagged_for_review": False
    }

    stages = [
        ("fetch",        "Fetching repository data",     "github_fetcher"),
        ("code",         "Analyzing code",               "code_analyzer"),
        ("dependencies", "Scanning dependencies",        "dependency_scanner"),
        ("activity",     "Analyzing activity",           "activity_analyzer"),
    ]

    # -- Stages 1-4: gather signals ------------------------------------------
    repo_data = {"repo": repo}  # shared context passed between agents

    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            for stage_key, description, agent_key in stages:
                task = progress.add_task(f"[cyan]{description}...", total=None)
                try:
                    result = agents[agent_key](repo_data)
                    repo_data[stage_key] = result
                    report["pipeline_stages"][stage_key] = result
                    progress.update(task, description=f"[green]✓ {description}")
                except Exception as e:
                    progress.update(task, description=f"[red]✗ {description} — {e}")
                    report["pipeline_stages"][stage_key] = {"error": str(e)}
                    if verbose:
                        console.print_exception()
                progress.stop_task(task)
    else:
        for stage_key, description, agent_key in stages:
            print(f"  → {description}...")
            try:
                result = agents[agent_key](repo_data)
                repo_data[stage_key] = result
                report["pipeline_stages"][stage_key] = result
            except Exception as e:
                print(f"    ERROR: {e}")
                report["pipeline_stages"][stage_key] = {"error": str(e)}

    # -- Stage 5: score -------------------------------------------------------
    try:
        score_result = agents["scorer"](report["pipeline_stages"])
        report.update({
            "recommendation": score_result.get("recommendation"),
            "confidence":     score_result.get("confidence"),
            "rationale":      score_result.get("rationale"),
            "flagged_for_review": score_result.get("flagged_for_review", False)
        })
    except Exception as e:
        report["recommendation"] = "ERROR"
        report["rationale"] = str(e)
        report["flagged_for_review"] = True

    return report


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

RECOMMENDATION_COLORS = {
    "RETAIN":   "bold green",
    "REHOST":   "bold blue",
    "REFACTOR": "bold yellow",
    "REWRITE":  "bold magenta",
    "RETIRE":   "bold red",
    "UNKNOWN":  "bold white",
    "ERROR":    "bold red",
}

def print_report(report: dict):
    """Pretty-print the final report to terminal."""
    rec = report.get("recommendation", "UNKNOWN")
    confidence = report.get("confidence", 0.0)
    confidence_pct = f"{confidence * 100:.0f}%" if confidence is not None else "N/A"

    if RICH_AVAILABLE:
        color = RECOMMENDATION_COLORS.get(rec, "white")
        panel_content = (
            f"[{color}]Recommendation: {rec}[/{color}]\n"
            f"Confidence: {confidence_pct}\n\n"
            f"[italic]{report.get('rationale', 'No rationale provided')}[/italic]"
        )
        if report.get("flagged_for_review"):
            panel_content += "\n\n[bold yellow]⚠ Flagged for human review[/bold yellow]"

        console.print(Panel(
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
        print(f"  Rationale:      {report.get('rationale')}")
        if report.get("flagged_for_review"):
            print("  ⚠ Flagged for human review")
        print(f"{'='*60}\n")


def save_report(report: dict, output_dir: str = "outputs/reports"):
    """Save report as JSON to the outputs directory."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    repo_safe = report["repo"].replace("/", "__")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = path / f"{repo_safe}__{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(report, f, indent=2)

    return filename


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="App Rationalization Agent — evaluate GitHub repos against the 5 Rs"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--repo",
        help="GitHub repository in owner/name format (e.g. facebook/react)"
    )
    group.add_argument(
        "--batch",
        help="Path to a text file with one repo per line"
    )
    parser.add_argument(
        "--output",
        choices=["pretty", "json"],
        default="pretty",
        help="Output format (default: pretty)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save report JSON to outputs/reports/"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed signal output from each agent"
    )
    return parser.parse_args()


def main():
    load_env()
    args = parse_args()
    agents = _import_agents()

    repos = []
    if args.repo:
        repos = [args.repo]
    elif args.batch:
        with open(args.batch) as f:
            repos = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if RICH_AVAILABLE:
        console.print(f"\n[bold cyan]App Rationalization Agent[/bold cyan] — analyzing {len(repos)} repo(s)\n")

    for repo in repos:
        if RICH_AVAILABLE:
            console.rule(f"[bold]{repo}[/bold]")
        else:
            print(f"\nAnalyzing: {repo}")

        report = run_pipeline(repo, agents, verbose=args.verbose)

        if args.output == "json":
            print(json.dumps(report, indent=2))
        else:
            print_report(report)

        if args.save:
            saved_path = save_report(report)
            if RICH_AVAILABLE:
                console.print(f"[dim]Report saved → {saved_path}[/dim]")
            else:
                print(f"Report saved → {saved_path}")


if __name__ == "__main__":
    main()
