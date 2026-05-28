"""
activity_analyzer.py — App Rationalization Agent
================================================
Analyzes the *activity* dimension of a repository for the 5-R rubric:
    - last_commit_days        — recency of work
    - commit_frequency_per_month — current cadence (last ~12 weeks)
    - contributor_count       — breadth of contribution

Reads weights and score buckets from config/scoring_rubric.json
(`activity_signals` section), and produces a per-signal score plus the
weighted category score.

Returned dict shape:
    {
        "agent": "activity_analyzer",
        "signals": {
            "last_commit_at": "2026-04-12T...Z",
            "last_commit_days": 46,
            "commit_frequency_per_month": 12.3,
            "weekly_commits_recent_12w": [3, 4, 2, ...],
            "contributor_count": 87,
        },
        "scored": {
            "last_commit_days":           { "value": 46,  "score": 0.8, "weight": 0.40 },
            "commit_frequency_per_month": { "value": 12,  "score": 0.8, "weight": 0.35 },
            "contributor_count":          { "value": 87,  "score": 1.0, "weight": 0.25 },
        },
        "category_score": 0.86,
        "errors": [ ... ]
    }
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx

# Reuse the centralized HTTP client + retry/rate-limit logic.
# All GitHub API traffic in this project goes through github_fetcher.
from github_fetcher import _build_client, _request

RUBRIC_PATH = Path(__file__).resolve().parent.parent / "config" / "scoring_rubric.json"

# Window used to compute "current" monthly commit cadence. 12 weeks ≈ 3 months
# of recent history — long enough to smooth weekly noise, short enough to
# reflect *current* activity rather than ancient bursts.
FREQUENCY_WINDOW_WEEKS = 12

# /stats/participation returns 202 while GitHub computes weekly counts.
# Retry a few times with backoff before giving up.
STATS_RETRIES = 4
STATS_RETRY_DELAY = 1.5


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def analyze(repo_data: dict) -> dict:
    """
    Score the activity signals for a repository.

    Args:
        repo_data: shared pipeline dict. Must contain `repo` ("owner/name").
                   If `fetch` (github_fetcher output) is present, its
                   `metadata.pushed_at` is used as a fallback for last commit.

    Returns:
        Structured dict (see module docstring).
    """
    repo = repo_data.get("repo")
    if not repo or "/" not in repo:
        raise ValueError(f"Expected 'owner/name' repo, got: {repo!r}")
    owner, name = repo.split("/", 1)

    fetch_data = repo_data.get("fetch") or {}
    pushed_at_fallback = (fetch_data.get("metadata") or {}).get("pushed_at")

    rubric = _load_rubric()
    activity_rubric: dict = rubric.get("activity_signals", {})

    signals: dict[str, Any] = {}
    errors: list[str] = []
    token = os.getenv("GITHUB_TOKEN")

    with _build_client(token) as client:
        # --- Last commit -----------------------------------------------------
        last_commit_iso: Optional[str] = None
        try:
            last_commit_iso = _fetch_last_commit_date(client, owner, name)
        except Exception as e:
            errors.append(f"last_commit: {e}")
        if not last_commit_iso:
            last_commit_iso = pushed_at_fallback

        signals["last_commit_at"] = last_commit_iso
        signals["last_commit_days"] = (
            _days_since(last_commit_iso) if last_commit_iso else None
        )

        # --- Commit frequency (weekly counts → per-month average) ------------
        try:
            weekly = _fetch_participation(client, owner, name)
            recent = weekly[-FREQUENCY_WINDOW_WEEKS:] if weekly else []
            signals["weekly_commits_recent_12w"] = recent
            signals["commit_frequency_per_month"] = (
                _avg_commits_per_month(recent) if recent else None
            )
        except Exception as e:
            errors.append(f"participation: {e}")
            signals["weekly_commits_recent_12w"] = []
            signals["commit_frequency_per_month"] = None

        # --- Contributor count -----------------------------------------------
        try:
            signals["contributor_count"] = _fetch_contributor_count(client, owner, name)
        except Exception as e:
            errors.append(f"contributors: {e}")
            signals["contributor_count"] = None

    # --- Score against the rubric --------------------------------------------
    scored: dict[str, dict] = {}
    weighted_sum = 0.0
    used_weight = 0.0

    for signal_name, config in activity_rubric.items():
        weight = float(config.get("weight", 0))
        raw = signals.get(signal_name)
        score = _score_from_map(raw, config.get("score_map", {})) if raw is not None else None
        scored[signal_name] = {"value": raw, "score": score, "weight": weight}
        if score is not None:
            weighted_sum += score * weight
            used_weight += weight

    category_score = round(weighted_sum / used_weight, 4) if used_weight > 0 else None

    return {
        "agent": "activity_analyzer",
        "signals": signals,
        "scored": scored,
        "category_score": category_score,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Rubric loading + score-map matching
# ---------------------------------------------------------------------------

def _load_rubric() -> dict:
    with open(RUBRIC_PATH) as f:
        return json.load(f)


def _score_from_map(value: Any, score_map: dict) -> Optional[float]:
    """
    Match a numeric value against the rubric's bucket keys.

    Supported key forms:
        "0-30"   → low <= value <= high
        "20+"    → value >= low
        "1"      → value == int(key)
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None

    for bucket, score in score_map.items():
        if _value_in_bucket(v, bucket):
            return float(score)
    return None


def _value_in_bucket(value: float, bucket: str) -> bool:
    bucket = bucket.strip()
    if bucket.endswith("+"):
        try:
            return value >= float(bucket[:-1])
        except ValueError:
            return False
    if "-" in bucket:
        try:
            low_s, high_s = bucket.split("-", 1)
            return float(low_s) <= value <= float(high_s)
        except ValueError:
            return False
    try:
        return value == float(bucket)
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# GitHub endpoint helpers
# ---------------------------------------------------------------------------

def _fetch_last_commit_date(client: httpx.Client, owner: str, name: str) -> Optional[str]:
    """Return the ISO date of the most recent commit on the default branch."""
    resp = _request(client, f"/repos/{owner}/{name}/commits", params={"per_page": "1"})
    if resp.status_code != 200:
        return None
    payload = resp.json()
    if not isinstance(payload, list) or not payload:
        return None
    commit = payload[0].get("commit") or {}
    # Prefer committer date (when the commit landed); fall back to author date.
    committer = commit.get("committer") or {}
    author = commit.get("author") or {}
    return committer.get("date") or author.get("date")


def _fetch_participation(client: httpx.Client, owner: str, name: str) -> list[int]:
    """
    Return the last 52 weeks of commit counts (all contributors).

    GitHub returns 202 while stats are being computed; we retry with backoff.
    """
    path = f"/repos/{owner}/{name}/stats/participation"
    for attempt in range(STATS_RETRIES):
        resp = _request(client, path)
        if resp.status_code == 200:
            data = resp.json()
            return list(data.get("all") or [])
        if resp.status_code == 202:
            time.sleep(STATS_RETRY_DELAY * (attempt + 1))
            continue
        return []
    return []


def _fetch_contributor_count(client: httpx.Client, owner: str, name: str) -> Optional[int]:
    """
    Total contributor count via the Link-header trick.

    Requesting per_page=1 lets GitHub tell us the page count in the Link
    header, which equals the total contributor count — far cheaper than
    paginating through every contributor.
    """
    resp = _request(
        client,
        f"/repos/{owner}/{name}/contributors",
        params={"per_page": "1", "anon": "false"},
    )
    if resp.status_code == 204:
        # GitHub returns 204 No Content for empty repos.
        return 0
    if resp.status_code != 200:
        return None

    link = resp.headers.get("Link")
    last_page = _parse_last_page(link)
    if last_page is not None:
        return last_page

    # No Link header → fewer than 2 pages of 1 result → count the body.
    payload = resp.json()
    return len(payload) if isinstance(payload, list) else None


def _parse_last_page(link_header: Optional[str]) -> Optional[int]:
    """
    Extract the page number from a GitHub Link header's rel="last" entry.

    Example: '<https://api.github.com/...?page=2>; rel="next",
              <https://api.github.com/...?page=87>; rel="last"'
    """
    if not link_header:
        return None
    for part in link_header.split(","):
        if 'rel="last"' not in part:
            continue
        url_segment = part.split(";", 1)[0].strip().strip("<>")
        # Parse `page=` from the query string.
        if "page=" not in url_segment:
            continue
        try:
            page_str = url_segment.split("page=", 1)[1].split("&", 1)[0]
            return int(page_str)
        except (ValueError, IndexError):
            continue
    return None


# ---------------------------------------------------------------------------
# Numeric utilities
# ---------------------------------------------------------------------------

def _days_since(iso_timestamp: str) -> Optional[int]:
    """Whole days between `iso_timestamp` and now (UTC)."""
    if not iso_timestamp:
        return None
    try:
        # fromisoformat handles "2026-04-12T13:45:01Z" on 3.11+; on 3.9+ we
        # need to swap the trailing "Z" for "+00:00".
        normalized = iso_timestamp.replace("Z", "+00:00")
        ts = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - ts
    return max(0, delta.days)


def _avg_commits_per_month(weekly_counts: list[int]) -> float:
    """Convert a window of weekly counts into an average per-month figure."""
    if not weekly_counts:
        return 0.0
    weeks = len(weekly_counts)
    total = sum(weekly_counts)
    # 4.345 weeks/month average.
    return round((total / weeks) * 4.345, 2)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Smoke-test activity_analyzer")
    parser.add_argument("--repo", required=True, help="owner/name")
    args = parser.parse_args()

    result = analyze({"repo": args.repo})
    print(json.dumps(result, indent=2, default=str))
