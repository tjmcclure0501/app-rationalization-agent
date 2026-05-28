"""
scorer.py — App Rationalization Agent
=====================================
Final stage of the pipeline. Consumes the aggregated `pipeline_stages`
dict produced by the orchestrator (github_fetcher → code_analyzer →
dependency_scanner → activity_analyzer) and produces a 5-R recommendation.

Decision flow:
    1. For each rubric category (activity, health, quality, complexity),
       pull the signals from the corresponding stage and score them via
       the score_map buckets in config/scoring_rubric.json.
    2. Weight per-category scores by category_weights → final_score.
    3. Map final_score to a recommendation via the rubric `thresholds`.
    4. Apply override rules (force_retire_if, force_rewrite_if,
       force_review_if) which can re-route the recommendation.
    5. Compute a confidence value from signal coverage (missing signals,
       low_signal stages, and stub agents reduce confidence). Apply the
       low_signal_penalty when coverage drops below 60%.
    6. Flag for human review if confidence falls below the rubric's
       flag_for_review_below threshold or any review override fired.

Public entrypoint matches the orchestrator's import contract:
    from scorer import score
    result = score(pipeline_stages)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

RUBRIC_PATH = Path(__file__).resolve().parent.parent / "config" / "scoring_rubric.json"

# Loaded once at import (per user requirement); falls back to an empty
# rubric only if the file is missing, in which case downstream calls
# return UNKNOWN with confidence 0 rather than crashing the pipeline.
try:
    with open(RUBRIC_PATH) as _f:
        _RUBRIC: dict = json.load(_f)
except (OSError, json.JSONDecodeError):
    _RUBRIC = {}

# Which pipeline stage owns which rubric category. Quality and complexity
# both come from code_analyzer (it fills both rubric sections).
_CATEGORY_TO_STAGE = {
    "activity":   "activity",
    "health":     "dependencies",
    "quality":    "code",
    "complexity": "code",
}

# Where each category's signals live in the rubric file.
_CATEGORY_RUBRIC_KEY = {
    "activity":   "activity_signals",
    "health":     "health_signals",
    "quality":    "quality_signals",
    "complexity": "complexity_signals",
}

# Recommendations in score-descending order — used both for threshold
# resolution and to constrain override re-routing (overrides only push
# *down* the ladder, never up).
_REC_ORDER = ("RETAIN", "REHOST", "REFACTOR", "REWRITE", "RETIRE")


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def score(pipeline_stages: dict) -> dict:
    """
    Produce a 5-R recommendation from the orchestrator's aggregated stages.

    Args:
        pipeline_stages: dict keyed by stage name ("fetch", "code",
                         "dependencies", "activity"). Each value is the
                         dict returned by the corresponding agent.

    Returns:
        dict with keys recommendation, confidence, rationale,
        signals_summary, overrides_applied, flagged_for_review.
    """
    if not _RUBRIC:
        return _empty_rubric_result()

    # --- 1. Score each rubric category from raw signals ----------------------
    category_results: dict[str, dict] = {}
    for category in _CATEGORY_TO_STAGE:
        category_results[category] = _score_category(category, pipeline_stages)

    # --- 2. Weighted final score across categories ---------------------------
    final_score = _aggregate_final_score(category_results)

    # --- 3. Threshold-based recommendation -----------------------------------
    base_recommendation = _map_to_recommendation(final_score)

    # --- 4. Confidence + low_signal_penalty ----------------------------------
    confidence, confidence_factors = _compute_confidence(category_results)

    # --- 5. Apply overrides (can re-route the recommendation, force review) -
    flat_signals = _flatten_signals(pipeline_stages)
    overrides_applied, recommendation, force_review_override = _apply_overrides(
        base_recommendation, flat_signals, category_results, confidence
    )

    # --- 6. Decide flagged_for_review ----------------------------------------
    flag_threshold = _RUBRIC.get("confidence", {}).get("flag_for_review_below", 0.55)
    rec_min_confidence = (
        _RUBRIC.get("thresholds", {})
        .get(recommendation.lower(), {})
        .get("min_confidence", 0.0)
    )
    flagged_for_review = bool(
        force_review_override
        or confidence < flag_threshold
        or confidence < rec_min_confidence
    )

    # --- 7. Build the human-readable rationale -------------------------------
    rationale = _build_rationale(
        recommendation=recommendation,
        base_recommendation=base_recommendation,
        final_score=final_score,
        category_results=category_results,
        confidence=confidence,
        confidence_factors=confidence_factors,
        overrides=overrides_applied,
        signals=flat_signals,
    )

    return {
        "recommendation":     recommendation,
        "confidence":         round(confidence, 4),
        "rationale":          rationale,
        "signals_summary": {
            "final_score":      round(final_score, 4),
            "base_recommendation": base_recommendation,
            "category_scores":  {
                c: (round(r["score"], 4) if r["score"] is not None else None)
                for c, r in category_results.items()
            },
            "category_weights": _RUBRIC.get("category_weights", {}),
            "per_signal":       {
                c: r["scored"] for c, r in category_results.items()
            },
        },
        "overrides_applied":  overrides_applied,
        "flagged_for_review": flagged_for_review,
    }


# ---------------------------------------------------------------------------
# Category scoring
# ---------------------------------------------------------------------------

def _score_category(category: str, pipeline_stages: dict) -> dict:
    """
    Score every signal in one rubric category from its source stage.

    Returns a dict with the weighted category score, per-signal detail,
    the list of missing signals, and stub/low_signal flags so the
    confidence calculation can use them downstream.
    """
    stage_key = _CATEGORY_TO_STAGE[category]
    stage = pipeline_stages.get(stage_key) or {}
    stage_signals = stage.get("signals") or {}
    rubric_section = _RUBRIC.get(_CATEGORY_RUBRIC_KEY[category], {})

    scored: dict[str, dict] = {}
    weighted_sum = 0.0
    used_weight = 0.0
    missing: list[str] = []

    for signal_name, config in rubric_section.items():
        weight = float(config.get("weight", 0))
        value = stage_signals.get(signal_name)
        signal_score = _score_from_map(value, config.get("score_map", {}))
        scored[signal_name] = {
            "value": value,
            "score": signal_score,
            "weight": weight,
        }
        if signal_score is not None:
            weighted_sum += signal_score * weight
            used_weight += weight
        else:
            missing.append(signal_name)

    # Re-normalize over used_weight so missing signals don't pull the
    # category toward zero — that's confidence's job, not score's.
    category_score = (
        round(weighted_sum / used_weight, 4) if used_weight > 0 else None
    )

    return {
        "score":      category_score,
        "scored":     scored,
        "missing":    missing,
        "stub":       stage.get("status") == "not_implemented",
        "low_signal": bool(stage.get("low_signal")),
        "errored":    "error" in stage,
    }


def _score_from_map(value: Any, score_map: dict) -> Optional[float]:
    if value is None:
        return None
    # Categorical first — covers strings like "yes", "modular", "eol".
    if isinstance(value, str) and value in score_map:
        return float(score_map[value])
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    for bucket, bucket_score in score_map.items():
        if _value_in_bucket(v, bucket):
            return float(bucket_score)
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
# Final score aggregation
# ---------------------------------------------------------------------------

def _aggregate_final_score(category_results: dict) -> float:
    """
    Weighted mean of category scores, normalized over categories that
    actually produced a score (missing categories are absorbed by the
    confidence calculation instead).
    """
    weights = _RUBRIC.get("category_weights", {})
    weighted_sum = 0.0
    used_weight = 0.0
    for category, result in category_results.items():
        cat_score = result.get("score")
        if cat_score is None:
            continue
        w = float(weights.get(category, 0))
        weighted_sum += cat_score * w
        used_weight += w
    return weighted_sum / used_weight if used_weight > 0 else 0.0


def _map_to_recommendation(final_score: float) -> str:
    """Pick the highest-tier recommendation whose min_score is satisfied."""
    thresholds = _RUBRIC.get("thresholds", {})
    ranked = sorted(
        thresholds.items(),
        key=lambda kv: kv[1].get("min_score", 0.0),
        reverse=True,
    )
    for name, config in ranked:
        if final_score >= float(config.get("min_score", 0.0)):
            return name.upper()
    return "RETIRE"


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

def _compute_confidence(category_results: dict) -> tuple[float, list[str]]:
    """
    Confidence reflects signal *coverage*: how much of the rubric we
    actually populated. Stub agents contribute zero coverage; low_signal
    stages contribute at half weight.
    """
    factors: list[str] = []
    expected = 0
    present = 0.0

    for category, result in category_results.items():
        rubric_section = _RUBRIC.get(_CATEGORY_RUBRIC_KEY[category], {})
        expected += len(rubric_section)

        if result["stub"]:
            factors.append(f"{category}: source agent not implemented")
            continue
        if result["errored"]:
            factors.append(f"{category}: source agent errored")
            continue

        scored = result["scored"]
        category_present = sum(
            1 for s in scored.values() if s["score"] is not None
        )

        # low_signal stages contributed partial info — count at half coverage
        # so the scorer doesn't take their numbers at face value.
        weight_factor = 0.5 if result["low_signal"] else 1.0
        present += category_present * weight_factor

        if result["low_signal"]:
            factors.append(f"{category}: source agent flagged low_signal")
        if result["missing"]:
            factors.append(
                f"{category}: missing {', '.join(result['missing'])}"
            )

    coverage = (present / expected) if expected > 0 else 0.0
    confidence = coverage

    # Per the rubric: penalize when fewer than 60% of expected signals
    # are present (low coverage = unreliable recommendation).
    penalty = float(_RUBRIC.get("confidence", {}).get("low_signal_penalty", 0.15))
    if coverage < 0.60:
        confidence -= penalty
        factors.append(
            f"low_signal_penalty applied ({coverage:.0%} coverage < 60%)"
        )

    confidence = max(0.0, min(1.0, confidence))
    return confidence, factors


# ---------------------------------------------------------------------------
# Override rules
# ---------------------------------------------------------------------------

def _flatten_signals(pipeline_stages: dict) -> dict[str, Any]:
    """Merge signals from every stage into one flat namespace for overrides."""
    merged: dict[str, Any] = {}
    for stage in pipeline_stages.values():
        if not isinstance(stage, dict):
            continue
        for k, v in (stage.get("signals") or {}).items():
            merged[k] = v
    return merged


def _apply_overrides(
    base_recommendation: str,
    signals: dict,
    category_results: dict,
    confidence: float,
) -> tuple[list[dict], str, bool]:
    """
    Evaluate the rubric's override rules. Retire/rewrite overrides can
    *demote* the recommendation; review overrides only set the review
    flag. Returns (overrides_applied, recommendation, force_review).
    """
    applied: list[dict] = []
    recommendation = base_recommendation
    force_review = False

    # --- force_retire_if -----------------------------------------------------
    for rule in _check_force_retire(signals):
        applied.append({"rule": rule, "action": "force_retire"})
        recommendation = _demote_to(recommendation, "RETIRE")

    # --- force_rewrite_if ----------------------------------------------------
    # Skip if we've already been demoted to RETIRE — retire dominates.
    if recommendation != "RETIRE":
        for rule in _check_force_rewrite(signals):
            applied.append({"rule": rule, "action": "force_rewrite"})
            recommendation = _demote_to(recommendation, "REWRITE")

    # --- force_review_if -----------------------------------------------------
    review_threshold = float(
        _RUBRIC.get("confidence", {}).get("flag_for_review_below", 0.55)
    )
    if confidence < review_threshold:
        applied.append({
            "rule": f"confidence < {review_threshold}",
            "action": "force_review",
        })
        force_review = True

    if _conflicting_signals(category_results):
        applied.append({
            "rule": "conflicting_signals_detected == true",
            "action": "force_review",
        })
        force_review = True

    return applied, recommendation, force_review


def _check_force_retire(signals: dict) -> list[str]:
    """force_retire_if rules from the rubric, encoded explicitly."""
    fired: list[str] = []

    last_commit_days = signals.get("last_commit_days")
    if isinstance(last_commit_days, (int, float)) and last_commit_days > 1095:
        fired.append("last_commit_days > 1095")

    contributor_count = signals.get("contributor_count")
    if (
        contributor_count == 1
        and isinstance(last_commit_days, (int, float))
        and last_commit_days > 365
    ):
        fired.append("contributor_count == 1 AND last_commit_days > 365")

    loc = signals.get("lines_of_code")
    if isinstance(loc, (int, float)) and loc < 50:
        fired.append("lines_of_code < 50")

    return fired


def _check_force_rewrite(signals: dict) -> list[str]:
    """force_rewrite_if rules from the rubric, encoded explicitly."""
    fired: list[str] = []

    # The rubric expresses "runtime_eol == true"; our signal is a string
    # ("current"|"lts"|"maintenance"|"eol") — interpret the truthy case as "eol".
    runtime_eol = signals.get("runtime_eol")
    dep_freshness = signals.get("dependency_freshness")
    if (
        runtime_eol == "eol"
        and isinstance(dep_freshness, (int, float))
        and dep_freshness < 0.3
    ):
        fired.append("runtime_eol == eol AND dependency_freshness < 0.3")

    # Similarly, "has_tests == false" → has_tests == "no" in our system.
    has_tests = signals.get("has_tests")
    loc = signals.get("lines_of_code")
    if (
        has_tests == "no"
        and isinstance(loc, (int, float))
        and loc > 50000
    ):
        fired.append("has_tests == no AND lines_of_code > 50000")

    return fired


def _conflicting_signals(category_results: dict) -> bool:
    """
    True when category scores point in very different directions — defined
    as a spread > 0.5 between the strongest and weakest scored category.
    Used by force_review_if to surface ambiguous repos for human review.
    """
    scores = [
        r["score"] for r in category_results.values() if r["score"] is not None
    ]
    if len(scores) < 2:
        return False
    return (max(scores) - min(scores)) > 0.5


def _demote_to(current: str, target: str) -> str:
    """
    Move the recommendation down the ladder toward `target`. Never *up*:
    overrides only express increased concern.
    """
    try:
        return target if _REC_ORDER.index(target) >= _REC_ORDER.index(current) else current
    except ValueError:
        return target


# ---------------------------------------------------------------------------
# Rationale generation
# ---------------------------------------------------------------------------

def _build_rationale(
    recommendation: str,
    base_recommendation: str,
    final_score: float,
    category_results: dict,
    confidence: float,
    confidence_factors: list[str],
    overrides: list[dict],
    signals: dict,
) -> str:
    """
    Single paragraph an architect can act on: the call, the strongest /
    weakest dimensions, salient signals, and any overrides or confidence
    caveats that influenced the outcome.
    """
    parts: list[str] = []

    parts.append(
        f"Recommendation: {recommendation} "
        f"(weighted score {final_score:.2f}, confidence {confidence:.0%})."
    )

    if recommendation != base_recommendation:
        parts.append(
            f"Threshold score would yield {base_recommendation}; "
            f"overrides demoted it to {recommendation}."
        )

    scored_categories = [
        (c, r["score"]) for c, r in category_results.items() if r["score"] is not None
    ]
    if scored_categories:
        scored_categories.sort(key=lambda x: x[1], reverse=True)
        best_name, best_score = scored_categories[0]
        worst_name, worst_score = scored_categories[-1]
        if best_name == worst_name:
            parts.append(
                f"Only {best_name} produced a score ({best_score:.2f})."
            )
        else:
            parts.append(
                f"Strongest dimension: {best_name} ({best_score:.2f}); "
                f"weakest: {worst_name} ({worst_score:.2f})."
            )

    notable = _notable_signals(signals)
    if notable:
        parts.append("Notable signals: " + "; ".join(notable) + ".")

    if overrides:
        rule_text = "; ".join(o["rule"] for o in overrides)
        parts.append(f"Overrides triggered: {rule_text}.")

    if confidence_factors:
        # Top three are enough to give the architect the texture.
        parts.append(
            "Confidence reduced by: " + "; ".join(confidence_factors[:3]) + "."
        )

    return " ".join(parts)


def _notable_signals(signals: dict) -> list[str]:
    """Cherry-pick the few signals most worth surfacing in plain English."""
    notable: list[str] = []

    last = signals.get("last_commit_days")
    if isinstance(last, (int, float)):
        if last <= 30:
            notable.append(f"actively maintained (last commit {int(last)}d ago)")
        elif last > 730:
            notable.append(f"dormant ({int(last)}d since last commit)")

    contributors = signals.get("contributor_count")
    if isinstance(contributors, (int, float)):
        if contributors >= 20:
            notable.append(f"{int(contributors)} contributors")
        elif contributors == 1:
            notable.append("single contributor")

    freq = signals.get("commit_frequency_per_month")
    if isinstance(freq, (int, float)):
        if freq >= 20:
            notable.append(f"{freq:.0f} commits/month")
        elif freq == 0:
            notable.append("no recent commits")

    runtime = signals.get("runtime_eol")
    if runtime == "eol":
        notable.append("runtime EOL")
    elif runtime == "current":
        notable.append("runtime current")

    freshness = signals.get("dependency_freshness")
    if isinstance(freshness, (int, float)):
        if freshness >= 0.9:
            notable.append(f"fresh dependencies ({freshness:.0%})")
        elif freshness < 0.5:
            notable.append(f"stale dependencies ({freshness:.0%})")

    has_tests = signals.get("has_tests")
    if has_tests == "no":
        notable.append("no tests")
    elif has_tests == "yes":
        notable.append("test coverage present")

    has_ci = signals.get("has_ci")
    if has_ci == "no":
        notable.append("no CI configured")

    loc = signals.get("lines_of_code")
    if isinstance(loc, (int, float)):
        if loc > 200_000:
            notable.append(f"large codebase ({int(loc):,} LOC)")
        elif loc < 1000:
            notable.append(f"very small codebase ({int(loc):,} LOC)")

    arch = signals.get("architecture_signals")
    if arch == "monolith":
        notable.append("monolithic structure")
    elif arch == "modular":
        notable.append("modular structure")

    return notable


# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------

def _empty_rubric_result() -> dict:
    return {
        "recommendation": "UNKNOWN",
        "confidence": 0.0,
        "rationale": (
            f"Scoring rubric could not be loaded from {RUBRIC_PATH}. "
            "No recommendation produced."
        ),
        "signals_summary": {
            "final_score": None,
            "base_recommendation": None,
            "category_scores": {},
            "category_weights": {},
            "per_signal": {},
        },
        "overrides_applied": [],
        "flagged_for_review": True,
    }


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

    parser = argparse.ArgumentParser(description="Smoke-test scorer end-to-end")
    parser.add_argument("--repo", required=True, help="owner/name")
    args = parser.parse_args()

    # Wire up the full pipeline locally so we can score a real repo.
    from github_fetcher import fetch_repo_data
    from code_analyzer import analyze as analyze_code
    from dependency_scanner import analyze as analyze_deps
    from activity_analyzer import analyze as analyze_activity

    repo_data = {"repo": args.repo}
    repo_data["fetch"] = fetch_repo_data(repo_data)
    repo_data["code"] = analyze_code(repo_data)
    repo_data["dependencies"] = analyze_deps(repo_data)
    repo_data["activity"] = analyze_activity(repo_data)

    pipeline_stages = {
        "fetch":        repo_data["fetch"],
        "code":         repo_data["code"],
        "dependencies": repo_data["dependencies"],
        "activity":     repo_data["activity"],
    }
    result = score(pipeline_stages)
    print(json.dumps(result, indent=2, default=str))
