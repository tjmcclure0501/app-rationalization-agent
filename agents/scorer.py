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
#
# REVIEW is the "we don't know enough to say" outcome. It sits past RETIRE
# in the demotion ladder so that *any* current value will demote to REVIEW
# when missing-signal warnings fire. A REVIEW report must never be treated
# as a portfolio decision — it's a request for human investigation.
_REC_ORDER = ("RETAIN", "REHOST", "REFACTOR", "REWRITE", "RETIRE", "REVIEW")


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
    # Pre-scoring classification (if the orchestrator ran repo_classifier).
    # Used to suppress override rules whose underlying signals are
    # structurally inapplicable to non-APPLICATION repo types.
    repo_type = (pipeline_stages.get("classification") or {}).get("repo_type")
    overrides_applied, recommendation, review_reasons = _apply_overrides(
        base_recommendation, flat_signals, category_results, confidence,
        repo_type=repo_type,
    )

    # --- 6. Decide flagged_for_review ----------------------------------------
    # The review_reasons list collected by _apply_overrides already covers
    # `confidence < flag_for_review_below` and `conflicting_signals_detected`.
    # Add the recommendation-tier min_confidence check here — it's not an
    # override rule, it's a per-tier confidence floor from the rubric.
    rec_min_confidence = (
        _RUBRIC.get("thresholds", {})
        .get(recommendation.lower(), {})
        .get("min_confidence", 0.0)
    )
    if confidence < rec_min_confidence:
        review_reasons.append(
            f"confidence {confidence:.0%} is below the {recommendation} tier's "
            f"min_confidence ({rec_min_confidence:.0%})"
        )

    flagged_for_review = bool(review_reasons)

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
        review_reasons=review_reasons,
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
        "overrides_applied":         overrides_applied,
        "flagged_for_review":        flagged_for_review,
        "flagged_for_review_reasons": review_reasons,
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


def _suppressed_rules_for(repo_type: Optional[str]) -> frozenset[str]:
    """
    Return the set of override rule full-names suppressed for this repo
    type per the rubric's `repo_type_overrides` section. Names are the
    same `force_retire_if: ...` / `force_rewrite_if: ...` strings
    emitted by _unevaluable_override_warnings.
    """
    if not repo_type:
        return frozenset()
    section = _RUBRIC.get("repo_type_overrides") or {}
    return frozenset(
        (section.get(repo_type) or {}).get("suppress_rules") or []
    )


def _apply_overrides(
    base_recommendation: str,
    signals: dict,
    category_results: dict,
    confidence: float,
    repo_type: Optional[str] = None,
) -> tuple[list[dict], str, list[str]]:
    """
    Evaluate the rubric's override rules. Retire/rewrite overrides can
    *demote* the recommendation; review overrides populate the
    review_reasons list. Each appended override carries both the raw
    rule text and a plain-English `reason` so downstream consumers
    (rationale, print_report) can surface what actually happened.

    Order: force_retire_if → force_rewrite_if → force_review_if. Retire
    takes precedence over rewrite (rewrite is skipped if retire fired);
    review never changes the recommendation, only flags it.

    Returns (overrides_applied, recommendation, review_reasons).
    """
    applied: list[dict] = []
    recommendation = base_recommendation
    review_reasons: list[str] = []

    # --- Suppress rules whose signals are structurally inapplicable --------
    # For non-APPLICATION repo types (DOCUMENTATION, CONFIGURATION, DATA)
    # the rubric names specific override rules that must not fire. We
    # record each suppression in `applied` for the audit trail.
    suppressed = _suppressed_rules_for(repo_type)
    for rule_name in sorted(suppressed):
        applied.append({
            "rule": rule_name,
            "action": "suppressed_by_repo_type",
            "reason": (
                f"rule suppressed for repo_type={repo_type} — signal "
                f"is structurally inapplicable per scoring_rubric.json "
                f"repo_type_overrides"
            ),
        })

    # --- Classify any unevaluable rules into demoting vs informational ------
    # Surface every unevaluable rule as a review reason so the gap is visible,
    # but only DEMOTING warnings — those where the missing signal is the only
    # thing preventing the rule from firing — count toward REVIEW demotion.
    # Suppressed rules are skipped entirely (their signals are not "missing",
    # they're not applicable).
    demoting_warnings, informational_warnings = _unevaluable_override_warnings(
        signals, suppressed=suppressed
    )
    review_reasons.extend(demoting_warnings)
    review_reasons.extend(informational_warnings)

    # --- force_retire_if -----------------------------------------------------
    retire_fired = False
    for rule in _check_force_retire(signals, suppressed=suppressed):
        applied.append({"rule": rule, "action": "force_retire", "reason": rule})
        recommendation = _demote_to(recommendation, "RETIRE")
        retire_fired = True

    # --- force_rewrite_if ----------------------------------------------------
    rewrite_fired = False
    if recommendation != "RETIRE":
        for rule in _check_force_rewrite(signals, suppressed=suppressed):
            applied.append({"rule": rule, "action": "force_rewrite", "reason": rule})
            recommendation = _demote_to(recommendation, "REWRITE")
            rewrite_fired = True

    # --- Conditional demote-to-REVIEW ---------------------------------------
    # Demote ONLY when:
    #   (a) at least one DEMOTING warning exists (a rule where the missing
    #       signal could have changed the outcome), AND
    #   (b) no retire/rewrite override has already fired definitively.
    # Informational warnings alone do not trigger REVIEW.
    if demoting_warnings and not retire_fired and not rewrite_fired:
        applied.append({
            "rule": "no_override_fired_with_demoting_unevaluable_rules",
            "action": "demote_to_review",
            "reason": (
                "no force_retire_if/force_rewrite_if rule fired, and one or "
                "more rules could not be evaluated due to missing signals "
                "that could have caused them to fire — threshold-based "
                "recommendation cannot be trusted"
            ),
        })
        recommendation = "REVIEW"

    # --- force_review_if: confidence below the rubric's flag threshold -------
    review_threshold = float(
        _RUBRIC.get("confidence", {}).get("flag_for_review_below", 0.55)
    )
    if confidence < review_threshold:
        reason = (
            f"confidence {confidence:.0%} is below the "
            f"flag_for_review_below threshold ({review_threshold:.0%})"
        )
        applied.append({
            "rule": f"confidence < {review_threshold}",
            "action": "force_review",
            "reason": reason,
        })
        review_reasons.append(reason)

    # --- force_review_if: category scores point in different directions -----
    spread = _category_score_spread(category_results)
    if spread is not None and spread["delta"] > 0.5:
        reason = (
            f"conflicting category scores — "
            f"{spread['lowest_cat']} at {spread['lowest_score']:.2f} "
            f"vs {spread['highest_cat']} at {spread['highest_score']:.2f} "
            f"(spread {spread['delta']:.2f} > 0.50)"
        )
        applied.append({
            "rule": "conflicting_signals_detected == true",
            "action": "force_review",
            "reason": reason,
        })
        review_reasons.append(reason)

    return applied, recommendation, review_reasons


# --- Override rule definitions ------------------------------------------
# A single declarative table that drives both:
#   - rule evaluation at runtime (_check_force_retire/_check_force_rewrite)
#   - missing-signal classification (_unevaluable_override_warnings)
# Keeping these in one place prevents drift between "what rules fire" and
# "what signals the rules depend on" — a previous bug source.
#
# Each entry: (kind, short_name, [(signal_name, op, target), ...])
# All conditions in a rule are AND'd. `op` is one of "==", "<", ">".
# Equality compares values directly (works for both strings and numbers);
# "<" and ">" coerce both sides to float.
_OVERRIDE_RULES: tuple = (
    ("force_retire",  "last_commit_days > 1095",
        [("last_commit_days", ">", 1095)]),
    ("force_retire",  "contributor_count == 1 AND last_commit_days > 365",
        [("contributor_count", "==", 1), ("last_commit_days", ">", 365)]),
    ("force_retire",  "lines_of_code < 50",
        [("lines_of_code", "<", 50)]),
    ("force_rewrite", "runtime_eol == eol AND dependency_freshness < 0.3",
        [("runtime_eol", "==", "eol"), ("dependency_freshness", "<", 0.3)]),
    ("force_rewrite", "has_tests == no AND lines_of_code > 50000",
        [("has_tests", "==", "no"), ("lines_of_code", ">", 50000)]),
)


def _eval_condition(signal_name: str, op: str, target: Any, signals: dict) -> str:
    """
    Evaluate one rule condition against the signal namespace.

    Returns:
        "pass"    — condition holds
        "fail"    — condition definitively does not hold (rule cannot fire)
        "unknown" — signal is absent or wrong type; outcome indeterminate
    """
    v = signals.get(signal_name)
    if v is None:
        return "unknown"
    if op == "==":
        return "pass" if v == target else "fail"
    try:
        v_num = float(v)
        t_num = float(target)
    except (TypeError, ValueError):
        return "unknown"
    if op == ">":
        return "pass" if v_num > t_num else "fail"
    if op == "<":
        return "pass" if v_num < t_num else "fail"
    return "unknown"


def _check_force_retire(
    signals: dict, suppressed: frozenset[str] = frozenset()
) -> list[str]:
    """Return the short names of every force_retire_if rule that fires.

    Rules in `suppressed` (passed as full names, e.g. 'force_retire_if:
    lines_of_code < 50') are skipped regardless of whether the
    underlying conditions hold."""
    return [
        name for kind, name, conds in _OVERRIDE_RULES
        if kind == "force_retire"
        and f"{kind}_if: {name}" not in suppressed
        and all(_eval_condition(s, op, t, signals) == "pass" for s, op, t in conds)
    ]


def _check_force_rewrite(
    signals: dict, suppressed: frozenset[str] = frozenset()
) -> list[str]:
    """Return the short names of every force_rewrite_if rule that fires."""
    return [
        name for kind, name, conds in _OVERRIDE_RULES
        if kind == "force_rewrite"
        and f"{kind}_if: {name}" not in suppressed
        and all(_eval_condition(s, op, t, signals) == "pass" for s, op, t in conds)
    ]


def _unevaluable_override_warnings(
    signals: dict, suppressed: frozenset[str] = frozenset()
) -> tuple[list[str], list[str]]:
    """
    Walk the override rule table and classify any rule that contains an
    unknown (missing) condition into one of two buckets:

      - **demoting**: every known condition passes, so the missing signal
        is the *only* thing preventing the rule from firing. Outcome is
        genuinely uncertain — this is the case that warrants a REVIEW
        demotion when no other override has fired.

      - **informational**: at least one *other* known condition already
        evaluates to "fail", so the AND rule cannot fire regardless of
        what the missing signal is. Worth surfacing as a transparency
        warning, but does NOT justify demoting the recommendation.

    Returns (demoting_warnings, informational_warnings).
    """
    demoting: list[str] = []
    informational: list[str] = []

    for kind, name, conditions in _OVERRIDE_RULES:
        full_name = f"{kind}_if: {name}"
        if full_name in suppressed:
            # Rule is structurally inapplicable to this repo type —
            # missing signals are not "unknown", they're irrelevant.
            continue
        statuses = [
            (s, _eval_condition(s, op, t, signals))
            for s, op, t in conditions
        ]
        if not any(status == "unknown" for _, status in statuses):
            continue  # rule is fully evaluable — no warning needed
        missing = [s for s, status in statuses if status == "unknown"]
        any_fail = any(status == "fail" for _, status in statuses)
        prefix = f"{kind}_if: {name}"
        msg_missing = "missing signal(s): " + ", ".join(missing)

        if any_fail:
            informational.append(
                f"{prefix} could not be fully evaluated — {msg_missing} "
                "(rule cannot fire — another condition in the AND already fails)"
            )
        else:
            demoting.append(
                f"{prefix} could not be fully evaluated — {msg_missing} "
                "(rule could fire if signal were present — outcome genuinely uncertain)"
            )

    return demoting, informational


def _category_score_spread(category_results: dict) -> Optional[dict]:
    """
    Inspect the per-category scores and return the strongest/weakest pair
    plus the absolute delta — so callers can both make the
    conflicting_signals decision *and* explain which categories caused it.

    Returns None when fewer than two categories produced a score.
    """
    scored: list[tuple[str, float]] = [
        (cat, r["score"])
        for cat, r in category_results.items()
        if r.get("score") is not None
    ]
    if len(scored) < 2:
        return None
    scored.sort(key=lambda x: x[1])
    lowest_cat, lowest_score = scored[0]
    highest_cat, highest_score = scored[-1]
    return {
        "lowest_cat":    lowest_cat,
        "lowest_score":  lowest_score,
        "highest_cat":   highest_cat,
        "highest_score": highest_score,
        "delta":         highest_score - lowest_score,
    }


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
    review_reasons: list[str],
) -> str:
    """
    Single paragraph an architect can act on: the call, the strongest /
    weakest dimensions, salient signals, any overrides that influenced
    the outcome, and — if the report is flagged for review — exactly why.
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

    # Render overrides using the descriptive `reason` (falling back to `rule`
    # for compatibility), so e.g. force_review entries surface the actual
    # numbers instead of an opaque "conflicting_signals_detected == true".
    if overrides:
        override_text = "; ".join(o.get("reason") or o["rule"] for o in overrides)
        parts.append(f"Overrides triggered: {override_text}.")

    # Dedicated, prominent sentence whenever the report is flagged for review,
    # so an architect skimming the rationale immediately sees *why*.
    if review_reasons:
        parts.append("Flagged for human review: " + "; ".join(review_reasons) + ".")

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
        "flagged_for_review_reasons": [
            f"scoring rubric could not be loaded from {RUBRIC_PATH}"
        ],
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


# ---------------------------------------------------------------------------
# Regression trace — php-fig/fig-standards
# ---------------------------------------------------------------------------
# Reference scenario used to verify the override pipeline. If a future
# refactor breaks the force_retire_if pathway, the deterministic trace
# below must still produce the expected output.
#
# Repo: php-fig/fig-standards — a finished PHP specification repository.
# No active development since ~2022; multiple historical contributors;
# composer.json pins a current PHP version; modest LOC (mostly markdown).
#
# Representative signals reaching the scorer (after all analyzers run):
#     pipeline_stages = {
#         "fetch":        { "signals": {} },
#         "code":         { "signals": {
#             "has_tests": "partial",
#             "has_readme": "detailed",
#             "has_ci": "yes",
#             "has_license": "yes",
#             "lines_of_code": 8_000,
#             "architecture_signals": "modular",
#         }},
#         "dependencies": { "signals": {
#             "runtime_eol": "current",
#             "dependency_freshness": 0.85,
#             "open_issues_ratio": 0.15,
#         }},
#         "activity":     { "signals": {
#             "last_commit_days": 1500,              # ~4 years dormant
#             "commit_frequency_per_month": 0.5,
#             "contributor_count": 30,
#         }},
#     }
#
# Expected execution path through score():
#   1. _score_category() runs for each rubric category. Activity scores
#      ~0.38 (low recency + low frequency offset by 30 contributors).
#      Health, quality, complexity all score high.
#   2. _aggregate_final_score() → weighted mean ≈ 0.74.
#   3. _map_to_recommendation(0.74) → "REHOST" (≥ 0.55, < 0.75).
#      ↑ This is base_recommendation. It is NOT the final answer.
#   4. _compute_confidence() → ~92% (all categories produce a score).
#   5. _apply_overrides("REHOST", signals, ...):
#        a. _unevaluable_override_warnings → [] (all signals present).
#        b. _check_force_retire(signals):
#             last_commit_days=1500 > 1095        →  rule fires
#             contributor_count=30, not == 1      →  rule does not fire
#             lines_of_code=8000, not < 50        →  rule does not fire
#           → ["last_commit_days > 1095"]
#           → recommendation = _demote_to("REHOST", "RETIRE") = "RETIRE"
#        c. _check_force_rewrite skipped (recommendation == "RETIRE").
#        d. force_review_if: category spread > 0.5 likely fires
#           (activity ~0.38 vs complexity ~1.00 = 0.62 spread).
#           → adds review_reasons entry, does not change recommendation.
#   6. Return:
#        recommendation:       "RETIRE"           (from override)
#        base_recommendation:  "REHOST"           (from threshold mapping)
#        flagged_for_review:   True               (conflicting signals)
#        flagged_for_review_reasons: ["conflicting category scores — …"]
#        overrides_applied:    [{ action: "force_retire", reason: "…" },
#                               { action: "force_review",  reason: "…" }]
#
# Second documented case — php-fig/fig-standards (docs-only repo):
#   Signals reaching the scorer:
#       last_commit_days=93, contributor_count=1, lines_of_code=0,
#       runtime_eol=None, dependency_freshness=None, has_tests="no"
#
#   _unevaluable_override_warnings classifies the rewrite rule
#   `runtime_eol == eol AND dependency_freshness < 0.3` as DEMOTING:
#   both conditions are unknown, neither already-failing, so the rule
#   could fire if the signals were present.
#   _check_force_retire:
#       last_commit_days=93 > 1095          → False
#       contributor_count==1 AND 93 > 365   → False (last_commit_days fails)
#       lines_of_code=0 < 50                → TRUE → recommendation = RETIRE
#   Because retire_fired is True, the demoting warning is surfaced as a
#   review_reasons entry but does NOT trigger the demote-to-REVIEW
#   branch. Final: recommendation = RETIRE, flagged_for_review = True.
#
# Third documented case — over-conservative REVIEW (the bug behind
# requiring _unevaluable_override_warnings to differentiate by impact):
#   Imagine a healthy active repo missing only runtime_eol but with
#   dependency_freshness = 0.95. The rewrite rule
#   `runtime_eol == eol AND dependency_freshness < 0.3` has:
#       runtime_eol            → unknown
#       0.95 < 0.3             → FAIL
#   The AND already cannot fire regardless of runtime_eol. This is
#   classified as INFORMATIONAL, not DEMOTING — it surfaces in
#   review_reasons for transparency but does NOT cause a REVIEW
#   demotion. Without this distinction the previous code over-demoted
#   healthy repos to REVIEW any time runtime_eol was absent (a common
#   tooling gap for Node projects that pin in .nvmrc rather than
#   package.json — which dependency_scanner now also reads).
#
# Failure mode behind the original RETAIN-at-82% bug report:
#   If `last_commit_days` is missing from pipeline_stages["activity"]
#   (e.g. activity_analyzer never ran), the single-condition rule
#   `last_commit_days > 1095` has no partner condition to disqualify
#   it → DEMOTING warning. If no retire/rewrite rule fires, demote to
#   REVIEW. The base high-score RETAIN can no longer pass through
#   silently — the report comes back REVIEW with explicit warnings.
