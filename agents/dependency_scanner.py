"""
dependency_scanner.py — App Rationalization Agent
=================================================
Analyzes the *health* dimension of a repository by inspecting the
dependency manifests pulled by github_fetcher. For each recognized
manifest (package.json, requirements.txt, pyproject.toml, pom.xml,
build.gradle(.kts), go.mod, composer.json) it extracts:

    - total declared dependencies
    - how many appear outdated based on version-pattern heuristics
    - the runtime/language version pin and its EOL status

Aggregates across all manifests into the three `health_signals`
defined in config/scoring_rubric.json:

    - runtime_eol           ("current" | "lts" | "maintenance" | "eol")
    - dependency_freshness  (up_to_date / classified ratio)
    - open_issues_ratio     (best-effort via /issues?state=closed Link header)

The returned dict mirrors the contract of activity_analyzer and
code_analyzer: a `signals` block matching rubric keys, a `scored`
block with per-signal weights, a `category_score` for "health", a
`raw` block with per-manifest detail, and `low_signal` / `errors`
flags so the scorer can flag low-confidence repos for human review.

No exceptions propagate — unparseable manifests are recorded in
`errors` and reduce the available signal rather than failing the run.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Reuse the centralized HTTP client + retry/rate-limit logic.
from github_fetcher import _build_client, _request

RUBRIC_PATH = Path(__file__).resolve().parent.parent / "config" / "scoring_rubric.json"

# Maximum number of outdated dependency examples surfaced per manifest in `raw`.
MAX_OUTDATED_EXAMPLES = 5


# ---------------------------------------------------------------------------
# Known package "current major" lookup tables.
# ---------------------------------------------------------------------------
# These give the dependency classifier a reference point for "what does
# fresh look like?" without external CVE / registry lookups. A dep is
# flagged outdated when its declared major is *more than one* below the
# current major (so being one major behind doesn't penalise — many teams
# pin one version back intentionally).
#
# Snapshot date: 2026-05-28. Update opportunistically; the rubric buckets
# are wide enough that minor staleness here doesn't move the score.

_NPM_CURRENT_MAJOR = {
    "react": 19,
    "react-dom": 19,
    "vue": 3,
    "@angular/core": 19,
    "@angular/cli": 19,
    "svelte": 5,
    "next": 15,
    "nuxt": 3,
    "gatsby": 5,
    "remix": 2,
    "express": 5,
    "fastify": 5,
    "koa": 2,
    "@nestjs/core": 11,
    "typescript": 5,
    "webpack": 5,
    "vite": 6,
    "rollup": 4,
    "jest": 29,
    "mocha": 10,
    "vitest": 2,
    "rxjs": 7,
    "lodash": 4,
    "axios": 1,
    "redux": 5,
    "moment": 2,    # deprecated but still on 2.x — flag for being behind alternatives elsewhere
    "jquery": 3,
    "bootstrap": 5,
    "tailwindcss": 3,
}

_PYPI_CURRENT_MAJOR = {
    "django": 5,
    "flask": 3,
    "tornado": 6,
    "pyramid": 2,
    "bottle": 0,        # historically 0.x by design
    "starlette": 0,     # 0.x by design
    "fastapi": 0,       # 0.x by design
    "sanic": 24,        # CalVer
    "aiohttp": 3,
    "pydantic": 2,
    "sqlalchemy": 2,
    "alembic": 1,
    "celery": 5,
    "numpy": 2,
    "pandas": 2,
    "tensorflow": 2,
    "torch": 2,
    "scikit-learn": 1,
    "requests": 2,
    "httpx": 0,         # 0.x by design (as of mid-2026)
    "pytest": 8,
    "black": 25,        # CalVer
    "ruff": 0,
    "boto3": 1,
}

_MAVEN_CURRENT_MAJOR = {
    "org.springframework:spring-core": 6,
    "org.springframework.boot:spring-boot": 3,
    "org.springframework.boot:spring-boot-starter": 3,
    "org.hibernate:hibernate-core": 6,
    "org.hibernate.orm:hibernate-core": 6,
    "junit:junit": 4,
    "org.junit.jupiter:junit-jupiter": 5,
    "com.fasterxml.jackson.core:jackson-databind": 2,
    "org.slf4j:slf4j-api": 2,
    "ch.qos.logback:logback-classic": 1,
}

_PACKAGIST_CURRENT_MAJOR = {
    "laravel/framework": 11,
    "symfony/symfony": 7,
    "symfony/framework-bundle": 7,
    "codeigniter/framework": 4,
    "yiisoft/yii2": 2,
    "cakephp/cakephp": 5,
}

_GO_CURRENT_MAJOR: dict[str, int] = {
    # Go modules embed major in import path (e.g. /v2). Catching staleness
    # by major here is unreliable, so the table is intentionally empty.
}

# Packages where 0.x is the *expected* current major. Excludes them from
# the "major == 0 means abandoned" heuristic.
_TOLERATED_ZERO_MAJOR = {
    "fastapi", "starlette", "httpx", "anyio", "trio",
    "bottle", "sanic", "ruff",
}


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def analyze(repo_data: dict) -> dict:
    """
    Score dependency health for a repository.

    Args:
        repo_data: shared pipeline dict; must contain `fetch` (from
                   github_fetcher.fetch_repo_data). `repo` is used for
                   the best-effort open_issues_ratio lookup.

    Returns:
        Structured dict (see module docstring).
    """
    fetch = repo_data.get("fetch") or {}
    manifests: dict = fetch.get("manifests") or {}
    metadata: dict = fetch.get("metadata") or {}

    errors: list[str] = []
    by_manifest: dict[str, dict] = {}
    runtime_findings: list[dict] = []

    # Dispatch each manifest to its per-format handler. Look up by
    # lower-cased basename so casing variants (Dockerfile, Gemfile,
    # PackagE.JSON in the wild) all match.
    handlers = {
        "package.json":     _analyze_package_json,
        "requirements.txt": _analyze_requirements_txt,
        "pyproject.toml":   _analyze_pyproject,
        "pipfile":          _analyze_pipfile,
        "pom.xml":          _analyze_pom_xml,
        "build.gradle":     _analyze_gradle,
        "build.gradle.kts": _analyze_gradle,
        "go.mod":           _analyze_gomod,
        "composer.json":    _analyze_composer,
    }

    for orig_name, data in manifests.items():
        handler = handlers.get(orig_name.lower())
        if not handler:
            continue
        content = (data or {}).get("content") or ""
        if not content:
            errors.append(f"{orig_name}: empty or unreadable")
            by_manifest[orig_name] = {"parse_error": "empty content"}
            continue
        try:
            result = handler(content)
        except Exception as e:
            errors.append(f"{orig_name}: {type(e).__name__}: {e}")
            by_manifest[orig_name] = {"parse_error": str(e)}
            continue
        by_manifest[orig_name] = result
        if result.get("runtime"):
            runtime_findings.append(result["runtime"])

    # --- Node runtime fallback: .nvmrc / .node-version ----------------------
    # Many JS projects (facebook/react is the canonical example) pin their
    # Node version in .nvmrc rather than package.json's `engines.node`.
    # Without this fallback, runtime_eol comes back null, which used to
    # silently kill the force_rewrite override checks for those repos.
    if not any(r.get("language") == "node" for r in runtime_findings):
        pin_lookup = {name.lower(): data for name, data in manifests.items()}
        for pin_name in (".nvmrc", ".node-version"):
            pin_data = pin_lookup.get(pin_name)
            if not pin_data:
                continue
            version_str = _parse_node_pin(pin_data.get("content") or "")
            if version_str:
                runtime_findings.append({
                    "language": "node",
                    "version": version_str,
                    "status":   _classify_node(version_str),
                    "source":   pin_name,
                })
                break

    # --- Aggregate signals ----------------------------------------------------
    total_up = sum(m.get("up_to_date", 0) for m in by_manifest.values())
    total_out = sum(m.get("outdated", 0) for m in by_manifest.values())
    total_unknown = sum(m.get("unknown", 0) for m in by_manifest.values())
    classified = total_up + total_out

    dependency_freshness: Optional[float] = (
        round(total_up / classified, 4) if classified > 0 else None
    )

    runtime_eol = _worst_runtime_status(runtime_findings)
    open_issues_ratio = _try_open_issues_ratio(repo_data.get("repo"), metadata, errors)

    # --- Low-signal heuristic -------------------------------------------------
    parseable = [name for name, m in by_manifest.items() if "parse_error" not in m]
    low_signal = (
        not parseable
        or (classified == 0 and runtime_eol is None)
    )

    signals: dict[str, Any] = {
        "runtime_eol":          runtime_eol,
        "dependency_freshness": dependency_freshness,
        "open_issues_ratio":    open_issues_ratio,
    }

    # --- Score against the rubric --------------------------------------------
    rubric = _load_rubric()
    scored, category_score = _score_category(signals, rubric.get("health_signals", {}))

    return {
        "agent": "dependency_scanner",
        "signals": signals,
        "scored": scored,
        "category_score": category_score,
        "raw": {
            "manifests_analyzed":    parseable,
            "manifests_unparseable": [n for n, m in by_manifest.items() if "parse_error" in m],
            "total_dependencies":    total_up + total_out + total_unknown,
            "classified":            classified,
            "up_to_date":            total_up,
            "outdated":              total_out,
            "unknown":               total_unknown,
            "runtime_findings":      runtime_findings,
            "by_manifest":           by_manifest,
        },
        "low_signal": low_signal,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Rubric loading + scoring helpers
# ---------------------------------------------------------------------------

def _load_rubric() -> dict:
    with open(RUBRIC_PATH) as f:
        return json.load(f)


def _score_category(signals: dict, category_rubric: dict) -> tuple[dict, Optional[float]]:
    scored: dict[str, dict] = {}
    weighted_sum = 0.0
    used_weight = 0.0
    for signal_name, config in category_rubric.items():
        weight = float(config.get("weight", 0))
        value = signals.get(signal_name)
        score = _score_from_map(value, config.get("score_map", {}))
        scored[signal_name] = {"value": value, "score": score, "weight": weight}
        if score is not None:
            weighted_sum += score * weight
            used_weight += weight
    category_score = round(weighted_sum / used_weight, 4) if used_weight > 0 else None
    return scored, category_score


def _score_from_map(value: Any, score_map: dict) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str) and value in score_map:
        return float(score_map[value])
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
            low, high = bucket.split("-", 1)
            return float(low) <= value <= float(high)
        except ValueError:
            return False
    try:
        return value == float(bucket)
    except ValueError:
        return False


def _worst_runtime_status(runtime_findings: list[dict]) -> Optional[str]:
    """Pick the most concerning runtime status across all manifests."""
    severity = {"current": 0, "lts": 1, "maintenance": 2, "eol": 3}
    statuses = [r.get("status") for r in runtime_findings if r.get("status")]
    if not statuses:
        return None
    return max(statuses, key=lambda s: severity.get(s, -1))


# ---------------------------------------------------------------------------
# Generic dependency classifier
# ---------------------------------------------------------------------------

_LOOSE_VERSION_LITERALS = {"*", "x", "x.x", "x.x.x", "latest", "any", ""}
_NON_REGISTRY_PREFIXES = (
    "git+", "git://", "git@",
    "file:", "../", "./",
    "http://", "https://",
    "github:", "gitlab:", "bitbucket:",
    "npm:", "workspace:",
)


def _classify_dep(name: str, version: str, ecosystem: str) -> str:
    """Return 'up_to_date' | 'outdated' | 'unknown' for a single dep."""
    if not version or not isinstance(version, str):
        return "unknown"
    v = version.strip().lower()

    if v in _LOOSE_VERSION_LITERALS:
        # Wildcards / "latest" are loose-by-default — a freshness anti-pattern.
        return "outdated"
    if v.startswith(_NON_REGISTRY_PREFIXES):
        return "unknown"

    # Find the leading major-version digit.
    m = re.search(r"(\d+)", v)
    if not m:
        return "unknown"
    major = int(m.group(1))

    known = _current_major_for(name, ecosystem)
    if known is not None:
        # >1 major behind a known-current package is meaningfully stale.
        if known - major > 1:
            return "outdated"
        return "up_to_date"

    # Unknown package, pinned at 0.x: ambiguous. Some libs ship 0.x forever
    # (FastAPI, Starlette). Don't penalise without a tolerated-list hit.
    if major == 0 and name.lower() not in _TOLERATED_ZERO_MAJOR:
        return "unknown"

    return "up_to_date"


def _current_major_for(name: str, ecosystem: str) -> Optional[int]:
    table = {
        "npm": _NPM_CURRENT_MAJOR,
        "pypi": _PYPI_CURRENT_MAJOR,
        "maven": _MAVEN_CURRENT_MAJOR,
        "packagist": _PACKAGIST_CURRENT_MAJOR,
        "go": _GO_CURRENT_MAJOR,
    }.get(ecosystem, {})
    return table.get(name.lower())


# ---------------------------------------------------------------------------
# Per-manifest handlers
# ---------------------------------------------------------------------------

def _empty_result() -> dict:
    return {
        "dependency_count": 0,
        "up_to_date": 0,
        "outdated": 0,
        "unknown": 0,
        "outdated_examples": [],
        "runtime": None,
    }


def _tally(result: dict, name: str, version: str, ecosystem: str, display: str) -> None:
    """Classify one dep and update the per-manifest counters in place."""
    status = _classify_dep(name, version, ecosystem)
    result["dependency_count"] += 1
    if status == "up_to_date":
        result["up_to_date"] += 1
    elif status == "outdated":
        result["outdated"] += 1
        if len(result["outdated_examples"]) < MAX_OUTDATED_EXAMPLES:
            result["outdated_examples"].append(display)
    else:
        result["unknown"] += 1


def _analyze_package_json(content: str) -> dict:
    pkg = json.loads(content)
    result = _empty_result()

    deps: dict[str, str] = {}
    for key in ("dependencies", "devDependencies", "peerDependencies"):
        deps.update(pkg.get(key) or {})

    for name, version in deps.items():
        _tally(result, name, str(version), "npm", f"{name}@{version}")

    node_pin = (pkg.get("engines") or {}).get("node")
    if node_pin:
        result["runtime"] = {
            "language": "node",
            "version": node_pin,
            "status": _classify_node(node_pin),
        }
    return result


def _analyze_requirements_txt(content: str) -> dict:
    result = _empty_result()
    for raw in content.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        # Skip flag lines (-r, -e, --hash, etc.) and direct URL/git installs.
        if line.startswith(("-", "git+", "http://", "https://")):
            continue
        m = re.match(r"^([A-Za-z0-9_.][A-Za-z0-9_.-]*)(?:\[[^\]]+\])?\s*(.*)$", line)
        if not m:
            continue
        name = m.group(1).lower()
        spec = m.group(2).strip().rstrip(";").strip()
        # Drop environment markers ("; python_version >= ...") before classify.
        if ";" in spec:
            spec = spec.split(";", 1)[0].strip()
        _tally(result, name, spec, "pypi", f"{name} {spec}".strip())
    return result


def _analyze_pyproject(content: str) -> dict:
    """
    Hand-rolled TOML peek — keeps Python 3.9 compatibility (no tomllib).
    Handles two common shapes:
        [project]                          (PEP 621)
            dependencies = ["pkg ==1.0", ...]
        [tool.poetry.dependencies]         (Poetry)
            pkg = "^1.0"
    """
    result = _empty_result()

    # PEP 621 dependencies list.
    pep621 = re.search(
        r'^\[project\][\s\S]*?dependencies\s*=\s*\[(.*?)\]',
        content, re.M | re.S,
    )
    if pep621:
        for entry in re.findall(r'"([^"]+)"', pep621.group(1)):
            m = re.match(r"^([A-Za-z0-9_.][A-Za-z0-9_.-]*)(?:\[[^\]]+\])?\s*(.*)$", entry.strip())
            if m:
                name = m.group(1).lower()
                spec = m.group(2).strip()
                if ";" in spec:
                    spec = spec.split(";", 1)[0].strip()
                _tally(result, name, spec, "pypi", f"{name} {spec}".strip())

    # Poetry-style table (lines like `requests = "^2.31"`).
    poetry = re.search(
        r'^\[tool\.poetry\.dependencies\](.*?)(?=^\[|\Z)',
        content, re.M | re.S,
    )
    if poetry:
        for line in poetry.group(1).splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r'^([A-Za-z0-9_.][A-Za-z0-9_.-]*)\s*=\s*"([^"]+)"', line)
            if m:
                name = m.group(1).lower()
                if name == "python":
                    continue
                _tally(result, name, m.group(2), "pypi", f"{name} {m.group(2)}")

    # Runtime: prefer PEP 621's requires-python, fall back to Poetry's.
    python_pin = None
    pp = re.search(r'requires-python\s*=\s*"([^"]+)"', content)
    if pp:
        python_pin = pp.group(1)
    else:
        pp = re.search(r'^\s*python\s*=\s*"([^"]+)"', content, re.M)
        if pp:
            python_pin = pp.group(1)
    if python_pin:
        result["runtime"] = {
            "language": "python",
            "version": python_pin,
            "status": _classify_python(python_pin),
        }
    return result


def _analyze_pipfile(content: str) -> dict:
    """Pipfile is TOML — extract [packages] and [dev-packages] sections."""
    result = _empty_result()
    for section in ("[packages]", "[dev-packages]"):
        m = re.search(rf'^{re.escape(section)}(.*?)(?=^\[|\Z)', content, re.M | re.S)
        if not m:
            continue
        for line in m.group(1).splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            entry = re.match(r'^([A-Za-z0-9_.][A-Za-z0-9_.-]*)\s*=\s*(.+)$', line)
            if not entry:
                continue
            name = entry.group(1).lower()
            spec_raw = entry.group(2).strip()
            # Spec is either `"^1.0"` or a {version = "...", extras = [...]} table.
            ver_match = re.search(r'"([^"]+)"', spec_raw)
            spec = ver_match.group(1) if ver_match else spec_raw
            _tally(result, name, spec, "pypi", f"{name} {spec}")
    return result


def _analyze_pom_xml(content: str) -> dict:
    result = _empty_result()

    for block in re.finditer(r"<dependency>(.*?)</dependency>", content, re.S):
        body = block.group(1)
        gid = re.search(r"<groupId>([^<]+)</groupId>", body)
        aid = re.search(r"<artifactId>([^<]+)</artifactId>", body)
        ver = re.search(r"<version>([^<]+)</version>", body)
        if not aid:
            continue
        name = f"{gid.group(1).strip()}:{aid.group(1).strip()}" if gid else aid.group(1).strip()
        # Skip Maven property placeholders — version can't be resolved here.
        version = ver.group(1).strip() if ver else ""
        if version.startswith("${"):
            version = ""
        _tally(result, name, version, "maven", f"{name}:{version or '?'}")

    java_m = re.search(
        r"<(?:maven\.compiler\.source|maven\.compiler\.target|java\.version)>([^<]+)</",
        content,
    )
    if java_m:
        ver = java_m.group(1).strip()
        result["runtime"] = {
            "language": "java",
            "version": ver,
            "status": _classify_java(ver),
        }
    return result


def _analyze_gradle(content: str) -> dict:
    """
    Tolerant pattern match for both Groovy and Kotlin DSL gradle files.
    Catches:
        implementation 'group:artifact:version'
        implementation "group:artifact:version"
        implementation("group:artifact:version")
        api 'g:a:v', testImplementation "g:a:v", etc.
    """
    result = _empty_result()
    pattern = re.compile(
        r"""(?:implementation|api|compile|testImplementation|runtimeOnly|"""
        r"""compileOnly|annotationProcessor)\s*\(?\s*['"]([^'"]+:[^'"]+:[^'"]+)['"]""",
    )
    for match in pattern.finditer(content):
        coord = match.group(1)
        parts = coord.split(":")
        if len(parts) < 3:
            continue
        name = f"{parts[0]}:{parts[1]}"
        version = parts[2]
        if version.startswith("$"):
            version = ""
        _tally(result, name, version, "maven", coord)

    # Java toolchain version — `sourceCompatibility = JavaVersion.VERSION_17` etc.
    m = re.search(
        r"(?:sourceCompatibility|targetCompatibility|javaVersion)\s*[=:]\s*"
        r"(?:JavaVersion\.VERSION_)?[\"']?(\d+(?:\.\d+)?)[\"']?",
        content,
    )
    if m:
        ver = m.group(1)
        result["runtime"] = {
            "language": "java",
            "version": ver,
            "status": _classify_java(ver),
        }
    return result


def _analyze_gomod(content: str) -> dict:
    result = _empty_result()

    # Multi-line `require ( ... )` block(s).
    for block in re.finditer(r"require\s*\((.*?)\)", content, re.S):
        for line in block.group(1).splitlines():
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                _tally(result, parts[0], parts[1], "go", f"{parts[0]} {parts[1]}")

    # Single-line `require module v1.2.3`.
    for m in re.finditer(r"^\s*require\s+(\S+)\s+(\S+)\s*$", content, re.M):
        _tally(result, m.group(1), m.group(2), "go", f"{m.group(1)} {m.group(2)}")

    go_m = re.search(r"^go\s+([\d.]+)", content, re.M)
    if go_m:
        result["runtime"] = {
            "language": "go",
            "version": go_m.group(1),
            "status": _classify_go(go_m.group(1)),
        }
    return result


def _analyze_composer(content: str) -> dict:
    comp = json.loads(content)
    result = _empty_result()

    deps: dict[str, str] = {}
    for key in ("require", "require-dev"):
        deps.update(comp.get(key) or {})

    php_pin = deps.pop("php", None)
    if php_pin:
        result["runtime"] = {
            "language": "php",
            "version": php_pin,
            "status": _classify_php(php_pin),
        }

    # Drop platform packages (ext-*, lib-*, etc.) — they're capability gates,
    # not real dependencies for freshness purposes.
    for name, version in deps.items():
        if name.startswith(("ext-", "lib-", "php-")) or "/" not in name:
            continue
        _tally(result, name, str(version), "packagist", f"{name}:{version}")
    return result


# ---------------------------------------------------------------------------
# Runtime EOL classifiers (snapshot: 2026-05-28)
# ---------------------------------------------------------------------------
# Each classifier maps a version specifier to one of:
#   "current" | "lts" | "maintenance" | "eol"
# Once a major (or major.minor) crosses its published EOL date we still
# treat it as "maintenance" for a configurable grace window — giving
# project teams time to respond before the rubric penalises them as EOL.

# EOL dates per runtime, used for the grace-period calculation. Keys
# vary by runtime: Node uses just the major; Python and PHP key on
# (major, minor) because each minor has its own date. Out-of-date
# entries just lose the grace benefit, they don't break classification.
_NODE_EOL_DATES: dict[int, str] = {
    18: "2025-04-30",
    20: "2026-04-30",
    22: "2027-04-30",
    24: "2028-04-30",
}

_PYTHON_EOL_DATES: dict[tuple[int, int], str] = {
    (3, 9):  "2025-10-31",
    (3, 10): "2026-10-31",
    (3, 11): "2027-10-31",
    (3, 12): "2028-10-31",
    (3, 13): "2029-10-31",
    (3, 14): "2030-10-31",
}

_PHP_EOL_DATES: dict[tuple[int, int], str] = {
    (8, 2): "2025-12-08",
    (8, 3): "2026-12-31",
    (8, 4): "2027-12-31",
    (8, 5): "2028-12-31",
}


def _eol_grace_period_days() -> int:
    """
    Grace window (days) after a runtime's official EOL date during
    which it's still classified as "maintenance" rather than "eol".
    Read from EOL_GRACE_PERIOD_DAYS in the environment; defaults to 90.
    Negative / unparseable values fall back to the default.
    """
    raw = os.environ.get("EOL_GRACE_PERIOD_DAYS", "90")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 90
    return value if value >= 0 else 90


def _within_eol_grace_period(eol_date_str: Optional[str]) -> bool:
    """True iff today is past the EOL date but within the grace window."""
    if not eol_date_str:
        return False
    try:
        eol_date = datetime.strptime(eol_date_str, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return False
    today = datetime.now(timezone.utc).date()
    days_past = (today - eol_date).days
    return 0 <= days_past <= _eol_grace_period_days()


def _extract_major_minor(spec: str) -> tuple[Optional[int], Optional[int]]:
    """Pull the leading X[.Y] from a version specifier like '^3.11' or '>=8.1'."""
    if not spec:
        return None, None
    m = re.search(r"(\d+)(?:\.(\d+))?", spec)
    if not m:
        return None, None
    major = int(m.group(1))
    minor = int(m.group(2)) if m.group(2) is not None else None
    return major, minor


def _parse_node_pin(content: str) -> Optional[str]:
    """
    Extract a Node version string from a .nvmrc or .node-version file.

    These files are typically a single line containing the version, with
    or without a leading "v" (e.g. "20", "18.17.0", "v22.5.1", or
    "lts/*"). We take the first non-empty line, strip the "v" prefix,
    and pass the rest to _classify_node which already handles partial
    versions like "lts/*" gracefully (returning None when nothing
    numeric is found).
    """
    if not content:
        return None
    for line in content.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped.lstrip("vV")
    return None


def _classify_node(spec: str) -> Optional[str]:
    major, _ = _extract_major_minor(spec)
    if major is None:
        return None
    # Grace-period check: if this major has only recently crossed its
    # published EOL date, downgrade to "maintenance" rather than "eol"
    # so projects aren't penalised the day after their runtime EOLs.
    if _within_eol_grace_period(_NODE_EOL_DATES.get(major)):
        return "maintenance"
    # Per the Node release calendar (Active LTS → Maintenance → EOL).
    if major >= 26:
        return "current"
    if major == 24:
        return "lts"           # Active LTS through Oct 2026
    if major == 22:
        return "maintenance"   # Maintenance LTS through Apr 2027
    # Everything else (≤20, plus odd non-LTS 21/23/25) is unsupported.
    return "eol"


def _classify_python(spec: str) -> Optional[str]:
    major, minor = _extract_major_minor(spec)
    if major is None:
        return None
    if major != 3:
        return "eol"
    if minor is None:
        return None
    # Grace-period check (per-minor EOL dates for Python).
    if _within_eol_grace_period(_PYTHON_EOL_DATES.get((major, minor))):
        return "maintenance"
    if minor >= 14:
        return "current"
    if minor == 13:
        return "lts"           # newest stable behind 3.14 — community LTS-equivalent
    if minor in (11, 12):
        return "maintenance"
    if minor == 10:
        return "maintenance"   # EOL Oct 2026 — still in security window
    return "eol"


def _classify_java(spec: str) -> Optional[str]:
    # Handle legacy "1.8" form.
    s = spec.strip()
    if s.startswith("1.") and len(s) >= 3 and s[2].isdigit():
        s = s[2:]
    major, _ = _extract_major_minor(s)
    if major is None:
        return None
    if major >= 25:
        return "current"
    if major == 21:
        return "lts"
    if major == 17:
        return "lts"
    return "eol"


def _classify_go(spec: str) -> Optional[str]:
    major, minor = _extract_major_minor(spec)
    if major is None or minor is None:
        return None
    if major != 1:
        return "eol"
    # Go's official support policy = two latest minors.
    if minor >= 25:
        return "current"
    return "eol"


def _classify_php(spec: str) -> Optional[str]:
    major, minor = _extract_major_minor(spec)
    if major is None:
        return None
    if major < 8 or minor is None:
        return "eol" if major is not None and major < 8 else None
    # Grace-period check (per-minor EOL dates for PHP).
    if _within_eol_grace_period(_PHP_EOL_DATES.get((major, minor))):
        return "maintenance"
    if minor >= 5:
        return "current"
    if minor == 4:
        return "lts"          # active support window
    if minor == 3:
        return "maintenance"  # security-only
    return "eol"


# ---------------------------------------------------------------------------
# Open issues ratio (best-effort, one extra API call)
# ---------------------------------------------------------------------------

def _try_open_issues_ratio(
    repo: Optional[str], metadata: dict, errors: list[str]
) -> Optional[float]:
    """
    Compute open / (open + closed) for issues, using the Link header on
    the closed-issues endpoint for total count. Falls back to None on
    any failure rather than raising.

    Caveat: GitHub counts open PRs inside open_issues_count, so the
    `state=closed&filter=all` page count is the consistent denominator.
    """
    if not repo or "/" not in repo:
        return None
    open_count = metadata.get("open_issues")
    if open_count is None:
        return None
    owner, name = repo.split("/", 1)

    try:
        token = os.getenv("GITHUB_TOKEN")
        with _build_client(token) as client:
            resp = _request(
                client,
                f"/repos/{owner}/{name}/issues",
                params={"state": "closed", "per_page": "1", "filter": "all"},
            )
            if resp.status_code != 200:
                return None
            closed_count = _parse_link_last_page(resp.headers.get("Link"))
            if closed_count is None:
                payload = resp.json()
                closed_count = len(payload) if isinstance(payload, list) else 0
    except Exception as e:
        errors.append(f"open_issues_ratio: {type(e).__name__}: {e}")
        return None

    total = open_count + closed_count
    if total <= 0:
        return None
    return round(open_count / total, 4)


def _parse_link_last_page(link_header: Optional[str]) -> Optional[int]:
    """
    Extract the `page` parameter from a GitHub Link header's rel="last"
    entry. Uses proper URL/query parsing so `page=` is not confused with
    `per_page=` — the prior string-split version returned `1` from the
    embedded `per_page=1` for high-issue-count repos.
    """
    if not link_header:
        return None
    for part in link_header.split(","):
        if 'rel="last"' not in part:
            continue
        url = part.split(";", 1)[0].strip().strip("<>")
        try:
            query = urllib.parse.urlparse(url).query
            params = urllib.parse.parse_qs(query)
            page_values = params.get("page")
            if page_values:
                return int(page_values[0])
        except (ValueError, IndexError):
            continue
    return None


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

    parser = argparse.ArgumentParser(description="Smoke-test dependency_scanner")
    parser.add_argument("--repo", required=True, help="owner/name")
    args = parser.parse_args()

    from github_fetcher import fetch_repo_data
    fetch = fetch_repo_data({"repo": args.repo})
    result = analyze({"repo": args.repo, "fetch": fetch})
    print(json.dumps(result, indent=2, default=str))
