"""
code_analyzer.py — App Rationalization Agent
============================================
Static-analysis agent for the 5-R rubric. Consumes the `fetch` payload
produced by github_fetcher (metadata, file_tree, readme, manifests) and
extracts:

    - primary_language / runtime_version
    - frameworks                 (parsed from dependency manifests)
    - lines_of_code              (estimated from file_tree sizes)
    - has_tests / has_readme /
      has_ci / has_license       → quality_signals
    - architecture_signals       → complexity_signals (modular/mixed/monolith)

Scores each signal against the rubric in config/scoring_rubric.json and
returns both per-category and per-signal scores. Makes no GitHub API
calls — all data comes from github_fetcher's structured output.

Returned dict shape:
    {
        "agent": "code_analyzer",
        "signals": { ... raw values ... },
        "scored": {
            "quality":    { signal: {value, score, weight}, ... },
            "complexity": { signal: {value, score, weight}, ... },
        },
        "category_scores": { "quality": 0.85, "complexity": 0.70 },
        "errors": [ ... ]
    }
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

RUBRIC_PATH = Path(__file__).resolve().parent.parent / "config" / "scoring_rubric.json"

# Bytes-per-line heuristic for LOC estimation. We're bucketing into wide
# ranges (0-999, 1000-50000, 50001-200000, 200001+), so a single language-
# agnostic constant is good enough — the rubric tolerates significant slop.
BYTES_PER_LINE = 32

# File extensions counted as "source" for LOC estimation.
SOURCE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".java", ".kt", ".kts", ".scala", ".groovy",
    ".go", ".rs", ".rb", ".php", ".swift", ".dart",
    ".cs", ".vb", ".fs",
    ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hxx",
    ".clj", ".cljs", ".ex", ".exs", ".elm", ".erl",
    ".lua", ".pl", ".pm", ".r", ".m", ".mm",
    ".vue", ".svelte", ".sh", ".bash", ".zsh",
}

# Path segments that almost always indicate vendored / generated code.
NON_SOURCE_SEGMENTS = (
    "node_modules/", "vendor/", ".venv/", "venv/", "env/",
    "dist/", "build/", "out/", "target/", "bin/", "obj/",
    "__pycache__/", "site-packages/",
    ".gradle/", ".idea/", ".vscode/", ".git/",
    ".next/", ".nuxt/", ".cache/", "coverage/",
)
MINIFIED_SUFFIXES = (".min.js", ".min.css", ".min.mjs", ".bundle.js")

# Recognized CI config locations / files.
CI_MARKERS = (
    ".github/workflows/",
    ".circleci/config",
    "jenkinsfile",
    ".travis.yml",
    "azure-pipelines.yml",
    "azure-pipelines.yaml",
    ".gitlab-ci.yml",
    "bitbucket-pipelines.yml",
    ".drone.yml",
    ".buildkite/",
    ".appveyor.yml",
    "appveyor.yml",
)

# Monorepo-shaped top-level directories.
MONOREPO_TOPDIRS = {"packages", "services", "apps", "libs", "modules", "workspaces", "crates"}
# "Organized but single package" top-level directories.
MIXED_TOPDIRS = {"src", "lib", "internal", "pkg", "cmd", "app"}


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def analyze(repo_data: dict) -> dict:
    """
    Run static analysis over the github_fetcher payload.

    Args:
        repo_data: shared pipeline dict; expects `fetch` populated by
                   github_fetcher.fetch_repo_data.

    Returns:
        Structured dict (see module docstring).
    """
    fetch = repo_data.get("fetch") or {}
    if not fetch:
        # The agent contract is to return signals; surface the missing
        # upstream dependency rather than crashing the pipeline.
        return {
            "agent": "code_analyzer",
            "signals": {},
            "scored": {"quality": {}, "complexity": {}},
            "category_scores": {"quality": None, "complexity": None},
            "errors": ["fetch data missing — github_fetcher must run first"],
        }

    metadata: dict = fetch.get("metadata") or {}
    file_tree: list[dict] = fetch.get("file_tree") or []
    manifests: dict = fetch.get("manifests") or {}
    readme: Optional[dict] = fetch.get("readme")

    errors: list[str] = []
    signals: dict[str, Any] = {}

    # --- Raw signal extraction ------------------------------------------------
    signals["primary_language"] = metadata.get("language")
    try:
        signals["runtime_version"] = _detect_runtime_version(manifests, file_tree)
    except Exception as e:
        errors.append(f"runtime_version: {e}")
        signals["runtime_version"] = None

    try:
        signals["frameworks"] = _detect_frameworks(manifests)
    except Exception as e:
        errors.append(f"frameworks: {e}")
        signals["frameworks"] = []

    try:
        signals["lines_of_code"] = _estimate_loc(file_tree)
    except Exception as e:
        errors.append(f"lines_of_code: {e}")
        signals["lines_of_code"] = None

    signals["has_tests"] = _detect_tests(file_tree)
    signals["has_readme"] = _detect_readme_quality(readme)
    signals["has_ci"] = _detect_ci(file_tree)
    signals["has_license"] = "yes" if metadata.get("license") else "no"
    signals["architecture_signals"] = _detect_architecture(file_tree)

    # --- Score against rubric -------------------------------------------------
    rubric = _load_rubric()
    quality_rubric = rubric.get("quality_signals", {})
    complexity_rubric = rubric.get("complexity_signals", {})

    quality_scored, quality_cat = _score_category(signals, quality_rubric)
    complexity_scored, complexity_cat = _score_category(signals, complexity_rubric)

    return {
        "agent": "code_analyzer",
        "signals": signals,
        "scored": {
            "quality":    quality_scored,
            "complexity": complexity_scored,
        },
        "category_scores": {
            "quality":    quality_cat,
            "complexity": complexity_cat,
        },
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Rubric loading + score-map matching
# ---------------------------------------------------------------------------

def _load_rubric() -> dict:
    with open(RUBRIC_PATH) as f:
        return json.load(f)


def _score_category(signals: dict, category_rubric: dict) -> tuple[dict, Optional[float]]:
    """Score every signal in a rubric category and return (per-signal, weighted)."""
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
    """
    Resolve a value to a score using the rubric's bucket keys.

    Supports:
      - direct categorical match ("yes", "modular", "detailed")
      - numeric ranges ("1000-50000")
      - open-ended ranges ("200001+")
      - exact numbers ("1", "0")
    """
    if value is None:
        return None

    # Categorical first — covers has_tests, has_readme, architecture, etc.
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
            low_s, high_s = bucket.split("-", 1)
            return float(low_s) <= value <= float(high_s)
        except ValueError:
            return False
    try:
        return value == float(bucket)
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Runtime version detection
# ---------------------------------------------------------------------------

def _detect_runtime_version(manifests: dict, file_tree: list[dict]) -> Optional[str]:
    """Best-effort: probe known manifests for an explicit runtime/language pin."""
    # Lookup manifests by lower-cased basename for case-insensitive matching.
    by_name = {name.lower(): data for name, data in manifests.items()}

    if "go.mod" in by_name:
        m = re.search(r"^go\s+([\d.]+)", by_name["go.mod"].get("content") or "", re.M)
        if m:
            return f"Go {m.group(1)}"

    if "package.json" in by_name:
        try:
            pkg = json.loads(by_name["package.json"].get("content") or "{}")
            node = (pkg.get("engines") or {}).get("node")
            if node:
                return f"Node {node}"
        except (json.JSONDecodeError, ValueError):
            pass

    if "pyproject.toml" in by_name:
        content = by_name["pyproject.toml"].get("content") or ""
        m = re.search(r'requires-python\s*=\s*"([^"]+)"', content)
        if m:
            return f"Python {m.group(1)}"
        m = re.search(r'python\s*=\s*"([^"]+)"', content)
        if m:
            return f"Python {m.group(1)}"

    if "pom.xml" in by_name:
        content = by_name["pom.xml"].get("content") or ""
        m = re.search(r"<(?:maven\.compiler\.source|java\.version)>([^<]+)</", content)
        if m:
            return f"Java {m.group(1).strip()}"

    if "composer.json" in by_name:
        try:
            comp = json.loads(by_name["composer.json"].get("content") or "{}")
            php = (comp.get("require") or {}).get("php")
            if php:
                return f"PHP {php}"
        except (json.JSONDecodeError, ValueError):
            pass

    if "cargo.toml" in by_name:
        m = re.search(r'rust-version\s*=\s*"([^"]+)"', by_name["cargo.toml"].get("content") or "")
        if m:
            return f"Rust {m.group(1)}"

    # Last-resort: version pin files at the repo root.
    pin_lookup = {
        ".nvmrc":          "Node",
        ".node-version":   "Node",
        ".python-version": "Python",
        ".ruby-version":   "Ruby",
        ".tool-versions":  None,  # asdf — too varied to parse here
    }
    root_files = {
        (e.get("path") or "").lower(): e
        for e in file_tree
        if e.get("type") == "blob" and "/" not in (e.get("path") or "")
    }
    for name, label in pin_lookup.items():
        if name in root_files and label:
            return f"{label} (see {name})"

    return None


# ---------------------------------------------------------------------------
# Framework detection
# ---------------------------------------------------------------------------

# Known framework markers per manifest. Values are display labels.
_JS_FRAMEWORKS = {
    "react": "React", "vue": "Vue", "@angular/core": "Angular",
    "svelte": "Svelte", "next": "Next.js", "nuxt": "Nuxt",
    "gatsby": "Gatsby", "remix": "Remix",
    "express": "Express", "fastify": "Fastify", "koa": "Koa",
    "@nestjs/core": "NestJS", "hapi": "Hapi",
    "typescript": "TypeScript",
    "jest": "Jest", "mocha": "Mocha", "vitest": "Vitest", "playwright": "Playwright",
    "webpack": "Webpack", "vite": "Vite", "rollup": "Rollup",
    "electron": "Electron", "react-native": "React Native",
}

_PY_FRAMEWORKS = {
    "django": "Django", "flask": "Flask", "fastapi": "FastAPI",
    "tornado": "Tornado", "pyramid": "Pyramid", "starlette": "Starlette",
    "sanic": "Sanic", "aiohttp": "aiohttp", "bottle": "Bottle",
    "pytest": "pytest", "nose": "nose",
    "numpy": "NumPy", "pandas": "pandas",
    "tensorflow": "TensorFlow", "torch": "PyTorch",
    "scikit-learn": "scikit-learn",
    "sqlalchemy": "SQLAlchemy", "celery": "Celery",
}

_GO_FRAMEWORKS = {
    "github.com/gin-gonic/gin":    "Gin",
    "github.com/gorilla/mux":      "Gorilla Mux",
    "github.com/labstack/echo":    "Echo",
    "github.com/gofiber/fiber":    "Fiber",
    "github.com/go-chi/chi":       "Chi",
    "gorm.io/gorm":                "GORM",
    "github.com/spf13/cobra":      "Cobra",
}

_JAVA_FRAMEWORKS = {
    "spring-boot":   "Spring Boot",
    "springframework": "Spring",
    "hibernate":     "Hibernate",
    "junit":         "JUnit",
    "quarkus":       "Quarkus",
    "micronaut":     "Micronaut",
    "dropwizard":    "Dropwizard",
}

_PHP_FRAMEWORKS = {
    "laravel/framework":    "Laravel",
    "symfony/symfony":      "Symfony",
    "symfony/framework-bundle": "Symfony",
    "codeigniter/framework": "CodeIgniter",
    "yiisoft/yii2":         "Yii",
    "cakephp/cakephp":      "CakePHP",
}

_RUBY_FRAMEWORKS = {
    "rails":   "Rails",
    "sinatra": "Sinatra",
    "rspec":   "RSpec",
    "rack":    "Rack",
}

_RUST_FRAMEWORKS = {
    "actix-web": "Actix Web",
    "rocket":    "Rocket",
    "axum":      "Axum",
    "warp":      "Warp",
    "tokio":     "Tokio",
    "serde":     "Serde",
}


def _detect_frameworks(manifests: dict) -> list[str]:
    """Inspect each manifest and union the detected framework labels."""
    found: set[str] = set()
    by_name = {name.lower(): data for name, data in manifests.items()}

    if "package.json" in by_name:
        found.update(_fw_from_package_json(by_name["package.json"].get("content") or ""))
    if "requirements.txt" in by_name:
        found.update(_fw_from_requirements(by_name["requirements.txt"].get("content") or ""))
    if "pyproject.toml" in by_name:
        found.update(_fw_from_pyproject(by_name["pyproject.toml"].get("content") or ""))
    if "pipfile" in by_name:
        found.update(_fw_from_pipfile(by_name["pipfile"].get("content") or ""))
    if "go.mod" in by_name:
        found.update(_fw_from_gomod(by_name["go.mod"].get("content") or ""))
    if "pom.xml" in by_name:
        found.update(_fw_from_pom(by_name["pom.xml"].get("content") or ""))
    for gradle in ("build.gradle", "build.gradle.kts"):
        if gradle in by_name:
            found.update(_fw_from_gradle(by_name[gradle].get("content") or ""))
    if "composer.json" in by_name:
        found.update(_fw_from_composer(by_name["composer.json"].get("content") or ""))
    if "gemfile" in by_name:
        found.update(_fw_from_gemfile(by_name["gemfile"].get("content") or ""))
    if "cargo.toml" in by_name:
        found.update(_fw_from_cargo(by_name["cargo.toml"].get("content") or ""))

    return sorted(found)


def _fw_from_package_json(content: str) -> set[str]:
    try:
        pkg = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return set()
    deps: dict = {}
    for key in ("dependencies", "devDependencies", "peerDependencies"):
        deps.update(pkg.get(key) or {})
    return {label for key, label in _JS_FRAMEWORKS.items() if key in deps}


def _fw_from_requirements(content: str) -> set[str]:
    found: set[str] = set()
    for line in content.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[<>=!~;\[\s]", line, 1)[0].lower()
        if name in _PY_FRAMEWORKS:
            found.add(_PY_FRAMEWORKS[name])
    return found


def _fw_from_pyproject(content: str) -> set[str]:
    found: set[str] = set()
    # Match either: package = "version"  (Poetry / PDM)
    #            or just bare names listed under [project.dependencies] arrays.
    for key in _PY_FRAMEWORKS:
        # Word-boundary match so "scikit-learn" isn't matched inside "scikit-learn-extra"
        if re.search(rf'(^|[\s,"\[]){re.escape(key)}(?=[\s"=<>~,!\]])', content, re.M | re.I):
            found.add(_PY_FRAMEWORKS[key])
    return found


def _fw_from_pipfile(content: str) -> set[str]:
    found: set[str] = set()
    for key in _PY_FRAMEWORKS:
        if re.search(rf'^\s*{re.escape(key)}\s*=', content, re.M | re.I):
            found.add(_PY_FRAMEWORKS[key])
    return found


def _fw_from_gomod(content: str) -> set[str]:
    found: set[str] = set()
    for key, label in _GO_FRAMEWORKS.items():
        if key in content:
            found.add(label)
    return found


def _fw_from_pom(content: str) -> set[str]:
    """Match <artifactId> entries and Spring Boot parent declarations."""
    found: set[str] = set()
    artifacts = re.findall(r"<artifactId>([^<]+)</artifactId>", content)
    haystack = " ".join(artifacts).lower() + " " + content.lower()
    for key, label in _JAVA_FRAMEWORKS.items():
        if key in haystack:
            found.add(label)
    return found


def _fw_from_gradle(content: str) -> set[str]:
    lowered = content.lower()
    return {label for key, label in _JAVA_FRAMEWORKS.items() if key in lowered}


def _fw_from_composer(content: str) -> set[str]:
    try:
        comp = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return set()
    deps: dict = {}
    for key in ("require", "require-dev"):
        deps.update(comp.get(key) or {})
    return {label for key, label in _PHP_FRAMEWORKS.items() if key in deps}


def _fw_from_gemfile(content: str) -> set[str]:
    found: set[str] = set()
    # gem "rails", "~> 7.0"  /  gem 'rails'
    gems = re.findall(r"""^\s*gem\s+['"]([^'"]+)['"]""", content, re.M)
    for gem in gems:
        if gem.lower() in _RUBY_FRAMEWORKS:
            found.add(_RUBY_FRAMEWORKS[gem.lower()])
    return found


def _fw_from_cargo(content: str) -> set[str]:
    found: set[str] = set()
    for key, label in _RUST_FRAMEWORKS.items():
        if re.search(rf'^\s*{re.escape(key)}\s*=', content, re.M):
            found.add(label)
    return found


# ---------------------------------------------------------------------------
# LOC estimation
# ---------------------------------------------------------------------------

def _estimate_loc(file_tree: list[dict]) -> int:
    total_bytes = 0
    for entry in file_tree:
        if entry.get("type") != "blob":
            continue
        path = (entry.get("path") or "").lower()
        if not path or any(seg in path for seg in NON_SOURCE_SEGMENTS):
            continue
        basename = path.rsplit("/", 1)[-1]
        if any(basename.endswith(s) for s in MINIFIED_SUFFIXES):
            continue
        if "." not in basename:
            continue
        ext = "." + basename.rsplit(".", 1)[-1]
        if ext not in SOURCE_EXTENSIONS:
            continue
        total_bytes += entry.get("size") or 0
    return total_bytes // BYTES_PER_LINE


# ---------------------------------------------------------------------------
# Test / CI / README / architecture detection
# ---------------------------------------------------------------------------

_TEST_DIR_PATTERNS = ("/test/", "/tests/", "/spec/", "/specs/", "/__tests__/")
_TEST_FILE_HINTS = ("test_", "_test.", "_test_", ".test.", ".spec.", "test.go", "tests.py")
_TEST_FILE_EXTENSIONS = (
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".java", ".kt",
    ".rb", ".rs", ".php", ".cs",
)


def _detect_tests(file_tree: list[dict]) -> str:
    """Classify test presence as yes/partial/no based on file_tree heuristics."""
    hits = 0
    for entry in file_tree:
        path = (entry.get("path") or "").lower()
        if not path:
            continue
        # Skip vendored code so we don't credit dependency test suites.
        if any(seg in path for seg in NON_SOURCE_SEGMENTS):
            continue

        if entry.get("type") == "tree":
            # Top-level or near-top "test"/"tests"/etc. directories.
            depth = path.count("/")
            if depth <= 2 and any(path.endswith(d.rstrip("/")) for d in
                                  ("test", "tests", "spec", "specs", "__tests__")):
                hits += 3  # a dedicated test dir is a strong signal
            continue

        if entry.get("type") != "blob":
            continue

        # Test directory in the file path.
        if any(seg in f"/{path}/" for seg in _TEST_DIR_PATTERNS):
            hits += 1
            continue

        basename = path.rsplit("/", 1)[-1]
        if not any(basename.endswith(ext) for ext in _TEST_FILE_EXTENSIONS):
            continue
        if any(hint in basename for hint in _TEST_FILE_HINTS):
            hits += 1

    if hits >= 5:
        return "yes"
    if hits >= 1:
        return "partial"
    return "no"


def _detect_ci(file_tree: list[dict]) -> str:
    for entry in file_tree:
        if entry.get("type") != "blob":
            continue
        path = (entry.get("path") or "").lower()
        if any(marker in path for marker in CI_MARKERS):
            return "yes"
    return "no"


def _detect_readme_quality(readme: Optional[dict]) -> str:
    if not readme:
        return "missing"
    content = readme.get("content") or ""
    size = readme.get("size") or len(content)
    if size <= 0:
        return "missing"
    # "detailed" if ≥2KB — long enough to cover setup, usage, and contributing.
    return "detailed" if size >= 2048 else "basic"


def _detect_architecture(file_tree: list[dict]) -> str:
    """
    Bucket the repo as modular / mixed / monolith based on layout.

    - modular:  monorepo dirs (packages/, services/, apps/, …) OR multiple
                non-vendored manifests nested under different paths
    - mixed:    single package but uses src/, lib/, pkg/, cmd/, internal/, app/
    - monolith: flat layout with sources at the root
    """
    manifest_paths: list[str] = []
    top_dirs: set[str] = set()

    for entry in file_tree:
        path = entry.get("path") or ""
        if not path:
            continue
        lower = path.lower()
        if any(seg in lower for seg in NON_SOURCE_SEGMENTS):
            continue

        if "/" in path:
            top_dirs.add(path.split("/", 1)[0].lower())

        if entry.get("type") != "blob":
            continue
        basename = path.rsplit("/", 1)[-1].lower()
        if basename in ("package.json", "pom.xml", "setup.py", "pyproject.toml",
                        "go.mod", "cargo.toml", "composer.json", "gemfile",
                        "build.gradle", "build.gradle.kts"):
            manifest_paths.append(path)

    nested_manifests = [p for p in manifest_paths if "/" in p]
    if len(nested_manifests) >= 2 or top_dirs & MONOREPO_TOPDIRS:
        return "modular"
    if top_dirs & MIXED_TOPDIRS:
        return "mixed"
    return "monolith"


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

    parser = argparse.ArgumentParser(description="Smoke-test code_analyzer")
    parser.add_argument("--repo", required=True, help="owner/name")
    args = parser.parse_args()

    # Need the github_fetcher payload to feed in.
    from github_fetcher import fetch_repo_data

    fetch = fetch_repo_data({"repo": args.repo})
    result = analyze({"repo": args.repo, "fetch": fetch})
    print(json.dumps(result, indent=2, default=str))
