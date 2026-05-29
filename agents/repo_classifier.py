"""
repo_classifier.py — App Rationalization Agent
==============================================
Pre-scoring stage. Consumes the `fetch` payload produced by
github_fetcher and classifies the repository into one of six types so
downstream stages (scoring rubric, system prompt) can adjust behavior
for repos that aren't deployable software.

Types:
    APPLICATION   — deployable software; primary target of 5-R analysis
    LIBRARY       — reusable package/framework consumed by other software
    DOCUMENTATION — pure docs/standards/reference; no executable code
    CONFIGURATION — IaC, dotfiles, config-only repos
    DATA          — datasets, assets, content; no runtime
    UNKNOWN       — signals are insufficient to classify

Why this exists: the 5-R rubric was built around runtime applications.
LOC, test presence, runtime EOL, dependency freshness are all
structurally inapplicable for a Markdown spec repo or a dotfiles
collection. Applying the rubric blindly to those repos produces
nonsense recommendations (e.g. RETIRE for an actively-maintained PSR
spec just because it has zero source-code LOC). The classifier lets
the scorer suppress those rules and lets Claude tailor its rationale.

Returned dict shape:
    {
        "repo_type":  "APPLICATION" | "LIBRARY" | "DOCUMENTATION"
                    | "CONFIGURATION" | "DATA" | "UNKNOWN",
        "confidence": float in [0, 1],
        "rationale":  "single-line human-readable explanation",
        "signals":    { ... per-signal counts and matches for debugging ... }
    }
"""

from __future__ import annotations

import json
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Extension classifications
# ---------------------------------------------------------------------------

DOC_EXTENSIONS = {
    ".md", ".markdown", ".rst", ".txt", ".asciidoc", ".adoc",
    ".rtf", ".tex", ".org", ".pod",
}

SOURCE_EXTENSIONS = {
    # Mainstream programming languages — files that contain executable code.
    ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".java", ".kt", ".kts", ".scala", ".groovy", ".clj", ".cljs",
    ".go", ".rs", ".rb", ".php", ".swift", ".dart",
    ".cs", ".vb", ".fs",
    ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hxx",
    ".ex", ".exs", ".elm", ".erl", ".lua", ".pl", ".pm",
    ".r", ".m", ".mm",
    ".vue", ".svelte",
}

DATA_EXTENSIONS = {
    ".csv", ".tsv", ".parquet", ".arrow", ".npz", ".pkl",
    ".feather", ".jsonl", ".ndjson",
    # Asset/media files
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg",
    ".mp3", ".wav", ".mp4", ".webm", ".pdf",
    ".xls", ".xlsx", ".ods",
}

CONFIG_EXTENSIONS = {
    ".yaml", ".yml", ".toml", ".ini", ".conf", ".cfg",
    ".env", ".envrc", ".tf", ".tfvars", ".hcl",
    ".nix", ".dhall",
    ".plist",
}

# ---------------------------------------------------------------------------
# Topic / description keyword sets
# ---------------------------------------------------------------------------

DOC_TOPICS = {
    "documentation", "docs", "specification", "spec", "standards",
    "awesome", "awesome-list", "curated-list", "tutorial", "guide",
    "book", "rfc", "psr", "manual", "handbook", "wiki",
}

CONFIG_TOPICS = {
    "dotfiles", "config", "configuration", "configs", "settings",
    "terraform", "ansible", "puppet", "chef", "saltstack",
    "infrastructure", "infrastructure-as-code", "iac",
    "kubernetes-config", "k8s-config", "helm-chart",
}

DATA_TOPICS = {
    "dataset", "datasets", "data", "corpus", "benchmark",
    "assets", "fonts", "icons", "wallpapers", "samples",
}

LIBRARY_TOPICS = {
    "library", "libraries", "framework", "sdk", "api",
    "package", "module", "plugin", "extension", "client",
    "react",   # libs often topic themselves after the ecosystem
    "vue", "angular", "django", "rails",
}

APPLICATION_TOPICS = {
    "application", "app", "server", "service", "microservice",
    "cli", "tui", "gui", "desktop-app", "web-app", "mobile-app",
    "daemon", "agent", "bot",
}

# Dependency manifest filenames (lowercased) — used to determine
# whether the repo carries runtime dependency metadata at all.
DEPENDENCY_MANIFESTS = {
    "package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "requirements.txt", "setup.py", "setup.cfg", "pyproject.toml",
    "pipfile", "pipfile.lock", "poetry.lock",
    "pom.xml", "build.gradle", "build.gradle.kts", "build.sbt",
    "gemfile", "gemfile.lock",
    "go.mod", "go.sum",
    "cargo.toml", "cargo.lock",
    "composer.json", "composer.lock",
    "packages.config", "paket.dependencies",
}


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def classify(repo_data: dict) -> dict:
    """
    Classify a repository into one of the six types using only signals
    already present in the github_fetcher output. No new API calls.
    """
    fetch = repo_data.get("fetch") or {}
    metadata = fetch.get("metadata") or {}
    file_tree = fetch.get("file_tree") or []
    manifests = fetch.get("manifests") or {}

    topics = {t.lower() for t in (metadata.get("topics") or []) if isinstance(t, str)}
    description = (metadata.get("description") or "").lower()

    # --- Tally file types ---------------------------------------------------
    counts = _count_extensions(file_tree)
    total_files = sum(counts.values())
    doc_files = sum(counts.get(ext, 0) for ext in DOC_EXTENSIONS)
    source_files = sum(counts.get(ext, 0) for ext in SOURCE_EXTENSIONS)
    data_files = sum(counts.get(ext, 0) for ext in DATA_EXTENSIONS)
    config_files = sum(counts.get(ext, 0) for ext in CONFIG_EXTENSIONS)

    doc_ratio    = doc_files    / total_files if total_files else 0.0
    source_ratio = source_files / total_files if total_files else 0.0
    data_ratio   = data_files   / total_files if total_files else 0.0
    config_ratio = config_files / total_files if total_files else 0.0

    has_dep_manifest = any(
        m.lower() in DEPENDENCY_MANIFESTS for m in manifests.keys()
    )

    signals = {
        "file_count":         total_files,
        "doc_ratio":          round(doc_ratio,    3),
        "source_ratio":       round(source_ratio, 3),
        "data_ratio":         round(data_ratio,   3),
        "config_ratio":       round(config_ratio, 3),
        "has_dep_manifest":   has_dep_manifest,
        "topics":             sorted(topics),
        "language":           metadata.get("language"),
    }

    # --- Empty / unreadable repo -------------------------------------------
    if total_files == 0:
        return _result("UNKNOWN", 0.0,
                       "file tree is empty or unreadable",
                       signals)

    # --- DOCUMENTATION ------------------------------------------------------
    # Strong: heavy doc content + no source manifests + little/no source code.
    if doc_ratio > 0.6 and source_ratio < 0.1 and not has_dep_manifest:
        return _result(
            "DOCUMENTATION", 0.90,
            f"{doc_files}/{total_files} files ({doc_ratio:.0%}) are docs; "
            f"no dependency manifests; only {source_ratio:.0%} source code",
            signals,
        )
    # Weaker: doc-related topic + no manifest + low source.
    if (DOC_TOPICS & topics) and not has_dep_manifest and source_ratio < 0.2:
        return _result(
            "DOCUMENTATION", 0.80,
            f"doc-related topics ({sorted(DOC_TOPICS & topics)}) and no "
            f"dependency manifests",
            signals,
        )
    # Description-driven fallback for specs/standards repos.
    doc_kw = ("specification", "specifications", "psr ", "rfc ", "standards",
              "documentation", "reference docs", "awesome list",
              "curated list")
    if any(kw in description for kw in doc_kw) and not has_dep_manifest and source_ratio < 0.2:
        return _result(
            "DOCUMENTATION", 0.75,
            f"description suggests docs/spec ({description[:80]!r}); "
            f"no dependency manifests",
            signals,
        )

    # --- CONFIGURATION ------------------------------------------------------
    if CONFIG_TOPICS & topics:
        return _result(
            "CONFIGURATION", 0.85,
            f"config/IaC topics: {sorted(CONFIG_TOPICS & topics)}",
            signals,
        )
    if (config_ratio > 0.5 and source_ratio < 0.2
            and any(kw in description for kw in ("dotfiles", "config", "infrastructure"))):
        return _result(
            "CONFIGURATION", 0.75,
            f"{config_ratio:.0%} config files; description matches "
            f"config keywords; only {source_ratio:.0%} source",
            signals,
        )

    # --- DATA ---------------------------------------------------------------
    if data_ratio > 0.5 and source_ratio < 0.15:
        return _result(
            "DATA", 0.80,
            f"{data_ratio:.0%} of files are data/asset; only "
            f"{source_ratio:.0%} source code",
            signals,
        )
    if DATA_TOPICS & topics and source_ratio < 0.2:
        return _result(
            "DATA", 0.70,
            f"dataset-related topics: {sorted(DATA_TOPICS & topics)}",
            signals,
        )

    # --- LIBRARY -----------------------------------------------------------
    lib = _looks_like_library(manifests, topics, description)
    if lib:
        return _result("LIBRARY", lib["confidence"], lib["reason"], signals)

    # --- APPLICATION (default for code repos) -------------------------------
    if source_ratio > 0.1 or has_dep_manifest:
        if APPLICATION_TOPICS & topics:
            return _result(
                "APPLICATION", 0.75,
                f"source code present and app-related topics: "
                f"{sorted(APPLICATION_TOPICS & topics)}",
                signals,
            )
        return _result(
            "APPLICATION", 0.60,
            f"source code present ({source_ratio:.0%}); no clear "
            f"library or non-code signals",
            signals,
        )

    # --- UNKNOWN -----------------------------------------------------------
    return _result(
        "UNKNOWN", 0.30,
        f"no clear classification signals; {total_files} files, "
        f"doc {doc_ratio:.0%} / source {source_ratio:.0%} / "
        f"data {data_ratio:.0%} / config {config_ratio:.0%}",
        signals,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result(repo_type: str, confidence: float, rationale: str, signals: dict) -> dict:
    return {
        "repo_type":  repo_type,
        "confidence": round(confidence, 2),
        "rationale":  rationale,
        "signals":    signals,
    }


def _count_extensions(file_tree: list) -> dict:
    """Return {extension: count} for file blobs in the tree."""
    counts: dict[str, int] = {}
    for entry in file_tree:
        if not isinstance(entry, dict) or entry.get("type") != "blob":
            continue
        path = (entry.get("path") or "").lower()
        if not path:
            continue
        basename = path.rsplit("/", 1)[-1]
        if "." not in basename:
            continue
        ext = "." + basename.rsplit(".", 1)[-1]
        counts[ext] = counts.get(ext, 0) + 1
    return counts


def _looks_like_library(manifests: dict, topics: set, description: str) -> Optional[dict]:
    """
    Detect "this repo IS a publishable package" rather than "this repo
    consumes packages." Library signals (in order of confidence):

      1. package.json with a public `name` (non-private)         → 0.85
      2. pyproject.toml declares a [project] section             → 0.80
      3. setup.py present                                        → 0.70
      4. Library-related topics                                  → 0.65
      5. Description explicitly says "library" or "framework"    → 0.55
    """
    # Case-insensitive manifest lookup so casing variants still match.
    by_name = {k.lower(): v for k, v in manifests.items()}

    # 1. package.json analysis
    pkg = by_name.get("package.json") or {}
    pkg_content = pkg.get("content") or ""
    if pkg_content:
        try:
            data = json.loads(pkg_content)
            name = data.get("name")
            is_private = bool(data.get("private"))
            has_entry_point = any(data.get(k) for k in ("main", "exports", "module"))
            if name and not is_private:
                return {
                    "confidence": 0.85,
                    "reason": f"package.json declares public package '{name}'",
                }
            if has_entry_point and not is_private:
                return {
                    "confidence": 0.75,
                    "reason": "package.json has library entry points (main/exports/module)",
                }
        except (json.JSONDecodeError, ValueError):
            pass

    # 2. pyproject.toml with [project] section
    pyproj = by_name.get("pyproject.toml") or {}
    pyproj_content = pyproj.get("content") or ""
    if pyproj_content and "[project]" in pyproj_content:
        return {
            "confidence": 0.80,
            "reason": "pyproject.toml declares a [project] section",
        }

    # 3. setup.py present
    if "setup.py" in by_name:
        return {
            "confidence": 0.70,
            "reason": "setup.py present (Python package)",
        }

    # 4. Topic-based fallback — repos that don't expose clean manifests
    # at the repo root (e.g. monorepos like facebook/react) often still
    # tag themselves with library/framework topics.
    matched = LIBRARY_TOPICS & topics
    if matched:
        return {
            "confidence": 0.65,
            "reason": f"library-related topics: {sorted(matched)}",
        }

    # 5. Description-based last resort
    lib_kw = ("library", "library for", "framework", "framework for",
              " sdk ", " sdk.", "javascript library", "python library",
              "package for", "client for")
    if any(kw in description for kw in lib_kw):
        return {
            "confidence": 0.55,
            "reason": f"description suggests library: {description[:80]!r}",
        }

    return None


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Smoke-test repo_classifier")
    parser.add_argument("--repo", required=True, help="owner/name")
    args = parser.parse_args()

    from github_fetcher import fetch_repo_data
    fetch = fetch_repo_data({"repo": args.repo})
    result = classify({"fetch": fetch})
    print(json.dumps(result, indent=2, default=str))
