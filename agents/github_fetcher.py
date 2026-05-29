"""
github_fetcher.py — App Rationalization Agent
=============================================
Fetches raw repository data from the GitHub REST API. This is the first
stage of the pipeline; downstream agents (code_analyzer, dependency_scanner,
activity_analyzer) consume the structured dict this module returns.

Returned dict shape:
    {
        "repo": "owner/name",
        "metadata": { ... },           # /repos/{owner}/{repo}, projected
        "default_branch": "main",
        "file_tree": [ { "path": ..., "type": ..., "size": ... }, ... ],
        "tree_truncated": bool,        # GitHub truncates trees > 100k entries
        "readme": { "name": ..., "content": str | None, "size": ... } | None,
        "manifests": {
            "package.json":      { "path": ..., "content": str, "size": int },
            "requirements.txt":  { "path": ..., "content": str, "size": int },
            ...
        },
        "rate_limit": { "remaining": int, "limit": int, "reset_at": iso8601 },
        "errors": [ "..." ]            # non-fatal issues encountered
    }
"""

from __future__ import annotations

import base64
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

GITHUB_API = "https://api.github.com"

# Manifest filenames we look for in the tree. Matching is case-insensitive
# on the *basename*. Lockfiles are included so dependency_scanner can use
# them when present (they pin resolved versions).
MANIFEST_FILENAMES = {
    # JavaScript / TypeScript
    "package.json",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    # Node version pin files — strictly not "manifests" but the
    # dependency_scanner reads them as a runtime fallback for repos
    # that pin Node outside of package.json's engines field.
    ".nvmrc",
    ".node-version",
    # Python
    "requirements.txt",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "pipfile",
    "pipfile.lock",
    "poetry.lock",
    # Java / Kotlin / Scala
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "settings.gradle",
    "build.sbt",
    # Ruby
    "gemfile",
    "gemfile.lock",
    # Go
    "go.mod",
    "go.sum",
    # Rust
    "cargo.toml",
    "cargo.lock",
    # PHP
    "composer.json",
    "composer.lock",
    # .NET
    "packages.config",
    "paket.dependencies",
    # Container / infra
    "dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
}

# .NET project files use a suffix rather than a fixed name.
MANIFEST_SUFFIXES = (".csproj", ".fsproj", ".vbproj")

DEFAULT_TIMEOUT = 30.0
MAX_RETRIES = 4
BACKOFF_BASE = 2.0
# Skip manifests larger than this — lockfiles can hit MBs and downstream
# scanners only need to parse modest manifests for signals.
MAX_MANIFEST_BYTES = 1_000_000


# ---------------------------------------------------------------------------
# HTTP client with retry / rate-limit handling
# ---------------------------------------------------------------------------

def _build_client(token: Optional[str]) -> httpx.Client:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "app-rationalization-agent",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return httpx.Client(base_url=GITHUB_API, headers=headers, timeout=DEFAULT_TIMEOUT)


def _request(client: httpx.Client, path: str, params: Optional[dict] = None) -> httpx.Response:
    """GET with exponential backoff on rate-limit and transient errors."""
    last_exc: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.get(path, params=params)
        except httpx.HTTPError as e:
            last_exc = e
            time.sleep(BACKOFF_BASE ** attempt)
            continue

        remaining = resp.headers.get("X-RateLimit-Remaining")
        if resp.status_code in (403, 429) and remaining == "0":
            reset = resp.headers.get("X-RateLimit-Reset")
            wait = _seconds_until(reset) if reset else BACKOFF_BASE ** attempt
            # Cap the wait so a misconfigured token doesn't stall the pipeline.
            time.sleep(min(wait, 60))
            continue

        if resp.status_code >= 500 or resp.status_code == 429:
            time.sleep(BACKOFF_BASE ** attempt)
            continue

        return resp

    if last_exc:
        raise last_exc
    raise RuntimeError(f"GitHub request to {path} exhausted retries")


def _seconds_until(epoch_str: str) -> float:
    try:
        target = int(epoch_str)
    except (TypeError, ValueError):
        return BACKOFF_BASE
    return max(0.0, target - time.time())


def _rate_limit_snapshot(resp: httpx.Response) -> dict:
    reset = resp.headers.get("X-RateLimit-Reset")
    reset_iso: Optional[str] = None
    if reset:
        try:
            reset_iso = datetime.fromtimestamp(int(reset), tz=timezone.utc).isoformat()
        except (TypeError, ValueError):
            reset_iso = None
    return {
        "remaining": _safe_int(resp.headers.get("X-RateLimit-Remaining")),
        "limit": _safe_int(resp.headers.get("X-RateLimit-Limit")),
        "reset_at": reset_iso,
    }


def _safe_int(v: Optional[str]) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Endpoint helpers
# ---------------------------------------------------------------------------

def _fetch_metadata(client: httpx.Client, owner: str, name: str) -> httpx.Response:
    return _request(client, f"/repos/{owner}/{name}")


def _fetch_tree(
    client: httpx.Client, owner: str, name: str, branch: str
) -> tuple[list[dict], bool]:
    """Returns (entries, truncated_flag). Entries are GitHub's raw tree items."""
    resp = _request(
        client,
        f"/repos/{owner}/{name}/git/trees/{branch}",
        params={"recursive": "1"},
    )
    if resp.status_code != 200:
        return [], False
    data = resp.json()
    return data.get("tree", []), bool(data.get("truncated", False))


def _fetch_readme(client: httpx.Client, owner: str, name: str) -> Optional[dict]:
    resp = _request(client, f"/repos/{owner}/{name}/readme")
    if resp.status_code != 200:
        return None
    data = resp.json()
    return {
        "name": data.get("name"),
        "path": data.get("path"),
        "size": data.get("size"),
        "content": _decode_content(data),
        "html_url": data.get("html_url"),
    }


def _fetch_file(client: httpx.Client, owner: str, name: str, path: str) -> Optional[dict]:
    resp = _request(client, f"/repos/{owner}/{name}/contents/{path}")
    if resp.status_code != 200:
        return None
    data = resp.json()
    if isinstance(data, list):
        # Path resolved to a directory; not a file we can decode.
        return None
    return {
        "path": data.get("path"),
        "size": data.get("size"),
        "sha": data.get("sha"),
        "content": _decode_content(data),
    }


def _decode_content(payload: dict) -> Optional[str]:
    """Decode the base64 content field returned by /contents and /readme."""
    raw = payload.get("content")
    encoding = payload.get("encoding", "base64")
    if not raw:
        return None
    if encoding != "base64":
        return raw
    try:
        return base64.b64decode(raw).decode("utf-8", errors="replace")
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Manifest selection
# ---------------------------------------------------------------------------

# Vendor / build directories we never want to pull manifests from.
_SKIP_PATH_SEGMENTS = (
    "node_modules/",
    "vendor/",
    ".venv/",
    "dist/",
    "build/",
    "site-packages/",
)


def _select_manifest_paths(tree_entries: list[dict]) -> list[str]:
    """
    From the recursive tree, pick file paths that look like dependency
    manifests. Keeps only the shallowest match per manifest type so we
    don't fetch every transitive package.json.
    """
    shallowest: dict[str, tuple[int, str]] = {}

    for entry in tree_entries:
        if entry.get("type") != "blob":
            continue
        path = entry.get("path") or ""
        if not path:
            continue

        lower_path = path.lower()
        if any(seg in lower_path for seg in _SKIP_PATH_SEGMENTS):
            continue

        basename = path.rsplit("/", 1)[-1].lower()
        suffix_match = next(
            (s for s in MANIFEST_SUFFIXES if basename.endswith(s)), None
        )
        is_manifest = basename in MANIFEST_FILENAMES or suffix_match is not None
        if not is_manifest:
            continue

        size = entry.get("size") or 0
        if size > MAX_MANIFEST_BYTES:
            continue

        depth = path.count("/")
        # Group .csproj/.fsproj/.vbproj by suffix so we pick one representative.
        key = suffix_match if suffix_match else basename
        prev = shallowest.get(key)
        if prev is None or depth < prev[0]:
            shallowest[key] = (depth, path)

    return [p for _, p in shallowest.values()]


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def fetch_repo_data(repo_data: dict) -> dict:
    """
    Fetch metadata, file tree, README, and dependency manifests for a repo.

    Args:
        repo_data: shared pipeline dict; must contain `repo` as "owner/name".

    Returns:
        Structured dict (see module docstring for shape).
    """
    repo = repo_data.get("repo")
    if not repo or "/" not in repo:
        raise ValueError(f"Expected 'owner/name' repo, got: {repo!r}")
    owner, name = repo.split("/", 1)

    token = os.getenv("GITHUB_TOKEN")
    errors: list[str] = []
    result: dict[str, Any] = {
        "repo": repo,
        "metadata": None,
        "default_branch": None,
        "file_tree": [],
        "tree_truncated": False,
        "readme": None,
        "manifests": {},
        "rate_limit": None,
        "errors": errors,
    }

    with _build_client(token) as client:
        # --- Metadata ---------------------------------------------------------
        meta_resp = _fetch_metadata(client, owner, name)
        result["rate_limit"] = _rate_limit_snapshot(meta_resp)
        if meta_resp.status_code == 404:
            raise FileNotFoundError(f"Repository not found: {repo}")
        if meta_resp.status_code == 401:
            raise PermissionError("GitHub authentication failed — check GITHUB_TOKEN")
        if meta_resp.status_code != 200:
            raise RuntimeError(
                f"GitHub metadata fetch failed ({meta_resp.status_code}): "
                f"{meta_resp.text[:200]}"
            )

        meta = meta_resp.json()
        default_branch = meta.get("default_branch") or "main"
        result["metadata"] = _summarize_metadata(meta)
        result["default_branch"] = default_branch

        # --- File tree --------------------------------------------------------
        tree: list[dict] = []
        try:
            tree, truncated = _fetch_tree(client, owner, name, default_branch)
            result["file_tree"] = [
                {
                    "path": t.get("path"),
                    "type": t.get("type"),
                    "size": t.get("size"),
                }
                for t in tree
            ]
            result["tree_truncated"] = truncated
        except Exception as e:
            errors.append(f"file_tree: {e}")

        # --- README -----------------------------------------------------------
        try:
            result["readme"] = _fetch_readme(client, owner, name)
        except Exception as e:
            errors.append(f"readme: {e}")

        # --- Manifests --------------------------------------------------------
        for path in _select_manifest_paths(tree):
            try:
                file_data = _fetch_file(client, owner, name, path)
                if file_data is None:
                    continue
                # Preserve original-case basename so consumers see Dockerfile,
                # Gemfile, Cargo.toml, etc. as they appear in the repo.
                basename = path.rsplit("/", 1)[-1]
                result["manifests"][basename] = file_data
            except Exception as e:
                errors.append(f"manifest {path}: {e}")

    return result


def _summarize_metadata(meta: dict) -> dict:
    """Project the GitHub repo payload down to the fields downstream agents use."""
    license_info = meta.get("license") or {}
    return {
        "full_name":      meta.get("full_name"),
        "description":    meta.get("description"),
        "html_url":       meta.get("html_url"),
        "homepage":       meta.get("homepage"),
        "language":       meta.get("language"),
        "topics":         meta.get("topics", []),
        "size_kb":        meta.get("size"),
        "stargazers":     meta.get("stargazers_count"),
        "forks":          meta.get("forks_count"),
        "watchers":       meta.get("subscribers_count"),
        "open_issues":    meta.get("open_issues_count"),
        "archived":       meta.get("archived"),
        "disabled":       meta.get("disabled"),
        "fork":           meta.get("fork"),
        "is_template":    meta.get("is_template"),
        "license":        license_info.get("spdx_id") if license_info else None,
        "default_branch": meta.get("default_branch"),
        "created_at":     meta.get("created_at"),
        "updated_at":     meta.get("updated_at"),
        "pushed_at":      meta.get("pushed_at"),
    }


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Smoke-test github_fetcher")
    parser.add_argument("--repo", required=True, help="owner/name")
    args = parser.parse_args()

    data = fetch_repo_data({"repo": args.repo})

    # Trim large content blobs so the preview is readable.
    preview = json.loads(json.dumps(data, default=str))
    if preview.get("readme") and preview["readme"].get("content"):
        preview["readme"]["content"] = preview["readme"]["content"][:300] + "..."
    for v in preview.get("manifests", {}).values():
        if v.get("content"):
            v["content"] = v["content"][:300] + "..."
    preview["file_tree"] = preview["file_tree"][:20]
    print(json.dumps(preview, indent=2))
