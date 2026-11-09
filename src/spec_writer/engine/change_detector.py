"""Change detection — git-first, hash-fallback.

Two routes:

- **Git** (``detect_via_git``) — runs ``git diff --name-status
  <since_commit>..HEAD`` and parses the output. Returns
  ``DetectionResult(added=[...], modified=[...], deleted=[...],
  baseline_lost=False)`` on success. If ``since_commit`` is
  unreachable from HEAD (force-push / branch reset),
  ``baseline_lost=True`` with empty lists.

- **Hash fallback** (``detect_via_hash``) — walks ``includes /
  excludes`` from scratch, computes a sha256 over sorted
  ``(path, size, mtime_ns)`` tuples, compares to the previous hash
  stored in :class:`SyncState`. Only tells "something changed" /
  "nothing changed" — does not return file lists. Returns
  :class:`DetectionResult` with ``hash_changed: bool``.

The agent picks the route at call time; both are exposed as a
single ``detect_code_changes`` MCP tool that auto-selects (git if
the repo has ``.git/``, else hash).
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


def _empty_strs() -> list[str]:
    return []


def _empty_rename_pairs() -> list[tuple[str, str]]:
    return []


@dataclass(frozen=True)
class DetectionResult:
    added: list[str] = field(default_factory=_empty_strs)
    modified: list[str] = field(default_factory=_empty_strs)
    deleted: list[str] = field(default_factory=_empty_strs)
    renamed: list[tuple[str, str]] = field(default_factory=_empty_rename_pairs)
    baseline_lost: bool = False
    hash_changed: bool = False
    method: str = "none"  # "git" / "hash" / "none"
    current_commit: str | None = None
    current_hash: str | None = None
    error: str | None = None


def is_git_repo(project_root: Path) -> bool:
    return (project_root / ".git").exists()


# ── git route ────────────────────────────────────────────────────────────


def detect_via_git(project_root: Path, since_commit: str | None) -> DetectionResult:
    head = _run_git(project_root, ["rev-parse", "HEAD"])
    if head.returncode != 0:
        return DetectionResult(error=head.stderr.strip(), method="git")
    current_commit = head.stdout.strip()

    if since_commit is None:
        # First-time sync: no baseline; nothing detected, just record HEAD.
        return DetectionResult(method="git", current_commit=current_commit)

    rev = _run_git(project_root, ["rev-parse", "--verify", f"{since_commit}^{{commit}}"])
    if rev.returncode != 0:
        return DetectionResult(
            baseline_lost=True,
            method="git",
            current_commit=current_commit,
            error=rev.stderr.strip() or "baseline commit unreachable",
        )

    diff = _run_git(
        project_root,
        ["diff", "--name-status", "--diff-filter=ACDMR", f"{since_commit}..HEAD"],
    )
    if diff.returncode != 0:
        return DetectionResult(
            error=diff.stderr.strip(),
            method="git",
            current_commit=current_commit,
        )

    added: list[str] = []
    modified: list[str] = []
    deleted: list[str] = []
    renamed: list[tuple[str, str]] = []
    for line in diff.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0][:1]
        if status == "A" and len(parts) == 2:
            added.append(parts[1])
        elif status == "M" and len(parts) == 2:
            modified.append(parts[1])
        elif status == "D" and len(parts) == 2:
            deleted.append(parts[1])
        elif status == "R" and len(parts) == 3:
            renamed.append((parts[1], parts[2]))

    return DetectionResult(
        added=added,
        modified=modified,
        deleted=deleted,
        renamed=renamed,
        method="git",
        current_commit=current_commit,
    )


def _run_git(project_root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )


# ── hash route ───────────────────────────────────────────────────────────


def detect_via_hash(
    project_root: Path,
    includes: list[str],
    excludes: list[str],
    previous_hash: str | None,
) -> DetectionResult:
    current = compute_code_scope_hash(project_root, includes, excludes)
    changed = previous_hash is not None and current != previous_hash
    return DetectionResult(
        hash_changed=changed,
        method="hash",
        current_hash=current,
    )


def compute_code_scope_hash(project_root: Path, includes: list[str], excludes: list[str]) -> str:
    """Stable hash of the project's tracked code surface.

    Deliberately ignores file contents — comparing sha256 over every
    file's bytes on a 10k-file repo is expensive. Path + size + mtime_ns
    is enough to catch real edits in practice; for git repos the agent
    should prefer :func:`detect_via_git` anyway.
    """
    import fnmatch

    if not project_root.exists():
        return ""

    digest = hashlib.sha256()
    for path in sorted(project_root.rglob("*")):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(project_root)
        except ValueError:
            continue
        rel_posix = str(rel).replace("\\", "/")
        if includes and not any(fnmatch.fnmatch(rel_posix, pat) for pat in includes):
            continue
        if any(fnmatch.fnmatch(rel_posix, pat) for pat in excludes):
            continue
        try:
            st = path.stat()
            entry = f"{rel_posix}\t{st.st_size}\t{st.st_mtime_ns}\n"
        except OSError:
            entry = f"{rel_posix}\t?\t?\n"
        digest.update(entry.encode("utf-8"))
    return digest.hexdigest()


__all__ = [
    "DetectionResult",
    "compute_code_scope_hash",
    "detect_via_git",
    "detect_via_hash",
    "is_git_repo",
]
