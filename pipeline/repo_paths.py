"""Repository path resolution for local external model checkouts."""

from __future__ import annotations

from pathlib import Path


PREFERRED_REPO_PATHS = {
    "vggt_omega": Path("external/vggt-omega"),
    "medicalsam3": Path("external/Medical-SAM3"),
}

LEGACY_REPO_PATHS = {
    "vggt_omega": Path("vggt-omega"),
    "medicalsam3": Path("Medical-SAM3"),
}


def resolve_repo_path(configured_path: str | Path, repo_key: str) -> Path:
    """Prefer the configured repo path, with legacy/default fallback.

    The project now documents model repos as Git submodules under
    ``external/``. Existing workspaces may still have root-level checkouts.
    This helper lets both layouts work without hard-coding absolute paths.
    """

    configured = Path(configured_path)
    preferred = PREFERRED_REPO_PATHS[repo_key]
    legacy = LEGACY_REPO_PATHS[repo_key]
    candidates = [configured]
    if configured == preferred:
        candidates.append(legacy)
    elif configured == legacy:
        candidates.append(preferred)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return configured

