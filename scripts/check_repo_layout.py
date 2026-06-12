"""Static repository layout checker.

This script uses only the Python standard library. It does not import the
prototype package, model repositories, Gradio, torch, OpenCV, or any heavy
dependencies.
"""

from __future__ import annotations

import argparse
from pathlib import Path


EXPECTED_FILES = [
    ".gitignore",
    ".gitmodules",
    ".env.example",
    "README.md",
    "pyproject.toml",
    "app.py",
    "docs/README.md",
    "docs/SETUP.md",
    "docs/GIT_WORKFLOW.md",
    "docs/SUBMODULES.md",
    "docs/UV_ENVIRONMENT.md",
    "docs/COMPATIBILITY_REPORT.md",
    "docs/CORRESPONDENCES.md",
    "docs/PIPELINE.md",
    "docs/REGISTRATION.md",
    "docs/GRADIO_DEMO.md",
    "docs/TROUBLESHOOTING.md",
    "examples/config.example.json",
    "pipeline/config.py",
    "pipeline/depth_vggtomega.py",
    "pipeline/segmentation_medicalsam3.py",
    "pipeline/registration.py",
]

EXPECTED_SUBMODULES = {
    "external/vggt-omega": "https://github.com/facebookresearch/vggt-omega.git",
    "external/Medical-SAM3": "https://github.com/AIM-Research-Lab/Medical-SAM3.git",
}

LEGACY_REPOS = [
    "vggt-omega",
    "Medical-SAM3",
]


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def appears_to_be_submodule(path: Path) -> bool:
    git_marker = path / ".git"
    if git_marker.is_file():
        return read_text(git_marker).strip().startswith("gitdir:")
    return False


def appears_to_be_nested_checkout(path: Path) -> bool:
    return (path / ".git").is_dir()


def check(root: Path) -> int:
    failures = 0
    warnings = 0

    print(f"Repository root: {root}")

    for relative in EXPECTED_FILES:
        path = root / relative
        if path.exists():
            print(f"OK   file: {relative}")
        else:
            print(f"FAIL missing file: {relative}")
            failures += 1

    gitmodules = read_text(root / ".gitmodules")
    for relative, url in EXPECTED_SUBMODULES.items():
        path = root / relative
        listed = relative in gitmodules and url in gitmodules
        if listed:
            print(f"OK   .gitmodules entry: {relative}")
        else:
            print(f"FAIL .gitmodules missing entry/url for: {relative}")
            failures += 1

        if not path.exists():
            print(f"WARN submodule path missing until initialized: {relative}")
            warnings += 1
        elif appears_to_be_submodule(path):
            print(f"OK   submodule checkout: {relative}")
        elif appears_to_be_nested_checkout(path):
            print(f"WARN nested Git checkout, not a parent-tracked submodule: {relative}")
            warnings += 1
        else:
            print(f"WARN path exists but does not look like a Git submodule: {relative}")
            warnings += 1

    for relative in LEGACY_REPOS:
        path = root / relative
        if path.exists():
            print(f"WARN legacy root-level external repo present: {relative}")
            warnings += 1

    for generated in ("outputs", "checkpoints", "cache", "runs", "demo_outputs"):
        path = root / generated
        if path.exists():
            print(f"WARN generated/heavy directory present: {generated}")
            warnings += 1

    print(f"Summary: {failures} failure(s), {warnings} warning(s)")
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Statically check prototype repository layout.")
    parser.add_argument("--root", default=".", help="Repository root to inspect.")
    args = parser.parse_args()
    raise SystemExit(check(Path(args.root).resolve()))


if __name__ == "__main__":
    main()
