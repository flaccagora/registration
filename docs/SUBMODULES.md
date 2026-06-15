# Submodule Management

The prototype expects upstream model and annotation repos under:

```text
external/vggt-omega/
external/Medical-SAM3/
external/manual-correspondences/
```

The upstream URLs are known and are recorded in the root `.gitmodules` file:

```text
https://github.com/facebookresearch/vggt-omega.git
https://github.com/AIM-Research-Lab/Medical-SAM3.git
https://github.com/flaccagora/label.git
```

The `.gitmodules` file records intended submodule URLs. The actual submodule
gitlinks are created only after `git submodule add ...` is run and committed in
the parent repository.

The current workspace may still contain legacy root-level checkouts:

```text
vggt-omega/
Medical-SAM3/
manual-correspondences/
```

The code prefers `external/...` paths but falls back to those legacy locations
when the external submodules have not been initialized yet.

## Convert Existing Root Checkouts To Submodules

Run later, after committing or backing up any local changes inside the external
repos:

```bash
mkdir -p external
git submodule add https://github.com/facebookresearch/vggt-omega.git external/vggt-omega
git submodule add https://github.com/AIM-Research-Lab/Medical-SAM3.git external/Medical-SAM3
git submodule add https://github.com/flaccagora/label.git external/manual-correspondences
git status
git add .gitmodules external/vggt-omega external/Medical-SAM3 external/manual-correspondences
git commit -m "chore: add external repos as submodules"
```

If `git submodule add` reports that a `.gitmodules` entry already exists, check
that the URL and path match this document, remove any duplicate entry Git added,
then continue with the single correct entry.

If root-level `vggt-omega/`, `Medical-SAM3/`, and `manual-correspondences/`
already exist, do not `git add` them. They are ignored as legacy local
checkouts. After the submodules work, archive or remove the root-level copies
manually if they are no longer needed.

## Clone With Submodules

```bash
git clone --recurse-submodules <prototype-repo-url>
cd registration
```

If already cloned:

```bash
git submodule update --init --recursive
```

## Pin Submodule Commits

Submodules are pinned by committing the gitlink from the parent repo.

```bash
cd external/vggt-omega
git status
git checkout <known-good-commit>
cd ../..
git status
git add external/vggt-omega
git commit -m "chore: pin VGGT-Omega submodule"
```

Repeat the same pattern for `external/Medical-SAM3` and
`external/manual-correspondences`.

## Update Submodules Safely

```bash
cd external/vggt-omega
git fetch origin
git checkout main
git pull --ff-only
cd ../..
git status
git add external/vggt-omega
git commit -m "chore: update VGGT-Omega submodule"
```

Inspect upstream release notes and rerun compatibility checks before updating
the pinned commit in shared branches.

## Recover From Detached HEAD

Submodules are often checked out at a detached commit. To make local changes:

```bash
cd external/vggt-omega
git switch -c experiment/my-vggt-change
```

To discard local submodule work and return to the parent-pinned commit:

```bash
git submodule update --init --recursive external/vggt-omega
```

Do this only after saving any local changes you care about.

## Avoid Committing Heavy Files

Keep checkpoints and generated outputs outside Git:

```text
checkpoints/
outputs/
runs/
demo_outputs/
cache/
```

Use `.env` for local paths and `.env.example` for shareable defaults.
