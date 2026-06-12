# Git Workflow

This project should track the prototype source, docs, small examples, and
configuration templates. External model repos, checkpoints, generated outputs,
and clinical/research data should stay out of the prototype Git history.

## First-Time Setup

Run later from the repository root:

```bash
git init
git status
git add .gitignore .gitmodules .env.example README.md pyproject.toml app.py pipeline docs examples scripts external/.gitkeep
git commit -m "chore: initialize surgical registration prototype"
```

If Git reports ignored files such as `vggt-omega/`, `Medical-SAM3/`,
`manual-correspondences/`, `outputs/`, or `checkpoints/`, that is expected.

## Branch Naming

Use short, scoped branch names:

```text
feature/gradio-registration-flow
fix/vggt-cache-shapes
docs/submodule-workflow
chore/repo-layout
```

Recommended prefixes:

- `feature/` for new prototype behavior.
- `fix/` for bug fixes.
- `docs/` for documentation-only changes.
- `chore/` for repository maintenance.
- `experiment/` for disposable research branches.

## Commit Messages

Use imperative, scoped messages:

```text
feat: add deformable RBF refinement
fix: normalize VGGT single-frame depth outputs
docs: document submodule setup
chore: ignore generated model outputs
```

Keep each commit focused. Do not mix generated outputs, checkpoints, or local
data with source changes.

## Day-to-Day Commands

```bash
git status
git diff
git add app.py pipeline/registration.py docs/REGISTRATION.md
git commit -m "fix: harden deformable control filtering"
```

Before committing, inspect staged changes:

```bash
git diff --cached
git status --short
```

## Do Not Commit

- Model checkpoints: `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`.
- Generated outputs: `outputs/`, `runs/`, `demo_outputs/`, `cache/`.
- Real clinical/research media or annotations.
- Root-level legacy external repo checkouts.
- `.env` files with local paths or secrets.

Use `.env.example` for documented environment variables.

