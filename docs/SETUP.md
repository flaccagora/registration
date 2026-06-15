# Setup Instructions

These are instructions for later use. No dependencies were installed and no
pipeline code was executed while creating this prototype.

## Python Environment

Recommended:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install this prototype and general dependencies later:

```bash
pip install -e .
```

The root `pyproject.toml` lists general dependencies used by the prototype:
NumPy, Pillow, OpenCV, trimesh, and Gradio. PyTorch is listed as an optional
model dependency because users often need a CUDA-specific wheel.

## Local Repositories

The prototype now prefers upstream model and annotation repos as Git submodules:

```text
external/vggt-omega/
external/Medical-SAM3/
external/manual-correspondences/
```

Existing root-level local checkouts are still supported as a fallback:

```text
vggt-omega/
Medical-SAM3/
manual-correspondences/
```

Use `.env.example` as the starting point for local paths:

```bash
cp .env.example .env
```

You can also override paths in the Gradio UI or in a config file such as
`examples/config.example.json`.

## Submodules

The root `.gitmodules` records the upstream model and annotation repository
URLs. Initialize them later with:

```bash
git submodule update --init --recursive
```

See `docs/SUBMODULES.md` for clone, update, pinning, and detached-HEAD
recovery workflows.

## VGGT-Omega

Follow the local `external/vggt-omega/README.md` for exact installation and
checkpoint access. If the submodule is not initialized yet, the wrapper can
fall back to a legacy root-level `vggt-omega/` checkout. The wrapper expects:

- importable `vggt_omega`
- a local checkpoint path, for example `checkpoints/vggt_omega_1b_512.pt`
- CUDA by default unless you select CPU

The prototype does not download or install VGGT-Omega checkpoints.

## MedicalSAM3

Follow `external/Medical-SAM3/README.md` for exact installation and checkpoint
access. If the submodule is not initialized yet, the wrapper can fall back to a
legacy root-level `Medical-SAM3/` checkout. The wrapper expects:

- importable MedicalSAM3/SAM3 dependencies
- optional checkpoint path for local weights
- a prompt: text, box, point, or existing mask
- CUDA for text, point, and box prompts because the local
  `inference/sam3_inference.py` helper enters CUDA autocast internally

If no checkpoint is supplied, the local MedicalSAM3 inference code may attempt
its own default loading behavior. Use an explicit checkpoint for reproducible
offline work.

Mask prompts bypass MedicalSAM3 model loading and can be used in CPU-only
structural tests.

## Manual Correspondences

The preferred annotation repo path is:

```text
external/manual-correspondences/
```

The registration pipeline accepts both its own canonical CSV/JSON schema and
the normalized registration JSON records produced by the `manual-correspondences`
repo. For normalized records, each landmark's `ct_landmark_id` is resolved via
the frame record's `ct_landmarks_path` or an embedded landmark catalog.

## Optional Dry Run

After installing dependencies, a synthetic geometry-only check is available:

```bash
python scripts/dry_run.py --out-dir outputs/dry_run
```

This does not call MedicalSAM3, VGGT-Omega, or Gradio. It only tests
correspondence parsing, PnP, metrics, and mesh export.

## Static Layout Check

After Git/submodule setup, a lightweight standard-library checker is available:

```bash
python scripts/check_repo_layout.py
```

It does not import model code or heavy dependencies.
