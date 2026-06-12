# Surgical Registration Prototype

This project is a research prototype for registering 2D surgical or medical
imagery to a 3D mesh surface. It combines sparse manual 2D-to-3D
correspondences, MedicalSAM3 segmentation, VGGT-Omega camera/depth estimates,
an initial rigid or similarity pose, and a final regularized deformable
refinement.

Warning: this prototype is not clinically validated, is not a medical device,
and must not be used for diagnosis, treatment, surgical navigation, or patient
care.

## Expected Folder Layout

```text
registration/
  app.py
  pyproject.toml
  pipeline/
    config.py
    io.py
    correspondences.py
    segmentation_medicalsam3.py
    depth_vggtomega.py
    registration.py
    visualization.py
    export.py
  docs/
  examples/
    config.example.json
    correspondences/
  scripts/
    dry_run.py
    check_repo_layout.py
  external/
    vggt-omega/
    Medical-SAM3/
```

The model repositories are expected to be Git submodules under `external/`.
Existing root-level `vggt-omega/` and `Medical-SAM3/` checkouts remain
supported as a legacy fallback until the submodules are initialized.

## Quick Start Later

Do not run these commands until the environment is prepared.

```bash
cp .env.example .env
git submodule update --init --recursive
```

```bash
python app.py
```

For a geometry-only dry run that does not call MedicalSAM3, VGGT-Omega, or
Gradio:

```bash
python scripts/dry_run.py --out-dir outputs/dry_run
```

## Documentation Map

- [`SETUP.md`](SETUP.md) - environment and dependency setup instructions.
- [`GIT_WORKFLOW.md`](GIT_WORKFLOW.md) - Git initialization and commit workflow.
- [`SUBMODULES.md`](SUBMODULES.md) - submodule setup, pinning, updating, and recovery.
- [`UV_ENVIRONMENT.md`](UV_ENVIRONMENT.md) - optional future `uv` workflow notes.
- [`COMPATIBILITY_REPORT.md`](COMPATIBILITY_REPORT.md) - static compatibility notes.
- [`CORRESPONDENCES.md`](CORRESPONDENCES.md) - manual correspondence schema.
- [`PIPELINE.md`](PIPELINE.md) - end-to-end stage description.
- [`REGISTRATION.md`](REGISTRATION.md) - initial pose and deformable refinement details.
- [`GRADIO_DEMO.md`](GRADIO_DEMO.md) - how to launch and use the demo later.
- [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) - common runtime problems.

## Current TODOs

- Validate coordinate-frame conventions on real VGGT-Omega outputs for the
  chosen clinical/research dataset.
- Add ARAP or embedded deformation graph refinement as an alternative to the
  current RBF control-point warp.
- Add mesh silhouette rendering and mask-overlap optimization.
- Add formal unit tests around PnP failure modes, malformed correspondence
  files, and deformation regularization.
- Add dataset-specific adapters for existing `manual-correspondences/` Label
  Studio exports.
- Replace legacy root-level model checkouts with initialized submodules under
  `external/` in shared clones.

## Known Limitations

- The deformable model is a smooth RBF warp, not a biomechanical model.
- Sparse deformation controls are filtered for non-finite values, very low
  weight, and optional outlier displacement, but this is only numerical
  stabilization.
- VGGT-Omega point maps may be produced at model preprocessing resolution. The
  prototype scales manual pixels into the point-map grid when image size is
  known, but real workflows should verify resize/crop conventions explicitly.
- The segmentation mask currently filters or weights sparse constraints; it is
  not yet optimized as a dense silhouette loss.
- Intrinsics estimated from image size are only a fallback. Calibrated camera
  intrinsics are preferred for meaningful reprojection metrics.
- The current local MedicalSAM3 inference helper uses CUDA autocast for
  model-backed prompts; CPU-only checks should use precomputed mask prompts.
- No clinical validation has been performed.
