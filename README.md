# Surgical Registration Prototype

Research prototype for registering a 2D surgical or medical image to a 3D mesh
surface using sparse manual 2D-to-3D correspondences, VGGT-Omega depth/camera
estimates, MedicalSAM3 segmentation masks, initial pose registration, and
deformable refinement.

Warning: this is not clinically validated, is not a medical device, and must
not be used for diagnosis, treatment, surgical navigation, or patient care.

Start with the documentation in [`docs/README.md`](docs/README.md).

Key entry points:

- [`app.py`](app.py) - Gradio demo, launched later by the user.
- [`pipeline/`](pipeline) - modular Python package.
- [`examples/correspondences/`](examples/correspondences) - CSV and JSON schema examples.
- [`scripts/dry_run.py`](scripts/dry_run.py) - synthetic geometry dry run, not executed during setup.
- [`docs/GIT_WORKFLOW.md`](docs/GIT_WORKFLOW.md) - Git workflow and first commit instructions.
- [`docs/SUBMODULES.md`](docs/SUBMODULES.md) - external model repository submodule workflow.

Preferred external model repo layout:

```text
external/vggt-omega/
external/Medical-SAM3/
```

Existing root-level `vggt-omega/` and `Medical-SAM3/` checkouts are supported
as a legacy fallback until submodules are initialized.
