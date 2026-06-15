# Troubleshooting

## Missing Checkpoints

Symptoms:

- `VGGT-Omega checkpoint_path is required`
- `checkpoint not found`
- model loading errors

Fix:

- Download checkpoints according to the upstream repository instructions.
- Pass checkpoint paths in the Gradio UI.
- Keep paths relative to the project root or use absolute paths.

## Import Errors

Symptoms:

- `VGGT-Omega dependencies are not importable`
- `Pillow is required`
- `opencv-python is required`
- `gradio is not installed`

Fix:

- Install the root prototype dependencies.
- Install local VGGT-Omega and MedicalSAM3 dependencies according to their
  READMEs.
- Use the same Python environment for the demo and local repos.

## CUDA Issues

Symptoms:

- CUDA requested but unavailable
- PyTorch cannot load CUDA kernels
- out-of-memory errors
- MedicalSAM3 text, box, or point prompt fails on CPU

Fix:

- Select `cpu` in the UI for structural debugging, understanding that model
  inference may be slow or unsupported.
- Use an existing mask prompt when doing CPU-only structural tests.
- Use `cuda` for MedicalSAM3 text, point, and box prompts with the current
  local inference helper.
- Install a PyTorch build matching the machine's CUDA version.
- Reduce VGGT image resolution to 256.
- Use fewer video frames.

## Bad Correspondence Files

Symptoms:

- missing numeric field errors
- PnP requires at least four correspondences
- non-finite values

Fix:

- Validate against `docs/CORRESPONDENCES.md`.
- Ensure pixel coordinates are in image pixels, not normalized percentages.
- Ensure 3D points are in the mesh coordinate frame.
- Use at least four non-coplanar points for robust PnP when possible.

## Registration Looks Shifted

Common causes:

- wrong camera intrinsics
- wrong image size after resizing/cropping
- mismatched `image_id` filter
- mesh coordinate frame differs from correspondence coordinate frame
- VGGT point-map frame does not match the assumed initial pose frame
- VGGT point-map resolution differs from the manual annotation image and the
  resize/crop convention has not been verified

Fix:

- Start with manual correspondence overlay.
- Check initial reprojection error table.
- Run with known calibration if available.
- Verify the same image was used for annotation, segmentation, and VGGT.

## Deformable Refinement Is Skipped

The RBF deformable stage needs VGGT point-map samples at manual correspondence
pixels. If VGGT has not run, or if point-map samples are invalid, the stage
exports the initial registered mesh and reports a skipped status.

If it fails with too few stable controls after filtering, inspect manual
correspondences, segmentation-mask filtering, and the VGGT point-map coordinate
mapping. The safety filters may have removed non-finite, low-weight, or
outlier controls.

## Path Issues

Avoid hard-coding absolute paths in code. Use UI fields, config JSON, or command
arguments. Relative paths are interpreted from the current working directory.

The preferred external repo paths are:

```text
external/vggt-omega/
external/Medical-SAM3/
external/manual-correspondences/
```

If submodules are missing, run:

```bash
git submodule update --init --recursive
```

Legacy root-level checkouts `vggt-omega/`, `Medical-SAM3/`, and
`manual-correspondences/` are still supported as fallbacks, but should not be
committed into the prototype Git history.

## Manual Correspondence Export Errors

Symptoms:

- `Could not resolve all manual-correspondences landmarks to 3D points`
- `Raw Label Studio export detected`
- `No correspondences were found`

Fix:

- Use the normalized registration export from `manual-correspondences`, not raw
  Label Studio JSON.
- Ensure every frame record has `ct_landmarks_path` or embedded `ct_landmarks`.
- Ensure each landmark's `ct_landmark_id` exists in the CT landmark catalog.
