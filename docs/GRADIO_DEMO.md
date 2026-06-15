# Gradio Demo

The demo entrypoint is `app.py`. It was created but not launched.

Run later after dependencies, external repos, and checkpoints are ready:

```bash
python app.py
```

## Inputs

The UI accepts:

- primary image
- optional image sequence
- optional video
- 3D mesh: `.obj`, `.ply`, or `.stl`
- manual correspondence CSV/JSON
- optional intrinsics JSON
- optional segmentation mask prompt

Runtime settings in the UI include:

- MedicalSAM3 repo path
- MedicalSAM3 checkpoint path
- VGGT-Omega repo path
- VGGT-Omega checkpoint path
- device
- VGGT image resolution
- output directory

The `cpu` device is useful for non-model stages and precomputed mask prompts.
The current local MedicalSAM3 inference helper uses CUDA autocast internally,
so text, point, and box MedicalSAM3 prompts require `cuda`.

Default repo paths come from `.env.example`:

```text
VGGT_OMEGA_REPO=external/vggt-omega
MEDICALSAM3_REPO=external/Medical-SAM3
MANUAL_CORRESPONDENCES_REPO=external/manual-correspondences
```

If these paths do not exist yet, the wrappers can fall back to legacy
root-level `vggt-omega/`, `Medical-SAM3/`, and `manual-correspondences/`
checkouts.

## Buttons

- `Run segmentation`: calls MedicalSAM3 or copies an existing mask prompt.
- `Run VGGT-Omega`: estimates depth, cameras, point maps, point cloud, and
  optional GLB scene.
- `Run initial pose registration`: estimates the coarse rigid/similarity pose.
- `Run deformable refinement`: fits the final regularized non-rigid warp.
- `Export results`: zips the registration output directory.

## Outputs

- segmentation overlay
- mask PNG
- depth visualization
- VGGT point cloud or GLB
- initial pose registration overlay
- initial pose transform JSON
- initial pose metrics
- final deformable registration overlay
- deformable result JSON
- registered mesh
- deformable metrics
- downloadable ZIP bundle

## Clinical Warning

The UI displays a warning that the prototype is not clinically validated and
must not be used for diagnosis, treatment, surgical navigation, or patient
care.
