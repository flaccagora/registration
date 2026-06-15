# Compatibility Report

Static review status only. No dependencies were installed, no Python code was
executed, and no external repos were launched.

## Repository Layout

Preferred external repo layout:

```text
external/vggt-omega/
external/Medical-SAM3/
external/manual-correspondences/
```

Legacy fallback layout currently supported by wrappers:

```text
vggt-omega/
Medical-SAM3/
manual-correspondences/
```

## Known Upstream URLs

```text
external/vggt-omega      https://github.com/facebookresearch/vggt-omega.git
external/Medical-SAM3    https://github.com/AIM-Research-Lab/Medical-SAM3.git
external/manual-correspondences  https://github.com/flaccagora/label.git
```

These URLs are recorded in `.gitmodules`.

## Static API Compatibility

- VGGT-Omega wrapper calls the local `vggt_omega.models.VGGTOmega`,
  `load_and_preprocess_images`, and `encoding_to_camera` APIs seen in the
  local demo.
- MedicalSAM3 wrapper calls the local `inference/sam3_inference.py` `SAM3Model`
  class and supports the local text and box prompt methods.
- MedicalSAM3 point prompts are converted to a small box because the inspected
  helper exposes text and box prediction methods, not a direct point method.
- Mask prompts bypass MedicalSAM3 and can be used for CPU-only structural
  testing.
- `pipeline.correspondences.load_correspondences` accepts normalized
  `manual-correspondences` registration exports by resolving `ct_landmark_id`
  against `ct_landmarks_path` or embedded landmark catalogs.

## Runtime Risks To Test Later

- Exact VGGT prediction tensor shapes and preprocessing coordinate mapping.
- Local MedicalSAM3 CUDA autocast behavior on the target machine.
- Submodule commits and checkpoint versions used by the prepared environment.
- Manual correspondence exports with missing or stale `ct_landmarks_path`
  values.
- PnP behavior on real sparse/manual correspondences.
- RBF deformation stability on noisy clinical/research annotations.
