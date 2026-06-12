# Pipeline

The prototype is organized as independent stages. Each stage saves artifacts so
later stages can be rerun without repeating expensive inference.

## Required Inputs

- Primary image, image sequence, or video.
- 3D mesh file: `.obj`, `.ply`, or `.stl`.
- Manual correspondence CSV or JSON.
- Optional camera intrinsics JSON.
- Optional segmentation prompt: text, point, box, or mask.
- VGGT-Omega checkpoint path when running depth/camera estimation.
- MedicalSAM3 checkpoint path when running local segmentation with explicit
  weights.

## Stage 1: Input Loading

`pipeline/io.py` handles images, masks, meshes, video frame extraction, JSON,
and basic point-cloud or mesh export.

Video inputs are sampled into `outputs/input_frames/` when the demo callback is
run. Mesh loading prefers `trimesh`, with a minimal OBJ fallback.

## Stage 2: Correspondences

`pipeline/correspondences.py` loads CSV/JSON correspondences into a canonical
`Correspondence` dataclass. It supports frame filtering and optional
segmentation-mask-based filtering or downweighting.

## Stage 3: MedicalSAM3 Segmentation

`pipeline/segmentation_medicalsam3.py` is a thin wrapper around the local
`external/Medical-SAM3/inference/sam3_inference.py` module. It lazy-loads the
repo only when segmentation is requested, and can fall back to a legacy
root-level `Medical-SAM3/` checkout.

Outputs:

- binary mask PNG
- binary mask `.npy`
- segmentation overlay PNG
- metadata including prompt type and mask area

Masks can filter correspondences or downweight correspondences outside the
target anatomy.

## Stage 4: VGGT-Omega Depth and Camera

`pipeline/depth_vggtomega.py` is a thin wrapper around the local
`external/vggt-omega/` Python API. It lazy-loads the model only when requested,
and can fall back to a legacy root-level `vggt-omega/` checkout.

Outputs:

- `predictions.npz`
- per-frame depth `.npy` saved as `H x W` float arrays
- per-frame depth visualization PNG
- camera JSON with `intrinsic` and `extrinsic_world_to_camera` matrices per
  frame
- point cloud PLY
- GLB scene when the local VGGT-Omega visualization helper is available
- metadata with cache key and any optional GLB export error

Outputs are cached by image paths, image resolution, checkpoint path, file
sizes, and modification times.

## Stage 5: Initial Pose Registration

`pipeline/registration.py` first estimates a coarse global alignment:

- RANSAC-PnP or PnP when intrinsics are available or estimated.
- Similarity transform from VGGT point-map samples when 3D image evidence is
  available and PnP is not appropriate.

Outputs:

- `initial_pose_registration.json` with
  `transform_matrix_mesh_to_camera`, intrinsics, inlier row indices, and
  reprojection errors
- reprojection error table
- initial pose overlay
- initial registered mesh/point cloud exports

## Stage 6: Deformable Refinement

After initial pose, the prototype can fit a regularized RBF deformation field.
Manual correspondences define sparse control anchors. VGGT point-map samples at
manual pixels provide depth-informed target points. MedicalSAM3 masks can
downweight anchors outside the segmented anatomy.

Before fitting, the implementation drops non-finite controls, controls below
the configured minimum weight, and optionally extreme displacement outliers.
This is a numerical safety step for sparse/noisy research annotations, not a
substitute for anatomical validation.

Outputs:

- `deformable_refinement.json`
- `deformed_mesh_frame.obj`
- `final_deformable_registered_mesh.obj`
- final deformable registered point cloud PLY
- final overlay
- deformation metrics

## Stage 7: Visualization and Export

`pipeline/visualization.py` creates overlays for:

- segmentation masks
- manual correspondences
- initial pose projected mesh points
- final deformable projected mesh points
- depth maps

`pipeline/export.py` saves JSON, OBJ, PLY, manifests, and ZIP result bundles.

## Output Directory

Default output layout:

```text
outputs/
  segmentation/
  vggt_omega/
  registration/
    initial_pose_registration.json
    initial_pose_metrics.json
    deformable_refinement.json
    deformable_metrics.json
    initial_registered_mesh.obj
    final_deformable_registered_mesh.obj
    manifest.json
```
