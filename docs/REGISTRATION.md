# Registration Details

The registration pipeline has two explicit stages: initial pose registration
and deformable refinement. UI labels, JSON outputs, and metrics preserve this
distinction.

## Initial Pose Registration

The initial stage estimates a global transform from the 3D mesh coordinate
frame to the camera or VGGT point-map frame.

Supported methods:

- `ransac_pnp`: default when at least four 2D-to-3D correspondences and camera
  intrinsics are available.
- `pnp`: non-RANSAC fallback when configured.
- `similarity_from_vggt_point_map`: estimates scale, rotation, and translation
  from manual mesh points to VGGT point-map samples at the same pixels.

If intrinsics are missing, the prototype can estimate a simple pinhole matrix
from image size:

```text
fx = fy = max(width, height)
cx = width / 2
cy = height / 2
```

This is only a fallback for coarse alignment. Real calibration is preferred.

Initial pose metrics include:

- correspondence count
- inlier count
- mean, median, weighted mean, and max reprojection error in pixels
- RANSAC threshold
- estimated scale for similarity initialization

The initial transform is exported as a 4 x 4 homogeneous
`transform_matrix_mesh_to_camera`. For PnP this is rigid. For VGGT point-map
similarity initialization the upper-left 3 x 3 block includes scale.

## Deformable Refinement

The final stage allows the mesh or surface model to deform after the initial
pose. The implemented model is a Gaussian RBF control-point deformation:

```text
deformed_vertex = vertex + sum_i phi(||vertex - control_i||) * coefficient_i
```

where `phi` is a Gaussian radial basis kernel. Coefficients are fit with
Tikhonov-style regularization. Higher correspondence weights reduce damping at
trusted controls.

The current constraints are:

- Manual correspondences define sparse control locations on the mesh.
- VGGT point-map samples at annotated pixels provide depth-informed 3D targets.
- The inverse initial pose maps those sampled 3D targets back into mesh
  coordinates.
- MedicalSAM3 masks downweight manual points outside the segmented anatomy.

Deformable metrics include:

- control point count
- mean and median control residual
- mean and max mesh displacement
- kernel radius
- regularization value
- smoothness energy
- number of low-weight, non-finite, or outlier controls dropped before fitting

## Regularization

The current RBF model penalizes unrealistic deformation by damping the RBF
coefficients. This is a smoothness prior, not a full biomechanical constraint.
The solver requires positive regularization and falls back to a least-squares
solve if the regularized kernel system is still singular.

Additional safeguards are configurable in `DeformableConfig`:

- `min_control_points`
- `min_control_weight`
- `trim_outlier_controls`
- `outlier_mad_multiplier`
- `max_control_displacement`

Future regularizers should include:

- embedded deformation graph with local rigidity penalties
- ARAP-style edge preservation on mesh adjacency
- silhouette/mask overlap terms
- depth consistency over visible mesh samples
- temporal consistency for video sequences

## Coordinate-Frame Caution

VGGT-Omega point maps must be interpreted in the same camera/world frame used
by the initial pose. The wrapper follows the local VGGT-Omega demo convention
for unprojecting depth into world points. Real datasets should verify axis
directions, scale, and camera-frame conventions before trusting quantitative
metrics.

VGGT point maps can also be at a model preprocessing resolution rather than the
original annotated image resolution. The implementation can scale manual pixels
into the point-map grid when the original image size is supplied, but users
should still verify whether the upstream preprocessing cropped, padded, or
resized the image in a way that changes coordinate mapping.
