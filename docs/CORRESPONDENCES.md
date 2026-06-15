# Manual Correspondence Schema

The pipeline supports CSV and JSON sparse 2D-to-3D correspondences. Each record
binds an image pixel to a 3D mesh surface point.

## Required Fields

- `image_id` or `frame_id`: image or video frame identifier.
- Pixel coordinate: either `u`, `v` columns or `pixel: {"u": ..., "v": ...}`.
- 3D point: either `x`, `y`, `z` columns or
  `point3d: {"x": ..., "y": ..., "z": ...}`.

## Optional Fields

- `label` or `class`: anatomical or annotation label.
- `confidence`: manual annotation confidence, usually 0 to 1.
- `weight`: explicit registration weight. If absent, `confidence` is used.

## CSV Example

```csv
image_id,u,v,x,y,z,label,confidence,weight
frame_0000,288.80,211.34,-30.0,-20.0,0.0,left_lower_surface,0.95,
frame_0000,371.54,210.27,30.0,-20.0,0.0,right_lower_surface,0.95,
```

Full example: `examples/correspondences/example_correspondences.csv`.

## JSON Example

```json
{
  "version": 1,
  "correspondences": [
    {
      "image_id": "frame_0000",
      "pixel": {"u": 288.8, "v": 211.34},
      "point3d": {"x": -30.0, "y": -20.0, "z": 0.0},
      "label": "left_lower_surface",
      "confidence": 0.95
    }
  ]
}
```

Full example: `examples/correspondences/example_correspondences.json`.

Normalized `manual-correspondences` export example:
`examples/correspondences/manual_correspondences_export.example.json`.

## Coordinate Conventions

Pixels use image coordinates: `u` increases to the right, `v` increases down,
and the origin is the top-left image pixel.

3D points are in the input mesh coordinate frame. Units are inherited from the
mesh and must be consistent with camera/depth estimates for metric results to
be meaningful.

## Existing Manual Correspondence Data

The `external/manual-correspondences/` repository contains Label Studio
workflows and normalized registration annotation exports. The pipeline can load
those normalized export records directly when they include:

- `frame_id`
- `ct_landmarks_path` or embedded `ct_landmarks`
- `landmarks` entries with `ct_landmark_id`, `image_x`, `image_y`, and
  optional `confidence`

Example normalized record:

```json
[
  {
    "frame_id": "case_000:10",
    "ct_landmarks_path": "ct_landmarks.json",
    "landmarks": [
      {
        "ct_landmark_id": "apex",
        "image_x": 420.0,
        "image_y": 180.0,
        "confidence": 0.9
      }
    ]
  }
]
```

The referenced landmark catalog maps IDs to 3D mesh points:

```json
{
  "landmarks": [
    {
      "id": "apex",
      "point3d": [12.0, 84.0, 620.0]
    }
  ]
}
```

`load_correspondences()` resolves these records into the canonical
`image_id,u,v,x,y,z,label,confidence,weight` format used by initial pose
registration and deformable refinement. Legacy root-level
`manual-correspondences/` checkouts are still supported as a local workspace
fallback, but should not be committed into this prototype repository.
