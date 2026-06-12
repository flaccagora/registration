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

## Coordinate Conventions

Pixels use image coordinates: `u` increases to the right, `v` increases down,
and the origin is the top-left image pixel.

3D points are in the input mesh coordinate frame. Units are inherited from the
mesh and must be consistent with camera/depth estimates for metric results to
be meaningful.

## Existing Manual Correspondence Data

The `manual-correspondences/` repository contains Label Studio workflows and
registration annotation exports. The new prototype uses a simpler CSV/JSON
schema first. A future adapter should convert the existing Label Studio export
schema into this canonical correspondence format.

