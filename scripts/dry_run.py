"""Synthetic dry run for the registration geometry.

This script deliberately does not call MedicalSAM3, VGGT-Omega, or Gradio. It
creates synthetic correspondences and runs the initial pose stage. If a VGGT
point map is later supplied, the same package functions can run deformable
refinement.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pipeline.correspondences import make_synthetic_correspondences, save_correspondences
from pipeline.export import export_registered_meshes, save_initial_pose_result, save_metrics
from pipeline.io import MeshData, ensure_dir, write_obj_mesh
from pipeline.registration import CameraIntrinsics, estimate_initial_pose, reprojection_error_table


def synthetic_mesh() -> MeshData:
    vertices = np.asarray(
        [
            [-30.0, -20.0, 0.0],
            [30.0, -20.0, 0.0],
            [30.0, 20.0, 0.0],
            [-30.0, 20.0, 0.0],
            [-25.0, -15.0, 40.0],
            [25.0, -15.0, 40.0],
            [25.0, 15.0, 40.0],
            [-25.0, 15.0, 40.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [1, 2, 6],
            [1, 6, 5],
            [0, 3, 7],
            [0, 7, 4],
        ],
        dtype=np.int64,
    )
    return MeshData(vertices=vertices, faces=faces)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a synthetic initial-pose dry run.")
    parser.add_argument("--out-dir", default="outputs/dry_run", help="Directory for generated dry-run files.")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    correspondences, metadata = make_synthetic_correspondences(image_id="synthetic_frame", image_size=(640, 480))
    csv_path = save_correspondences(correspondences, out_dir / "synthetic_correspondences.csv")
    json_path = save_correspondences(correspondences, out_dir / "synthetic_correspondences.json")
    mesh = synthetic_mesh()
    mesh_path = write_obj_mesh(mesh.vertices, mesh.faces, out_dir / "synthetic_mesh.obj")

    intrinsics = CameraIntrinsics(
        camera_matrix=np.asarray(metadata["camera_matrix"], dtype=np.float64),
        distortion=np.zeros((0, 1), dtype=np.float64),
        source="synthetic",
    )
    result = estimate_initial_pose(correspondences, intrinsics=intrinsics, image_size=(640, 480))
    pose_path = save_initial_pose_result(result, out_dir)
    metrics = result.metrics | {"reprojection_table": reprojection_error_table(correspondences, result)}
    metrics_path = save_metrics(metrics, out_dir, "dry_run_metrics.json")
    artifacts = export_registered_meshes(mesh, result, out_dir)

    print("Synthetic dry run complete.")
    print(f"Correspondences CSV: {csv_path}")
    print(f"Correspondences JSON: {json_path}")
    print(f"Mesh: {mesh_path}")
    print(f"Initial pose JSON: {pose_path}")
    print(f"Metrics JSON: {metrics_path}")
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

