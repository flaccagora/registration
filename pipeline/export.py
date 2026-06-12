"""Export transforms, metrics, registered meshes, and result bundles."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from pipeline.io import MeshData, ensure_dir, write_json, write_obj_mesh, write_ply_points
from pipeline.registration import DeformableRegistrationResult, InitialPoseResult


def save_initial_pose_result(result: InitialPoseResult, output_dir: str | Path) -> Path:
    """Save the initial pose transform and metrics as JSON."""

    output = ensure_dir(output_dir)
    return write_json(result.to_dict(), output / "initial_pose_registration.json")


def save_deformable_result(result: DeformableRegistrationResult, output_dir: str | Path) -> Path:
    """Save deformable refinement metadata as JSON."""

    output = ensure_dir(output_dir)
    return write_json(result.to_dict(), output / "deformable_refinement.json")


def save_metrics(metrics: dict[str, Any], output_dir: str | Path, name: str = "metrics.json") -> Path:
    """Save a metrics dictionary."""

    return write_json(metrics, Path(output_dir) / name)


def export_registered_meshes(
    mesh: MeshData,
    initial_pose: InitialPoseResult,
    output_dir: str | Path,
    deformable_result: DeformableRegistrationResult | None = None,
) -> dict[str, Path]:
    """Export initial and final registered surfaces.

    OBJ files preserve faces when available. PLY point clouds are also written
    so users can inspect the registered surface even when faces are absent.
    """

    output = ensure_dir(output_dir)
    initial_vertices = _apply_transform(mesh.vertices, initial_pose.transform_matrix)
    artifacts = {
        "initial_registered_obj": write_obj_mesh(initial_vertices, mesh.faces, output / "initial_registered_mesh.obj"),
        "initial_registered_ply": write_ply_points(initial_vertices, output / "initial_registered_points.ply"),
    }
    if deformable_result is not None:
        artifacts.update(
            {
                "deformed_mesh_frame_obj": write_obj_mesh(
                    deformable_result.deformed_vertices_mesh_frame,
                    mesh.faces,
                    output / "deformed_mesh_frame.obj",
                ),
                "final_deformable_registered_obj": write_obj_mesh(
                    deformable_result.final_vertices_camera_frame,
                    mesh.faces,
                    output / "final_deformable_registered_mesh.obj",
                ),
                "final_deformable_registered_ply": write_ply_points(
                    deformable_result.final_vertices_camera_frame,
                    output / "final_deformable_registered_points.ply",
                ),
            }
        )
    return artifacts


def _apply_transform(points: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([np.asarray(points).reshape(-1, 3), np.ones(len(points), dtype=np.float64)])
    return (homogeneous @ np.asarray(transform_matrix).reshape(4, 4).T)[:, :3]


def export_result_bundle(output_dir: str | Path, archive_path: str | Path | None = None) -> Path:
    """Create a zip archive of a result directory."""

    output = Path(output_dir)
    if not output.exists() or not output.is_dir():
        raise FileNotFoundError(f"No result directory to export: {output}")
    if not any(output.iterdir()):
        raise FileNotFoundError(f"Result directory is empty: {output}")
    if archive_path is None:
        archive_path = output.with_suffix(".zip")
    archive = Path(archive_path)
    if archive.suffix == ".zip":
        archive_base = archive.with_suffix("")
    else:
        archive_base = archive
        archive = archive.with_suffix(".zip")
    shutil.make_archive(str(archive_base), "zip", output)
    return archive


def manifest(output_dir: str | Path, artifacts: dict[str, Any]) -> Path:
    """Write a compact manifest of generated files."""

    normalized = {}
    for key, value in artifacts.items():
        if isinstance(value, Path):
            normalized[key] = str(value)
        elif isinstance(value, list):
            normalized[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            normalized[key] = value
    target = Path(output_dir) / "manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    return target
