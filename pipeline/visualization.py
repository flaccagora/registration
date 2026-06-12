"""Visualization helpers for segmentation, correspondences, depth, and registration."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from pipeline.correspondences import Correspondence, correspondence_arrays
from pipeline.errors import MissingDependencyError
from pipeline.io import MeshData, blend_mask, colorize_depth, load_rgb_image, sample_points, save_rgb_image
from pipeline.registration import CameraIntrinsics, InitialPoseResult, project_points_pinhole


PALETTE = [
    (230, 57, 70),
    (42, 157, 143),
    (38, 70, 83),
    (244, 162, 97),
    (69, 123, 157),
    (131, 56, 236),
    (255, 183, 3),
]


def segmentation_overlay(image_path: str | Path, mask: np.ndarray, output_path: str | Path) -> Path:
    """Save a segmentation mask overlay."""

    image = load_rgb_image(image_path)
    return save_rgb_image(blend_mask(image, mask), output_path)


def depth_visualization(depth: np.ndarray, output_path: str | Path) -> Path:
    """Save a colorized depth map."""

    return save_rgb_image(colorize_depth(depth), output_path)


def draw_manual_correspondences(
    image_path: str | Path,
    correspondences: Iterable[Correspondence],
    output_path: str | Path,
    radius: int = 5,
) -> Path:
    """Draw manual 2D correspondence points on the source image."""

    try:
        import cv2
    except ImportError as exc:
        raise MissingDependencyError("opencv-python is required for correspondence visualization.") from exc

    image = load_rgb_image(image_path).copy()
    for idx, item in enumerate(correspondences):
        color = PALETTE[idx % len(PALETTE)]
        center = (int(round(item.u)), int(round(item.v)))
        cv2.circle(image, center, radius, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(image, center, radius + 2, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        label = item.label or str(idx)
        cv2.putText(image, label, (center[0] + 7, center[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return save_rgb_image(image, output_path)


def registration_overlay(
    image_path: str | Path,
    mesh: MeshData,
    initial_pose: InitialPoseResult,
    output_path: str | Path,
    *,
    correspondences: Iterable[Correspondence] | None = None,
    deformed_vertices_mesh_frame: np.ndarray | None = None,
    sample_count: int = 6000,
) -> Path:
    """Overlay projected mesh vertices and optional correspondence residuals."""

    try:
        import cv2
    except ImportError as exc:
        raise MissingDependencyError("opencv-python is required for registration visualization.") from exc

    if initial_pose.intrinsics is None:
        raise ValueError("Registration overlay requires camera intrinsics.")
    image = load_rgb_image(image_path).copy()
    vertices = deformed_vertices_mesh_frame if deformed_vertices_mesh_frame is not None else mesh.vertices
    vertices = sample_points(np.asarray(vertices, dtype=np.float64), sample_count)
    projected = project_points_pinhole(vertices, initial_pose.transform_matrix, initial_pose.intrinsics.camera_matrix)
    height, width = image.shape[:2]
    valid = (
        np.isfinite(projected).all(axis=1)
        & (projected[:, 0] >= 0)
        & (projected[:, 0] < width)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < height)
    )
    for point in projected[valid]:
        cv2.circle(image, (int(round(point[0])), int(round(point[1]))), 1, (15, 220, 130), thickness=-1)

    if correspondences is not None:
        object_points, image_points, _ = correspondence_arrays(list(correspondences))
        projected_corr = project_points_pinhole(object_points, initial_pose.transform_matrix, initial_pose.intrinsics.camera_matrix)
        for idx, (observed, predicted) in enumerate(zip(image_points, projected_corr)):
            if not np.isfinite(predicted).all():
                continue
            color = PALETTE[idx % len(PALETTE)]
            obs = (int(round(observed[0])), int(round(observed[1])))
            pred = (int(round(predicted[0])), int(round(predicted[1])))
            cv2.circle(image, obs, 5, color, thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(image, pred, 5, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
            cv2.line(image, obs, pred, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    return save_rgb_image(image, output_path)


def side_by_side(paths: list[str | Path], output_path: str | Path) -> Path:
    """Create a horizontal side-by-side comparison image."""

    images = [load_rgb_image(path) for path in paths if path]
    if not images:
        raise ValueError("No images provided for side_by_side visualization.")
    max_height = max(image.shape[0] for image in images)
    padded = []
    for image in images:
        if image.shape[0] == max_height:
            padded.append(image)
            continue
        pad_height = max_height - image.shape[0]
        pad = np.full((pad_height, image.shape[1], 3), 255, dtype=np.uint8)
        padded.append(np.vstack([image, pad]))
    combined = np.hstack(padded)
    return save_rgb_image(combined, output_path)


def camera_frame_projection_overlay(
    image_path: str | Path,
    vertices_camera_frame: np.ndarray,
    intrinsics: CameraIntrinsics,
    output_path: str | Path,
    *,
    sample_count: int = 6000,
) -> Path:
    """Overlay points that have already been transformed to camera space."""

    try:
        import cv2
    except ImportError as exc:
        raise MissingDependencyError("opencv-python is required for projection visualization.") from exc

    image = load_rgb_image(image_path).copy()
    vertices = sample_points(np.asarray(vertices_camera_frame, dtype=np.float64), sample_count)
    z = vertices[:, 2]
    valid_z = np.isfinite(z) & (z > 1e-9)
    projected = np.full((len(vertices), 2), np.nan, dtype=np.float64)
    k = intrinsics.camera_matrix
    projected[valid_z, 0] = k[0, 0] * vertices[valid_z, 0] / z[valid_z] + k[0, 2]
    projected[valid_z, 1] = k[1, 1] * vertices[valid_z, 1] / z[valid_z] + k[1, 2]
    height, width = image.shape[:2]
    valid = (
        np.isfinite(projected).all(axis=1)
        & (projected[:, 0] >= 0)
        & (projected[:, 0] < width)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < height)
    )
    for point in projected[valid]:
        cv2.circle(image, (int(round(point[0])), int(round(point[1]))), 1, (255, 170, 20), thickness=-1)
    return save_rgb_image(image, output_path)
