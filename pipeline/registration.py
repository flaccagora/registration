"""Initial pose registration and deformable surface refinement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from pipeline.config import DeformableConfig, InitialPoseConfig
from pipeline.correspondences import Correspondence, correspondence_arrays, filter_correspondences
from pipeline.errors import InvalidInputError, MissingDependencyError
from pipeline.io import MeshData, read_json


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics and optional distortion coefficients."""

    camera_matrix: np.ndarray
    distortion: np.ndarray
    source: str = "provided"

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_matrix": self.camera_matrix.astype(float).tolist(),
            "distortion_coefficients": self.distortion.reshape(-1).astype(float).tolist(),
            "source": self.source,
        }


@dataclass
class InitialPoseResult:
    """Coarse global registration result."""

    method: str
    transform_matrix: np.ndarray
    intrinsics: CameraIntrinsics | None
    inlier_indices: list[int]
    reprojection_errors_px: np.ndarray
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": "initial_pose_registration",
            "method": self.method,
            "transform_matrix_mesh_to_camera": self.transform_matrix.astype(float).tolist(),
            "intrinsics": self.intrinsics.to_dict() if self.intrinsics else None,
            "inlier_indices": [int(idx) for idx in self.inlier_indices],
            "reprojection_errors_px": self.reprojection_errors_px.astype(float).tolist(),
            "metrics": self.metrics,
        }


@dataclass
class RBFDeformationModel:
    """Smooth control-point RBF deformation in mesh coordinates."""

    control_points: np.ndarray
    coefficients: np.ndarray
    kernel_radius: float

    def transform_points(self, points: np.ndarray, chunk_size: int = 50000) -> np.ndarray:
        """Apply the learned smooth displacement field to points."""

        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        output = points.copy()
        for start in range(0, len(points), chunk_size):
            chunk = points[start : start + chunk_size]
            kernel = gaussian_rbf(chunk, self.control_points, self.kernel_radius)
            output[start : start + chunk_size] = chunk + kernel @ self.coefficients
        return output

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "rbf",
            "control_points": self.control_points.astype(float).tolist(),
            "coefficients": self.coefficients.astype(float).tolist(),
            "kernel_radius": float(self.kernel_radius),
        }


@dataclass
class DeformableRegistrationResult:
    """Final deformable registration result."""

    method: str
    initial_pose: InitialPoseResult
    deformation_model: RBFDeformationModel | None
    deformed_vertices_mesh_frame: np.ndarray
    final_vertices_camera_frame: np.ndarray
    control_source_points: np.ndarray
    control_target_points: np.ndarray
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": "deformable_refinement",
            "method": self.method,
            "initial_pose": self.initial_pose.to_dict(),
            "deformation_model": self.deformation_model.to_dict() if self.deformation_model else None,
            "control_source_points": self.control_source_points.astype(float).tolist(),
            "control_target_points": self.control_target_points.astype(float).tolist(),
            "metrics": self.metrics,
        }


def load_intrinsics(path: str | Path) -> CameraIntrinsics:
    """Load camera intrinsics from JSON."""

    payload = read_json(path)
    if "camera_matrix" in payload:
        camera_matrix = np.asarray(payload["camera_matrix"], dtype=np.float64)
    else:
        required = ("fx", "fy", "cx", "cy")
        missing = [key for key in required if key not in payload]
        if missing:
            raise InvalidInputError(f"Intrinsics JSON missing keys: {', '.join(missing)}")
        camera_matrix = np.asarray(
            [
                [float(payload["fx"]), 0.0, float(payload["cx"])],
                [0.0, float(payload["fy"]), float(payload["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    if camera_matrix.shape != (3, 3):
        raise InvalidInputError("camera_matrix must be 3x3.")
    distortion = np.asarray(payload.get("distortion_coefficients") or payload.get("distortion") or [], dtype=np.float64)
    return CameraIntrinsics(camera_matrix=camera_matrix, distortion=distortion.reshape(-1, 1), source=str(path))


def estimate_default_intrinsics(image_size: tuple[int, int], focal_length_px: float | None = None) -> CameraIntrinsics:
    """Create a conservative pinhole camera matrix from image dimensions."""

    width, height = image_size
    focal = float(focal_length_px) if focal_length_px else float(max(width, height))
    camera_matrix = np.asarray([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return CameraIntrinsics(camera_matrix=camera_matrix, distortion=np.zeros((0, 1), dtype=np.float64), source="estimated_from_image_size")


def make_transform(rotation: np.ndarray, translation: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Build a homogeneous matrix mapping mesh points into camera/world space."""

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = float(scale) * np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return matrix


def apply_transform(points: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """Apply a homogeneous rigid/similarity transform to 3D points."""

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    matrix = np.asarray(transform_matrix, dtype=np.float64).reshape(4, 4)
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    transformed = homogeneous @ matrix.T
    return transformed[:, :3]


def project_points_pinhole(points_3d: np.ndarray, transform_matrix: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    """Project mesh points through a mesh-to-camera transform and pinhole camera.

    Distortion is intentionally not applied here so the same path supports both
    pure PnP and similarity transforms whose 3x3 block may include scale.
    """

    camera_points = apply_transform(points_3d, transform_matrix)
    z = camera_points[:, 2]
    projected = np.full((len(camera_points), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(z) & (np.abs(z) > 1e-9)
    projected[valid, 0] = camera_matrix[0, 0] * camera_points[valid, 0] / z[valid] + camera_matrix[0, 2]
    projected[valid, 1] = camera_matrix[1, 1] * camera_points[valid, 1] / z[valid] + camera_matrix[1, 2]
    return projected


def weighted_umeyama(source_points: np.ndarray, target_points: np.ndarray, weights: np.ndarray | None = None, estimate_scale: bool = True) -> tuple[np.ndarray, np.ndarray, float]:
    """Estimate weighted similarity transform from ``source`` to ``target``.

    The returned transform minimizes 3D point residuals. It is used for VGGT
    point-map assisted initialization and can also support external depth data.
    """

    source = np.asarray(source_points, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)
    if len(source) != len(target) or len(source) < 3:
        raise InvalidInputError("Similarity estimation requires at least three paired 3D points.")
    if weights is None:
        w = np.ones(len(source), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = np.maximum(w, 0.0)
    if float(w.sum()) <= 0:
        raise InvalidInputError("Similarity estimation received only zero weights.")
    w = w / w.sum()
    source_mean = np.sum(source * w[:, None], axis=0)
    target_mean = np.sum(target * w[:, None], axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = (source_centered * w[:, None]).T @ target_centered
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(vt.T @ u.T) < 0:
        correction[-1, -1] = -1.0
    rotation = vt.T @ correction @ u.T
    if estimate_scale:
        variance = np.sum(w * np.sum(source_centered * source_centered, axis=1))
        scale = float(np.sum(singular_values * np.diag(correction)) / max(variance, 1e-12))
    else:
        scale = 1.0
    if not np.isfinite(scale) or scale <= 0:
        raise InvalidInputError("Similarity estimation produced a non-positive or non-finite scale.")
    translation = target_mean - scale * rotation @ source_mean
    return rotation, translation, scale


def sample_point_map(
    point_map: np.ndarray,
    image_points: np.ndarray,
    image_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a dense VGGT point map at sparse pixel locations."""

    points = np.asarray(point_map, dtype=np.float64)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise InvalidInputError("point_map must have shape H x W x 3.")
    height, width = points.shape[:2]
    scale_x = 1.0
    scale_y = 1.0
    if image_size is not None:
        image_width, image_height = image_size
        scale_x = width / max(float(image_width), 1.0)
        scale_y = height / max(float(image_height), 1.0)
    sampled = []
    valid = []
    for pixel in np.asarray(image_points, dtype=np.float64).reshape(-1, 2):
        u = int(round(pixel[0] * scale_x))
        v = int(round(pixel[1] * scale_y))
        ok = 0 <= u < width and 0 <= v < height
        value = points[v, u] if ok else np.asarray([np.nan, np.nan, np.nan])
        ok = ok and np.isfinite(value).all()
        sampled.append(value)
        valid.append(ok)
    return np.asarray(sampled, dtype=np.float64), np.asarray(valid, dtype=bool)


def estimate_initial_pose(
    correspondences: Iterable[Correspondence],
    *,
    intrinsics: CameraIntrinsics | None = None,
    image_size: tuple[int, int] | None = None,
    point_map: np.ndarray | None = None,
    config: InitialPoseConfig | None = None,
    method: str = "auto",
) -> InitialPoseResult:
    """Estimate the initial rigid/similarity registration from sparse matches."""

    config = config or InitialPoseConfig()
    if config.ransac_reprojection_error_px <= 0:
        raise InvalidInputError("ransac_reprojection_error_px must be greater than zero.")
    if config.ransac_iterations <= 0:
        raise InvalidInputError("ransac_iterations must be greater than zero.")
    items = list(correspondences)
    object_points_all, image_points_all, weights_all = correspondence_arrays(items)
    valid_indices = np.flatnonzero(weights_all > 0)
    if len(valid_indices) == 0:
        raise InvalidInputError("All correspondences have zero registration weight.")
    object_points = object_points_all[valid_indices]
    image_points = image_points_all[valid_indices]
    weights = weights_all[valid_indices]
    weighted_items = [items[int(idx)] for idx in valid_indices]
    method = method.lower().strip()

    if method in {"auto", "pnp", "ransac_pnp"}:
        if intrinsics is None:
            if not config.estimate_intrinsics_if_missing or image_size is None:
                if point_map is not None and config.allow_similarity_from_vggt_points:
                    result = estimate_initial_similarity_from_point_map(weighted_items, point_map=point_map, config=config, image_size=image_size)
                    return _remap_initial_result(result, valid_indices, len(items), len(items) - len(valid_indices))
                raise InvalidInputError("Initial PnP registration requires intrinsics or image_size for default intrinsics.")
            intrinsics = estimate_default_intrinsics(image_size, config.default_focal_length_px)
        if len(object_points) >= 4:
            result = _estimate_pnp(object_points, image_points, weights, intrinsics, config)
            return _remap_initial_result(result, valid_indices, len(items), len(items) - len(valid_indices))
        if method != "auto":
            raise InvalidInputError("PnP requires at least four 2D-to-3D correspondences.")

    if point_map is not None and method in {"auto", "similarity", "vggt_similarity"}:
        result = estimate_initial_similarity_from_point_map(
            weighted_items,
            point_map=point_map,
            config=config,
            intrinsics=intrinsics,
            image_size=image_size,
        )
        return _remap_initial_result(result, valid_indices, len(items), len(items) - len(valid_indices))

    raise InvalidInputError("Could not estimate initial pose. Provide at least four correspondences or a VGGT point map.")


def _remap_initial_result(
    result: InitialPoseResult,
    valid_indices: np.ndarray,
    total_count: int,
    omitted_count: int,
) -> InitialPoseResult:
    """Map result arrays from positive-weight controls back to input rows."""

    local_errors = np.asarray(result.reprojection_errors_px, dtype=np.float64)
    full_errors = np.full(total_count, np.nan, dtype=np.float64)
    if len(local_errors) == len(valid_indices):
        full_errors[valid_indices] = local_errors
    result.reprojection_errors_px = full_errors
    result.inlier_indices = [int(valid_indices[idx]) for idx in result.inlier_indices if 0 <= int(idx) < len(valid_indices)]
    result.metrics["zero_weight_correspondence_count"] = int(omitted_count)
    return result


def _estimate_pnp(
    object_points: np.ndarray,
    image_points: np.ndarray,
    weights: np.ndarray,
    intrinsics: CameraIntrinsics,
    config: InitialPoseConfig,
) -> InitialPoseResult:
    try:
        import cv2
    except ImportError as exc:
        raise MissingDependencyError("opencv-python is required for PnP registration.") from exc

    object_points = object_points.astype(np.float64)
    image_points = image_points.astype(np.float64)
    camera_matrix = intrinsics.camera_matrix.astype(np.float64)
    distortion = intrinsics.distortion.astype(np.float64)

    if config.use_ransac:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=float(config.ransac_reprojection_error_px),
            iterationsCount=int(config.ransac_iterations),
        )
        inlier_indices = list(range(len(object_points))) if inliers is None else [int(idx) for idx in inliers.reshape(-1)]
    else:
        ok, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, distortion, flags=cv2.SOLVEPNP_EPNP)
        inlier_indices = list(range(len(object_points)))
    if not ok:
        raise InvalidInputError("OpenCV solvePnP failed for the provided correspondences.")

    if len(inlier_indices) >= 4 and hasattr(cv2, "solvePnPRefineLM"):
        refine_objects = object_points[inlier_indices]
        refine_images = image_points[inlier_indices]
        rvec, tvec = cv2.solvePnPRefineLM(refine_objects, refine_images, camera_matrix, distortion, rvec, tvec)

    rotation, _ = cv2.Rodrigues(rvec)
    transform = make_transform(rotation, tvec.reshape(3), scale=1.0)
    projected = project_points_pinhole(object_points, transform, camera_matrix)
    errors = np.linalg.norm(projected - image_points, axis=1)
    weighted_error = float(np.average(errors, weights=np.maximum(weights, 1e-12)))
    metrics = {
        "correspondence_count": int(len(object_points)),
        "inlier_count": int(len(inlier_indices)),
        "mean_reprojection_error_px": float(np.nanmean(errors)),
        "median_reprojection_error_px": float(np.nanmedian(errors)),
        "weighted_mean_reprojection_error_px": weighted_error,
        "max_reprojection_error_px": float(np.nanmax(errors)),
        "ransac_reprojection_error_px": float(config.ransac_reprojection_error_px),
        "scale": 1.0,
    }
    return InitialPoseResult(
        method="ransac_pnp" if config.use_ransac else "pnp",
        transform_matrix=transform,
        intrinsics=intrinsics,
        inlier_indices=inlier_indices,
        reprojection_errors_px=errors,
        metrics=metrics,
    )


def estimate_initial_similarity_from_point_map(
    correspondences: Iterable[Correspondence],
    *,
    point_map: np.ndarray,
    config: InitialPoseConfig | None = None,
    intrinsics: CameraIntrinsics | None = None,
    image_size: tuple[int, int] | None = None,
) -> InitialPoseResult:
    """Estimate a similarity transform using VGGT 3D points sampled at pixels."""

    del config
    object_points, image_points, weights = correspondence_arrays(correspondences)
    target_points, valid = sample_point_map(point_map, image_points, image_size=image_size)
    if int(valid.sum()) < 3:
        raise InvalidInputError("At least three valid VGGT point-map samples are required for similarity initialization.")
    rotation, translation, scale = weighted_umeyama(object_points[valid], target_points[valid], weights[valid], estimate_scale=True)
    transform = make_transform(rotation, translation, scale=scale)
    residuals = np.linalg.norm(apply_transform(object_points[valid], transform) - target_points[valid], axis=1)
    metrics = {
        "correspondence_count": int(len(object_points)),
        "valid_point_map_count": int(valid.sum()),
        "mean_3d_residual": float(np.mean(residuals)),
        "median_3d_residual": float(np.median(residuals)),
        "max_3d_residual": float(np.max(residuals)),
        "scale": float(scale),
    }
    return InitialPoseResult(
        method="similarity_from_vggt_point_map",
        transform_matrix=transform,
        intrinsics=intrinsics,
        inlier_indices=np.flatnonzero(valid).astype(int).tolist(),
        reprojection_errors_px=np.full(len(object_points), np.nan, dtype=np.float64),
        metrics=metrics,
    )


def gaussian_rbf(points: np.ndarray, centers: np.ndarray, radius: float) -> np.ndarray:
    """Evaluate Gaussian radial basis functions between points and centers."""

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    radius = max(float(radius), 1e-9)
    diff = points[:, None, :] - centers[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    return np.exp(-dist2 / (2.0 * radius * radius))


def fit_rbf_deformation(
    control_points: np.ndarray,
    target_points: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    regularization: float = 1e-2,
    kernel_radius: float | None = None,
) -> RBFDeformationModel:
    """Fit a regularized RBF displacement field in mesh coordinates."""

    source = np.asarray(control_points, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)
    if len(source) != len(target) or len(source) < 3:
        raise InvalidInputError("RBF deformation requires at least three source/target control pairs.")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise InvalidInputError("RBF deformation controls contain non-finite coordinates.")
    if regularization <= 0:
        raise InvalidInputError("RBF regularization must be greater than zero.")
    displacement = target - source
    if weights is None:
        w = np.ones(len(source), dtype=np.float64)
    else:
        w = np.maximum(np.asarray(weights, dtype=np.float64).reshape(-1), 1e-6)
    if kernel_radius is None:
        if len(source) > 1:
            pairwise = np.sqrt(np.sum((source[:, None, :] - source[None, :, :]) ** 2, axis=-1))
            nonzero = pairwise[pairwise > 1e-9]
            kernel_radius = float(np.median(nonzero)) if len(nonzero) else 1.0
        else:
            kernel_radius = 1.0
    elif kernel_radius <= 0:
        raise InvalidInputError("RBF kernel_radius must be greater than zero when provided.")
    kernel = gaussian_rbf(source, source, kernel_radius)
    # Larger correspondence weights reduce the diagonal damping for that anchor.
    damping = float(regularization) * np.diag(1.0 / w)
    system = kernel + damping
    try:
        coefficients = np.linalg.solve(system, displacement)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(system, displacement, rcond=None)[0]
    if not np.isfinite(coefficients).all():
        raise InvalidInputError("RBF deformation solve produced non-finite coefficients.")
    return RBFDeformationModel(control_points=source, coefficients=coefficients, kernel_radius=float(kernel_radius))


def _robust_control_subset(
    source_points: np.ndarray,
    target_points: np.ndarray,
    weights: np.ndarray,
    config: DeformableConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Drop unusable or extreme deformation controls before fitting the warp."""

    source = np.asarray(source_points, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)
    control_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1) & np.isfinite(control_weights)
    finite &= control_weights >= float(config.min_control_weight)
    source = source[finite]
    target = target[finite]
    control_weights = control_weights[finite]
    dropped_nonfinite_or_low_weight = int((~finite).sum())

    displacement_norm = np.linalg.norm(target - source, axis=1)
    keep = np.ones(len(source), dtype=bool)
    if config.max_control_displacement is not None:
        keep &= displacement_norm <= float(config.max_control_displacement)
    if config.trim_outlier_controls and len(source) >= max(5, config.min_control_points + 2):
        median = float(np.median(displacement_norm))
        mad = float(np.median(np.abs(displacement_norm - median)))
        robust_sigma = 1.4826 * mad
        if robust_sigma > 1e-9:
            keep &= displacement_norm <= median + float(config.outlier_mad_multiplier) * robust_sigma

    dropped_outlier = int((~keep).sum())
    source = source[keep]
    target = target[keep]
    control_weights = control_weights[keep]
    if len(source) < int(config.min_control_points):
        raise InvalidInputError(
            "Deformable refinement has too few stable controls after filtering. "
            "Check manual correspondences, segmentation filtering, and VGGT point-map alignment."
        )
    metrics = {
        "dropped_nonfinite_or_low_weight_controls": dropped_nonfinite_or_low_weight,
        "dropped_outlier_controls": dropped_outlier,
        "max_control_displacement": None if config.max_control_displacement is None else float(config.max_control_displacement),
        "trim_outlier_controls": bool(config.trim_outlier_controls),
    }
    return source, target, control_weights, metrics


def build_deformation_constraints(
    correspondences: Iterable[Correspondence],
    *,
    initial_pose: InitialPoseResult,
    point_map: np.ndarray,
    image_size: tuple[int, int] | None = None,
    segmentation_mask: np.ndarray | None = None,
    outside_mask_weight: float = 0.25,
    max_control_points: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert manual 2D points plus VGGT point map into 3D deformation anchors.

    For each manual correspondence, the VGGT 3D point at the annotated pixel is
    transformed back into the mesh coordinate frame using the inverse initial
    pose. The RBF model then learns a smooth warp from the original mesh anchor
    to that depth-informed target.
    """

    filtered = filter_correspondences(
        correspondences,
        mask=segmentation_mask,
        outside_mask_weight=outside_mask_weight if segmentation_mask is not None else None,
    )
    object_points, image_points, weights = correspondence_arrays(filtered)
    sampled, valid = sample_point_map(point_map, image_points, image_size=image_size)
    if int(valid.sum()) < 3:
        raise InvalidInputError("Deformable refinement requires at least three valid VGGT point-map samples at manual correspondences.")
    inverse_initial = np.linalg.inv(initial_pose.transform_matrix)
    target_mesh_points = apply_transform(sampled[valid], inverse_initial)
    source_mesh_points = object_points[valid]
    valid_weights = weights[valid]

    if len(source_mesh_points) > max_control_points:
        order = np.argsort(-valid_weights)[:max_control_points]
        source_mesh_points = source_mesh_points[order]
        target_mesh_points = target_mesh_points[order]
        valid_weights = valid_weights[order]
    return source_mesh_points, target_mesh_points, valid_weights


def deformable_refinement(
    mesh: MeshData,
    correspondences: Iterable[Correspondence],
    *,
    initial_pose: InitialPoseResult,
    point_map: np.ndarray | None,
    image_size: tuple[int, int] | None = None,
    segmentation_mask: np.ndarray | None = None,
    config: DeformableConfig | None = None,
) -> DeformableRegistrationResult:
    """Run the final regularized deformable refinement stage."""

    config = config or DeformableConfig()
    if config.model != "rbf":
        raise InvalidInputError(f"Unsupported deformable model '{config.model}'. Only 'rbf' is implemented.")
    if config.regularization <= 0:
        raise InvalidInputError("deformable regularization must be greater than zero.")
    if config.min_control_points < 3:
        raise InvalidInputError("min_control_points must be at least 3 for 3D deformation.")
    if config.max_control_points < config.min_control_points:
        raise InvalidInputError("max_control_points must be greater than or equal to min_control_points.")
    if config.min_control_weight < 0:
        raise InvalidInputError("min_control_weight must be non-negative.")
    if config.outlier_mad_multiplier <= 0:
        raise InvalidInputError("outlier_mad_multiplier must be greater than zero.")
    if config.chunk_size <= 0:
        raise InvalidInputError("deformable chunk_size must be greater than zero.")
    vertices = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)
    if point_map is None:
        final_vertices = apply_transform(vertices, initial_pose.transform_matrix)
        metrics = {
            "status": "skipped_no_point_map",
            "control_point_count": 0,
            "mean_displacement": 0.0,
            "max_displacement": 0.0,
            "note": "Provide VGGT-Omega point maps to enable non-rigid refinement.",
        }
        return DeformableRegistrationResult(
            method="rbf_deformation_skipped",
            initial_pose=initial_pose,
            deformation_model=None,
            deformed_vertices_mesh_frame=vertices.copy(),
            final_vertices_camera_frame=final_vertices,
            control_source_points=np.empty((0, 3), dtype=np.float64),
            control_target_points=np.empty((0, 3), dtype=np.float64),
            metrics=metrics,
        )

    source_controls, target_controls, weights = build_deformation_constraints(
        correspondences,
        initial_pose=initial_pose,
        point_map=point_map,
        image_size=image_size,
        segmentation_mask=segmentation_mask,
        outside_mask_weight=config.segmentation_outside_weight,
        max_control_points=config.max_control_points,
    )
    source_controls, target_controls, weights, filter_metrics = _robust_control_subset(
        source_controls,
        target_controls,
        weights,
        config,
    )
    model = fit_rbf_deformation(
        source_controls,
        target_controls,
        weights=weights,
        regularization=config.regularization,
        kernel_radius=config.kernel_radius,
    )
    deformed_vertices = model.transform_points(vertices, chunk_size=config.chunk_size)
    final_vertices = apply_transform(deformed_vertices, initial_pose.transform_matrix)
    displacements = deformed_vertices - vertices
    control_residuals = np.linalg.norm(model.transform_points(source_controls) - target_controls, axis=1)
    metrics = {
        "status": "ok",
        "regularization_model": "gaussian_rbf",
        "control_point_count": int(len(source_controls)),
        "mean_control_residual": float(np.mean(control_residuals)),
        "median_control_residual": float(np.median(control_residuals)),
        "mean_displacement": float(np.mean(np.linalg.norm(displacements, axis=1))),
        "max_displacement": float(np.max(np.linalg.norm(displacements, axis=1))),
        "kernel_radius": float(model.kernel_radius),
        "regularization": float(config.regularization),
        "smoothness_energy": float(config.regularization * np.mean(model.coefficients * model.coefficients)),
        **filter_metrics,
    }
    return DeformableRegistrationResult(
        method="rbf_deformation",
        initial_pose=initial_pose,
        deformation_model=model,
        deformed_vertices_mesh_frame=deformed_vertices,
        final_vertices_camera_frame=final_vertices,
        control_source_points=source_controls,
        control_target_points=target_controls,
        metrics=metrics,
    )


def reprojection_error_table(
    correspondences: Iterable[Correspondence],
    initial_pose: InitialPoseResult,
) -> list[dict[str, Any]]:
    """Build a per-correspondence reprojection error table."""

    items = list(correspondences)
    if initial_pose.intrinsics is None:
        return [
            {
                "index": idx,
                "image_id": item.image_id,
                "label": item.label,
                "u": item.u,
                "v": item.v,
                "error_px": None,
                "note": "No intrinsics available for reprojection.",
            }
            for idx, item in enumerate(items)
        ]
    object_points, image_points, _ = correspondence_arrays(items)
    projected = project_points_pinhole(object_points, initial_pose.transform_matrix, initial_pose.intrinsics.camera_matrix)
    errors = np.linalg.norm(projected - image_points, axis=1)
    rows = []
    inlier_set = set(initial_pose.inlier_indices)
    for idx, (item, proj, error) in enumerate(zip(items, projected, errors)):
        rows.append(
            {
                "index": idx,
                "image_id": item.image_id,
                "label": item.label,
                "u": float(item.u),
                "v": float(item.v),
                "projected_u": float(proj[0]) if np.isfinite(proj[0]) else None,
                "projected_v": float(proj[1]) if np.isfinite(proj[1]) else None,
                "error_px": float(error) if np.isfinite(error) else None,
                "inlier": idx in inlier_set,
                "weight": item.effective_weight,
            }
        )
    return rows


def segmentation_projection_fraction(
    vertices_camera_frame: np.ndarray,
    intrinsics: CameraIntrinsics | None,
    segmentation_mask: np.ndarray | None,
    *,
    max_points: int = 10000,
) -> float | None:
    """Estimate how much of the projected surface falls inside a mask."""

    if intrinsics is None or segmentation_mask is None or len(vertices_camera_frame) == 0:
        return None
    vertices = np.asarray(vertices_camera_frame, dtype=np.float64).reshape(-1, 3)
    if len(vertices) > max_points:
        indices = np.linspace(0, len(vertices) - 1, max_points).astype(int)
        vertices = vertices[indices]
    z = vertices[:, 2]
    valid = np.isfinite(z) & (z > 1e-9)
    if not valid.any():
        return 0.0
    camera_matrix = intrinsics.camera_matrix
    u = camera_matrix[0, 0] * vertices[valid, 0] / z[valid] + camera_matrix[0, 2]
    v = camera_matrix[1, 1] * vertices[valid, 1] / z[valid] + camera_matrix[1, 2]
    mask = np.squeeze(segmentation_mask).astype(bool)
    height, width = mask.shape[:2]
    ui = np.round(u).astype(int)
    vi = np.round(v).astype(int)
    inside_image = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    if not inside_image.any():
        return 0.0
    inside_mask = mask[vi[inside_image], ui[inside_image]]
    return float(np.mean(inside_mask))
