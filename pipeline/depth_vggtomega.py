"""Thin VGGT-Omega depth, camera, and point-map wrapper."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

from pipeline.errors import InvalidInputError, MissingCheckpointError, MissingDependencyError
from pipeline.io import colorize_depth, ensure_dir, save_rgb_image, stable_hash, write_ply_points
from pipeline.repo_paths import resolve_repo_path


@dataclass
class VGGTOmegaResult:
    """Saved VGGT-Omega prediction artifacts."""

    output_dir: Path
    predictions_npz_path: Path
    camera_json_path: Path
    depth_paths: list[Path]
    depth_visualization_paths: list[Path]
    point_cloud_path: Path | None
    glb_path: Path | None
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "predictions_npz_path": str(self.predictions_npz_path),
            "camera_json_path": str(self.camera_json_path),
            "depth_paths": [str(path) for path in self.depth_paths],
            "depth_visualization_paths": [str(path) for path in self.depth_visualization_paths],
            "point_cloud_path": str(self.point_cloud_path) if self.point_cloud_path else None,
            "glb_path": str(self.glb_path) if self.glb_path else None,
            "metadata": self.metadata,
        }


class VGGTOmegaRunner:
    """Lazy-loading adapter around the local VGGT-Omega repository."""

    def __init__(
        self,
        repo_path: str | Path = "external/vggt-omega",
        checkpoint_path: str | Path | None = None,
        device: str = "cuda",
        image_resolution: int = 512,
        output_dir: str | Path = "outputs/vggt_omega",
        cache: bool = True,
    ) -> None:
        self.repo_path = resolve_repo_path(repo_path, "vggt_omega")
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.device = device
        self.image_resolution = int(image_resolution)
        self.output_dir = Path(output_dir)
        self.cache = bool(cache)
        self._model = None

    def _validate_runtime_paths(self) -> None:
        if not self.repo_path.exists():
            raise FileNotFoundError(f"VGGT-Omega repo not found: {self.repo_path}")
        if self.checkpoint_path is None:
            raise MissingCheckpointError("VGGT-Omega checkpoint_path is required.")
        if not self.checkpoint_path.exists():
            raise MissingCheckpointError(f"VGGT-Omega checkpoint not found: {self.checkpoint_path}")

    def _prepare_imports(self) -> None:
        self._validate_runtime_paths()
        if str(self.repo_path) not in sys.path:
            sys.path.insert(0, str(self.repo_path))

    def _load_model(self):
        if self._model is not None:
            return self._model
        self._prepare_imports()
        try:
            import torch
            from vggt_omega.models import VGGTOmega
        except ImportError as exc:
            raise MissingDependencyError("VGGT-Omega dependencies are not importable. Install the local repo first.") from exc
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise MissingDependencyError("CUDA was requested for VGGT-Omega, but torch.cuda.is_available() is false.")
        model = VGGTOmega().eval()
        state_dict = torch.load(str(self.checkpoint_path), map_location="cpu")
        model.load_state_dict(state_dict)
        self._model = model.to(self.device)
        return self._model

    def run(self, image_paths: Iterable[str | Path], output_name: str | None = None) -> VGGTOmegaResult:
        """Run VGGT-Omega and save depth/camera/point-map artifacts."""

        paths = [Path(path) for path in image_paths]
        if not paths:
            raise InvalidInputError("At least one image is required for VGGT-Omega.")
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"VGGT-Omega image not found: {path}")
        self._validate_runtime_paths()

        cache_key = stable_hash(paths, {"resolution": self.image_resolution, "checkpoint": str(self.checkpoint_path)})
        run_name = output_name or f"vggt_{cache_key}"
        output_dir = ensure_dir(self.output_dir / run_name)
        predictions_path = output_dir / "predictions.npz"
        camera_json_path = output_dir / "cameras.json"
        metadata_path = output_dir / "metadata.json"
        if self.cache and predictions_path.exists() and camera_json_path.exists() and metadata_path.exists():
            return self._result_from_cache(output_dir, predictions_path, camera_json_path, metadata_path)

        self._prepare_imports()
        try:
            import torch
            from vggt_omega.utils.load_fn import load_and_preprocess_images
            from vggt_omega.utils.pose_enc import encoding_to_camera
        except ImportError as exc:
            raise MissingDependencyError("VGGT-Omega dependencies are not importable. Install the local repo first.") from exc

        model = self._load_model()
        images = load_and_preprocess_images([str(path) for path in paths], image_resolution=self.image_resolution).to(self.device)
        with torch.inference_mode():
            predictions = model(images)
        extrinsic, intrinsic = encoding_to_camera(predictions["pose_enc"], predictions["images"].shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        predictions_np: dict[str, np.ndarray] = {}
        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                array = value.detach().float().cpu().numpy()
                if array.shape[0] == 1:
                    array = array[0]
                predictions_np[key] = array
        missing = [key for key in ("depth", "extrinsic", "intrinsic") if key not in predictions_np]
        if missing:
            raise InvalidInputError(f"VGGT-Omega predictions are missing required keys: {', '.join(missing)}")

        predictions_np["depth"] = normalize_depth_sequence(predictions_np["depth"])
        predictions_np["extrinsic"] = normalize_extrinsic_sequence(predictions_np["extrinsic"])
        predictions_np["intrinsic"] = normalize_intrinsic_sequence(predictions_np["intrinsic"])
        predictions_np["world_points_from_depth"] = unproject_depth_map_to_point_map(
            predictions_np["depth"],
            predictions_np["extrinsic"],
            predictions_np["intrinsic"],
        )
        np.savez(predictions_path, **predictions_np)

        depth_paths, depth_vis_paths = _save_depth_outputs(predictions_np["depth"], output_dir)
        camera_json_path.write_text(
            json.dumps(_camera_payload(predictions_np["extrinsic"], predictions_np["intrinsic"], paths), indent=2),
            encoding="utf-8",
        )
        point_cloud_path = _save_point_cloud(predictions_np.get("world_points_from_depth"), output_dir)
        glb_path, glb_error = _save_glb_if_available(predictions_np, output_dir)
        metadata = {
            "image_paths": [str(path) for path in paths],
            "image_resolution": self.image_resolution,
            "checkpoint_path": str(self.checkpoint_path),
            "cache_key": cache_key,
            "depth_paths": [str(path) for path in depth_paths],
            "depth_visualization_paths": [str(path) for path in depth_vis_paths],
            "point_cloud_path": str(point_cloud_path) if point_cloud_path else None,
            "glb_path": str(glb_path) if glb_path else None,
            "glb_error": glb_error,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return VGGTOmegaResult(
            output_dir=output_dir,
            predictions_npz_path=predictions_path,
            camera_json_path=camera_json_path,
            depth_paths=depth_paths,
            depth_visualization_paths=depth_vis_paths,
            point_cloud_path=point_cloud_path,
            glb_path=glb_path,
            metadata=metadata,
        )

    def _result_from_cache(self, output_dir: Path, predictions_path: Path, camera_json_path: Path, metadata_path: Path) -> VGGTOmegaResult:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return VGGTOmegaResult(
            output_dir=output_dir,
            predictions_npz_path=predictions_path,
            camera_json_path=camera_json_path,
            depth_paths=[Path(path) for path in metadata.get("depth_paths", [])],
            depth_visualization_paths=[Path(path) for path in metadata.get("depth_visualization_paths", [])],
            point_cloud_path=Path(metadata["point_cloud_path"]) if metadata.get("point_cloud_path") else None,
            glb_path=Path(metadata["glb_path"]) if metadata.get("glb_path") else None,
            metadata=metadata | {"cache_hit": True},
        )


def unproject_depth_map_to_point_map(depth_map: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """Unproject depth into VGGT world coordinates.

    VGGT-Omega returns camera extrinsics that map world points into camera
    coordinates. This helper mirrors the local demo logic and applies the
    inverse transform after pinhole backprojection.
    """

    depth = normalize_depth_sequence(depth_map)
    extrinsic = normalize_extrinsic_sequence(extrinsic)
    intrinsic = normalize_intrinsic_sequence(intrinsic)
    num_frames, height, width = depth.shape
    extrinsic = _match_frame_count(extrinsic, num_frames, "extrinsic")
    intrinsic = _match_frame_count(intrinsic, num_frames, "intrinsic")

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]
    camera_points = np.stack([(x - cx) / fx * depth, (y - cy) / fy * depth, depth], axis=-1)

    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum("sij,shwj->shwi", np.transpose(rotation, (0, 2, 1)), camera_points - translation[:, None, None, :])


def normalize_depth_sequence(depth: np.ndarray) -> np.ndarray:
    """Normalize VGGT depth arrays to ``N x H x W``."""

    depth_array = np.asarray(depth)
    if depth_array.ndim == 4 and depth_array.shape[-1] == 1:
        depth_array = depth_array[..., 0]
    elif depth_array.ndim == 3 and depth_array.shape[-1] == 1:
        depth_array = depth_array[..., 0]
    if depth_array.ndim == 2:
        depth_array = depth_array[None, ...]
    if depth_array.ndim != 3:
        raise InvalidInputError(f"Expected VGGT depth shape N x H x W or H x W x 1, got {depth_array.shape}.")
    return depth_array


def normalize_intrinsic_sequence(intrinsic: np.ndarray) -> np.ndarray:
    """Normalize camera intrinsics to ``N x 3 x 3``."""

    array = np.asarray(intrinsic, dtype=np.float64)
    if array.ndim == 2 and array.shape == (3, 3):
        array = array[None, ...]
    if array.ndim != 3 or array.shape[-2:] != (3, 3):
        raise InvalidInputError(f"Expected VGGT intrinsic shape N x 3 x 3 or 3 x 3, got {array.shape}.")
    return array


def normalize_extrinsic_sequence(extrinsic: np.ndarray) -> np.ndarray:
    """Normalize camera extrinsics to ``N x 3 x 4`` or ``N x 4 x 4``."""

    array = np.asarray(extrinsic, dtype=np.float64)
    if array.ndim == 2 and array.shape in ((3, 4), (4, 4)):
        array = array[None, ...]
    if array.ndim != 3 or array.shape[-2:] not in ((3, 4), (4, 4)):
        raise InvalidInputError(f"Expected VGGT extrinsic shape N x 3 x 4, N x 4 x 4, 3 x 4, or 4 x 4, got {array.shape}.")
    return array


def _match_frame_count(array: np.ndarray, num_frames: int, name: str) -> np.ndarray:
    if len(array) == num_frames:
        return array
    if len(array) == 1:
        return np.repeat(array, num_frames, axis=0)
    raise InvalidInputError(f"VGGT {name} frame count {len(array)} does not match depth frame count {num_frames}.")


def _save_depth_outputs(depth: np.ndarray, output_dir: Path) -> tuple[list[Path], list[Path]]:
    depth_array = normalize_depth_sequence(depth)
    depth_paths: list[Path] = []
    vis_paths: list[Path] = []
    for idx, frame_depth in enumerate(depth_array):
        depth_path = output_dir / f"depth_{idx:04d}.npy"
        np.save(depth_path, frame_depth.astype(np.float32))
        vis_path = output_dir / f"depth_{idx:04d}.png"
        save_rgb_image(colorize_depth(frame_depth), vis_path)
        depth_paths.append(depth_path)
        vis_paths.append(vis_path)
    return depth_paths, vis_paths


def _camera_payload(extrinsic: np.ndarray, intrinsic: np.ndarray, image_paths: list[Path]) -> dict[str, Any]:
    extrinsic = normalize_extrinsic_sequence(extrinsic)
    intrinsic = normalize_intrinsic_sequence(intrinsic)
    frame_count = max(len(image_paths), len(extrinsic), len(intrinsic))
    extrinsic = _match_frame_count(extrinsic, frame_count, "extrinsic")
    intrinsic = _match_frame_count(intrinsic, frame_count, "intrinsic")
    frames = []
    for idx in range(frame_count):
        image_path = image_paths[idx] if idx < len(image_paths) else None
        frames.append(
            {
                "frame_index": idx,
                "image_path": str(image_path) if image_path is not None else None,
                "intrinsic": np.asarray(intrinsic[idx]).astype(float).tolist(),
                "extrinsic_world_to_camera": np.asarray(extrinsic[idx]).astype(float).tolist(),
            }
        )
    return {"frames": frames}


def _save_point_cloud(point_map: np.ndarray | None, output_dir: Path, max_points: int = 200000) -> Path | None:
    if point_map is None:
        return None
    points = np.asarray(point_map).reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    if len(points) == 0:
        return None
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points).astype(int)
        points = points[indices]
    return write_ply_points(points, output_dir / "vggt_point_cloud.ply")


def _save_glb_if_available(predictions_np: dict[str, np.ndarray], output_dir: Path) -> tuple[Path | None, str | None]:
    try:
        from visual_util import predictions_to_glb
    except ImportError as exc:
        return None, f"visual_util import failed: {exc}"
    try:
        glb_predictions = dict(predictions_np)
        if "depth" in glb_predictions and glb_predictions["depth"].ndim == 3:
            glb_predictions["depth"] = glb_predictions["depth"][..., None]
        scene = predictions_to_glb(glb_predictions, target_dir=str(output_dir), max_points=200000)
        glb_path = output_dir / "vggt_scene.glb"
        scene.export(file_obj=glb_path)
        return glb_path, None
    except Exception as exc:
        return None, f"GLB export failed: {exc}"


def load_first_point_map(predictions_npz_path: str | Path) -> np.ndarray | None:
    """Load the first VGGT point map from a saved ``predictions.npz`` file."""

    payload = np.load(predictions_npz_path)
    for key in ("world_points_from_depth", "points", "point_map"):
        if key in payload:
            point_map = payload[key]
            return point_map[0] if point_map.ndim == 4 else point_map
    return None


def load_first_depth(predictions_npz_path: str | Path) -> np.ndarray | None:
    """Load the first VGGT depth map from a saved ``predictions.npz`` file."""

    payload = np.load(predictions_npz_path)
    if "depth" not in payload:
        return None
    return normalize_depth_sequence(payload["depth"])[0]
