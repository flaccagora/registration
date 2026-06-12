"""Input/output helpers for images, meshes, arrays, and JSON artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from pipeline.errors import InvalidInputError, MissingDependencyError


@dataclass
class MeshData:
    """Lightweight mesh container used by registration and export code."""

    vertices: np.ndarray
    faces: np.ndarray | None = None
    path: Path | None = None
    metadata: dict[str, Any] | None = None


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a ``Path``."""

    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_json(path: str | Path) -> Any:
    """Read JSON from disk."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(payload: Any, path: str | Path) -> Path:
    """Write JSON to disk and return the path."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return target


def file_path(file_obj: Any) -> str:
    """Extract a local path from Gradio-style file objects or raw strings."""

    if file_obj is None:
        return ""
    if isinstance(file_obj, (str, Path)):
        return str(file_obj)
    if isinstance(file_obj, dict):
        for key in ("name", "path"):
            if file_obj.get(key):
                return str(file_obj[key])
        if file_obj.get("video") is not None:
            return file_path(file_obj["video"])
    if hasattr(file_obj, "name"):
        return str(file_obj.name)
    if hasattr(file_obj, "path"):
        return str(file_obj.path)
    return str(file_obj)


def stable_hash(paths: Iterable[str | Path], extra: dict[str, Any] | None = None) -> str:
    """Create a stable cache key from file paths, mtimes, sizes, and options."""

    digest = hashlib.sha256()
    for raw_path in paths:
        path = Path(raw_path)
        digest.update(str(path.resolve()).encode("utf-8", errors="ignore"))
        if path.exists():
            stat = path.stat()
            digest.update(str(stat.st_size).encode())
            digest.update(str(int(stat.st_mtime_ns)).encode())
    if extra:
        digest.update(json.dumps(extra, sort_keys=True, default=str).encode())
    return digest.hexdigest()[:16]


def load_rgb_image(path: str | Path) -> np.ndarray:
    """Load an image as an RGB ``uint8`` NumPy array."""

    try:
        from PIL import Image
    except ImportError as exc:
        raise MissingDependencyError("Pillow is required to load images.") from exc

    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def save_rgb_image(image: np.ndarray, path: str | Path) -> Path:
    """Save an RGB ``uint8`` image."""

    try:
        from PIL import Image
    except ImportError as exc:
        raise MissingDependencyError("Pillow is required to save images.") from exc

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(target)
    return target


def save_mask(mask: np.ndarray, png_path: str | Path, npy_path: str | Path | None = None) -> tuple[Path, Path | None]:
    """Save a binary mask as PNG and optionally as a NumPy array."""

    mask_array = (np.squeeze(mask) > 0).astype(np.uint8)
    png_target = Path(png_path)
    png_target.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except ImportError as exc:
        raise MissingDependencyError("Pillow is required to save masks.") from exc
    Image.fromarray(mask_array * 255).save(png_target)

    npy_target = None
    if npy_path is not None:
        npy_target = Path(npy_path)
        npy_target.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_target, mask_array)
    return png_target, npy_target


def load_mask(path: str | Path) -> np.ndarray:
    """Load a mask from ``.npy`` or image file and return a binary array."""

    path = Path(path)
    if path.suffix.lower() == ".npy":
        return (np.load(path) > 0).astype(np.uint8)
    image = load_rgb_image(path)
    if image.ndim == 3:
        image = image[..., 0]
    return (image > 0).astype(np.uint8)


def blend_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (0, 210, 130), alpha: float = 0.45) -> np.ndarray:
    """Overlay a binary mask on an RGB image."""

    rgb = np.asarray(image, dtype=np.float32).copy()
    mask_bool = np.squeeze(mask).astype(bool)
    if mask_bool.shape[:2] != rgb.shape[:2]:
        raise InvalidInputError("Mask shape does not match image shape.")
    overlay = np.zeros_like(rgb)
    overlay[:, :] = np.asarray(color, dtype=np.float32)
    rgb[mask_bool] = (1.0 - alpha) * rgb[mask_bool] + alpha * overlay[mask_bool]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Convert a depth map into a visible RGB heatmap."""

    depth_array = np.asarray(depth, dtype=np.float32)
    if depth_array.ndim == 3:
        depth_array = depth_array[..., 0]
    finite = np.isfinite(depth_array) & (depth_array > 0)
    if not finite.any():
        return np.zeros((*depth_array.shape[:2], 3), dtype=np.uint8)
    low, high = np.percentile(depth_array[finite], [2, 98])
    if high <= low:
        high = low + 1.0
    normalized = np.clip((depth_array - low) / (high - low), 0.0, 1.0)
    gray = (normalized * 255).astype(np.uint8)
    try:
        import cv2

        colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    except ImportError:
        return np.stack([gray, 255 - gray, np.full_like(gray, 128)], axis=-1)


def load_mesh(path: str | Path) -> MeshData:
    """Load ``.obj``, ``.ply``, or ``.stl`` mesh data.

    ``trimesh`` is preferred for broad format support. A minimal OBJ fallback
    keeps the synthetic and documentation examples usable in smaller envs.
    """

    mesh_path = Path(path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    try:
        import trimesh

        loaded = trimesh.load(mesh_path, force="mesh", process=False)
        vertices = np.asarray(loaded.vertices, dtype=np.float64)
        faces = np.asarray(loaded.faces, dtype=np.int64) if getattr(loaded, "faces", None) is not None else None
        return MeshData(vertices=vertices, faces=faces, path=mesh_path, metadata={"loader": "trimesh"})
    except ImportError:
        if mesh_path.suffix.lower() != ".obj":
            raise MissingDependencyError("trimesh is required to load PLY/STL meshes.") from None
        return _load_obj_fallback(mesh_path)


def _load_obj_fallback(path: Path) -> MeshData:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                indices = []
                for token in line.strip().split()[1:]:
                    index = token.split("/")[0]
                    if index:
                        indices.append(int(index) - 1)
                if len(indices) >= 3:
                    faces.append(indices[:3])
    if not vertices:
        raise InvalidInputError(f"No vertices found in OBJ mesh: {path}")
    return MeshData(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64) if faces else None,
        path=path,
        metadata={"loader": "obj_fallback"},
    )


def sample_points(points: np.ndarray, max_points: int, seed: int = 7) -> np.ndarray:
    """Deterministically subsample a point array."""

    points = np.asarray(points)
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(points), size=max_points, replace=False))
    return points[indices]


def write_ply_points(points: np.ndarray, path: str | Path, colors: np.ndarray | None = None) -> Path:
    """Write a point cloud as ASCII PLY."""

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    has_color = colors is not None
    if has_color:
        color_array = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        if len(color_array) != len(points):
            raise InvalidInputError("PLY colors must have the same length as points.")
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_color:
        lines.extend(["property uchar red", "property uchar green", "property uchar blue"])
    lines.append("end_header")
    for idx, point in enumerate(points):
        if has_color:
            color = color_array[idx]
            lines.append(f"{point[0]} {point[1]} {point[2]} {int(color[0])} {int(color[1])} {int(color[2])}")
        else:
            lines.append(f"{point[0]} {point[1]} {point[2]}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def write_obj_mesh(vertices: np.ndarray, faces: np.ndarray | None, path: str | Path) -> Path:
    """Write a simple OBJ mesh."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"v {vertex[0]} {vertex[1]} {vertex[2]}" for vertex in np.asarray(vertices).reshape(-1, 3)]
    if faces is not None:
        for face in np.asarray(faces, dtype=np.int64).reshape(-1, 3):
            lines.append(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def extract_video_frames(video_path: str | Path, output_dir: str | Path, sample_fps: float = 1.0, max_frames: int | None = None) -> list[Path]:
    """Extract sampled video frames for later VGGT-Omega processing."""

    try:
        import cv2
    except ImportError as exc:
        raise MissingDependencyError("opencv-python is required to extract video frames.") from exc

    output = ensure_dir(output_dir)
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise InvalidInputError(f"Could not open video: {video_path}")
    fps = video.get(cv2.CAP_PROP_FPS)
    interval = max(int(round((fps if fps and fps > 0 else 1.0) / max(float(sample_fps), 0.1))), 1)
    frame_paths: list[Path] = []
    frame_idx = 0
    saved_idx = 0
    while True:
        ok, frame = video.read()
        if not ok:
            break
        if frame_idx % interval == 0:
            target = output / f"{saved_idx:06d}.png"
            cv2.imwrite(str(target), frame)
            frame_paths.append(target)
            saved_idx += 1
            if max_frames is not None and saved_idx >= max_frames:
                break
        frame_idx += 1
    video.release()
    if not frame_paths:
        raise InvalidInputError(f"No frames extracted from video: {video_path}")
    return frame_paths

