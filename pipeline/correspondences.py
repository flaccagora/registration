"""Manual 2D-to-3D correspondence schema and parsing utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from pipeline.errors import InvalidInputError


@dataclass(frozen=True)
class Correspondence:
    """One sparse manual 2D-to-3D registration constraint.

    Coordinates are image pixel ``(u, v)`` and mesh/world surface point
    ``(x, y, z)`` in the mesh coordinate frame.
    """

    image_id: str
    u: float
    v: float
    x: float
    y: float
    z: float
    label: str | None = None
    confidence: float = 1.0
    weight: float | None = None

    @property
    def point2d(self) -> np.ndarray:
        return np.asarray([self.u, self.v], dtype=np.float64)

    @property
    def point3d(self) -> np.ndarray:
        return np.asarray([self.x, self.y, self.z], dtype=np.float64)

    @property
    def effective_weight(self) -> float:
        value = self.confidence if self.weight is None else self.weight
        return float(max(0.0, value))

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def _as_float(record: dict[str, Any], keys: Iterable[str], *, required: bool = True, default: float | None = None) -> float | None:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip() != "":
            return float(value)
    if required:
        raise InvalidInputError(f"Missing numeric field. Expected one of: {', '.join(keys)}")
    return default


def _pixel_from_record(record: dict[str, Any]) -> tuple[float, float]:
    if isinstance(record.get("pixel"), dict):
        pixel = record["pixel"]
        return float(pixel.get("u", pixel.get("x"))), float(pixel.get("v", pixel.get("y")))
    if isinstance(record.get("pixel"), (list, tuple)) and len(record["pixel"]) >= 2:
        return float(record["pixel"][0]), float(record["pixel"][1])
    if isinstance(record.get("image_point"), (list, tuple)) and len(record["image_point"]) >= 2:
        return float(record["image_point"][0]), float(record["image_point"][1])
    u = _as_float(record, ("u", "image_x", "x_px", "pixel_x"))
    v = _as_float(record, ("v", "image_y", "y_px", "pixel_y"))
    return float(u), float(v)


def _point3d_from_record(record: dict[str, Any]) -> tuple[float, float, float]:
    for key in ("point3d", "mesh_point", "world_point", "surface_point"):
        value = record.get(key)
        if isinstance(value, dict):
            return float(value["x"]), float(value["y"]), float(value["z"])
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            return float(value[0]), float(value[1]), float(value[2])
    x = _as_float(record, ("x", "mesh_x", "world_x", "X"))
    y = _as_float(record, ("y", "mesh_y", "world_y", "Y"))
    z = _as_float(record, ("z", "mesh_z", "world_z", "Z"))
    return float(x), float(y), float(z)


def _direct_point3d_from_record(record: dict[str, Any]) -> tuple[float, float, float] | None:
    try:
        return _point3d_from_record(record)
    except (KeyError, TypeError, ValueError, InvalidInputError):
        return None


def correspondence_from_record(record: dict[str, Any], default_image_id: str = "frame_0000") -> Correspondence:
    """Normalize a dictionary into the canonical correspondence dataclass."""

    image_id = str(record.get("image_id") or record.get("frame_id") or record.get("image") or default_image_id)
    u, v = _pixel_from_record(record)
    x, y, z = _point3d_from_record(record)
    confidence = _as_float(record, ("confidence", "score"), required=False, default=1.0)
    weight = _as_float(record, ("weight",), required=False, default=None)
    return Correspondence(
        image_id=image_id,
        u=u,
        v=v,
        x=x,
        y=y,
        z=z,
        label=record.get("label") or record.get("class") or record.get("name"),
        confidence=float(confidence if confidence is not None else 1.0),
        weight=float(weight) if weight is not None else None,
    )


def load_correspondences(path: str | Path, default_image_id: str = "frame_0000") -> list[Correspondence]:
    """Load correspondences from CSV, canonical JSON, or manual-correspondences JSON.

    The `manual-correspondences` registration export stores 2D landmark pixels
    by `ct_landmark_id` and references 3D points through `ct_landmarks_path`.
    This loader resolves that catalog so those exports become canonical
    `Correspondence` rows for the registration pipeline.
    """

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Correspondence file not found: {source}")
    suffix = source.suffix.lower()
    if suffix == ".csv":
        with source.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    elif suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows = payload.get("correspondences") or payload.get("points") or payload.get("records") or payload.get("data")
        else:
            rows = payload
        if not isinstance(rows, list):
            raise InvalidInputError("JSON correspondence file must contain a list or a 'correspondences' list.")
    else:
        raise InvalidInputError("Correspondence file must be .csv or .json")
    rows = [row for row in rows if _record_has_content(row)]
    if suffix == ".json" and _looks_like_manual_correspondence_export(rows):
        correspondences = correspondences_from_manual_export_records(rows, source.parent)
        validate_correspondences(correspondences)
        return correspondences
    if suffix == ".json" and _looks_like_raw_label_studio_export(rows):
        raise InvalidInputError(
            "Raw Label Studio export detected. Convert it with the "
            "manual-correspondences parser first, then pass the normalized "
            "registration export containing frame_id, landmarks, and "
            "ct_landmarks_path."
        )
    correspondences = [correspondence_from_record(row, default_image_id=default_image_id) for row in rows]
    validate_correspondences(correspondences)
    return correspondences


def _record_has_content(record: Any) -> bool:
    if not isinstance(record, dict):
        return record is not None
    return any(value is not None and str(value).strip() != "" for value in record.values())


def _looks_like_manual_correspondence_export(rows: list[Any]) -> bool:
    """Return true for normalized manual-correspondences registration records."""

    for row in rows:
        if not isinstance(row, dict):
            continue
        has_landmark_records = isinstance(row.get("landmarks"), list) or isinstance(row.get("pseudo_labels"), list)
        has_direct_3d = _direct_point3d_from_record(row) is not None
        if has_landmark_records and not has_direct_3d:
            return True
    return False


def _looks_like_raw_label_studio_export(rows: list[Any]) -> bool:
    """Return true for raw Label Studio task exports before normalization."""

    for row in rows:
        if not isinstance(row, dict):
            continue
        has_task_data = isinstance(row.get("data"), dict)
        has_annotations = isinstance(row.get("annotations"), list) or isinstance(row.get("completions"), list)
        if has_task_data and has_annotations:
            return True
    return False


def correspondences_from_manual_export_records(records: list[dict[str, Any]], base_dir: str | Path | None = None) -> list[Correspondence]:
    """Convert `manual-correspondences` normalized exports to canonical rows.

    Expected record shape is the output of
    `surgvu_annotator.registration.parse_registration_label_studio_export`:
    each frame record has `frame_id`, `landmarks`, and usually
    `ct_landmarks_path`. Landmark IDs are resolved against that 3D landmark
    catalog.
    """

    base = Path(base_dir or ".")
    correspondences: list[Correspondence] = []
    unresolved: list[str] = []
    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        image_id = str(record.get("frame_id") or record.get("image_id") or f"record_{record_index:04d}")
        catalog = _landmark_catalog_from_record(record, base)
        for landmark in _iter_manual_landmark_records(record):
            if not isinstance(landmark, dict):
                continue
            landmark_id = landmark.get("ct_landmark_id") or landmark.get("landmark_id") or landmark.get("label")
            if landmark_id is None:
                unresolved.append(f"{image_id}: landmark without ct_landmark_id")
                continue
            point = _point3d_from_manual_landmark(landmark)
            if point is None:
                point = catalog.get(str(landmark_id))
            if point is None:
                unresolved.append(f"{image_id}: {landmark_id}")
                continue
            u, v = _pixel_from_record(landmark)
            confidence = _as_float(landmark, ("confidence", "score"), required=False, default=1.0)
            weight_value = _as_float(landmark, ("weight",), required=False, default=None)
            correspondences.append(
                Correspondence(
                    image_id=image_id,
                    u=float(u),
                    v=float(v),
                    x=float(point[0]),
                    y=float(point[1]),
                    z=float(point[2]),
                    label=str(landmark_id),
                    confidence=float(confidence if confidence is not None else 1.0),
                    weight=float(weight_value) if weight_value is not None else None,
                )
            )
    if unresolved:
        sample = "; ".join(unresolved[:8])
        raise InvalidInputError(
            "Could not resolve all manual-correspondences landmarks to 3D points. "
            "Ensure each record has a valid ct_landmarks_path or embedded point3d. "
            f"Unresolved examples: {sample}"
        )
    return correspondences


def _iter_manual_landmark_records(record: dict[str, Any]) -> list[dict[str, Any]]:
    landmarks = [item for item in record.get("landmarks") or [] if isinstance(item, dict)]
    for pseudo in record.get("pseudo_labels") or []:
        if isinstance(pseudo, dict) and pseudo.get("type") == "correspondence":
            landmarks.append(pseudo)
    return landmarks


def _landmark_catalog_from_record(record: dict[str, Any], base_dir: Path) -> dict[str, np.ndarray]:
    embedded = record.get("ct_landmarks") or record.get("landmark_catalog")
    if embedded is not None:
        return _parse_landmark_catalog(embedded)
    path_value = record.get("ct_landmarks_path")
    if not path_value:
        return {}
    path = _resolve_catalog_path(path_value, base_dir)
    if path is None:
        return {}
    return _parse_landmark_catalog(json.loads(path.read_text(encoding="utf-8")))


def _resolve_catalog_path(path_value: Any, base_dir: Path) -> Path | None:
    path = Path(str(path_value))
    candidates = [path] if path.is_absolute() else [base_dir / path, Path.cwd() / path, path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _parse_landmark_catalog(payload: Any) -> dict[str, np.ndarray]:
    if isinstance(payload, dict) and isinstance(payload.get("landmarks"), list):
        payload = payload["landmarks"]

    catalog: dict[str, np.ndarray] = {}
    if isinstance(payload, dict):
        for landmark_id, value in payload.items():
            point = _point3d_like_value(value)
            if point is not None:
                catalog[str(landmark_id)] = point
    elif isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            landmark_id = item.get("id") or item.get("name") or item.get("label") or item.get("ct_landmark_id")
            point = _point3d_like_value(item)
            if landmark_id is not None and point is not None:
                catalog[str(landmark_id)] = point
    return catalog


def _point3d_like_value(value: Any) -> np.ndarray | None:
    """Parse common 3D point shapes used by canonical and annotation exports."""

    point = value
    if isinstance(value, dict):
        for key in ("point3d", "mesh_point", "world_point", "surface_point", "position_mm", "xyz"):
            nested = value.get(key)
            if nested is not None:
                point = nested
                break
        else:
            if all(axis in value for axis in ("x", "y", "z")):
                point = [value["x"], value["y"], value["z"]]
            else:
                return None
    try:
        array = np.asarray(point, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size != 3 or not np.all(np.isfinite(array)):
        return None
    return array


def _point3d_from_manual_landmark(landmark: dict[str, Any]) -> np.ndarray | None:
    return _point3d_like_value(landmark)


def save_correspondences(correspondences: Iterable[Correspondence], path: str | Path) -> Path:
    """Save correspondences as CSV or JSON based on file suffix."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [item.to_record() for item in correspondences]
    if target.suffix.lower() == ".csv":
        fieldnames = ["image_id", "u", "v", "x", "y", "z", "label", "confidence", "weight"]
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    elif target.suffix.lower() == ".json":
        target.write_text(
            json.dumps({"version": 1, "correspondences": rows}, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
    else:
        raise InvalidInputError("Output correspondence file must be .csv or .json")
    return target


def validate_correspondences(correspondences: Iterable[Correspondence]) -> None:
    """Check numeric validity and minimum structural requirements."""

    items = list(correspondences)
    if not items:
        raise InvalidInputError("No correspondences were found.")
    for idx, item in enumerate(items):
        values = [item.u, item.v, item.x, item.y, item.z, item.confidence]
        if not np.all(np.isfinite(values)):
            raise InvalidInputError(f"Correspondence {idx} contains non-finite values.")
        if item.confidence < 0:
            raise InvalidInputError(f"Correspondence {idx} has negative confidence.")


def filter_correspondences(
    correspondences: Iterable[Correspondence],
    image_id: str | None = None,
    mask: np.ndarray | None = None,
    outside_mask_weight: float | None = None,
) -> list[Correspondence]:
    """Filter or downweight correspondences by frame id and segmentation mask."""

    selected = [item for item in correspondences if image_id is None or item.image_id == image_id]
    if mask is None:
        return selected
    mask_bool = np.squeeze(mask).astype(bool)
    filtered: list[Correspondence] = []
    height, width = mask_bool.shape[:2]
    for item in selected:
        u = int(round(item.u))
        v = int(round(item.v))
        inside = 0 <= u < width and 0 <= v < height and bool(mask_bool[v, u])
        if inside:
            filtered.append(item)
        elif outside_mask_weight is not None and outside_mask_weight > 0:
            filtered.append(
                Correspondence(
                    image_id=item.image_id,
                    u=item.u,
                    v=item.v,
                    x=item.x,
                    y=item.y,
                    z=item.z,
                    label=item.label,
                    confidence=item.confidence,
                    weight=item.effective_weight * outside_mask_weight,
                )
            )
    return filtered


def correspondence_arrays(correspondences: Iterable[Correspondence]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``object_points``, ``image_points``, and weights arrays."""

    items = list(correspondences)
    validate_correspondences(items)
    object_points = np.asarray([item.point3d for item in items], dtype=np.float64)
    image_points = np.asarray([item.point2d for item in items], dtype=np.float64)
    weights = np.asarray([item.effective_weight for item in items], dtype=np.float64)
    return object_points, image_points, weights


def make_synthetic_correspondences(
    image_id: str = "synthetic_frame",
    image_size: tuple[int, int] = (640, 480),
) -> tuple[list[Correspondence], dict[str, Any]]:
    """Create a tiny deterministic non-clinical correspondence set.

    The points are a cube-like target projected with a known camera. This is
    intended for dry runs and geometry smoke tests only.
    """

    width, height = image_size
    object_points = np.asarray(
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
    camera_matrix = np.asarray(
        [[720.0, 0.0, width / 2.0], [0.0, 720.0, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    angle = np.deg2rad(8.0)
    rotation = np.asarray(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ],
        dtype=np.float64,
    )
    translation = np.asarray([5.0, -3.0, 520.0], dtype=np.float64)
    camera_points = (rotation @ object_points.T).T + translation
    image_points = np.column_stack(
        [
            camera_matrix[0, 0] * camera_points[:, 0] / camera_points[:, 2] + camera_matrix[0, 2],
            camera_matrix[1, 1] * camera_points[:, 1] / camera_points[:, 2] + camera_matrix[1, 2],
        ]
    )
    correspondences = [
        Correspondence(
            image_id=image_id,
            u=float(pixel[0]),
            v=float(pixel[1]),
            x=float(point[0]),
            y=float(point[1]),
            z=float(point[2]),
            label=f"synthetic_{idx:02d}",
            confidence=1.0,
        )
        for idx, (pixel, point) in enumerate(zip(image_points, object_points))
    ]
    metadata = {
        "image_size": {"width": width, "height": height},
        "camera_matrix": camera_matrix.tolist(),
        "rotation_matrix": rotation.tolist(),
        "translation": translation.tolist(),
        "clinical_use": False,
    }
    return correspondences, metadata
