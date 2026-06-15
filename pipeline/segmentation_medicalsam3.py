"""Thin MedicalSAM3 segmentation wrapper.

The wrapper isolates repository-specific imports and only loads the model when
``segment`` is called. This keeps the registration package importable on
machines where MedicalSAM3 dependencies or checkpoints are not installed yet.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np

from pipeline.errors import InvalidInputError, MissingCheckpointError, MissingDependencyError
from pipeline.io import blend_mask, ensure_dir, load_mask, load_rgb_image, save_mask, save_rgb_image
from pipeline.repo_paths import resolve_repo_path


@dataclass
class SegmentationPrompt:
    """Prompt passed to MedicalSAM3."""

    prompt_type: str = "text"
    text: str | None = None
    point: tuple[float, float] | None = None
    points: list[tuple[float, float]] | None = None
    box: tuple[float, float, float, float] | None = None
    boxes: list[tuple[float, float, float, float]] | None = None
    mask_path: Path | None = None


@dataclass
class SegmentationResult:
    """Saved segmentation outputs."""

    mask: np.ndarray
    mask_png_path: Path
    mask_npy_path: Path | None
    overlay_path: Path
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mask_png_path": str(self.mask_png_path),
            "mask_npy_path": str(self.mask_npy_path) if self.mask_npy_path else None,
            "overlay_path": str(self.overlay_path),
            "metadata": self.metadata,
        }


class MedicalSAM3Segmenter:
    """Lazy-loading adapter around the local MedicalSAM3 repository."""

    def __init__(
        self,
        repo_path: str | Path = "external/Medical-SAM3",
        checkpoint_path: str | Path | None = None,
        device: str = "cuda",
        confidence_threshold: float = 0.1,
        output_dir: str | Path = "outputs/segmentation",
    ) -> None:
        self.repo_path = resolve_repo_path(repo_path, "medicalsam3")
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.device = device
        self.confidence_threshold = float(confidence_threshold)
        self.output_dir = Path(output_dir)
        self._module = None
        self._model = None

    def _validate_runtime_paths(self) -> None:
        if not self.repo_path.exists():
            raise FileNotFoundError(f"MedicalSAM3 repo not found: {self.repo_path}")
        if self.checkpoint_path is not None and not self.checkpoint_path.exists():
            raise MissingCheckpointError(f"MedicalSAM3 checkpoint not found: {self.checkpoint_path}")

    def _load_module(self):
        self._validate_runtime_paths()
        if self._module is not None:
            return self._module
        module_path = self.repo_path / "inference" / "sam3_inference.py"
        if not module_path.exists():
            raise FileNotFoundError(f"MedicalSAM3 inference module not found: {module_path}")
        for path in (self.repo_path, self.repo_path / "inference"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        spec = importlib.util.spec_from_file_location("medical_sam3_local_inference", module_path)
        if spec is None or spec.loader is None:
            raise MissingDependencyError(f"Could not load MedicalSAM3 module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except ImportError as exc:
            raise MissingDependencyError(
                "MedicalSAM3 dependencies are not importable. Install the local Medical-SAM3 repo "
                "and its requirements before running model-backed segmentation."
            ) from exc
        if not hasattr(module, "SAM3_ROOT"):
            module.SAM3_ROOT = self.repo_path
        self._module = module
        return module

    def _load_model(self):
        if self._model is not None:
            return self._model
        if not self.device.startswith("cuda"):
            raise MissingDependencyError(
                "The local MedicalSAM3 inference helper uses CUDA autocast internally. "
                "Use device='cuda' for text/box/point prompts, or use prompt_type='mask' "
                "to provide a precomputed mask without loading MedicalSAM3."
            )
        try:
            import torch
        except ImportError as exc:
            raise MissingDependencyError("PyTorch is required for MedicalSAM3 model-backed segmentation.") from exc
        if not torch.cuda.is_available():
            raise MissingDependencyError(
                "MedicalSAM3 model-backed segmentation requires CUDA with the current local inference helper. "
                "Use a precomputed mask prompt for CPU-only checks."
            )
        module = self._load_module()
        if not hasattr(module, "SAM3Model"):
            raise MissingDependencyError("MedicalSAM3 inference module does not expose SAM3Model.")
        self._model = module.SAM3Model(
            confidence_threshold=self.confidence_threshold,
            device=self.device,
            checkpoint_path=str(self.checkpoint_path) if self.checkpoint_path else None,
        )
        return self._model

    def segment(self, image_path: str | Path, prompt: SegmentationPrompt, output_name: str | None = None) -> SegmentationResult:
        """Segment an image and save mask artifacts.

        Supported direct MedicalSAM3 prompts are text and box. Point prompts are
        converted into small boxes around each point. Multiple point or box
        prompts are evaluated independently and unioned. Mask prompts bypass
        model inference and are copied into the standard output format.
        """

        image = load_rgb_image(image_path)
        height, width = image.shape[:2]
        output_dir = ensure_dir(self.output_dir)
        stem = output_name or Path(image_path).stem
        prompt_type = prompt.prompt_type.lower().strip()

        if prompt_type == "mask":
            if prompt.mask_path is None:
                raise InvalidInputError("Mask prompt selected but no mask_path was provided.")
            mask = load_mask(prompt.mask_path)
            metadata = {"method": "provided_mask", "prompt_type": "mask"}
        else:
            model = self._load_model()
            inference_state = model.encode_image(image)
            if prompt_type == "text":
                if not prompt.text:
                    raise InvalidInputError("Text prompt selected but no text prompt was provided.")
                mask = model.predict_text(inference_state, prompt.text)
                metadata = {"method": "medicalsam3_text", "prompt_type": "text", "text": prompt.text}
            elif prompt_type == "box":
                boxes = prompt.boxes or ([prompt.box] if prompt.box is not None else [])
                if not boxes:
                    raise InvalidInputError("Box prompt selected but no box was provided.")
                int_boxes = [_normalize_box(box, width, height) for box in boxes]
                mask = _union_box_predictions(model, inference_state, int_boxes, (height, width))
                metadata = {
                    "method": "medicalsam3_box" if len(int_boxes) == 1 else "medicalsam3_boxes_union",
                    "prompt_type": "box",
                    "box": list(int_boxes[0]) if len(int_boxes) == 1 else None,
                    "boxes": [list(box) for box in int_boxes],
                }
            elif prompt_type == "point":
                points = prompt.points or ([prompt.point] if prompt.point is not None else [])
                if not points:
                    raise InvalidInputError("Point prompt selected but no point was provided.")
                radius = max(12, int(round(0.03 * max(width, height))))
                boxes = [_box_around_point(point, radius, width, height) for point in points]
                mask = _union_box_predictions(model, inference_state, boxes, (height, width))
                point_values = [[float(x), float(y)] for x, y in points]
                metadata = {
                    "method": "medicalsam3_point_as_box" if len(boxes) == 1 else "medicalsam3_points_as_boxes_union",
                    "prompt_type": "point",
                    "point": point_values[0] if len(point_values) == 1 else None,
                    "points": point_values,
                    "box": list(boxes[0]) if len(boxes) == 1 else None,
                    "boxes": [list(box) for box in boxes],
                }
            else:
                raise InvalidInputError(f"Unsupported segmentation prompt type: {prompt.prompt_type}")
            if mask is None:
                raise InvalidInputError("MedicalSAM3 returned no mask for the provided prompt.")

        mask = np.asarray(mask)
        if mask.shape[:2] != (height, width):
            mask = _resize_binary_mask(mask, (height, width))

        mask_png, mask_npy = save_mask(mask, output_dir / f"{stem}_mask.png", output_dir / f"{stem}_mask.npy")
        overlay = blend_mask(image, mask)
        overlay_path = save_rgb_image(overlay, output_dir / f"{stem}_segmentation_overlay.png")
        metadata.update(
            {
                "image_path": str(image_path),
                "mask_area_px": int((mask > 0).sum()),
                "image_shape": [int(height), int(width)],
                "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            }
        )
        return SegmentationResult(
            mask=(mask > 0).astype(np.uint8),
            mask_png_path=mask_png,
            mask_npy_path=mask_npy,
            overlay_path=overlay_path,
            metadata=metadata,
        )


def _resize_binary_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize a mask with nearest-neighbor sampling without importing MedicalSAM3."""

    try:
        from PIL import Image
    except ImportError as exc:
        raise MissingDependencyError("Pillow is required to resize masks.") from exc
    mask_array = (np.squeeze(mask) > 0).astype(np.uint8) * 255
    target_height, target_width = target_shape
    resized = Image.fromarray(mask_array).resize((target_width, target_height), Image.NEAREST)
    return (np.asarray(resized) > 127).astype(np.uint8)


def _box_around_point(point: tuple[float, float], radius: int, width: int, height: int) -> tuple[int, int, int, int]:
    x, y = point
    return (
        max(0, int(round(x - radius))),
        max(0, int(round(y - radius))),
        min(width - 1, int(round(x + radius))),
        min(height - 1, int(round(y + radius))),
    )


def _normalize_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (float(value) for value in box)
    left = max(0, min(width - 1, int(round(min(x0, x1)))))
    top = max(0, min(height - 1, int(round(min(y0, y1)))))
    right = max(0, min(width - 1, int(round(max(x0, x1)))))
    bottom = max(0, min(height - 1, int(round(max(y0, y1)))))
    if right <= left or bottom <= top:
        raise InvalidInputError("Box prompt must cover a non-empty image region.")
    return left, top, right, bottom


def _union_box_predictions(
    model: Any,
    inference_state: dict[str, Any],
    boxes: list[tuple[int, int, int, int]],
    image_size: tuple[int, int],
) -> np.ndarray | None:
    union_mask: np.ndarray | None = None
    for box in boxes:
        prediction = model.predict_box(inference_state, box, image_size)
        if prediction is None:
            continue
        prediction_array = np.asarray(prediction)
        if prediction_array.shape[:2] != image_size:
            prediction_array = _resize_binary_mask(prediction_array, image_size)
        prediction_array = prediction_array > 0
        if union_mask is None:
            union_mask = prediction_array
        else:
            union_mask = union_mask | prediction_array
    return union_mask.astype(np.uint8) if union_mask is not None else None
