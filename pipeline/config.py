"""Configuration objects for the registration prototype."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


def _path_or_none(value: str | Path | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return Path(value)


def _float_or_none(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


@dataclass
class MedicalSAM3Config:
    """Runtime options for the MedicalSAM3 wrapper."""

    repo_path: Path = Path("external/Medical-SAM3")
    checkpoint_path: Path | None = None
    device: str = "cuda"
    confidence_threshold: float = 0.1


@dataclass
class VGGTOmegaConfig:
    """Runtime options for the VGGT-Omega wrapper."""

    repo_path: Path = Path("external/vggt-omega")
    checkpoint_path: Path | None = None
    device: str = "cuda"
    image_resolution: int = 512
    cache: bool = True


@dataclass
class InitialPoseConfig:
    """Options for sparse initial pose registration."""

    use_ransac: bool = True
    ransac_reprojection_error_px: float = 12.0
    ransac_iterations: int = 200
    estimate_intrinsics_if_missing: bool = True
    default_focal_length_px: float | None = None
    allow_similarity_from_vggt_points: bool = True


@dataclass
class DeformableConfig:
    """Options for regularized non-rigid refinement."""

    model: str = "rbf"
    regularization: float = 1e-2
    kernel_radius: float | None = None
    max_control_points: int = 128
    min_control_points: int = 3
    min_control_weight: float = 1e-6
    trim_outlier_controls: bool = True
    outlier_mad_multiplier: float = 4.0
    max_control_displacement: float | None = None
    segmentation_outside_weight: float = 0.25
    chunk_size: int = 50000


@dataclass
class OutputConfig:
    """Output location and export options."""

    output_dir: Path = Path("outputs")
    save_intermediates: bool = True
    export_registered_mesh: bool = True


@dataclass
class PipelineConfig:
    """Top-level configuration for the end-to-end prototype."""

    medsam3: MedicalSAM3Config = field(default_factory=MedicalSAM3Config)
    vggt: VGGTOmegaConfig = field(default_factory=VGGTOmegaConfig)
    initial_pose: InitialPoseConfig = field(default_factory=InitialPoseConfig)
    deformable: DeformableConfig = field(default_factory=DeformableConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineConfig":
        """Load configuration from a JSON file."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PipelineConfig":
        """Create a config object from a nested dictionary."""

        medsam = payload.get("medsam3", {})
        vggt = payload.get("vggt", {})
        pose = payload.get("initial_pose", {})
        deform = payload.get("deformable", {})
        output = payload.get("output", {})
        return cls(
            medsam3=MedicalSAM3Config(
                repo_path=Path(medsam.get("repo_path", "external/Medical-SAM3")),
                checkpoint_path=_path_or_none(medsam.get("checkpoint_path")),
                device=medsam.get("device", "cuda"),
                confidence_threshold=float(medsam.get("confidence_threshold", 0.1)),
            ),
            vggt=VGGTOmegaConfig(
                repo_path=Path(vggt.get("repo_path", "external/vggt-omega")),
                checkpoint_path=_path_or_none(vggt.get("checkpoint_path")),
                device=vggt.get("device", "cuda"),
                image_resolution=int(vggt.get("image_resolution", 512)),
                cache=bool(vggt.get("cache", True)),
            ),
            initial_pose=InitialPoseConfig(
                use_ransac=bool(pose.get("use_ransac", True)),
                ransac_reprojection_error_px=float(pose.get("ransac_reprojection_error_px", 12.0)),
                ransac_iterations=int(pose.get("ransac_iterations", 200)),
                estimate_intrinsics_if_missing=bool(pose.get("estimate_intrinsics_if_missing", True)),
                default_focal_length_px=_float_or_none(pose.get("default_focal_length_px")),
                allow_similarity_from_vggt_points=bool(pose.get("allow_similarity_from_vggt_points", True)),
            ),
            deformable=DeformableConfig(
                model=deform.get("model", "rbf"),
                regularization=float(deform.get("regularization", 1e-2)),
                kernel_radius=deform.get("kernel_radius"),
                max_control_points=int(deform.get("max_control_points", 128)),
                min_control_points=int(deform.get("min_control_points", 3)),
                min_control_weight=float(deform.get("min_control_weight", 1e-6)),
                trim_outlier_controls=bool(deform.get("trim_outlier_controls", True)),
                outlier_mad_multiplier=float(deform.get("outlier_mad_multiplier", 4.0)),
                max_control_displacement=_float_or_none(deform.get("max_control_displacement")),
                segmentation_outside_weight=float(deform.get("segmentation_outside_weight", 0.25)),
                chunk_size=int(deform.get("chunk_size", 50000)),
            ),
            output=OutputConfig(
                output_dir=Path(output.get("output_dir", "outputs")),
                save_intermediates=bool(output.get("save_intermediates", True)),
                export_registered_mesh=bool(output.get("export_registered_mesh", True)),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration dictionary."""

        def normalize(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {key: normalize(item) for key, item in value.items()}
            if isinstance(value, list):
                return [normalize(item) for item in value]
            return value

        return normalize(asdict(self))

    def save_json(self, path: str | Path) -> None:
        """Write this configuration to disk."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def resolve_paths(self, base_dir: str | Path) -> "PipelineConfig":
        """Return a copy with relative paths resolved against ``base_dir``."""

        base = Path(base_dir)
        payload = self.to_dict()
        for section, keys in {
            "medsam3": ("repo_path", "checkpoint_path"),
            "vggt": ("repo_path", "checkpoint_path"),
            "output": ("output_dir",),
        }.items():
            for key in keys:
                value = payload[section].get(key)
                if value and not Path(value).is_absolute():
                    payload[section][key] = str(base / value)
        return PipelineConfig.from_dict(payload)
