"""Gradio demo for the surgical registration research prototype.

The demo is not launched during repository setup. Run it later with:

    python app.py
"""

from __future__ import annotations

import os
from pathlib import Path
import traceback
from typing import Any

import numpy as np

from pipeline.config import DeformableConfig, InitialPoseConfig
from pipeline.correspondences import filter_correspondences, load_correspondences
from pipeline.depth_vggtomega import VGGTOmegaRunner, load_first_point_map
from pipeline.export import (
    export_registered_meshes,
    export_result_bundle,
    manifest,
    save_deformable_result,
    save_initial_pose_result,
    save_metrics,
)
from pipeline.io import ensure_dir, extract_video_frames, file_path, load_mask, load_mesh, load_rgb_image
from pipeline.registration import deformable_refinement, estimate_initial_pose, load_intrinsics, reprojection_error_table
from pipeline.segmentation_medicalsam3 import MedicalSAM3Segmenter, SegmentationPrompt
from pipeline.visualization import (
    camera_frame_projection_overlay,
    draw_manual_correspondences,
    registration_overlay,
)

try:
    import gradio as gr
except ImportError:  # pragma: no cover - exercised only when launching the demo without gradio installed.
    gr = None


CLINICAL_WARNING = (
    "Research prototype only. This demo is not clinically validated, is not a "
    "medical device, and must not be used for diagnosis, treatment, surgical "
    "navigation, or patient care."
)

DEFAULT_VGGT_REPO = os.environ.get("VGGT_OMEGA_REPO", "external/vggt-omega")
DEFAULT_MEDICALSAM3_REPO = os.environ.get("MEDICALSAM3_REPO", "external/Medical-SAM3")
DEFAULT_OUTPUT_DIR = os.environ.get("REGISTRATION_OUTPUT_DIR", "outputs")
DEFAULT_VGGT_CHECKPOINT = os.environ.get("VGGT_OMEGA_CHECKPOINT", "")
DEFAULT_MEDICALSAM3_CHECKPOINT = os.environ.get("MEDICALSAM3_CHECKPOINT", "")


def _error_payload(exc: Exception) -> str:
    return f"Error: {exc}\n\n{traceback.format_exc(limit=2)}"


def _parse_point(text: str | None) -> tuple[float, float] | None:
    if not text:
        return None
    values = [float(part.strip()) for part in text.replace(";", ",").split(",") if part.strip()]
    if len(values) != 2:
        raise ValueError("Point prompt must be 'u,v'.")
    return values[0], values[1]


def _parse_box(text: str | None) -> tuple[float, float, float, float] | None:
    if not text:
        return None
    values = [float(part.strip()) for part in text.replace(";", ",").split(",") if part.strip()]
    if len(values) != 4:
        raise ValueError("Box prompt must be 'x_min,y_min,x_max,y_max'.")
    return values[0], values[1], values[2], values[3]


def _prepare_image_paths(
    image_file: Any,
    image_sequence: list[Any] | None,
    video_file: Any,
    output_dir: str | Path,
    sample_fps: float,
) -> list[Path]:
    paths: list[Path] = []
    if image_file:
        paths.append(Path(file_path(image_file)))
    for item in image_sequence or []:
        raw_path = file_path(item)
        if raw_path:
            paths.append(Path(raw_path))
    if video_file:
        video_path = Path(file_path(video_file))
        frame_dir = Path(output_dir) / "input_frames" / video_path.stem
        paths.extend(extract_video_frames(video_path, frame_dir, sample_fps=sample_fps))
    if not paths:
        raise ValueError("Upload at least one image or video.")
    return paths


def _required_file_path(file_obj: Any, label: str) -> str:
    path = file_path(file_obj)
    if not path:
        raise ValueError(f"{label} is required.")
    return path


def _load_segmentation_mask(segmentation_state: dict[str, Any] | None) -> np.ndarray | None:
    if not segmentation_state or not segmentation_state.get("mask_npy_path"):
        return None
    return load_mask(segmentation_state["mask_npy_path"])


def _load_point_map(vggt_state: dict[str, Any] | None) -> np.ndarray | None:
    if not vggt_state or not vggt_state.get("predictions_npz_path"):
        return None
    return load_first_point_map(vggt_state["predictions_npz_path"])


def run_segmentation_ui(
    image_file,
    image_sequence,
    video_file,
    video_sample_fps,
    prompt_type,
    text_prompt,
    point_prompt,
    box_prompt,
    mask_prompt_file,
    medsam_repo,
    medsam_checkpoint,
    device,
    output_dir,
):
    """Gradio callback for MedicalSAM3 segmentation."""

    try:
        output = ensure_dir(Path(output_dir) / "segmentation")
        image_paths = _prepare_image_paths(image_file, image_sequence, video_file, output_dir, video_sample_fps)
        first_image = image_paths[0]
        prompt_kind = str(prompt_type).lower().strip()
        prompt = SegmentationPrompt(
            prompt_type=prompt_kind,
            text=(text_prompt or None) if prompt_kind == "text" else None,
            point=_parse_point(point_prompt) if prompt_kind == "point" else None,
            box=_parse_box(box_prompt) if prompt_kind == "box" else None,
            mask_path=Path(file_path(mask_prompt_file)) if prompt_kind == "mask" and mask_prompt_file else None,
        )
        segmenter = MedicalSAM3Segmenter(
            repo_path=medsam_repo,
            checkpoint_path=medsam_checkpoint or None,
            device=device,
            output_dir=output,
        )
        result = segmenter.segment(first_image, prompt)
        state = result.to_dict()
        status = f"Segmentation complete. Mask area: {result.metadata['mask_area_px']} px."
        return str(result.overlay_path), str(result.mask_png_path), status, state
    except Exception as exc:
        return None, None, _error_payload(exc), None


def run_vggt_ui(
    image_file,
    image_sequence,
    video_file,
    video_sample_fps,
    vggt_repo,
    vggt_checkpoint,
    image_resolution,
    device,
    output_dir,
):
    """Gradio callback for VGGT-Omega inference."""

    try:
        image_paths = _prepare_image_paths(image_file, image_sequence, video_file, output_dir, video_sample_fps)
        runner = VGGTOmegaRunner(
            repo_path=vggt_repo,
            checkpoint_path=vggt_checkpoint or None,
            device=device,
            image_resolution=int(image_resolution),
            output_dir=Path(output_dir) / "vggt_omega",
            cache=True,
        )
        result = runner.run(image_paths)
        depth_preview = str(result.depth_visualization_paths[0]) if result.depth_visualization_paths else None
        scene_file = result.glb_path or result.point_cloud_path
        status = f"VGGT-Omega complete for {len(image_paths)} image(s)."
        if result.metadata.get("cache_hit"):
            status += " Used cached outputs."
        return depth_preview, str(scene_file) if scene_file else None, status, result.to_dict()
    except Exception as exc:
        return None, None, _error_payload(exc), None


def run_initial_pose_ui(
    image_file,
    image_sequence,
    video_file,
    video_sample_fps,
    mesh_file,
    correspondence_file,
    intrinsics_file,
    image_id_filter,
    use_segmentation_mask,
    output_dir,
    segmentation_state,
    vggt_state,
):
    """Gradio callback for initial rigid/similarity pose registration."""

    try:
        output = ensure_dir(Path(output_dir) / "registration")
        image_paths = _prepare_image_paths(image_file, image_sequence, video_file, output_dir, video_sample_fps)
        first_image = image_paths[0]
        image = load_rgb_image(first_image)
        height, width = image.shape[:2]
        mesh = load_mesh(_required_file_path(mesh_file, "3D mesh file"))
        correspondences = load_correspondences(_required_file_path(correspondence_file, "Manual correspondence file"))
        mask = _load_segmentation_mask(segmentation_state) if use_segmentation_mask else None
        selected = filter_correspondences(
            correspondences,
            image_id=image_id_filter or None,
            mask=mask,
            outside_mask_weight=0.25 if mask is not None else None,
        )
        if not selected:
            raise ValueError("No correspondences remain after image_id and segmentation filtering.")
        intrinsics = load_intrinsics(_required_file_path(intrinsics_file, "Camera intrinsics JSON")) if intrinsics_file else None
        point_map = _load_point_map(vggt_state)
        config = InitialPoseConfig()
        result = estimate_initial_pose(
            selected,
            intrinsics=intrinsics,
            image_size=(width, height),
            point_map=point_map,
            config=config,
            method="auto",
        )
        pose_json = save_initial_pose_result(result, output)
        table = reprojection_error_table(selected, result)
        metrics = result.metrics | {"reprojection_table": table}
        metrics_json = save_metrics(metrics, output, "initial_pose_metrics.json")
        manual_overlay = draw_manual_correspondences(first_image, selected, output / "manual_correspondences_overlay.png")
        overlay_path = None
        if result.intrinsics is not None:
            overlay_path = registration_overlay(
                first_image,
                mesh,
                result,
                output / "initial_pose_registration_overlay.png",
                correspondences=selected,
            )
        state = {
            "pose_json_path": str(pose_json),
            "metrics_json_path": str(metrics_json),
            "result": result.to_dict(),
            "selected_correspondence_count": len(selected),
            "manual_overlay_path": str(manual_overlay),
        }
        manifest(output, {"initial_pose": pose_json, "initial_pose_metrics": metrics_json, "initial_pose_overlay": overlay_path})
        status = f"Initial pose registration complete with {len(selected)} correspondence(s)."
        return str(overlay_path or manual_overlay), str(pose_json), metrics, status, state
    except Exception as exc:
        return None, None, None, _error_payload(exc), None


def _initial_result_from_state(initial_pose_state: dict[str, Any]):
    from pipeline.registration import CameraIntrinsics, InitialPoseResult

    payload = initial_pose_state["result"]
    intrinsics_payload = payload.get("intrinsics")
    intrinsics = None
    if intrinsics_payload:
        intrinsics = CameraIntrinsics(
            camera_matrix=np.asarray(intrinsics_payload["camera_matrix"], dtype=np.float64),
            distortion=np.asarray(intrinsics_payload.get("distortion_coefficients") or [], dtype=np.float64).reshape(-1, 1),
            source=intrinsics_payload.get("source", "state"),
        )
    return InitialPoseResult(
        method=payload["method"],
        transform_matrix=np.asarray(payload["transform_matrix_mesh_to_camera"], dtype=np.float64),
        intrinsics=intrinsics,
        inlier_indices=[int(idx) for idx in payload.get("inlier_indices", [])],
        reprojection_errors_px=np.asarray(payload.get("reprojection_errors_px", []), dtype=np.float64),
        metrics=payload.get("metrics", {}),
    )


def run_deformable_ui(
    image_file,
    image_sequence,
    video_file,
    video_sample_fps,
    mesh_file,
    correspondence_file,
    image_id_filter,
    use_segmentation_mask,
    regularization,
    max_control_points,
    output_dir,
    segmentation_state,
    vggt_state,
    initial_pose_state,
):
    """Gradio callback for non-rigid/deformable refinement."""

    try:
        if not initial_pose_state:
            raise ValueError("Run initial pose registration before deformable refinement.")
        output = ensure_dir(Path(output_dir) / "registration")
        image_paths = _prepare_image_paths(image_file, image_sequence, video_file, output_dir, video_sample_fps)
        first_image = image_paths[0]
        image = load_rgb_image(first_image)
        height, width = image.shape[:2]
        mesh = load_mesh(_required_file_path(mesh_file, "3D mesh file"))
        correspondences = load_correspondences(_required_file_path(correspondence_file, "Manual correspondence file"))
        selected = filter_correspondences(correspondences, image_id=image_id_filter or None)
        if not selected:
            raise ValueError("No correspondences match the selected image/frame id.")
        point_map = _load_point_map(vggt_state)
        mask = _load_segmentation_mask(segmentation_state) if use_segmentation_mask else None
        initial_result = _initial_result_from_state(initial_pose_state)
        config = DeformableConfig(regularization=float(regularization), max_control_points=int(max_control_points))
        result = deformable_refinement(
            mesh,
            selected,
            initial_pose=initial_result,
            point_map=point_map,
            image_size=(width, height),
            segmentation_mask=mask,
            config=config,
        )
        deform_json = save_deformable_result(result, output)
        mesh_artifacts = export_registered_meshes(mesh, initial_result, output, deformable_result=result)
        overlay_path = None
        if initial_result.intrinsics is not None:
            overlay_path = camera_frame_projection_overlay(
                first_image,
                result.final_vertices_camera_frame,
                initial_result.intrinsics,
                output / "final_deformable_registration_overlay.png",
            )
        metrics_json = save_metrics(result.metrics, output, "deformable_metrics.json")
        state = {
            "deformable_json_path": str(deform_json),
            "metrics_json_path": str(metrics_json),
            "result": result.to_dict(),
            "mesh_artifacts": {key: str(value) for key, value in mesh_artifacts.items()},
        }
        manifest(output, {"deformable": deform_json, "deformable_metrics": metrics_json, **mesh_artifacts})
        status = "Deformable refinement complete."
        if result.metrics.get("status", "") != "ok":
            status = f"Deformable refinement skipped: {result.metrics.get('note', result.metrics.get('status'))}"
        final_mesh = mesh_artifacts.get("final_deformable_registered_obj") or mesh_artifacts.get("initial_registered_obj")
        return str(overlay_path) if overlay_path else None, str(deform_json), str(final_mesh), result.metrics, status, state
    except Exception as exc:
        return None, None, None, None, _error_payload(exc), None


def export_results_ui(output_dir):
    """Gradio callback to zip current outputs."""

    try:
        archive = export_result_bundle(Path(output_dir) / "registration")
        return str(archive), f"Exported result bundle: {archive}"
    except Exception as exc:
        return None, _error_payload(exc)


def build_demo():
    """Build the Gradio Blocks application."""

    if gr is None:
        raise RuntimeError("gradio is not installed. Install it before launching the demo.")

    with gr.Blocks(title="Surgical 2D-to-3D Registration Prototype") as demo:
        gr.Markdown("# Surgical 2D-to-3D Registration Prototype")
        gr.Markdown(f"**Warning:** {CLINICAL_WARNING}")

        segmentation_state = gr.State(None)
        vggt_state = gr.State(None)
        initial_pose_state = gr.State(None)
        deformable_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Inputs")
                image_file = gr.File(label="Primary image", file_types=[".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"])
                image_sequence = gr.Files(label="Optional image sequence", file_types=[".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"])
                video_file = gr.File(label="Optional video", file_types=[".mp4", ".mov", ".avi", ".mkv"])
                video_sample_fps = gr.Number(label="Video sample FPS", value=1.0, precision=2)
                mesh_file = gr.File(label="3D mesh (.obj, .ply, .stl)", file_types=[".obj", ".ply", ".stl"])
                correspondence_file = gr.File(label="Manual correspondence CSV/JSON", file_types=[".csv", ".json"])
                intrinsics_file = gr.File(label="Optional camera intrinsics JSON", file_types=[".json"])
                image_id_filter = gr.Textbox(label="Optional image/frame id filter", placeholder="frame_0000")

                with gr.Accordion("Runtime paths and model settings", open=False):
                    medsam_repo = gr.Textbox(label="MedicalSAM3 repo path", value=DEFAULT_MEDICALSAM3_REPO)
                    medsam_checkpoint = gr.Textbox(label="MedicalSAM3 checkpoint path", value=DEFAULT_MEDICALSAM3_CHECKPOINT)
                    vggt_repo = gr.Textbox(label="VGGT-Omega repo path", value=DEFAULT_VGGT_REPO)
                    vggt_checkpoint = gr.Textbox(label="VGGT-Omega checkpoint path", value=DEFAULT_VGGT_CHECKPOINT)
                    device = gr.Dropdown(label="Device", choices=["cuda", "cpu"], value="cuda")
                    image_resolution = gr.Dropdown(label="VGGT image resolution", choices=[256, 512], value=512)
                    output_dir = gr.Textbox(label="Output directory", value=DEFAULT_OUTPUT_DIR)

            with gr.Column(scale=1):
                gr.Markdown("## MedicalSAM3 Segmentation")
                prompt_type = gr.Dropdown(label="Prompt type", choices=["text", "point", "box", "mask"], value="text")
                text_prompt = gr.Textbox(label="Text prompt", value="target anatomy or surgical region")
                point_prompt = gr.Textbox(label="Point prompt u,v", placeholder="320,240")
                box_prompt = gr.Textbox(label="Box prompt x_min,y_min,x_max,y_max", placeholder="100,80,420,360")
                mask_prompt_file = gr.File(label="Existing mask prompt", file_types=[".png", ".jpg", ".jpeg", ".npy"])
                run_segmentation = gr.Button("Run segmentation")
                segmentation_overlay_output = gr.Image(label="Segmentation overlay", type="filepath")
                segmentation_mask_file = gr.File(label="Mask PNG")
                segmentation_status = gr.Textbox(label="Segmentation status", lines=5)

                gr.Markdown("## VGGT-Omega Depth and Camera")
                run_vggt = gr.Button("Run VGGT-Omega")
                depth_output = gr.Image(label="Depth visualization", type="filepath")
                vggt_scene_file = gr.File(label="VGGT point cloud or GLB")
                vggt_status = gr.Textbox(label="VGGT-Omega status", lines=5)

            with gr.Column(scale=1):
                gr.Markdown("## Initial Pose Registration")
                use_segmentation_mask = gr.Checkbox(label="Use segmentation mask to filter/downweight correspondences", value=True)
                run_initial_pose = gr.Button("Run initial pose registration")
                initial_overlay = gr.Image(label="Initial pose registration overlay", type="filepath")
                initial_pose_json = gr.File(label="Initial pose transform JSON")
                initial_metrics = gr.JSON(label="Initial pose metrics")
                initial_status = gr.Textbox(label="Initial pose status", lines=5)

                gr.Markdown("## Deformable Refinement")
                regularization = gr.Number(label="RBF regularization", value=0.01, precision=4)
                max_control_points = gr.Number(label="Max deformation control points", value=128, precision=0)
                run_deformable = gr.Button("Run deformable refinement")
                deformable_overlay = gr.Image(label="Final deformable registration overlay", type="filepath")
                deformable_json = gr.File(label="Deformable result JSON")
                final_mesh_file = gr.File(label="Registered mesh")
                deformable_metrics = gr.JSON(label="Deformable metrics")
                deformable_status = gr.Textbox(label="Deformable status", lines=5)

                gr.Markdown("## Export")
                export_button = gr.Button("Export results")
                result_bundle = gr.File(label="Result bundle ZIP")
                export_status = gr.Textbox(label="Export status", lines=3)

        run_segmentation.click(
            run_segmentation_ui,
            inputs=[
                image_file,
                image_sequence,
                video_file,
                video_sample_fps,
                prompt_type,
                text_prompt,
                point_prompt,
                box_prompt,
                mask_prompt_file,
                medsam_repo,
                medsam_checkpoint,
                device,
                output_dir,
            ],
            outputs=[segmentation_overlay_output, segmentation_mask_file, segmentation_status, segmentation_state],
        )
        run_vggt.click(
            run_vggt_ui,
            inputs=[
                image_file,
                image_sequence,
                video_file,
                video_sample_fps,
                vggt_repo,
                vggt_checkpoint,
                image_resolution,
                device,
                output_dir,
            ],
            outputs=[depth_output, vggt_scene_file, vggt_status, vggt_state],
        )
        run_initial_pose.click(
            run_initial_pose_ui,
            inputs=[
                image_file,
                image_sequence,
                video_file,
                video_sample_fps,
                mesh_file,
                correspondence_file,
                intrinsics_file,
                image_id_filter,
                use_segmentation_mask,
                output_dir,
                segmentation_state,
                vggt_state,
            ],
            outputs=[initial_overlay, initial_pose_json, initial_metrics, initial_status, initial_pose_state],
        )
        run_deformable.click(
            run_deformable_ui,
            inputs=[
                image_file,
                image_sequence,
                video_file,
                video_sample_fps,
                mesh_file,
                correspondence_file,
                image_id_filter,
                use_segmentation_mask,
                regularization,
                max_control_points,
                output_dir,
                segmentation_state,
                vggt_state,
                initial_pose_state,
            ],
            outputs=[deformable_overlay, deformable_json, final_mesh_file, deformable_metrics, deformable_status, deformable_state],
        )
        export_button.click(export_results_ui, inputs=[output_dir], outputs=[result_bundle, export_status])

    return demo


if __name__ == "__main__":
    build_demo().launch()
