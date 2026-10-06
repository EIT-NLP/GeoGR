from __future__ import annotations

import argparse
import importlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm

from lmms_eval.tasks._task_utils.scannet3d.config import get_data_paths
from lmms_eval.tasks._task_utils.scannet3d.coord_cache import coords_cache_path
from lmms_eval.tasks._task_utils.scannet3d.data import get_asset_manager


TASK_BUILDERS = {
    "scanqa_val": ("build_scanqa_docs", "val"),
    "sqa3d_test": ("build_sqa3d_docs", "test"),
    "scanrefer_val": ("build_scanrefer_docs", "val"),
    "scan2cap_val": ("build_scan2cap_docs", "val"),
    "multi3drefer_val": ("build_multi3drefer_docs", "val"),
}


def _parse_tasks(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_shape(raw: str) -> Tuple[int, int]:
    parts = [int(part.strip()) for part in raw.split(",")]
    if len(parts) != 2:
        raise ValueError("--frame-shape must be formatted as H,W")
    return parts[0], parts[1]


def _build_docs(task_name: str, config_path: str, max_frames: int, sampling_strategy: str) -> list[dict]:
    if task_name not in TASK_BUILDERS:
        raise ValueError(f"Unsupported task for coord precompute: {task_name}")
    builder_name, split = TASK_BUILDERS[task_name]
    module = importlib.import_module("lmms_eval.tasks._task_utils.scannet3d.data")
    builder = getattr(module, builder_name)
    paths = get_data_paths(config_path)
    assets = get_asset_manager(config_path)
    return builder(paths, assets, split=split, max_frames=max_frames, strategy=sampling_strategy)


def _load_train_rows(path: Path) -> list[dict]:
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            rows = json.load(handle)
    elif path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    else:
        raise ValueError(f"Unsupported Video3D training dataset format: {path}")
    if not isinstance(rows, list):
        raise TypeError(f"Video3D training dataset must contain a JSON list: {path}")
    return rows


def _build_video3d_train_docs(
    data_yaml_path: str,
    assets,
    video_folder: str,
    max_frames: int,
    sampling_strategy: str,
) -> list[dict]:
    """Build exact VideoProcessor frame sets for the configured training data."""
    if sampling_strategy != "uniform":
        raise ValueError(
            "Video3D training cache currently supports only uniform frame sampling; "
            f"got {sampling_strategy!r}."
        )
    if max_frames <= 0:
        raise ValueError(f"Video3D training cache requires max_frames > 0, got {max_frames}")

    yaml_path = Path(data_yaml_path)
    with open(yaml_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    datasets = config.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"Video3D training YAML has no datasets: {yaml_path}")

    video_ids = set()
    for dataset in datasets:
        dataset_path = Path(str(dataset.get("json_path", "")))
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Video3D training dataset does not exist: {dataset_path}")
        for row in _load_train_rows(dataset_path):
            video_id = row.get("video")
            if not video_id:
                raise KeyError(f"Video3D training sample has no 'video' field: {dataset_path}")
            video_ids.add(str(video_id))

    # VideoProcessor keeps the configured video_folder spelling when constructing
    # frame paths. Do not resolve symlinks here: cache keys include absolute paths,
    # so resolving /3d-com/data would produce a different key from training.
    video_root = Path(os.path.abspath(os.path.expanduser(video_folder)))
    docs = []
    for video_id in sorted(video_ids):
        scene = assets.get_scene(video_id)
        all_frames = [
            os.path.abspath(video_root / str(image_info["img_path"]))
            for image_info in scene.get("images", [])
        ]
        if not all_frames:
            raise ValueError(f"No frames found in Video3D metadata for {video_id}")
        missing = next((path for path in all_frames if not os.path.isfile(path)), None)
        if missing is not None:
            raise FileNotFoundError(
                f"Video3D training frame is missing under --video-folder={video_root}: {missing}"
            )
        sampled_indices = np.linspace(0, len(all_frames) - 1, max_frames, dtype=int)
        docs.append(
            {
                "video_id": video_id,
                "frame_files": [all_frames[index] for index in sampled_indices],
            }
        )
    return docs


def _rewrite_docs_to_video_folder(docs: list[dict], assets, video_folder: str) -> list[dict]:
    """Use the same absolute frame spelling as VideoProcessor(video_folder=...)."""
    video_root = Path(os.path.abspath(os.path.expanduser(video_folder)))
    rewritten = []
    for doc in docs:
        scene = assets.get_scene(doc["video_id"])
        by_resolved_path = {
            os.path.abspath(assets._resolve_scannet_path(image_info["img_path"])): image_info["img_path"]
            for image_info in scene.get("images", [])
        }
        copied = dict(doc)
        frame_files = []
        for frame_file in doc.get("frame_files", []):
            image_path = by_resolved_path.get(os.path.abspath(frame_file))
            if image_path is None:
                raise KeyError(
                    f"Cannot map evaluation frame to metadata for {doc['video_id']}: {frame_file}"
                )
            frame_files.append(os.path.abspath(video_root / str(image_path)))
        copied["frame_files"] = frame_files
        rewritten.append(copied)
    return rewritten


def _unique_docs(docs_by_task: dict[str, list[dict]]) -> list[dict]:
    by_key = {}
    tasks_by_key = defaultdict(list)
    counts_by_key = defaultdict(int)
    for task_name, docs in docs_by_task.items():
        for doc in docs:
            frame_files = tuple(os.path.abspath(path) for path in doc.get("frame_files", []))
            key = (doc.get("video_id"), frame_files)
            if key not in by_key:
                copied = dict(doc)
                copied["frame_files"] = list(frame_files)
                by_key[key] = copied
            tasks_by_key[key].append(task_name)
            counts_by_key[key] += 1

    unique = []
    for key, doc in by_key.items():
        copied = dict(doc)
        copied["_coord_cache_tasks"] = sorted(set(tasks_by_key[key]))
        copied["_coord_cache_doc_count"] = counts_by_key[key]
        unique.append(copied)
    unique.sort(key=lambda item: (item.get("video_id", ""), item.get("frame_files", [""])[0] if item.get("frame_files") else ""))
    return unique


def _scene_frame_lookup(assets, video_id: str) -> dict[str, dict]:
    scene = assets.get_scene(video_id)
    lookup = {}
    for image_info in scene.get("images", []):
        resolved = os.path.abspath(assets._resolve_scannet_path(image_info["img_path"]))
        lookup[resolved] = image_info
        lookup[os.path.normpath(image_info["img_path"])] = image_info
    return lookup


def _resolve_frame_info(assets, lookup: dict[str, dict], frame_file: str) -> dict:
    abs_frame = os.path.abspath(frame_file)
    if abs_frame in lookup:
        return lookup[abs_frame]
    normalized = os.path.normpath(frame_file)
    if normalized in lookup:
        return lookup[normalized]
    for value in lookup.values():
        if normalized.endswith(os.path.normpath(value.get("img_path", ""))):
            return value
    raise KeyError(f"Missing frame metadata for {frame_file}")


def _unproject_ov(intrinsics: torch.Tensor, poses: torch.Tensor, depths: torch.Tensor, depth_scale: float) -> torch.Tensor:
    num_frames, height, width = depths.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    xx = xx.reshape(1, -1).repeat(num_frames, 1)
    yy = yy.reshape(1, -1).repeat(num_frames, 1)
    z = depths.reshape(num_frames, -1) / float(depth_scale)

    fx = intrinsics[:, 0, 0].unsqueeze(-1)
    fy = intrinsics[:, 1, 1].unsqueeze(-1)
    cx = intrinsics[:, 0, 2].unsqueeze(-1)
    cy = intrinsics[:, 1, 2].unsqueeze(-1)
    x = (xx - cx) * z / fx.clamp_min(1e-6)
    y = (yy - cy) * z / fy.clamp_min(1e-6)
    cam_xyz = torch.stack((x, y, z, torch.ones_like(z)), dim=-1)
    world = torch.bmm(poses, cam_xyz.transpose(1, 2)).transpose(1, 2)
    world = world[..., :3] / world[..., 3:].clamp_min(1e-6)
    return world.reshape(num_frames, height, width, 3)


def _unproject_video3d(intrinsics: torch.Tensor, poses: torch.Tensor, depths: torch.Tensor, depth_scale: float) -> torch.Tensor:
    num_frames, height, width = depths.shape
    y = torch.arange(0, height).to(depths.device)
    x = torch.arange(0, width).to(depths.device)
    y, x = torch.meshgrid(y, x)

    x = x.unsqueeze(0).repeat(num_frames, 1, 1).view(num_frames, height * width)
    y = y.unsqueeze(0).repeat(num_frames, 1, 1).view(num_frames, height * width)

    fx = intrinsics[:, 0, 0].unsqueeze(-1).repeat(1, height * width)
    fy = intrinsics[:, 1, 1].unsqueeze(-1).repeat(1, height * width)
    cx = intrinsics[:, 0, 2].unsqueeze(-1).repeat(1, height * width)
    cy = intrinsics[:, 1, 2].unsqueeze(-1).repeat(1, height * width)

    z = depths.view(num_frames, height * width) / float(depth_scale)
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy
    cam_coords = torch.stack([x, y, z, torch.ones_like(x)], -1)

    world_coords = (poses @ cam_coords.permute(0, 2, 1)).permute(0, 2, 1)
    world_coords = world_coords[..., :3] / world_coords[..., 3].unsqueeze(-1)
    return world_coords.view(num_frames, height, width, 3)


def _load_raw_world_coords(assets, paths, doc: dict, depth_scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    scene = assets.get_scene(doc["video_id"])
    lookup = _scene_frame_lookup(assets, doc["video_id"])
    axis_align_ov = torch.as_tensor(np.array(scene.get("axis_align_matrix", np.eye(4))), dtype=torch.float32)
    intrinsic_ov = torch.as_tensor(np.array(scene.get("depth_cam2img", scene.get("cam2img"))), dtype=torch.float32)
    axis_align_video3d = torch.from_numpy(np.array(scene.get("axis_align_matrix", np.eye(4))))
    intrinsic_video3d = torch.from_numpy(np.array(scene.get("depth_cam2img", scene.get("cam2img"))))

    depths_ov = []
    depths_video3d = []
    poses_ov = []
    poses_video3d = []
    source_shape = None
    for frame_file in doc["frame_files"]:
        frame_info = _resolve_frame_info(assets, lookup, frame_file)
        depth_path = frame_info.get("depth_path") or frame_file.replace(".jpg", ".png")
        if not os.path.isabs(depth_path):
            depth_path = assets._resolve_scannet_path(depth_path)
        if not os.path.exists(depth_path):
            depth_path = os.path.join(paths.scannet_root, os.path.normpath(depth_path).lstrip("/"))
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Cannot find depth image: {depth_path}")

        with Image.open(depth_path) as depth_img:
            depth_np = np.asarray(depth_img)
            current_shape = (int(depth_np.shape[0]), int(depth_np.shape[1]))
            if source_shape is None:
                source_shape = current_shape
            elif source_shape != current_shape:
                raise ValueError(f"Video3D cache expects a fixed depth shape per frame set, got {source_shape} and {current_shape}")
            depths_ov.append(torch.from_numpy(depth_np.astype(np.float32)))
            depths_video3d.append(torch.from_numpy(depth_np.astype(np.int32)))

        pose_path = frame_file.replace(".jpg", ".txt")
        if os.path.exists(pose_path):
            cam2global = np.loadtxt(pose_path)
        else:
            cam2global = np.array(frame_info["cam2global"])
        poses_ov.append(axis_align_ov @ torch.as_tensor(cam2global, dtype=torch.float32))
        poses_video3d.append(axis_align_video3d @ torch.from_numpy(np.array(cam2global)))

    depths_ov_t = torch.stack(depths_ov, dim=0)
    poses_ov_t = torch.stack(poses_ov, dim=0)
    intrinsic_ov_t = intrinsic_ov.unsqueeze(0).repeat(depths_ov_t.shape[0], 1, 1)
    ov_world_coords = _unproject_ov(intrinsic_ov_t, poses_ov_t, depths_ov_t, depth_scale=depth_scale)

    depths_video3d_t = torch.stack(depths_video3d, dim=0)
    poses_video3d_t = torch.stack(poses_video3d, dim=0)
    intrinsic_video3d_t = intrinsic_video3d.unsqueeze(0).repeat(depths_video3d_t.shape[0], 1, 1)
    video3d_world_coords = _unproject_video3d(
        intrinsic_video3d_t.float(),
        poses_video3d_t.float(),
        depths_video3d_t.float(),
        depth_scale=depth_scale,
    )

    if source_shape is None:
        raise ValueError("Cannot infer source depth shape from an empty frame list")
    return ov_world_coords, video3d_world_coords, depths_ov_t > 0, source_shape


def _masked_adaptive_avg_pool2d(
    coords: torch.Tensor,
    valid_mask: torch.Tensor,
    output_size: Tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    weighted_coords = torch.nan_to_num(coords.to(dtype=torch.float32)) * valid_mask
    pooled_sum = F.adaptive_avg_pool2d(weighted_coords, output_size=output_size)
    pooled_count = F.adaptive_avg_pool2d(valid_mask.to(dtype=torch.float32), output_size=output_size)
    pooled = pooled_sum / pooled_count.clamp_min(1e-6)
    fallback = F.adaptive_avg_pool2d(torch.nan_to_num(coords.to(dtype=torch.float32)), output_size=output_size)
    pooled = torch.where((pooled_count > 0).expand_as(pooled), pooled, fallback)
    return pooled, pooled_count[:, 0]


def _ov_pad_avg14(
    world_coords: torch.Tensor,
    crop_size: int,
    frame_shape: Tuple[int, int],
    depth_valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    _, height, width, _ = world_coords.shape
    coords = world_coords.permute(0, 3, 1, 2).contiguous()
    valid_mask = None if depth_valid_mask is None else depth_valid_mask[:, None, :, :].to(dtype=torch.float32)
    if width > height:
        pad_top = (width - height) // 2
        pad_bottom = width - height - pad_top
        coords = F.pad(coords, (0, 0, pad_top, pad_bottom), mode="replicate")
        if valid_mask is not None:
            valid_mask = F.pad(valid_mask, (0, 0, pad_top, pad_bottom), mode="replicate")
    elif height > width:
        pad_left = (height - width) // 2
        pad_right = height - width - pad_left
        coords = F.pad(coords, (pad_left, pad_right, 0, 0), mode="replicate")
        if valid_mask is not None:
            valid_mask = F.pad(valid_mask, (pad_left, pad_right, 0, 0), mode="replicate")
    coords = F.interpolate(coords, size=(crop_size, crop_size), mode="nearest")
    if valid_mask is None:
        coords = F.adaptive_avg_pool2d(torch.nan_to_num(coords.to(dtype=torch.float32)), output_size=frame_shape)
        stats = {"zero_valid_cells": 0, "min_valid_ratio": None, "mean_valid_ratio": None}
    else:
        valid_mask = F.interpolate(valid_mask, size=(crop_size, crop_size), mode="nearest")
        coords, pooled_count = _masked_adaptive_avg_pool2d(coords, valid_mask, frame_shape)
        stats = {
            "zero_valid_cells": int((pooled_count <= 0).sum().item()),
            "min_valid_ratio": float(pooled_count.min().item()),
            "mean_valid_ratio": float(pooled_count.mean().item()),
        }
    return coords.permute(0, 2, 3, 1).contiguous(), stats


def _video3d_center_crop_384(world_coords: torch.Tensor, crop_size: int) -> torch.Tensor:
    _, height, width, _ = world_coords.shape
    new_height = crop_size
    new_width = int(width * (crop_size / height))
    left = (new_width - crop_size) // 2
    top = (new_height - crop_size) // 2
    resized = []
    for coords in world_coords:
        resized_coords = cv2.resize(coords.numpy(), (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        resized.append(resized_coords[top : top + crop_size, left : left + crop_size, :])
    return torch.from_numpy(np.stack(resized)).to(dtype=torch.float32).contiguous()


def _video3d_center_crop_rgb_uint8(frame_files: list[str], source_shape: tuple[int, int], crop_size: int) -> torch.Tensor:
    source_height, source_width = source_shape
    new_height = crop_size
    new_width = int(source_width * (crop_size / source_height))
    left = (new_width - crop_size) // 2
    top = (new_height - crop_size) // 2
    cropped = []
    for frame_file in frame_files:
        with Image.open(frame_file) as img:
            frame = img.convert("RGB")
            frame = frame.resize((new_width, new_height))
            frame = frame.crop((left, top, left + crop_size, top + crop_size))
            cropped.append(np.asarray(frame, dtype=np.uint8))
    return torch.from_numpy(np.stack(cropped)).to(dtype=torch.uint8).contiguous()


def _video3d_center_crop_mask_384(depth_valid_mask: torch.Tensor, crop_size: int) -> torch.Tensor:
    _, height, width = depth_valid_mask.shape
    new_height = crop_size
    new_width = int(width * (crop_size / height))
    left = (new_width - crop_size) // 2
    top = (new_height - crop_size) // 2
    cropped = []
    for mask in depth_valid_mask:
        resized_mask = cv2.resize(mask.numpy().astype(np.float32), (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        cropped.append(resized_mask[top : top + crop_size, left : left + crop_size] > 0.5)
    return torch.from_numpy(np.stack(cropped)).to(dtype=torch.bool).contiguous()


def _video3d_avg14(
    video3d_coords: torch.Tensor,
    frame_shape: Tuple[int, int],
    depth_valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    crop_size = int(video3d_coords.shape[1])
    patch_h = crop_size // frame_shape[0]
    patch_w = crop_size // frame_shape[1]
    usable_h = patch_h * frame_shape[0]
    usable_w = patch_w * frame_shape[1]
    coords = video3d_coords[:, :usable_h, :usable_w, :].permute(0, 3, 1, 2).contiguous()
    if depth_valid_mask is None:
        coords = F.avg_pool2d(coords, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))
        stats = {"zero_valid_cells": 0, "min_valid_ratio": None, "mean_valid_ratio": None}
    else:
        valid_mask = depth_valid_mask[:, None, :usable_h, :usable_w].to(dtype=torch.float32)
        coords, pooled_count = _masked_adaptive_avg_pool2d(coords, valid_mask, frame_shape)
        stats = {
            "zero_valid_cells": int((pooled_count <= 0).sum().item()),
            "min_valid_ratio": float(pooled_count.min().item()),
            "mean_valid_ratio": float(pooled_count.mean().item()),
        }
    return coords.permute(0, 2, 3, 1).contiguous(), stats


def _validate_payload(payload: dict, frame_count: int, crop_size: int, frame_shape: Tuple[int, int]) -> None:
    expected_ov = (frame_count, frame_shape[0], frame_shape[1], 3)
    expected_video3d = (frame_count, crop_size, crop_size, 3)
    expected_video3d_avg = (frame_count, frame_shape[0], frame_shape[1], 3)
    checks = {
        "ov_pad_avg14": expected_ov,
        "video3d_center_crop_384": expected_video3d,
        "video3d_center_crop_rgb_uint8": expected_video3d,
        "video3d_center_crop_avg14": expected_video3d_avg,
    }
    for field_name, expected in checks.items():
        value = payload.get(field_name)
        if not torch.is_tensor(value):
            raise TypeError(f"{field_name} must be a tensor")
        if tuple(value.shape) != expected:
            raise ValueError(f"{field_name} shape mismatch: {tuple(value.shape)} vs {expected}")
        if field_name == "video3d_center_crop_rgb_uint8" and value.dtype != torch.uint8:
            raise TypeError(f"{field_name} must be torch.uint8, got {value.dtype}")
    source_shape = payload.get("video3d_source_shape")
    if not isinstance(source_shape, (list, tuple)) or len(source_shape) != 2:
        raise ValueError("video3d_source_shape must be [height, width]")
    if int(source_shape[0]) <= 0 or int(source_shape[1]) <= 0:
        raise ValueError(f"video3d_source_shape must be positive, got {source_shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute shared ScanNet3D coordinate cache for OV and Video3D voxel methods.")
    parser.add_argument("--tasks", default="scan2cap_val,scanqa_val,sqa3d_test,multi3drefer_val,scanrefer_val")
    parser.add_argument(
        "--output-root",
        default=str(Path(__file__).resolve().parents[1] / "cache"),
    )
    parser.add_argument("--three-d-config", default="lmms-eval/lmms_eval/tasks/_task_utils/scannet3d/data_paths.local.yaml")
    parser.add_argument("--max-frames-num", type=int, default=32)
    parser.add_argument("--sampling-strategy", default="uniform")
    parser.add_argument("--frame-shape", default="14,14")
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--limit-unique", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--upgrade-incompatible",
        action="store_true",
        help="Rewrite cache files that exist but do not satisfy the current V2 payload schema.",
    )
    parser.add_argument(
        "--video3d-train-data-yaml",
        default=None,
        help="Video3D training YAML. Its unique scenes are added using exact VideoProcessor frame sampling.",
    )
    parser.add_argument(
        "--video-folder",
        default=None,
        help="VideoProcessor video_folder used with --video3d-train-data-yaml, usually <project>/data.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Split sorted unique frame sets into this many disjoint shards for parallel precompute.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index used with --shard-count.",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Optional manifest output path. Sharded runs default to per-shard manifests.",
    )
    parser.add_argument("--masked-depth-pool", action="store_true")
    args = parser.parse_args()

    frame_shape = _parse_shape(args.frame_shape)
    if args.video3d_train_data_yaml and not args.video_folder:
        parser.error("--video-folder is required with --video3d-train-data-yaml")
    if args.shard_count <= 0:
        parser.error("--shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, --shard-count)")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    os.environ["LMMS_EVAL_SCANNET3D_CONFIG"] = args.three_d_config

    paths = get_data_paths(args.three_d_config)
    assets = get_asset_manager(args.three_d_config)
    docs_by_task = {}
    for task_name in _parse_tasks(args.tasks):
        docs_by_task[task_name] = _build_docs(task_name, args.three_d_config, args.max_frames_num, args.sampling_strategy)
    if args.video_folder:
        for task_name in list(docs_by_task):
            docs_by_task[task_name] = _rewrite_docs_to_video_folder(
                docs_by_task[task_name],
                assets,
                args.video_folder,
            )
    if args.video3d_train_data_yaml:
        docs_by_task["video3d_train"] = _build_video3d_train_docs(
            args.video3d_train_data_yaml,
            assets,
            args.video_folder,
            args.max_frames_num,
            args.sampling_strategy,
        )

    unique_docs = _unique_docs(docs_by_task)
    if args.limit_unique is not None:
        unique_docs = unique_docs[: args.limit_unique]
    total_unique_docs = len(unique_docs)
    unique_docs = unique_docs[args.shard_index :: args.shard_count]

    manifest = {
        "version": 2,
        "output_root": str(output_root.resolve()),
        "tasks": {task_name: len(docs) for task_name, docs in docs_by_task.items()},
        "total_docs": sum(len(docs) for docs in docs_by_task.values()),
        "unique_frame_sets": len(unique_docs),
        "total_unique_frame_sets": total_unique_docs,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "frame_shape": list(frame_shape),
        "crop_size": int(args.crop_size),
        "depth_scale": float(args.depth_scale),
        "sampling_strategy": args.sampling_strategy,
        "max_frames_num": int(args.max_frames_num),
        "pooling_mode": "masked_depth" if args.masked_depth_pool else "plain_average",
        "entries": [],
    }

    written = 0
    skipped = 0
    failed = []
    for doc in tqdm(unique_docs, desc="precompute coords"):
        cache_path = coords_cache_path(
            output_root,
            video_id=doc["video_id"],
            frame_files=doc["frame_files"],
            crop_size=args.crop_size,
            frame_shape=frame_shape,
            depth_scale=args.depth_scale,
        )
        if cache_path is None:
            raise ValueError("Cache path is disabled")

        entry = {
            "video_id": doc["video_id"],
            "frame_files": doc["frame_files"],
            "tasks": doc["_coord_cache_tasks"],
            "doc_count": int(doc["_coord_cache_doc_count"]),
            "cache_path": str(cache_path),
        }
        try:
            should_write = args.overwrite or not cache_path.exists()
            if cache_path.exists() and not args.overwrite:
                try:
                    payload = torch.load(cache_path, map_location="cpu")
                    if not isinstance(payload, dict):
                        raise TypeError(f"Existing cache is not a dict: {cache_path}")
                    _validate_payload(payload, len(doc["frame_files"]), args.crop_size, frame_shape)
                    skipped += 1
                except (TypeError, ValueError, KeyError):
                    if not args.upgrade_incompatible:
                        raise
                    should_write = True

            if should_write:
                ov_world_coords, video3d_world_coords, depth_valid_mask, video3d_source_shape = _load_raw_world_coords(
                    assets,
                    paths,
                    doc,
                    depth_scale=args.depth_scale,
                )
                video3d_coords = _video3d_center_crop_384(video3d_world_coords, crop_size=args.crop_size)
                video3d_valid_mask = None
                ov_valid_mask = None
                if args.masked_depth_pool:
                    ov_valid_mask = depth_valid_mask
                    video3d_valid_mask = _video3d_center_crop_mask_384(depth_valid_mask, crop_size=args.crop_size)
                ov_avg14, ov_pool_stats = _ov_pad_avg14(
                    ov_world_coords,
                    crop_size=args.crop_size,
                    frame_shape=frame_shape,
                    depth_valid_mask=ov_valid_mask,
                )
                video3d_avg14, video3d_pool_stats = _video3d_avg14(
                    video3d_coords,
                    frame_shape=frame_shape,
                    depth_valid_mask=video3d_valid_mask,
                )
                payload = {
                    "version": 2,
                    "video_id": doc["video_id"],
                    "frame_files": list(doc["frame_files"]),
                    "frame_shape": list(frame_shape),
                    "crop_size": int(args.crop_size),
                    "depth_scale": float(args.depth_scale),
                    "pooling_mode": "masked_depth" if args.masked_depth_pool else "plain_average",
                    "pool_stats": {
                        "ov_pad_avg14": ov_pool_stats,
                        "video3d_center_crop_avg14": video3d_pool_stats,
                    },
                    "tasks": doc["_coord_cache_tasks"],
                    "doc_count": int(doc["_coord_cache_doc_count"]),
                    "video3d_source_shape": [int(video3d_source_shape[0]), int(video3d_source_shape[1])],
                    "ov_pad_avg14": ov_avg14,
                    "video3d_center_crop_384": video3d_coords,
                    "video3d_center_crop_rgb_uint8": _video3d_center_crop_rgb_uint8(
                        list(doc["frame_files"]),
                        video3d_source_shape,
                        crop_size=args.crop_size,
                    ),
                    "video3d_center_crop_avg14": video3d_avg14,
                }
                _validate_payload(payload, len(doc["frame_files"]), args.crop_size, frame_shape)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, cache_path)
                written += 1
            manifest["entries"].append(entry)
        except Exception as exc:
            entry["error"] = repr(exc)
            failed.append(entry)
            manifest["entries"].append(entry)

    manifest["written"] = written
    manifest["skipped"] = skipped
    manifest["failed"] = failed
    if args.manifest_path:
        manifest_path = Path(args.manifest_path)
    elif args.shard_count > 1:
        manifest_path = output_root / f"scannet3d_coord_cache_manifest.shard{args.shard_index:02d}-of-{args.shard_count:02d}.json"
    else:
        manifest_path = output_root / "scannet3d_coord_cache_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_root": str(output_root.resolve()),
                "manifest": str(manifest_path),
                "total_docs": manifest["total_docs"],
                "unique_frame_sets": manifest["unique_frame_sets"],
                "written": written,
                "skipped": skipped,
                "failed": len(failed),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
