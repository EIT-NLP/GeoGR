#!/usr/bin/env python3
"""Precompute OV pooled ScanNet coordinates for compressed post-training."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LMMS_ROOT = PROJECT_ROOT / "lmms-eval"
if str(LMMS_ROOT) not in sys.path:
    sys.path.insert(0, str(LMMS_ROOT))

from lmms_eval.tasks._task_utils.scannet3d.config import get_data_paths  # noqa: E402
from lmms_eval.tasks._task_utils.scannet3d.coord_cache import coords_cache_path  # noqa: E402


def _parse_frame_shape(raw: str) -> tuple[int, int]:
    parts = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"--frame-shape must be formatted as H,W, got {raw!r}.")
    return parts[0], parts[1]


def _normalize_video_id(video_id: str) -> str:
    if video_id.startswith("shareVideoGPTV/"):
        video_id = video_id[len("shareVideoGPTV/") :]
    if video_id.startswith("scannet__"):
        return video_id.replace("__", "/", 1)
    if video_id.startswith("scannet/"):
        return video_id
    return video_id.replace("__", "/", 1)


def _cache_key_frame_path(frame_path: Path) -> str:
    if frame_path.is_symlink():
        return os.readlink(frame_path)
    return str(frame_path)


def _load_yaml_json_paths(input_yaml: Path) -> list[Path]:
    input_yaml = input_yaml.expanduser().resolve()
    with input_yaml.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    datasets = data.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError(f"Expected a datasets list in {input_yaml}")
    json_paths = []
    for item in datasets:
        path = Path(os.path.expandvars(item["json_path"])).expanduser()
        if not path.is_absolute():
            path = input_yaml.parent / path
        json_paths.append(path.resolve())
    return json_paths


def _collect_unique_videos(input_yaml: Path, frame_root: Path) -> list[tuple[str, Path]]:
    seen: set[str] = set()
    videos: list[tuple[str, Path]] = []
    for json_path in _load_yaml_json_paths(input_yaml):
        with json_path.open("r", encoding="utf-8") as handle:
            samples = json.load(handle)
        for sample in samples:
            raw_video = sample.get("video")
            if not isinstance(raw_video, str) or raw_video in seen:
                continue
            video_dir = frame_root / raw_video
            if not video_dir.exists():
                raise FileNotFoundError(f"Missing prepared frame directory for {raw_video}: {video_dir}")
            seen.add(raw_video)
            videos.append((raw_video, video_dir))
    return videos


def _load_embodiedscan_scenes(embodiedscan_root: str) -> dict[str, dict[str, Any]]:
    scenes: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        info_path = Path(embodiedscan_root) / f"embodiedscan_infos_{split}.pkl"
        if not info_path.exists():
            continue
        with info_path.open("rb") as handle:
            payload = pickle.load(handle)
        for item in payload.get("data_list", []):
            sample_idx = item.get("sample_idx")
            if isinstance(sample_idx, str):
                scenes[sample_idx] = item
    return scenes


def _resolve_scannet_path(relative_path: str, paths) -> str:
    if os.path.isabs(relative_path):
        return relative_path

    rel = relative_path.lstrip("/")
    candidates = []
    if rel.startswith("scannet/posed_images/"):
        candidates.append(os.path.join(paths.video_3d_llm_root, rel[len("scannet/") :]))
    candidates.extend(
        [
            os.path.join(paths.scannet_root, rel),
            os.path.join(paths.video_3d_llm_root, rel),
            os.path.join(paths.video_3d_llm_root, "data", "scannet", rel),
        ]
    )
    if rel.startswith("scannet/"):
        tail = rel[len("scannet/") :]
        candidates.extend(
            [
                os.path.join(paths.scannet_root, tail),
                os.path.join(paths.video_3d_llm_root, tail),
                os.path.join(paths.video_3d_llm_root, "data", "scannet", tail),
            ]
        )
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def _build_frame_info_cache(scene: dict[str, Any], paths) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for image_info in scene.get("images", []):
        img_path = image_info.get("img_path")
        if not img_path:
            continue
        resolved = os.path.abspath(_resolve_scannet_path(img_path, paths))
        cache[resolved] = image_info
        cache[os.path.normpath(img_path)] = image_info
    return cache


def _resolve_frame_info(cache: dict[str, dict[str, Any]], frame_file: str) -> dict[str, Any] | None:
    abs_frame = os.path.abspath(frame_file)
    if abs_frame in cache:
        return cache[abs_frame]
    normalized = os.path.normpath(frame_file)
    if normalized in cache:
        return cache[normalized]
    for image_info in cache.values():
        img_path = image_info.get("img_path", "")
        if img_path and normalized.endswith(os.path.normpath(img_path)):
            return image_info
    return None


def _unproject(intrinsics: torch.Tensor, poses: torch.Tensor, depths: torch.Tensor, depth_scale: float) -> torch.Tensor:
    num_frames, height, width = depths.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    xx = xx.reshape(1, -1).repeat(num_frames, 1)
    yy = yy.reshape(1, -1).repeat(num_frames, 1)
    z = depths.reshape(num_frames, -1)
    if z.numel() > 0 and float(z.max().item()) > 100.0:
        z = z / float(depth_scale)

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


def _resize_world_coords_like_ov_pad(world_coords: torch.Tensor, crop_size: int) -> torch.Tensor:
    _, height, width, _ = world_coords.shape
    coords = world_coords.permute(0, 3, 1, 2).contiguous()
    if width > height:
        pad_top = (width - height) // 2
        pad_bottom = width - height - pad_top
        coords = F.pad(coords, (0, 0, pad_top, pad_bottom), mode="replicate")
    elif height > width:
        pad_left = (height - width) // 2
        pad_right = height - width - pad_left
        coords = F.pad(coords, (pad_left, pad_right, 0, 0), mode="replicate")
    coords = F.interpolate(coords, size=(crop_size, crop_size), mode="nearest")
    return coords.permute(0, 2, 3, 1).contiguous()


def _pool_world_coords(world_coords: torch.Tensor, frame_shape: tuple[int, int]) -> torch.Tensor:
    coords = torch.nan_to_num(world_coords.to(dtype=torch.float32))
    coords = coords.permute(0, 3, 1, 2).contiguous()
    coords = F.adaptive_avg_pool2d(coords, output_size=frame_shape)
    return coords.permute(0, 2, 3, 1).contiguous()


def _compute_pooled_coords(
    *,
    video_id: str,
    frame_files: list[str],
    scenes: dict[str, dict[str, Any]],
    paths,
    crop_size: int,
    frame_shape: tuple[int, int],
    depth_scale: float,
) -> torch.Tensor:
    if video_id not in scenes:
        raise KeyError(f"Scene metadata not found for {video_id}")
    scene = scenes[video_id]
    frame_info_cache = _build_frame_info_cache(scene, paths)
    axis_align = torch.as_tensor(np.array(scene.get("axis_align_matrix", np.eye(4))), dtype=torch.float32)
    intrinsic = torch.as_tensor(np.array(scene.get("depth_cam2img", scene.get("cam2img"))), dtype=torch.float32)

    depths = []
    poses = []
    for frame_file in frame_files:
        frame_info = _resolve_frame_info(frame_info_cache, frame_file)
        if frame_info is None:
            raise KeyError(f"Missing frame metadata for {video_id}: {frame_file}")

        depth_path = frame_info.get("depth_path") or frame_file.replace(".jpg", ".png")
        depth_path = _resolve_scannet_path(depth_path, paths)
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Cannot find depth image for {video_id}: {depth_path}")

        with Image.open(depth_path) as depth_img:
            depths.append(torch.from_numpy(np.asarray(depth_img).astype(np.float32)))

        cam2global = torch.as_tensor(np.array(frame_info["cam2global"]), dtype=torch.float32)
        poses.append(axis_align @ cam2global)

    depths_t = torch.stack(depths, dim=0)
    poses_t = torch.stack(poses, dim=0)
    intrinsic_t = intrinsic.unsqueeze(0).repeat(depths_t.shape[0], 1, 1)
    world_coords = _unproject(intrinsic_t, poses_t, depths_t, depth_scale=depth_scale)
    world_coords = _resize_world_coords_like_ov_pad(world_coords, crop_size=crop_size)
    return _pool_world_coords(world_coords, frame_shape=frame_shape)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute OV pooled ScanNet coordinates for compressed training.")
    parser.add_argument("--input-yaml", required=True, type=Path)
    parser.add_argument("--frame-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--three-d-config", required=True, type=Path)
    parser.add_argument("--frame-shape", default="14,14")
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    frame_shape = _parse_frame_shape(args.frame_shape)
    paths = get_data_paths(str(args.three_d_config))
    scenes = _load_embodiedscan_scenes(paths.embodiedscan_root)
    videos = _collect_unique_videos(args.input_yaml, args.frame_root)
    if args.limit is not None:
        videos = videos[: args.limit]

    args.output_root.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    for raw_video, video_dir in tqdm(videos, desc="precompute train pooled coords"):
        video_id = _normalize_video_id(raw_video)
        symlink_frames = sorted(path for path in video_dir.iterdir() if path.is_file())
        frame_files = [_cache_key_frame_path(path) for path in symlink_frames]
        cache_path = coords_cache_path(
            args.output_root,
            video_id=video_id,
            frame_files=frame_files,
            crop_size=args.crop_size,
            frame_shape=frame_shape,
            depth_scale=args.depth_scale,
        )
        if cache_path is None:
            raise ValueError("coords cache path is disabled")
        if cache_path.exists() and not args.overwrite:
            skipped += 1
            continue

        coords = _compute_pooled_coords(
            video_id=video_id,
            frame_files=frame_files,
            scenes=scenes,
            paths=paths,
            crop_size=args.crop_size,
            frame_shape=frame_shape,
            depth_scale=args.depth_scale,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": 1,
                "video_id": video_id,
                "frame_files": frame_files,
                "frame_shape": list(frame_shape),
                "crop_size": int(args.crop_size),
                "depth_scale": float(args.depth_scale),
                "ov_pad_avg14": coords.cpu().contiguous(),
                "world_coords": coords.cpu().contiguous(),
            },
            cache_path,
        )
        written += 1

    print(
        f"[precompute] videos={len(videos)} written={written} skipped={skipped} "
        f"output_root={args.output_root}"
    )


if __name__ == "__main__":
    main()
