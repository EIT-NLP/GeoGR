from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Optional, Tuple


DEFAULT_COORDS_CACHE_ENV = "SCANNET3D_COORDS_CACHE_ROOT"
DEFAULT_FRAME_SHAPE = (14, 14)
DEFAULT_CROP_SIZE = 384
DEFAULT_DEPTH_SCALE = 1000.0


def normalize_video_id(video_id: str) -> str:
    return str(video_id or "unknown").replace("/", "__")


def coords_cache_key(
    *,
    video_id: str,
    frame_files: Iterable[str],
    frame_shape: Tuple[int, int] = DEFAULT_FRAME_SHAPE,
    crop_size: int = DEFAULT_CROP_SIZE,
    depth_scale: float = DEFAULT_DEPTH_SCALE,
) -> str:
    payload = {
        "version": 1,
        "video_id": video_id,
        "frame_files": [os.path.abspath(path) for path in frame_files],
        "crop_size": int(crop_size),
        "frame_shape": [int(frame_shape[0]), int(frame_shape[1])],
        "depth_scale": float(depth_scale),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def coords_cache_path(
    root: str | os.PathLike[str] | None,
    *,
    video_id: str,
    frame_files: Iterable[str],
    frame_shape: Tuple[int, int] = DEFAULT_FRAME_SHAPE,
    crop_size: int = DEFAULT_CROP_SIZE,
    depth_scale: float = DEFAULT_DEPTH_SCALE,
) -> Optional[Path]:
    if not root:
        return None
    key = coords_cache_key(
        video_id=video_id,
        frame_files=frame_files,
        frame_shape=frame_shape,
        crop_size=crop_size,
        depth_scale=depth_scale,
    )
    return Path(root) / normalize_video_id(video_id) / f"{key}.pt"
