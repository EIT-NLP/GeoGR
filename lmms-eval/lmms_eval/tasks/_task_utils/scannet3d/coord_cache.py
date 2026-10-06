from pathlib import Path
from typing import Iterable, Optional

import torch

from lmms_eval.scannet3d_coord_cache import (
    DEFAULT_COORDS_CACHE_ENV,
    DEFAULT_CROP_SIZE,
    DEFAULT_DEPTH_SCALE,
    DEFAULT_FRAME_SHAPE,
    coords_cache_key,
    coords_cache_path,
    normalize_video_id,
)


def load_tensor_field(path: Path, field_names: Iterable[str]) -> Optional[torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if torch.is_tensor(payload):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(f"Cached coordinates must be a tensor or dict: {path}")
    for field_name in field_names:
        value = payload.get(field_name)
        if value is not None:
            if not torch.is_tensor(value):
                raise TypeError(f"Cached field {field_name!r} must be a tensor: {path}")
            return value
    return None
