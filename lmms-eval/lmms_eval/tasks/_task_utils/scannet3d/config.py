from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


_CONFIG_ENV = "LMMS_EVAL_SCANNET3D_CONFIG"
_DEFAULT_CONFIG = Path(__file__).resolve().parent / "data_paths.local.yaml"
_EXAMPLE_CONFIG = Path(__file__).resolve().parents[4] / "scannet3d_data_paths.example.yaml"


@dataclass(frozen=True)
class Scannet3DPaths:
    scanqa_root: str
    sqa3d_root: str
    scanrefer_root: str
    scan2cap_root: str
    multi3drefer_root: str
    scannet_root: str
    embodiedscan_root: str
    metadata_root: str
    video_3d_llm_root: str


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"3D data config at {path} must be a mapping.")
    return data


def get_data_paths(config_path: str | None = None) -> Scannet3DPaths:
    resolved = Path(config_path or os.getenv(_CONFIG_ENV) or _DEFAULT_CONFIG)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Missing ScanNet3D config file: {resolved}. "
            f"Create it from {_EXAMPLE_CONFIG}."
        )
    data = _load_yaml(resolved)
    required = {
        "scanqa_root",
        "sqa3d_root",
        "scanrefer_root",
        "scan2cap_root",
        "multi3drefer_root",
        "scannet_root",
        "embodiedscan_root",
        "metadata_root",
        "video_3d_llm_root",
    }
    missing = sorted(required.difference(data))
    if missing:
        raise KeyError(f"Missing required ScanNet3D config keys in {resolved}: {missing}")
    normalized = {}
    for key in required:
        value = os.path.expandvars(os.path.expanduser(str(data[key])))
        path = Path(value)
        if not path.is_absolute():
            path = resolved.parent / path
        normalized[key] = str(path.resolve())
    return Scannet3DPaths(**normalized)
