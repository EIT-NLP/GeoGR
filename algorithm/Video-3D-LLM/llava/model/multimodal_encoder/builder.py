"""Vision-tower factory for the project-supported SigLIP encoder."""

import json
import os

from .siglip_encoder import SigLipVisionTower


def _is_siglip_vision_tower(vision_tower):
    if "siglip" in str(vision_tower).lower():
        return True
    config_path = os.path.join(str(vision_tower), "config.json")
    if not os.path.isfile(config_path):
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    model_type = str(config.get("model_type", "")).lower()
    vision_model_type = str(config.get("vision_config", {}).get("model_type", "")).lower()
    return "siglip" in model_type or "siglip" in vision_model_type


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None))
    if not vision_tower:
        raise ValueError("A SigLIP vision tower must be specified with mm_vision_tower.")

    siglip_model_path = os.environ.get("SIGLIP_MODEL_PATH")
    if siglip_model_path:
        vision_tower = siglip_model_path
        setattr(vision_tower_cfg, "mm_vision_tower", vision_tower)

    if not _is_siglip_vision_tower(vision_tower):
        raise ValueError(
            "This release supports only SigLIP vision towers; "
            f"received {vision_tower!r}. Set SIGLIP_MODEL_PATH to a local SigLIP checkpoint."
        )
    return SigLipVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)
