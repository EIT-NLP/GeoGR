import json
from typing import Any, Dict, Optional, Tuple

from .base import BaseCompressor, CompressorConfig, IdentityCompressor
from .registry import get_compressor, is_registered


def _parse_compressor_config(raw_config: Any, field_name: str) -> Dict[str, Any]:
    compressor_config = raw_config

    if isinstance(compressor_config, str):
        try:
            compressor_config = json.loads(compressor_config)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid {field_name} JSON: {compressor_config}") from None

    if compressor_config is None:
        return {}
    if not isinstance(compressor_config, dict):
        raise TypeError(
            f"{field_name} must be a dict or a JSON object string, "
            f"got {type(compressor_config).__name__}."
        )
    return compressor_config


def get_compressor_spec_from_llava_config(llava_config, location: str = "projector") -> Tuple[Optional[str], Dict[str, Any]]:
    if location not in {"projector", "llm"}:
        raise ValueError(f"Unsupported compressor location '{location}'.")

    if location == "projector":
        explicit_type = getattr(llava_config, "mm_projector_compressor_type", None)
        explicit_config = getattr(llava_config, "mm_projector_compressor_config", None)
        legacy_type = getattr(llava_config, "mm_compressor_type", None)
        legacy_config = getattr(llava_config, "mm_compressor_config", None)

        explicit_set = explicit_type not in (None, "")
        legacy_set = legacy_type not in (None, "")
        if explicit_set and legacy_set:
            raise ValueError(
                "Both mm_projector_compressor_type and legacy mm_compressor_type are set. "
                "Use only one projector-level compressor entrypoint."
            )
        compressor_type = explicit_type if explicit_set else legacy_type
        raw_config = explicit_config if explicit_set else legacy_config
        field_name = "mm_projector_compressor_config" if explicit_set else "mm_compressor_config"
    else:
        compressor_type = getattr(llava_config, "mm_llm_compressor_type", None)
        raw_config = getattr(llava_config, "mm_llm_compressor_config", None)
        field_name = "mm_llm_compressor_config"

    return compressor_type, _parse_compressor_config(raw_config, field_name)


def build_compressor(
    compressor_type: Optional[str],
    compressor_config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> BaseCompressor:
    if compressor_type in ("none", "identity", None, ""):
        position = kwargs.pop("position", "after_projector")
        return IdentityCompressor(CompressorConfig(position=position))

    if compressor_type == "group_wise_skip_recovery" and not is_registered(compressor_type):
        from . import group_wise_skip_recovery  # noqa: F401
    elif compressor_type in {"segpruner", "spatial_kcenter_merge"} and not is_registered(compressor_type):
        from . import spatial_selection  # noqa: F401

    if not is_registered(compressor_type):
        raise ValueError(f"Unknown compressor type: {compressor_type}")

    compressor_cls = get_compressor(compressor_type)
    config = compressor_config or {}
    config = config.copy()
    config.update(kwargs)
    return compressor_cls(config)


def build_compressor_from_llava_config(llava_config, location: str = "projector") -> BaseCompressor:
    compressor_type, compressor_config = get_compressor_spec_from_llava_config(llava_config, location=location)
    default_position = "after_projector" if location == "projector" else "llm"
    return build_compressor(compressor_type, compressor_config, position=default_position)
