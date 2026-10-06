from .base import (
    BaseCompressor,
    CompressorConfig,
    CompressorOutput,
    IdentityCompressor,
    LLMForwardContext,
    LLMForwardOutput,
)
from .builder import build_compressor, build_compressor_from_llava_config, get_compressor_spec_from_llava_config
from .late_entry_early_exit import LateEntryEarlyExitCompressor, LateEntryEarlyExitConfig
from .late_entry_early_exit_adapter import (
    LateEntryEarlyExitCompressor as Video3DLateEntryEarlyExitCompressor,
    LateEntryEarlyExitConfig as Video3DLateEntryEarlyExitConfig,
)
from .spatial_selection import (
    SegPrunerProjectorCompressor,
    SegPrunerProjectorConfig,
    SpatialKCenterMergeCompressor,
    SpatialKCenterMergeConfig,
)
from .vispruner import VisPrunerCompressor, VisPrunerConfig
from .visionzip import VisionZipCompressor, VisionZipConfig
from .voxel_dtc import VoxelDTCCompressor, VoxelDTCConfig, VoxelVTCCompressor, VoxelVTCConfig
from .voxel_geosem import (
    VoxelGeoSemAnchorMergeCompressor,
    VoxelGeoSemAnchorMergeConfig,
    VoxelGeoSemAnchorPruneCompressor,
    VoxelGeoSemAnchorPruneConfig,
)
from .voxel_vtc_post import (
    VoxelVTCAttnPruneCompressor,
    VoxelVTCAttnPruneConfig,
    VoxelVTCRandomCompressor,
    VoxelVTCRandomConfig,
    VoxelVTCToMeCompressor,
    VoxelVTCToMeConfig,
    VoxelVTCVisionZipCompressor,
    VoxelVTCVisionZipConfig,
    VoxelVTCVisPrunerCompressor,
    VoxelVTCVisPrunerConfig,
)
from .registry import COMPRESSOR_REGISTRY, get_compressor, is_registered, list_compressors, register_compressor

# The scheduler is shared with OV, but Video3D binds its own model adapter.
COMPRESSOR_REGISTRY["late_entry_early_exit"] = Video3DLateEntryEarlyExitCompressor
LateEntryEarlyExitCompressor = Video3DLateEntryEarlyExitCompressor
LateEntryEarlyExitConfig = Video3DLateEntryEarlyExitConfig

__all__ = [
    "BaseCompressor",
    "CompressorConfig",
    "CompressorOutput",
    "IdentityCompressor",
    "LLMForwardContext",
    "LLMForwardOutput",
    "build_compressor",
    "build_compressor_from_llava_config",
    "get_compressor_spec_from_llava_config",
    "register_compressor",
    "get_compressor",
    "list_compressors",
    "is_registered",
    "COMPRESSOR_REGISTRY",
    "LateEntryEarlyExitCompressor",
    "LateEntryEarlyExitConfig",
    "SegPrunerProjectorCompressor",
    "SegPrunerProjectorConfig",
    "SpatialKCenterMergeCompressor",
    "SpatialKCenterMergeConfig",
    "VisPrunerCompressor",
    "VisPrunerConfig",
    "VisionZipCompressor",
    "VisionZipConfig",
    "VoxelDTCCompressor",
    "VoxelDTCConfig",
    "VoxelVTCCompressor",
    "VoxelVTCConfig",
    "VoxelGeoSemAnchorMergeCompressor",
    "VoxelGeoSemAnchorMergeConfig",
    "VoxelGeoSemAnchorPruneCompressor",
    "VoxelGeoSemAnchorPruneConfig",
    "VoxelVTCRandomCompressor",
    "VoxelVTCRandomConfig",
    "VoxelVTCAttnPruneCompressor",
    "VoxelVTCAttnPruneConfig",
    "VoxelVTCVisPrunerCompressor",
    "VoxelVTCVisPrunerConfig",
    "VoxelVTCVisionZipCompressor",
    "VoxelVTCVisionZipConfig",
    "VoxelVTCToMeCompressor",
    "VoxelVTCToMeConfig",
]
