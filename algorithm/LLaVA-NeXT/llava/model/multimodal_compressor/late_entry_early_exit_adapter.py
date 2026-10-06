"""LLaVA-OneVision adapter for the shared late-entry/early-exit scheduler.

The scheduler is shared, while this module owns OV-specific call conventions.
In particular, OV uses stock Qwen2 one-dimensional position IDs and its
training collator may pass fixed-mask sample IDs.
"""

import torch

from _video3d_shared_multimodal_compressor.late_entry_early_exit.compressor import (
    LateEntryEarlyExitCompressor as _SharedLateEntryEarlyExitCompressor,
    LateEntryEarlyExitConfig,
)


class LateEntryEarlyExitCompressor(_SharedLateEntryEarlyExitCompressor):
    """OV adapter for stock Qwen2's one-dimensional rotary positions."""

    def _expects_3d_position_ids(self, backbone):
        if bool(getattr(backbone, "expects_3d_position_ids", False)):
            raise TypeError("The LLaVA-OV adapter cannot drive a three-axis Qwen2 backbone.")
        return False

    def _make_decode_position_ids(self, *, backbone, position, device):
        self._expects_3d_position_ids(backbone)
        return torch.full((1, 1), position, device=device, dtype=torch.long)

    def prepare_training_compression(self, *, sample_cache_ids=None, **kwargs):
        # Fixed-mask IDs are meaningful to group-wise analysis, not to this
        # deterministic layer-window scheduler.
        del sample_cache_ids
        return super().prepare_training_compression(**kwargs)


__all__ = ["LateEntryEarlyExitCompressor", "LateEntryEarlyExitConfig"]
