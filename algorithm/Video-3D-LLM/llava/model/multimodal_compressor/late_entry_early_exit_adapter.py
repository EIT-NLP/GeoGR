"""Video3D adapter for the shared late-entry/early-exit scheduler."""

import torch

from .late_entry_early_exit.compressor import (
    LateEntryEarlyExitCompressor as _SharedLateEntryEarlyExitCompressor,
    LateEntryEarlyExitConfig,
)


class LateEntryEarlyExitCompressor(_SharedLateEntryEarlyExitCompressor):
    """Video3D adapter for its three-axis Qwen2 rotary position interface."""

    def _expects_3d_position_ids(self, backbone):
        if not bool(getattr(backbone, "expects_3d_position_ids", False)):
            raise TypeError("The Video3D adapter requires a three-axis Qwen2 backbone.")
        return True

    def _make_decode_position_ids(self, *, backbone, position, device):
        self._expects_3d_position_ids(backbone)
        return torch.full((1, 1, 3), position, device=device, dtype=torch.long)

    def _next_decode_position(self, *, backbone, full_positions, state):
        self._expects_3d_position_ids(backbone)
        del full_positions
        # Video3D generation continues text by prompt length. Its visual mRoPE
        # xyz values are spatial coordinates and are not text sequence offsets.
        return int(state["full_prompt_length"])

    # Video3D does not use the OV fixed-mask sample-id side channel.
    def prepare_training_compression(self, **kwargs):
        kwargs.pop("sample_cache_ids", None)
        return super().prepare_training_compression(**kwargs)


__all__ = ["LateEntryEarlyExitCompressor", "LateEntryEarlyExitConfig"]
