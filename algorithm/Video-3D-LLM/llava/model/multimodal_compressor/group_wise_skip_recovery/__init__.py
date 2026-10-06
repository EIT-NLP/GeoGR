"""GeoGR GroupRoute registration.

The implementation is kept beside the compressor registry.  It deliberately
imports the active backend's ``late_entry_early_exit_adapter`` through the
public ``llava`` package: the same source file is loaded by LLaVA-OneVision
and Video3D, while the adapter supplies their different position-id rules.
"""

from .compressor import GroupWiseSkipRecoveryCompressor, GroupWiseSkipRecoveryConfig

__all__ = ["GroupWiseSkipRecoveryCompressor", "GroupWiseSkipRecoveryConfig"]
