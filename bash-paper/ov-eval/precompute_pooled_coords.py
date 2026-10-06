from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path

import torch
from tqdm import tqdm

from lmms_eval.tasks._task_utils.scannet3d.config import get_data_paths
from lmms_eval.tasks._task_utils.scannet3d.coord_cache import coords_cache_path
from lmms_eval.tasks._task_utils.scannet3d.data import get_asset_manager
from ov_lmms_plugin.models.llava_onevision_3d import LlavaOneVision3D


TASK_BUILDERS = {
    "scanqa_val": ("build_scanqa_docs", "val"),
    "sqa3d_test": ("build_sqa3d_docs", "test"),
    "scanrefer_val": ("build_scanrefer_docs", "val"),
    "scan2cap_val": ("build_scan2cap_docs", "val"),
    "multi3drefer_val": ("build_multi3drefer_docs", "val"),
}


def _build_docs(task_name: str, config_path: str, max_frames: int, sampling_strategy: str):
    if task_name not in TASK_BUILDERS:
        raise ValueError(f"Unsupported task for pooled coord precompute: {task_name}")
    builder_name, split = TASK_BUILDERS[task_name]
    module = importlib.import_module("lmms_eval.tasks._task_utils.scannet3d.data")
    builder = getattr(module, builder_name)
    paths = get_data_paths(config_path)
    assets = get_asset_manager(config_path)
    return builder(paths, assets, split=split, max_frames=max_frames, strategy=sampling_strategy)


def _parse_tasks(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute OV pooled 3D coordinates for voxel compression.")
    parser.add_argument("--tasks", default="scanqa_val")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--three-d-config", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--siglip-model-path", default=os.environ.get("SIGLIP_MODEL_PATH", ""))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-frames-num", type=int, default=32)
    parser.add_argument("--sampling-strategy", default="uniform")
    parser.add_argument("--frame-shape", default="14,14")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-name", default="llava_qwen")
    parser.add_argument("--conv-template", default="qwen_1_5")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--video-decode-backend", default="decord")
    args = parser.parse_args()

    frame_shape = tuple(int(part) for part in args.frame_shape.split(","))
    if len(frame_shape) != 2:
        raise ValueError("--frame-shape must be formatted as H,W")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    os.environ["LMMS_EVAL_SCANNET3D_CONFIG"] = args.three_d_config
    if args.siglip_model_path:
        os.environ["SIGLIP_MODEL_PATH"] = args.siglip_model_path

    model = LlavaOneVision3D(
        pretrained=args.pretrained,
        device=args.device,
        device_map=args.device,
        model_name=args.model_name,
        conv_template=args.conv_template,
        attn_implementation=args.attn_implementation,
        max_frames_num=args.max_frames_num,
        video_decode_backend=args.video_decode_backend,
        three_d_config=args.three_d_config,
        enable_3d_aux=True,
        mm_projector_compressor_type=None,
        mm_projector_compressor_config=None,
        pooled_coords_root=str(output_root),
    )

    written = 0
    skipped = 0
    for task_name in _parse_tasks(args.tasks):
        docs = _build_docs(task_name, args.three_d_config, args.max_frames_num, args.sampling_strategy)
        if args.limit is not None:
            docs = docs[: args.limit]
        for doc in tqdm(docs, desc=f"precompute {task_name}"):
            cache_path = model._pooled_coords_cache_path(doc, frame_shape)
            if cache_path is None:
                raise ValueError("pooled coords cache path is disabled")
            if cache_path.exists():
                skipped += 1
                continue
            coords = model.compute_pooled_world_coords_for_doc(doc, frame_shape=frame_shape)
            shared_cache_path = coords_cache_path(
                output_root,
                video_id=doc["video_id"],
                frame_files=doc["frame_files"],
                crop_size=int(model._image_processor.crop_size.get("width", 384)),
                frame_shape=frame_shape,
                depth_scale=float(getattr(model.model.config, "mm_depth_scale", 1000.0)),
            )
            if shared_cache_path is not None:
                cache_path = shared_cache_path
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "version": 1,
                    "video_id": doc["video_id"],
                    "frame_files": list(doc["frame_files"]),
                    "frame_shape": list(frame_shape),
                    "ov_pad_avg14": coords.cpu().contiguous(),
                    "world_coords": coords.cpu().contiguous(),
                },
                cache_path,
            )
            written += 1

    print(f"written={written} skipped={skipped} output_root={output_root}")


if __name__ == "__main__":
    main()
