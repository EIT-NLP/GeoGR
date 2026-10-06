#!/usr/bin/env python3
"""Prepare ScanNet frame folders for LLaVA-OneVision training."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import shutil
from pathlib import Path
from typing import Any

import yaml


def load_embodiedscan_index(data_root: Path) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        info_path = data_root / "embodiedscan" / f"embodiedscan_infos_{split}.pkl"
        if not info_path.exists():
            continue
        with info_path.open("rb") as handle:
            payload = pickle.load(handle)
        for item in payload.get("data_list", []):
            sample_idx = item.get("sample_idx")
            if not isinstance(sample_idx, str) or not sample_idx.startswith("scannet/"):
                continue
            frame_paths = []
            for image_info in item.get("images", []):
                image_path = image_info.get("img_path")
                if image_path:
                    frame_paths.append(str(data_root / image_path))
            if frame_paths:
                index[sample_idx] = frame_paths
    return index


def natural_key(path: str) -> tuple[Any, ...]:
    stem = Path(path).stem
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem)


def resolve_manifest_path(path_value: str, manifest_path: Path) -> Path:
    path = Path(os.path.expandvars(path_value)).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def discover_scene_frames(data_root: Path, video_id: str) -> list[str]:
    scene_id = video_id.split("/")[-1]
    frame_dir = data_root / "scannet" / "posed_images" / scene_id
    if not frame_dir.exists():
        return []
    return sorted((str(p) for p in frame_dir.glob("*.jpg")), key=natural_key)


def sample_frames(frame_paths: list[str], frames_upbound: int) -> list[str]:
    if not frame_paths:
        return []
    if frames_upbound <= 0:
        return frame_paths
    indices = _linspace_floor(0, len(frame_paths) - 1, frames_upbound)
    return [frame_paths[i] for i in indices]


def _linspace_floor(start: int, stop: int, num: int) -> list[int]:
    if num <= 1:
        return [start]
    step = (stop - start) / float(num - 1)
    return [math.floor(start + step * i) for i in range(num)]


def reset_dir(path: Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_scene_frames(
    dst_scene_dir: Path,
    frame_paths: list[str],
    overwrite: bool,
) -> None:
    if overwrite:
        reset_dir(dst_scene_dir)
    else:
        dst_scene_dir.mkdir(parents=True, exist_ok=True)

    for idx, src in enumerate(frame_paths):
        src_path = Path(src)
        if not src_path.exists():
            raise FileNotFoundError(f"Missing sampled frame: {src_path}")
        dst = dst_scene_dir / f"{idx:05d}.jpg"
        if dst.exists() or dst.is_symlink():
            try:
                dst.unlink()
            except FileNotFoundError:
                pass
        os.symlink(src_path, dst)


def convert_json(
    input_json: Path,
    output_json: Path,
    frame_root: Path,
    data_root: Path,
    scene_index: dict[str, list[str]],
    frames_upbound: int,
    overwrite_links: bool,
) -> dict[str, Any]:
    with input_json.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    converted = []
    scene_to_frames: dict[str, list[str]] = {}
    missing_scenes: set[str] = set()
    for sample in data:
        item = dict(sample)
        video_id = item.get("video")
        if isinstance(video_id, str) and video_id.startswith("scannet/"):
            if video_id not in scene_to_frames:
                source_frames = scene_index.get(video_id) or discover_scene_frames(data_root, video_id)
                sampled_frames = sample_frames(source_frames, frames_upbound)
                if not sampled_frames:
                    missing_scenes.add(video_id)
                scene_to_frames[video_id] = sampled_frames
            item["video"] = f"shareVideoGPTV/{video_id.replace('/', '__')}"
        converted.append(item)

    if missing_scenes:
        preview = ", ".join(sorted(missing_scenes)[:10])
        raise FileNotFoundError(
            f"{input_json} has {len(missing_scenes)} scenes without frames. First missing: {preview}"
        )

    for video_id, sampled_frames in sorted(scene_to_frames.items()):
        scene_dir = frame_root / "shareVideoGPTV" / video_id.replace("/", "__")
        link_scene_frames(scene_dir, sampled_frames, overwrite=overwrite_links)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(converted, handle, ensure_ascii=False, indent=2)

    return {
        "input_json": str(input_json),
        "output_json": str(output_json),
        "num_samples": len(converted),
        "num_scenes": len(scene_to_frames),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-yaml", required=True, type=Path)
    parser.add_argument("--output-yaml", required=True, type=Path)
    parser.add_argument("--output-json-dir", required=True, type=Path)
    parser.add_argument("--frame-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--frames-upbound", type=int, default=32)
    parser.add_argument("--overwrite-links", action="store_true")
    args = parser.parse_args()

    input_yaml = args.input_yaml.expanduser().resolve()
    with input_yaml.open("r", encoding="utf-8") as handle:
        train_yaml = yaml.safe_load(handle)
    datasets = train_yaml.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError(f"Expected a datasets list in {input_yaml}")

    scene_index = load_embodiedscan_index(args.data_root)
    output_datasets = []
    summaries = []
    for dataset in datasets:
        json_path = resolve_manifest_path(dataset["json_path"], input_yaml)
        output_json = args.output_json_dir / json_path.name
        summary = convert_json(
            input_json=json_path,
            output_json=output_json,
            frame_root=args.frame_root,
            data_root=args.data_root,
            scene_index=scene_index,
            frames_upbound=args.frames_upbound,
            overwrite_links=args.overwrite_links,
        )
        summaries.append(summary)
        new_dataset = dict(dataset)
        new_dataset["json_path"] = str(output_json)
        output_datasets.append(new_dataset)

    args.output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with args.output_yaml.open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"datasets": output_datasets}, handle, sort_keys=False)

    print(f"[prepare] wrote yaml: {args.output_yaml}")
    print(f"[prepare] frame root: {args.frame_root}")
    for summary in summaries:
        print(
            "[prepare] {output_json}: samples={num_samples}, scenes={num_scenes}".format(
                **summary
            )
        )


if __name__ == "__main__":
    main()
