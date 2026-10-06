from __future__ import annotations

import csv
import json
import os
import pickle
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
from PIL import Image

from .box_utils import best_iou, box_iou
from .config import Scannet3DPaths, get_data_paths


KNOWN_MULTI3D_TYPES = ["zt_wo_d", "zt_w_d", "st_wo_d", "st_w_d", "mt"]
KNOWN_SQA3D_TYPES = ["what", "is", "how", "can", "which", "others"]


def infer_question_type(question: str) -> str:
    question = (question or "").strip().lower()
    if question.startswith("what"):
        return "what"
    if question.startswith("is"):
        return "is"
    if question.startswith("how"):
        return "how"
    if question.startswith("can"):
        return "can"
    if question.startswith("which"):
        return "which"
    return "others"


def locate_first(root: str, candidates: Iterable[str]) -> str:
    for candidate in candidates:
        path = os.path.join(root, candidate)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find any of {list(candidates)} under {root}")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_scene_boxes(path: str) -> dict[str, list[float]]:
    scene = torch.load(path, map_location="cpu")
    object_ids = scene["aabb_obj_ids"].tolist()
    corners = scene["aabb_corner_xyz"].tolist()

    result = {}
    for object_id, corner_set in zip(object_ids, corners):
        xs, ys, zs = zip(*corner_set)
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        z_min, z_max = min(zs), max(zs)
        result[str(object_id)] = [
            (x_min + x_max) / 2.0,
            (y_min + y_max) / 2.0,
            (z_min + z_max) / 2.0,
            x_max - x_min,
            y_max - y_min,
            z_max - z_min,
        ]
    return result


def frame_number_from_path(path: str) -> int:
    return int(Path(path).stem)


@lru_cache(maxsize=1)
def get_asset_manager(config_path: str | None = None) -> "Scannet3DAssetManager":
    return Scannet3DAssetManager(get_data_paths(config_path))


class Scannet3DAssetManager:
    def __init__(self, paths: Scannet3DPaths):
        self.paths = paths
        self._scene_cache = {}
        self._scene_split = {}
        self._gt_box_cache = {}
        self._pred_box_cache = None
        self._mc_sampling = None
        self._semantic_mapping = None
        self._resolved_path_cache = {}
        self._scene_frames_cache = {}
        self._load_embodiedscan()

    def _load_embodiedscan(self):
        for split in ("train", "val", "test"):
            file_path = os.path.join(self.paths.embodiedscan_root, f"embodiedscan_infos_{split}.pkl")
            with open(file_path, "rb") as handle:
                records = pickle.load(handle)["data_list"]
            for record in records:
                scene_key = record["sample_idx"]
                self._scene_cache[scene_key] = record
                self._scene_split[scene_key] = split

    def get_scene(self, scene_key: str) -> dict:
        return self._scene_cache[scene_key]

    def get_scene_split(self, scene_key: str) -> str:
        return self._scene_split[scene_key]

    def _resolve_scannet_path(self, relative_path: str) -> str:
        """Resolve a ScanNet-relative path across known dataset layouts."""
        if os.path.isabs(relative_path):
            return relative_path

        rel = relative_path.lstrip("/")
        cached = self._resolved_path_cache.get(rel)
        if cached is not None:
            return cached

        # Fast path for the common EmbodiedScan image layout:
        # "scannet/posed_images/<scene>/<frame>.jpg" -> "<video_3d_llm_root>/posed_images/<scene>/<frame>.jpg"
        if rel.startswith("scannet/posed_images/"):
            fast_candidate = os.path.join(self.paths.video_3d_llm_root, rel[len("scannet/") :])
            if os.path.exists(fast_candidate):
                self._resolved_path_cache[rel] = fast_candidate
                return fast_candidate

        candidates = [
            os.path.join(self.paths.scannet_root, rel),
            os.path.join(self.paths.video_3d_llm_root, rel),
            os.path.join(self.paths.video_3d_llm_root, "data", "scannet", rel),
        ]

        # EmbodiedScan commonly stores image paths with a "scannet/" prefix.
        if rel.startswith("scannet/"):
            tail = rel[len("scannet/") :]
            candidates.extend(
                [
                    os.path.join(self.paths.scannet_root, tail),
                    os.path.join(self.paths.video_3d_llm_root, tail),
                    os.path.join(self.paths.video_3d_llm_root, "data", "scannet", tail),
                ]
            )

        for candidate in candidates:
            if os.path.exists(candidate):
                self._resolved_path_cache[rel] = candidate
                return candidate
        self._resolved_path_cache[rel] = candidates[0]
        return candidates[0]

    def sample_frame_files(self, scene_key: str, max_frames: int = 32, strategy: str = "uniform") -> list[str]:
        if "mc" in strategy:
            return self._sample_frame_files_mc(scene_key, max_frames=max_frames, strategy=strategy)

        all_frames = self._scene_frames_cache.get(scene_key)
        if all_frames is None:
            scene = self.get_scene(scene_key)
            all_frames = [self._resolve_scannet_path(image_info["img_path"]) for image_info in scene["images"]]
            self._scene_frames_cache[scene_key] = all_frames
        if not all_frames:
            return []
        sampled_indices = np.linspace(0, len(all_frames) - 1, max_frames, dtype=int)
        return [all_frames[index] for index in sampled_indices]

    def _sample_frame_files_mc(self, scene_key: str, max_frames: int, strategy: str) -> list[str]:
        if self._mc_sampling is None:
            sampling_path = os.path.join(self.paths.metadata_root, "scannet_select_frames.json")
            records = load_json(sampling_path)
            self._mc_sampling = {record["video_id"]: record for record in records}

        if scene_key not in self._mc_sampling:
            return self.sample_frame_files(scene_key, max_frames=max_frames, strategy="uniform")

        record = self._mc_sampling[scene_key]
        frame_files = record["frame_files"][:max_frames]
        voxel_nums = record["voxel_nums"][:max_frames]
        ratio = 1.0
        if "ratio95" in strategy:
            ratio = 0.95
        elif "ratio90" in strategy:
            ratio = 0.90
        if ratio != 1.0:
            total = record["num_all_voxels"]
            covered = 0
            trimmed = []
            for frame_file, voxel_num in zip(frame_files, voxel_nums):
                trimmed.append(frame_file)
                covered += voxel_num
                if covered >= total * ratio:
                    break
            frame_files = trimmed
        frame_files = sorted(frame_files, key=frame_number_from_path)
        return [self._resolve_scannet_path(frame_file) for frame_file in frame_files]

    def open_images(self, frame_files: Iterable[str]) -> list[Image.Image]:
        images = []
        for frame_file in frame_files:
            with Image.open(frame_file) as handle:
                images.append(handle.convert("RGB"))
        return images

    def get_gt_boxes(self, scene_id: str, split: str) -> dict[str, list[float]]:
        cache_key = (split, scene_id)
        if cache_key not in self._gt_box_cache:
            box_candidates = [
                os.path.join(self.paths.scannet_root, "pcd_with_object_aabbs", split, f"{scene_id}.pth"),
                os.path.join(self.paths.video_3d_llm_root, "data", "scannet", "pcd_with_object_aabbs", split, f"{scene_id}.pth"),
            ]
            box_path = None
            for candidate in box_candidates:
                if os.path.exists(candidate):
                    box_path = candidate
                    break
            if box_path is None:
                box_path = box_candidates[0]
            self._gt_box_cache[cache_key] = load_scene_boxes(box_path)
        return self._gt_box_cache[cache_key]

    def get_pred_boxes(self, scene_key: str) -> list[list[float]]:
        if self._pred_box_cache is None:
            path = os.path.join(self.paths.metadata_root, "scannet_val_pred_box.json")
            self._pred_box_cache = {key: value for key, value in load_json(path).items()}
        return [[float(item) for item in box[:6]] for box in self._pred_box_cache.get(scene_key, [])]

    def select_best_proposal(self, scene_key: str, gt_box: list[float], threshold: float = 0.5) -> list[float] | None:
        proposals = self.get_pred_boxes(scene_key)
        if not proposals:
            return None
        score, box = best_iou(gt_box, proposals)
        if score < threshold:
            return None
        return box

    def semantic_label_mapping(self) -> dict[str, int]:
        if self._semantic_mapping is not None:
            return self._semantic_mapping

        candidate_paths = [
            os.path.join(self.paths.scannet_root, "pcd_with_object_aabbs", "metadata", "scannetv2-labels.combined.tsv"),
            os.path.join(self.paths.video_3d_llm_root, "data", "scannet", "pcd_with_object_aabbs", "metadata", "scannetv2-labels.combined.tsv"),
            os.path.join(self.paths.scanqa_root, "scannet", "meta_data", "scannetv2-labels.combined.tsv"),
        ]
        mapping_path = None
        for candidate in candidate_paths:
            if os.path.exists(candidate):
                mapping_path = candidate
                break
        if mapping_path is None:
            raise FileNotFoundError("Could not locate scannetv2-labels.combined.tsv for ScanRefer eval_type generation.")

        mapping = {}
        with open(mapping_path, "r", encoding="utf-8") as handle:
            tsv = csv.reader(handle, delimiter="\t")
            next(tsv)
            for row in tsv:
                mapping[row[1]] = int(row[4])
        self._semantic_mapping = mapping
        return mapping


def _raw_scanrefer_records(paths: Scannet3DPaths, split: str) -> list[dict]:
    scanrefer_root = locate_first(paths.scanrefer_root, ["raw/scanrefer", "scanrefer", "."])
    return load_json(os.path.join(scanrefer_root, f"ScanRefer_filtered_{split}.json"))


def _raw_scan2cap_records(paths: Scannet3DPaths, split: str) -> list[dict]:
    scan2cap_root = locate_first(paths.scan2cap_root, ["raw/scan2cap", "scan2cap", "."])
    return load_json(os.path.join(scan2cap_root, f"ScanRefer_filtered_{split}.json"))


def compute_scanrefer_eval_types(records: list[dict], assets: Scannet3DAssetManager) -> dict[tuple[str, str], str]:
    label_mapping = assets.semantic_label_mapping()
    valid_semantics = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39}
    scene_semantic_counts = defaultdict(int)
    seen_instances = set()
    for item in records:
        key = (item["scene_id"], item["object_id"])
        if key in seen_instances:
            continue
        seen_instances.add(key)
        obj_name = item["object_name"].replace("_", " ")
        semantic = label_mapping.get(obj_name, 39)
        if semantic not in valid_semantics:
            semantic = 39
        scene_semantic_counts[(item["scene_id"], semantic)] += 1

    eval_types = {}
    for item in records:
        obj_name = item["object_name"].replace("_", " ")
        semantic = label_mapping.get(obj_name, 39)
        if semantic not in valid_semantics:
            semantic = 39
        eval_types[(item["scene_id"], item["object_id"])] = "unique" if scene_semantic_counts[(item["scene_id"], semantic)] == 1 else "multiple"
    return eval_types


def build_scanqa_docs(paths: Scannet3DPaths, assets: Scannet3DAssetManager, split: str = "val", max_frames: int = 32, strategy: str = "uniform") -> list[dict]:
    data = load_json(os.path.join(paths.scanqa_root, "ScanQA", f"ScanQA_v1.0_{split}.json"))
    docs = []
    for item in data:
        scene_id = item["scene_id"]
        scene_key = f"scannet/{scene_id}"
        docs.append(
            {
                "sample_id": item["question_id"],
                "scene_id": scene_id,
                "video_id": scene_key,
                "frame_files": assets.sample_frame_files(scene_key, max_frames=max_frames, strategy=strategy),
                "question": item["question"],
                "answers": list(item["answers"]),
                "question_type": infer_question_type(item["question"]),
                "task_kind": "qa",
            }
        )
    return docs


def build_sqa3d_docs(paths: Scannet3DPaths, assets: Scannet3DAssetManager, split: str = "test", max_frames: int = 32, strategy: str = "uniform") -> list[dict]:
    questions = load_json(os.path.join(paths.sqa3d_root, "balanced", f"v1_balanced_questions_{split}_scannetv2.json"))["questions"]
    annotations = load_json(os.path.join(paths.sqa3d_root, "balanced", f"v1_balanced_sqa_annotations_{split}_scannetv2.json"))["annotations"]
    question_by_id = {item["question_id"]: item for item in questions}

    docs = []
    for item in annotations:
        question = question_by_id[item["question_id"]]
        scene_id = item["scene_id"]
        scene_key = f"scannet/{scene_id}"
        situation = question["situation"]
        answers = [answer["answer"] for answer in item["answers"]]
        docs.append(
            {
                "sample_id": str(item["question_id"]),
                "scene_id": scene_id,
                "video_id": scene_key,
                "frame_files": assets.sample_frame_files(scene_key, max_frames=max_frames, strategy=strategy),
                "question": question["question"],
                "situation": situation,
                "answers": answers,
                "question_type": infer_question_type(question["question"]),
                "task_kind": "qa",
            }
        )
    return docs


def build_scanrefer_docs(paths: Scannet3DPaths, assets: Scannet3DAssetManager, split: str = "val", max_frames: int = 32, strategy: str = "uniform") -> list[dict]:
    records = _raw_scanrefer_records(paths, split)
    eval_types = compute_scanrefer_eval_types(records, assets)
    docs = []
    for item in records:
        scene_id = item["scene_id"]
        scene_key = f"scannet/{scene_id}"
        gt_box = assets.get_gt_boxes(scene_id, split)[str(item["object_id"])]
        proposal_boxes = assets.get_pred_boxes(scene_key)
        docs.append(
            {
                "sample_id": f"{scene_id}_{item['object_id']}_{item['ann_id']}",
                "scene_id": scene_id,
                "video_id": scene_key,
                "frame_files": assets.sample_frame_files(scene_key, max_frames=max_frames, strategy=strategy),
                "description": item["description"].capitalize(),
                "question": item["description"].capitalize(),
                "answers": [gt_box],
                "target_boxes": [gt_box],
                "proposal_boxes": proposal_boxes,
                "question_type": eval_types[(scene_id, str(item["object_id"]))],
                "ann_id": item["ann_id"],
                "object_id": str(item["object_id"]),
                "object_name": item["object_name"],
                "task_kind": "grounding_single",
            }
        )
    return docs


def build_scan2cap_docs(paths: Scannet3DPaths, assets: Scannet3DAssetManager, split: str = "val", max_frames: int = 32, strategy: str = "uniform") -> list[dict]:
    records = _raw_scan2cap_records(paths, split)
    eval_types = compute_scanrefer_eval_types(records, assets)
    annotations_by_instance = defaultdict(list)
    if split == "val":
        for item in records:
            key = f"{item['scene_id']}|{item['object_id']}|{item['object_name']}"
            annotations_by_instance[key].append(item["description"])

    visible_instances = set()
    docs = []
    prompt = "<image> Given an object located at <coord> , describe the object in detail."
    for index, item in enumerate(records):
        instance_key = f"{item['scene_id']}|{item['object_id']}|{item['object_name']}"
        if split != "train" and instance_key in visible_instances:
            continue
        visible_instances.add(instance_key)
        scene_id = item["scene_id"]
        scene_key = f"scannet/{scene_id}"
        gt_box = assets.get_gt_boxes(scene_id, split)[str(item["object_id"])]
        if split == "train":
            box_input = list(gt_box)
            answers = [item["description"]]
        else:
            box_input = assets.select_best_proposal(scene_key, gt_box, threshold=0.5)
            answers = list(annotations_by_instance[instance_key])
        docs.append(
            {
                "sample_id": index,
                "scene_id": scene_id,
                "video_id": scene_key,
                "frame_files": assets.sample_frame_files(scene_key, max_frames=max_frames, strategy=strategy),
                "question": prompt,
                "native_prompt": prompt,
                "description": item["description"].capitalize(),
                "answers": answers,
                "box_input": box_input,
                "target_boxes": [gt_box] if gt_box is not None else [],
                "proposal_boxes": [],
                "question_type": eval_types[(scene_id, str(item["object_id"]))],
                "ann_id": item["ann_id"],
                "object_id": str(item["object_id"]),
                "object_name": item["object_name"],
                "task_kind": "caption",
            }
        )
    return docs


def build_multi3drefer_docs(paths: Scannet3DPaths, assets: Scannet3DAssetManager, split: str = "val", max_frames: int = 32, strategy: str = "uniform") -> list[dict]:
    records = load_json(os.path.join(paths.multi3drefer_root, f"multi3drefer_{split}.json"))
    docs = []
    for index, item in enumerate(records):
        scene_id = item["scene_id"]
        scene_key = f"scannet/{scene_id}"
        gt_lookup = assets.get_gt_boxes(scene_id, split)
        target_boxes = [gt_lookup[str(object_id)] for object_id in item["object_ids"]]
        proposal_boxes = assets.get_pred_boxes(scene_key)
        docs.append(
            {
                "sample_id": f"{scene_id}_{item['ann_id']}",
                "scene_id": scene_id,
                "video_id": scene_key,
                "frame_files": assets.sample_frame_files(scene_key, max_frames=max_frames, strategy=strategy),
                "description": item["description"].capitalize(),
                "question": item["description"].capitalize(),
                "answers": target_boxes,
                "target_boxes": target_boxes,
                "proposal_boxes": proposal_boxes,
                "question_type": item["eval_type"],
                "ann_id": item["ann_id"],
                "object_ids": [str(object_id) for object_id in item["object_ids"]],
                "object_name": item["object_name"],
                "task_kind": "grounding_multi",
                "index": index,
            }
        )
    return docs
