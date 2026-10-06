#!/usr/bin/env python3
"""Create deterministic, disjoint ScanQA/SQA3D subsets for staged OV training.

The input YAML must already use the prepared OV video layout
(``shareVideoGPTV/...``).  The generated JSON files preserve every sample
exactly; only their membership and order are changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_samples(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError(f"Expected a JSON list of objects: {path}")
    for index, sample in enumerate(data):
        video = sample.get("video")
        if not isinstance(video, str) or not video.startswith("shareVideoGPTV/"):
            raise ValueError(
                f"Input must be prepared OV data with shareVideoGPTV/... videos; "
                f"found {video!r} at {path}[{index}]"
            )
    return data


def sample_identity(sample: dict[str, Any], index: int) -> str:
    if "id" in sample:
        return json.dumps(sample["id"], ensure_ascii=False, sort_keys=True)
    return f"__source_index__:{index}"


def split_indices(samples: list[dict[str, Any]], seed: int) -> tuple[list[int], list[int]]:
    # Keep repeated variants of the same question ID in one half.  SQA3D can
    # contain multiple language variants under one ID, so an index-only split
    # would leak the same underlying question into both training stages.
    groups: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        groups.setdefault(sample_identity(sample, index), []).append(index)

    grouped_indices = list(groups.values())
    random.Random(seed).shuffle(grouped_indices)
    target_first = (len(samples) + 1) // 2
    target_second = len(samples) // 2
    first: list[int] = []
    second: list[int] = []
    for group in grouped_indices:
        first_error = abs(target_first - (len(first) + len(group))) + abs(target_second - len(second))
        second_error = abs(target_first - len(first)) + abs(target_second - (len(second) + len(group)))
        if first_error < second_error or (first_error == second_error and len(first) <= len(second)):
            first.extend(group)
        else:
            second.extend(group)

    first.sort()
    second.sort()
    if set(first) & set(second):
        raise AssertionError("Dataset split has overlapping source indices")
    if len(first) + len(second) != len(samples):
        raise AssertionError("Dataset split does not cover all source samples")
    first_ids = {sample_identity(samples[index], index) for index in first}
    second_ids = {sample_identity(samples[index], index) for index in second}
    if first_ids & second_ids:
        raise AssertionError("Dataset split leaks a repeated sample ID across halves")
    return first, second


def write_json(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_input_datasets(input_yaml: Path) -> list[dict[str, Any]]:
    with input_yaml.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"Expected a non-empty datasets list in {input_yaml}")
    normalized = []
    for entry in datasets:
        if not isinstance(entry, dict) or not entry.get("json_path"):
            raise ValueError(f"Every dataset entry needs json_path: {input_yaml}")
        normalized.append(entry)
    return normalized


def build_split(input_yaml: Path, output_root: Path, seed: int, overwrite: bool) -> dict[str, Any]:
    datasets = load_input_datasets(input_yaml)
    manifest_path = output_root / "split_manifest.json"
    if manifest_path.exists() and not overwrite:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("seed") != seed or manifest.get("source_yaml") != str(input_yaml.resolve()):
            raise RuntimeError(
                f"Existing split manifest does not match seed/source: {manifest_path}. "
                "Use SPLIT_OVERWRITE=1 to regenerate explicitly."
            )
        if manifest.get("version") != 2:
            raise RuntimeError(
                f"Existing split uses an obsolete format: {manifest_path}. "
                "Use SPLIT_OVERWRITE=1 to regenerate explicitly."
            )
        for item in manifest.get("datasets", []):
            source_path = Path(item["source_json"])
            if not source_path.is_file() or sha256_file(source_path) != item.get("source_sha256"):
                raise RuntimeError(
                    f"Source data changed after the split was created: {source_path}. "
                    "Use SPLIT_OVERWRITE=1 to regenerate explicitly."
                )
            for split_name in ("half_a", "half_b"):
                split_path = Path(item[f"{split_name}_json"])
                if not split_path.is_file() or sha256_file(split_path) != item.get(f"{split_name}_sha256"):
                    raise RuntimeError(f"Generated split file is missing or modified: {split_path}")
        for split_name in ("half_a", "half_b"):
            if not (output_root / f"{split_name}.yaml").is_file():
                raise RuntimeError(f"Incomplete existing split: {output_root / f'{split_name}.yaml'}")
        print(f"[split] reuse existing deterministic split: {output_root}")
        return manifest

    output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "version": 2,
        "source_yaml": str(input_yaml.resolve()),
        "seed": seed,
        "split_rule": "per dataset, group repeated sample IDs, shuffle groups by seed + dataset ordinal, greedily balance A/B",
        "datasets": [],
    }
    output_entries = {"half_a": [], "half_b": []}

    for ordinal, entry in enumerate(datasets):
        source_path = Path(entry["json_path"]).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source JSON: {source_path}")
        samples = load_samples(source_path)
        first_indices, second_indices = split_indices(samples, seed + ordinal * 1009)
        split_samples = {
            "half_a": [samples[index] for index in first_indices],
            "half_b": [samples[index] for index in second_indices],
        }
        source_name = source_path.name
        output_paths = {
            split: output_root / split / "json" / source_name for split in ("half_a", "half_b")
        }
        for split, path in output_paths.items():
            write_json(path, split_samples[split])
            split_entry = dict(entry)
            split_entry["json_path"] = str(path.resolve())
            split_entry["sampling_strategy"] = "all"
            output_entries[split].append(split_entry)

        manifest["datasets"].append(
            {
                "source_json": str(source_path),
                "source_sha256": sha256_file(source_path),
                "source_count": len(samples),
                "half_a_count": len(first_indices),
                "half_b_count": len(second_indices),
                "half_a_source_index_sha256": sha256_json(first_indices),
                "half_b_source_index_sha256": sha256_json(second_indices),
                "half_a_unique_ids": len(
                    {sample_identity(samples[index], index) for index in first_indices}
                ),
                "half_b_unique_ids": len(
                    {sample_identity(samples[index], index) for index in second_indices}
                ),
                "half_a_json": str(output_paths["half_a"].resolve()),
                "half_b_json": str(output_paths["half_b"].resolve()),
                "half_a_sha256": sha256_file(output_paths["half_a"]),
                "half_b_sha256": sha256_file(output_paths["half_b"]),
            }
        )

    for split, entries in output_entries.items():
        yaml_path = output_root / f"{split}.yaml"
        with yaml_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump({"datasets": entries}, handle, sort_keys=False)
        manifest[f"{split}_yaml"] = str(yaml_path.resolve())
        manifest[f"{split}_count"] = sum(
            item[f"{split}_count"] for item in manifest["datasets"]
        )

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"[split] wrote half_a={manifest['half_a_count']} samples, "
        f"half_b={manifest['half_b_count']} samples, seed={seed}: {output_root}"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-yaml", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_split(args.input_yaml.resolve(), args.output_root.resolve(), args.seed, args.overwrite)


if __name__ == "__main__":
    main()
