from __future__ import annotations

from lmms_eval.tasks._task_utils.scannet3d.base import Scannet3DConfigurableTask
from lmms_eval.tasks._task_utils.scannet3d.data import build_scanrefer_docs, get_asset_manager
from lmms_eval.tasks._task_utils.scannet3d.metrics import aggregate_scanrefer_iou


def scanrefer_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    del lmms_eval_specific_kwargs
    return (
        "Identify the object according to the following description.\n"
        f"{doc['description']}"
    )


def scanrefer_doc_to_target(doc):
    return doc["target_boxes"][0]


def scanrefer_process_results(doc, results, **kwargs):
    del kwargs
    pred = results[0].strip() if results else ""
    payload = {
        "sample_id": doc["sample_id"],
        "pred": pred,
        "target_boxes": doc["target_boxes"],
        "question_type": doc["question_type"],
    }
    return {
        "scanrefer_iou25": payload,
        "scanrefer_iou50": payload,
    }


def _agg(threshold):
    return lambda items: aggregate_scanrefer_iou(items, threshold=threshold, question_type=None)


class ScanReferTask(Scannet3DConfigurableTask):
    BENCHMARK_NAME = "scanrefer"
    DEFAULT_SPLIT = "val"

    def __init__(self, config=None, model_name=None):
        defaults = {
            "doc_to_text": scanrefer_doc_to_text,
            "doc_to_target": scanrefer_doc_to_target,
            "process_results": scanrefer_process_results,
            "metric_list": [
                {"metric": "scanrefer_iou25", "aggregation": _agg(0.25), "higher_is_better": True},
                {"metric": "scanrefer_iou50", "aggregation": _agg(0.50), "higher_is_better": True},
            ],
            "generation_kwargs": {"max_new_tokens": 128, "temperature": 0.0},
        }
        merged = {**defaults, **(config or {})}
        super().__init__(config=merged, model_name=model_name)

    def build_docs(self, split: str, max_frames: int, strategy: str):
        return build_scanrefer_docs(get_asset_manager().paths, get_asset_manager(), split=split, max_frames=max_frames, strategy=strategy)
