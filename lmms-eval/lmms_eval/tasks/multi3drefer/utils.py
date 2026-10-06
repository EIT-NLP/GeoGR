from __future__ import annotations

from lmms_eval.tasks._task_utils.scannet3d.base import Scannet3DConfigurableTask
from lmms_eval.tasks._task_utils.scannet3d.data import build_multi3drefer_docs, get_asset_manager
from lmms_eval.tasks._task_utils.scannet3d.metrics import aggregate_multi3drefer_f1


def multi3drefer_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    del lmms_eval_specific_kwargs
    return (
        "Identify the object according to the following description.\n"
        f"{doc['description']}\n"
        "There may be no corresponding object, or there may be one or more objects."
    )


def multi3drefer_doc_to_target(doc):
    return doc["target_boxes"]


def multi3drefer_process_results(doc, results, **kwargs):
    del kwargs
    pred = results[0].strip() if results else ""
    payload = {
        "sample_id": doc["sample_id"],
        "pred": pred,
        "target_boxes": doc["target_boxes"],
        "question_type": doc["question_type"],
    }
    return {"multi3drefer_f1_25": payload, "multi3drefer_f1_50": payload}


def _agg(threshold):
    return lambda items: aggregate_multi3drefer_f1(items, threshold=threshold, question_type=None)


class Multi3DReferTask(Scannet3DConfigurableTask):
    BENCHMARK_NAME = "multi3drefer"
    DEFAULT_SPLIT = "val"

    def __init__(self, config=None, model_name=None):
        defaults = {
            "doc_to_text": multi3drefer_doc_to_text,
            "doc_to_target": multi3drefer_doc_to_target,
            "process_results": multi3drefer_process_results,
            "metric_list": [
                {"metric": "multi3drefer_f1_25", "aggregation": _agg(0.25), "higher_is_better": True},
                {"metric": "multi3drefer_f1_50", "aggregation": _agg(0.50), "higher_is_better": True},
            ],
        }
        merged = {**defaults, **(config or {})}
        super().__init__(config=merged, model_name=model_name)

    def build_docs(self, split: str, max_frames: int, strategy: str):
        return build_multi3drefer_docs(get_asset_manager().paths, get_asset_manager(), split=split, max_frames=max_frames, strategy=strategy)
