from __future__ import annotations

from lmms_eval.tasks._task_utils.scannet3d.base import Scannet3DConfigurableTask
from lmms_eval.tasks._task_utils.scannet3d.data import build_scan2cap_docs, get_asset_manager
from lmms_eval.tasks._task_utils.scannet3d.metrics import aggregate_scan2cap_metric


def scan2cap_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    del lmms_eval_specific_kwargs
    return doc["native_prompt"]


def scan2cap_doc_to_target(doc):
    return list(doc["answers"])


def scan2cap_process_results(doc, results, **kwargs):
    del kwargs
    pred = results[0].strip() if results else ""
    payload = {
        "sample_id": doc["sample_id"],
        "pred": pred,
        "answers": list(doc["answers"]),
        "question_type": doc["question_type"],
    }
    return {
        "scan2cap_bleu_4": payload,
        "scan2cap_meteor": payload,
        "scan2cap_rouge_l": payload,
        "scan2cap_cider": payload,
    }


def _caption(metric_name):
    return lambda items: aggregate_scan2cap_metric(items, metric_name)


class Scan2CapTask(Scannet3DConfigurableTask):
    BENCHMARK_NAME = "scan2cap"
    DEFAULT_SPLIT = "val"

    def __init__(self, config=None, model_name=None):
        defaults = {
            "doc_to_text": scan2cap_doc_to_text,
            "doc_to_target": scan2cap_doc_to_target,
            "process_results": scan2cap_process_results,
            "generation_kwargs": {"max_new_tokens": 512, "temperature": 0.0},
            "metric_list": [
                {"metric": "scan2cap_bleu_4", "aggregation": _caption("bleu_4"), "higher_is_better": True},
                {"metric": "scan2cap_meteor", "aggregation": _caption("meteor"), "higher_is_better": True},
                {"metric": "scan2cap_rouge_l", "aggregation": _caption("rouge_l"), "higher_is_better": True},
                {"metric": "scan2cap_cider", "aggregation": _caption("cider"), "higher_is_better": True},
            ],
        }
        merged = {**defaults, **(config or {})}
        super().__init__(config=merged, model_name=model_name)

    def build_docs(self, split: str, max_frames: int, strategy: str):
        return build_scan2cap_docs(get_asset_manager().paths, get_asset_manager(), split=split, max_frames=max_frames, strategy=strategy)
