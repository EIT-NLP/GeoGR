from __future__ import annotations

from lmms_eval.tasks._task_utils.scannet3d.base import Scannet3DConfigurableTask
from lmms_eval.tasks._task_utils.scannet3d.data import build_scanqa_docs, get_asset_manager
from lmms_eval.tasks._task_utils.scannet3d.metrics import aggregate_caption_metric, aggregate_scanqa_em


def scanqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    del lmms_eval_specific_kwargs
    return f"{doc['question']} Answer the question simply."


def scanqa_doc_to_target(doc):
    return list(doc["answers"])


def _scanqa_payload(doc, results):
    pred = results[0].strip() if results else ""
    return {
        "sample_id": doc["sample_id"],
        "pred": pred,
        "answers": list(doc["answers"]),
        "question_type": doc["question_type"],
    }


def scanqa_process_results(doc, results, **kwargs):
    del kwargs
    payload = _scanqa_payload(doc, results)
    return {
        "scanqa_em": payload,
        "scanqa_bleu_1": payload,
        "scanqa_bleu_2": payload,
        "scanqa_bleu_3": payload,
        "scanqa_bleu_4": payload,
        "scanqa_meteor": payload,
        "scanqa_rouge_l": payload,
        "scanqa_cider": payload,
    }


def _caption(metric_name):
    return lambda items: aggregate_caption_metric(items, metric_name)


class ScanQATask(Scannet3DConfigurableTask):
    BENCHMARK_NAME = "scanqa"
    DEFAULT_SPLIT = "val"

    def __init__(self, config=None, model_name=None):
        defaults = {
            "doc_to_text": scanqa_doc_to_text,
            "doc_to_target": scanqa_doc_to_target,
            "process_results": scanqa_process_results,
            "generation_kwargs": {"max_new_tokens": 512, "temperature": 0.0},
            "metric_list": [
                {"metric": "scanqa_em", "aggregation": aggregate_scanqa_em, "higher_is_better": True},
                {"metric": "scanqa_bleu_1", "aggregation": _caption("bleu_1"), "higher_is_better": True},
                {"metric": "scanqa_bleu_2", "aggregation": _caption("bleu_2"), "higher_is_better": True},
                {"metric": "scanqa_bleu_3", "aggregation": _caption("bleu_3"), "higher_is_better": True},
                {"metric": "scanqa_bleu_4", "aggregation": _caption("bleu_4"), "higher_is_better": True},
                {"metric": "scanqa_meteor", "aggregation": _caption("meteor"), "higher_is_better": True},
                {"metric": "scanqa_rouge_l", "aggregation": _caption("rouge_l"), "higher_is_better": True},
                {"metric": "scanqa_cider", "aggregation": _caption("cider"), "higher_is_better": True},
            ],
        }
        merged = {**defaults, **(config or {})}
        super().__init__(config=merged, model_name=model_name)

    def build_docs(self, split: str, max_frames: int, strategy: str):
        return build_scanqa_docs(get_asset_manager().paths, get_asset_manager(), split=split, max_frames=max_frames, strategy=strategy)
