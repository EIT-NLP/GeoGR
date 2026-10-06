from __future__ import annotations

from lmms_eval.tasks._task_utils.scannet3d.base import Scannet3DConfigurableTask
from lmms_eval.tasks._task_utils.scannet3d.data import build_sqa3d_docs, get_asset_manager
from lmms_eval.tasks._task_utils.scannet3d.metrics import aggregate_sqa3d_em


SQA3D_QUESTION_TYPES = ("all", "what", "is", "how", "can", "which", "others")


def sqa3d_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    del lmms_eval_specific_kwargs
    return f"{doc['situation']} {doc['question']} Answer the question using a single word or phrase."


def sqa3d_doc_to_target(doc):
    return list(doc["answers"])


def sqa3d_process_results(doc, results, **kwargs):
    del kwargs
    pred = results[0].strip() if results else ""
    payload = {
        "sample_id": doc["sample_id"],
        "pred": pred,
        "answers": list(doc["answers"]),
        "question_type": doc["question_type"],
    }
    metrics = {"sqa3d_em": payload}
    for question_type in SQA3D_QUESTION_TYPES:
        metrics[f"sqa3d_{question_type}_em"] = payload
    return metrics


def _agg():
    return lambda items: aggregate_sqa3d_em(items, question_type=None)


def _agg_type(question_type):
    return lambda items: aggregate_sqa3d_em(items, question_type=question_type)


class SQA3DTask(Scannet3DConfigurableTask):
    BENCHMARK_NAME = "sqa3d"
    DEFAULT_SPLIT = "test"

    def __init__(self, config=None, model_name=None):
        metric_list = [
            {"metric": "sqa3d_em", "aggregation": _agg(), "higher_is_better": True},
        ]
        for question_type in SQA3D_QUESTION_TYPES:
            scoped_type = None if question_type == "all" else question_type
            metric_list.append(
                {
                    "metric": f"sqa3d_{question_type}_em",
                    "aggregation": _agg_type(scoped_type),
                    "higher_is_better": True,
                }
            )
        defaults = {
            "doc_to_text": sqa3d_doc_to_text,
            "doc_to_target": sqa3d_doc_to_target,
            "process_results": sqa3d_process_results,
            "generation_kwargs": {"max_new_tokens": 512, "temperature": 0.0},
            "metric_list": metric_list,
        }
        merged = {**defaults, **(config or {})}
        super().__init__(config=merged, model_name=model_name)

    def build_docs(self, split: str, max_frames: int, strategy: str):
        return build_sqa3d_docs(get_asset_manager().paths, get_asset_manager(), split=split, max_frames=max_frames, strategy=strategy)
