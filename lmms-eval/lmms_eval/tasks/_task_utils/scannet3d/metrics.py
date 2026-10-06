from __future__ import annotations

import os
import re
import string
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Iterable, List

from .box_utils import box_iou, max_weight_assignment, normalize_box, normalize_boxes
from .data import KNOWN_MULTI3D_TYPES, KNOWN_SQA3D_TYPES


@lru_cache(maxsize=1)
def _load_caption_metric_classes():
    try:
        from pycocoevalcap.eval import Bleu, Cider, Meteor, Rouge
    except Exception as error:
        raise ImportError(
            "ScanNet3D caption metrics require `pycocoevalcap`. "
            "Install the dependencies declared by lmms-eval."
        ) from error
    return Bleu, Cider, Rouge, Meteor


def _caption_metric_inputs(items):
    normalize_scanqa = _normalize_scanqa_metrics_enabled()
    refs = {}
    preds = {}
    for item in items:
        sample_id = item["sample_id"]
        if normalize_scanqa:
            pred = normalize_scanqa_metric_text(item["pred"])
            answers = [normalize_scanqa_metric_text(answer) for answer in item["answers"]]
        else:
            pred = _caption_eval_text(item["pred"]).rstrip(".")
            answers = [_caption_eval_text(answer) for answer in item["answers"]]
        refs[sample_id] = answers
        preds[sample_id] = [pred]
    return refs, preds


def _caption_eval_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.replace("|||", " ")
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _scan2cap_text_normalize(text: str) -> str:
    text = _caption_eval_text(text).replace(".", " . ").replace(",", " , ").lower()
    text = re.sub(r"\s+", " ", text).strip()
    return f"sos {text} eos"


def _scan2cap_metric_inputs(items):
    refs = {}
    preds = {}
    for item in items:
        sample_id = item["sample_id"]
        pred = _scan2cap_text_normalize(item["pred"])
        answers = [_scan2cap_text_normalize(answer) for answer in item["answers"]]
        refs[sample_id] = answers
        preds[sample_id] = [pred]
    return refs, preds


def aggregate_caption_metric(items, metric_name: str) -> float:
    if not items:
        return 0.0
    Bleu, Cider, Rouge, Meteor = _load_caption_metric_classes()
    refs, preds = _caption_metric_inputs(items)
    if metric_name == "bleu_1":
        return Bleu().compute_score(refs, preds)[0][0] * 100.0
    if metric_name == "bleu_2":
        return Bleu().compute_score(refs, preds)[0][1] * 100.0
    if metric_name == "bleu_3":
        return Bleu().compute_score(refs, preds)[0][2] * 100.0
    if metric_name == "bleu_4":
        return Bleu().compute_score(refs, preds)[0][3] * 100.0
    if metric_name == "cider":
        return Cider().compute_score(refs, preds)[0] * 100.0
    if metric_name == "rouge_l":
        return Rouge().compute_score(refs, preds)[0] * 100.0
    if metric_name == "meteor":
        return Meteor().compute_score(refs, preds)[0] * 100.0
    raise KeyError(metric_name)


def aggregate_scan2cap_metric(items, metric_name: str) -> float:
    if not items:
        return 0.0
    Bleu, Cider, Rouge, Meteor = _load_caption_metric_classes()
    refs, preds = _scan2cap_metric_inputs(items)
    if metric_name == "bleu_4":
        return Bleu().compute_score(refs, preds)[0][3] * 100.0
    if metric_name == "cider":
        return Cider().compute_score(refs, preds)[0] * 100.0
    if metric_name == "rouge_l":
        return Rouge().compute_score(refs, preds)[0] * 100.0
    if metric_name == "meteor":
        return Meteor().compute_score(refs, preds)[0] * 100.0
    raise KeyError(metric_name)


def simple_answer_normalize(text: str) -> str:
    text = (text or "").strip()
    if text.endswith("."):
        text = text[:-1]
    return re.sub(r"\s+", " ", text).strip().lower()


def _normalize_scanqa_metrics_enabled() -> bool:
    value = os.environ.get("SCANNET3D_NORMALIZE_SCANQA_METRICS", "")
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def normalize_scanqa_metric_text(text: str) -> str:
    text = _caption_eval_text(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"[\s\.!,?;:]+$", "", text)


def clean_sqa3d_answer(text: str) -> str:
    text = (text or "").lower()
    text = re.sub("[ ]+$", "", text)
    text = re.sub("^[ ]+", "", text)
    text = re.sub(" {2,}", " ", text)
    text = re.sub(r"\.[ ]{2,}", ". ", text)
    text = re.sub(r"[^a-zA-Z0-9,'\s\-:]+", "", text)
    replacements = {
        "ç": "c",
        "’": "'",
        "letf": "left",
        "let": "left",
        "tehre": "there",
        "rigth": "right",
        "rght": "right",
        "behine": "behind",
        "tv": "TV",
        "chai": "chair",
        "wasing": "washing",
        "waslked": "walked",
        "oclock": "o'clock",
    }
    for src, tgt in replacements.items():
        text = re.sub(rf"\b{re.escape(src)}\b", tgt, text)
    text = re.sub(r"\bo'[ ]+clock\b", "o'clock", text)
    digit_map = {
        "0": "zero",
        "none": "zero",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
        "10": "ten",
        "11": "eleven",
        "12": "twelve",
        "13": "thirteen",
        "14": "fourteen",
        "15": "fifteen",
        "16": "sixteen",
        "17": "seventeen",
        "18": "eighteen",
        "19": "nineteen",
        "20": "twenty",
        "23": "twenty-three",
    }
    for src, tgt in digit_map.items():
        text = re.sub(rf"\b{re.escape(src)}\b", tgt, text)
    text = re.sub(r"\b([a-zA-Z]+)([0-9])\b", r"\1", text)
    text = re.sub(r"\ba\b ([a-zA-Z]+)", r"\1", text)
    text = re.sub(r"\ban\b ([a-zA-Z]+)", r"\1", text)
    text = re.sub(r"\bthe\b ([a-zA-Z]+)", r"\1", text)
    text = re.sub(r"\bbackwards\b", "backward", text)
    return text


def aggregate_scanqa_em(items) -> float:
    if not items:
        return 0.0
    normalize_scanqa = _normalize_scanqa_metrics_enabled()
    correct = 0
    for item in items:
        if normalize_scanqa:
            pred = normalize_scanqa_metric_text(item["pred"])
            answers = [normalize_scanqa_metric_text(answer) for answer in item["answers"]]
        else:
            pred = item["pred"].strip()
            answers = list(item["answers"])
        if pred in answers:
            correct += 1
    return correct / len(items) * 100.0


def aggregate_sqa3d_em(items, question_type: str | None = None) -> float:
    filtered = [item for item in items if question_type is None or item["question_type"] == question_type]
    if not filtered:
        return 0.0
    correct = 0
    for item in filtered:
        pred = clean_sqa3d_answer(item["pred"])
        answers = [clean_sqa3d_answer(answer) for answer in item["answers"]]
        if pred in answers:
            correct += 1
    return correct / len(filtered) * 100.0


def aggregate_scanrefer_iou(items, threshold: float, question_type: str | None = None) -> float:
    filtered = [item for item in items if question_type is None or item["question_type"] == question_type]
    if not filtered:
        return 0.0
    hits = 0
    for item in filtered:
        pred = normalize_box(item["pred"])
        if pred is None:
            continue
        gt = normalize_box(item["target_boxes"][0])
        if gt is None:
            continue
        if box_iou(pred, gt) >= threshold:
            hits += 1
    return hits / len(filtered) * 100.0


def _multi3d_f1(pred_boxes: list[list[float]], gt_boxes: list[list[float]], threshold: float) -> float:
    pred_count = len(pred_boxes)
    gt_count = len(gt_boxes)
    if pred_count == 0 and gt_count == 0:
        return 1.0
    if pred_count == 0 or gt_count == 0:
        return 0.0
    matrix = [[box_iou(pred_box, gt_box) for gt_box in gt_boxes] for pred_box in pred_boxes]
    matches = max_weight_assignment(matrix)
    tp = 0
    for pred_index, gt_index in matches:
        if pred_index >= pred_count or gt_index >= gt_count:
            continue
        if matrix[pred_index][gt_index] >= threshold:
            tp += 1
    return 2.0 * tp / (pred_count + gt_count)


def aggregate_multi3drefer_f1(items, threshold: float, question_type: str | None = None) -> float:
    filtered = [item for item in items if question_type is None or item["question_type"] == question_type]
    if not filtered:
        return 0.0
    scores = []
    for item in filtered:
        pred_boxes = normalize_boxes(item["pred"])
        gt_boxes = normalize_boxes(item["target_boxes"])
        scores.append(_multi3d_f1(pred_boxes, gt_boxes, threshold))
    return sum(scores) / len(scores) * 100.0
