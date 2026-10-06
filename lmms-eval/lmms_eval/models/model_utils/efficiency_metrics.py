import math
from typing import Any, Dict, List, Optional


def _summarize_task_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_input_tokens = 0.0
    total_output_tokens = 0.0
    docs_with_tokens = 0

    for sample in samples:
        token_entries = sample.get("token_counts")
        sample_input = 0.0
        sample_output = 0.0
        has_token_data = False

        if isinstance(token_entries, list):
            for entry in token_entries:
                if not isinstance(entry, dict):
                    continue
                sample_input += float(entry.get("input_tokens") or 0)
                sample_output += float(entry.get("output_tokens") or 0)
                has_token_data = True

        if has_token_data:
            docs_with_tokens += 1
            total_input_tokens += sample_input
            total_output_tokens += sample_output

    total_tokens = total_input_tokens + total_output_tokens
    avg_output = total_output_tokens / docs_with_tokens if docs_with_tokens > 0 else 0.0

    return {
        "docs": float(len(samples)),
        "docs_with_token_counts": float(docs_with_tokens),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "avg_output_tokens_per_sample": avg_output,
    }


def build_efficiency_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    samples_by_task = results.get("samples")
    if not isinstance(samples_by_task, dict) or not samples_by_task:
        return {}

    by_task: Dict[str, Any] = {}
    overall: Dict[str, Any] = {
        "docs": 0.0,
        "docs_with_token_counts": 0.0,
        "total_input_tokens": 0.0,
        "total_output_tokens": 0.0,
        "total_tokens": 0.0,
    }

    for task_name, task_samples in samples_by_task.items():
        if not isinstance(task_samples, list):
            continue

        task_summary = _summarize_task_samples(task_samples)
        by_task[task_name] = task_summary

        overall["docs"] += task_summary["docs"]
        overall["docs_with_token_counts"] += task_summary["docs_with_token_counts"]
        overall["total_input_tokens"] += task_summary["total_input_tokens"]
        overall["total_output_tokens"] += task_summary["total_output_tokens"]
        overall["total_tokens"] += task_summary["total_tokens"]

    docs_with_tokens = overall["docs_with_token_counts"]
    overall["avg_output_tokens_per_sample"] = (overall["total_output_tokens"] / docs_with_tokens) if docs_with_tokens > 0 else 0.0

    return {
        "by_task": by_task,
        "overall": overall,
    }


def _extract_compression_efficiency(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    direct = sample.get("compression_efficiency")
    if isinstance(direct, dict):
        return direct

    generation_metadata = sample.get("generation_metadata")
    if isinstance(generation_metadata, list) and generation_metadata:
        first = generation_metadata[0]
        if isinstance(first, dict):
            nested = first.get("compression_efficiency")
            if isinstance(nested, dict):
                return nested
    return None


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _summarize_numeric(values: List[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    ordered = [float(value) for value in values]
    return {
        "count": float(len(ordered)),
        "mean": float(sum(ordered) / len(ordered)),
        "median": float(_percentile(ordered, 0.5)),
        "p90": float(_percentile(ordered, 0.9)),
        "min": float(min(ordered)),
        "max": float(max(ordered)),
    }


def _summarize_compression_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics: Dict[str, List[float]] = {}
    docs_with_efficiency = 0
    metric_fields = (
        "projector_stage_input_tokens",
        "projector_stage_output_tokens",
        "projector_stage_keep_ratio",
        "projector_compression_time_ms",
        "num_voxels_before_post",
        "input_tokens_before_post",
        "nominal_target_tokens",
        "budget_limited_by_voxel_count",
        "target_tokens",
        "dominant_ratio",
        "requested_dominant_tokens",
        "requested_contextual_tokens",
        "dominant_tokens",
        "contextual_tokens",
        "num_residual_merged",
        "llm_stage_input_tokens",
        "llm_stage_output_tokens",
        "llm_stage_keep_ratio",
        "late_entry_layer",
        "early_exit_layer",
        "active_visual_layers",
        "llm_prune_layer",
        "prefill_final_visual_tokens",
        "kv_cache_mb",
        "ttft_ms",
        "ttft_wall_ms",
        "ttft_cuda_ms",
        "e2e_wall_ms",
        "e2e_cuda_ms",
        "tpot_wall_ms",
        "tpot_cuda_ms",
        "decode_cuda_ms",
        "multimodal_prepare_time_ms",
        "post_prepare_ttft_wall_ms",
        "llm_prefill_cuda_ms",
        "llm_prefill_wall_ms",
        "llm_prefill_sequence_length",
        "llm_decode_cuda_ms",
        "llm_decode_forward_count",
        "llm_forward_count",
        "peak_memory_allocated_mb",
        "peak_memory_reserved_mb",
        "peak_memory_allocated_delta_mb",
        "peak_memory_reserved_delta_mb",
        "tflops",
    )

    for sample in samples:
        efficiency = _extract_compression_efficiency(sample)
        if not isinstance(efficiency, dict):
            continue
        docs_with_efficiency += 1

        for field in metric_fields:
            value = efficiency.get(field)
            if isinstance(value, (int, float)):
                metrics.setdefault(field, []).append(float(value))

    summary: Dict[str, Any] = {
        "docs": float(len(samples)),
        "docs_with_compression_efficiency": float(docs_with_efficiency),
    }

    for field, values in metrics.items():
        field_summary = _summarize_numeric(values)
        if field_summary is not None:
            summary[field] = field_summary

    return summary


def build_compression_efficiency_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    samples_by_task = results.get("samples")
    if not isinstance(samples_by_task, dict) or not samples_by_task:
        return {}

    by_task: Dict[str, Any] = {}
    overall_samples: List[Dict[str, Any]] = []

    for task_name, task_samples in samples_by_task.items():
        if not isinstance(task_samples, list):
            continue
        by_task[task_name] = _summarize_compression_samples(task_samples)
        overall_samples.extend(task_samples)

    overall = _summarize_compression_samples(overall_samples)
    if overall.get("docs_with_compression_efficiency", 0.0) == 0.0:
        return {}

    return {
        "by_task": by_task,
        "overall": overall,
    }
