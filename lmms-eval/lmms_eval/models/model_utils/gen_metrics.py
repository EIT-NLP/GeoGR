import time
from numbers import Number
from typing import Any, Callable, Dict, List, Optional

import torch
from loguru import logger as eval_logger
from transformers.generation.streamers import BaseStreamer

_THROUGHPUT_METRICS_HISTORY: List[Dict[str, Any]] = []


def reset_logged_metrics() -> None:
    """Reset in-memory throughput metrics captured during a run."""

    _THROUGHPUT_METRICS_HISTORY.clear()


def get_logged_metrics_history() -> List[Dict[str, Any]]:
    """Return captured throughput metrics in collection order."""

    return list(_THROUGHPUT_METRICS_HISTORY)


def summarize_logged_metrics() -> Dict[str, Any]:
    """Aggregate captured throughput metrics for final reporting."""

    if not _THROUGHPUT_METRICS_HISTORY:
        return {}

    total_gen_tokens = 0.0
    total_elapsed_time = 0.0
    total_requests = 0.0
    avg_speed_vals: List[float] = []
    additional_numeric: Dict[str, List[float]] = {}

    for metric in _THROUGHPUT_METRICS_HISTORY:
        token_val = metric.get("total_gen_tokens")
        latency_val = metric.get("total_elapsed_time")
        speed_val = metric.get("avg_speed")
        requests_val = metric.get("total_requests")
        if requests_val is None:
            requests_val = metric.get("request_count")
        if requests_val is None:
            requests_val = metric.get("num_requests")

        if isinstance(token_val, Number):
            total_gen_tokens += float(token_val)
        if isinstance(latency_val, Number):
            total_elapsed_time += float(latency_val)
        if isinstance(speed_val, Number):
            avg_speed_vals.append(float(speed_val))
        if isinstance(requests_val, Number):
            total_requests += float(requests_val)

        for key, value in metric.items():
            if key in {"total_gen_tokens", "total_elapsed_time", "avg_speed", "total_requests", "request_count", "num_requests"}:
                continue
            if isinstance(value, Number):
                additional_numeric.setdefault(key, []).append(float(value))

    summary: Dict[str, Any] = {
        "total_gen_tokens": int(total_gen_tokens) if total_gen_tokens.is_integer() else total_gen_tokens,
        "total_elapsed_time": total_elapsed_time,
        "avg_speed": (total_gen_tokens / total_elapsed_time) if total_elapsed_time > 0 else (sum(avg_speed_vals) / len(avg_speed_vals) if avg_speed_vals else 0.0),
    }

    if total_requests > 0:
        summary["avg_latency"] = total_elapsed_time / total_requests

    for key, values in additional_numeric.items():
        if values:
            summary[f"avg_{key}"] = sum(values) / len(values)

    return summary


def _record_metrics(
    total_elapsed_time: float,
    total_gen_tokens: int,
    avg_speed: float,
    additional_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "total_elapsed_time": total_elapsed_time,
        "total_gen_tokens": total_gen_tokens,
        "avg_speed": avg_speed,
    }
    if additional_metrics:
        payload.update(additional_metrics)
    _THROUGHPUT_METRICS_HISTORY.append(payload)


def space_tokenizer(text: str) -> float:
    """
    A simple tokenizer that counts the token by the splitted space
    Then a rough estimate of the token count is returned. (times 1.5)
    Args:
        text (str): The input text to tokenize.
    """
    return len(text.split(" ")) * 1.5


def calculate_token_throughput(token_count: float, duration: float) -> float:
    """
    Calculate the token throughput.

    Args:
        token_count (float): The number of tokens processed.
        duration (float): The time taken to process the tokens in seconds.

    Returns:
        float: The token throughput in tokens per second.
    """
    if duration <= 0:
        return 0.0
    return token_count / duration


def log_metrics(
    total_elapsed_time: float,
    total_gen_tokens: int,
    avg_speed: float,
    additional_metrics: Optional[Dict[str, Any]] = None,
):
    """
    Log the metrics in a structured format.

    Args:
        total_elapsed_time (float): Sum of generation latencies in seconds.
        total_gen_tokens (int): The total number of generated tokens.
        avg_speed (float): The average speed in tokens per second.
        additional_metrics (Dict[str, Any]): Additional metrics to log.
    """
    required_stats = f"Metric summary - Total elapsed time: {total_elapsed_time:.3f}s, Total gen tokens: {total_gen_tokens}, Avg speed: {avg_speed:.1f} tokens/s"
    if additional_metrics is not None:
        required_stats += ", Additional metrics: "
        required_stats += ", ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in additional_metrics.items())
    eval_logger.info(required_stats)
    _record_metrics(
        total_elapsed_time=total_elapsed_time,
        total_gen_tokens=total_gen_tokens,
        avg_speed=avg_speed,
        additional_metrics=additional_metrics,
    )


class GenMetrics:
    """
    A class to manage the generation of metrics for model evaluation.
    """

    def __init__(self, tokenize_fn: Callable = space_tokenizer):
        self.tokenize_fn = tokenize_fn

    def __enter__(self):
        """
        Initialize the context manager.
        """
        self.metrics = {}
        self.start_time = time.perf_counter()
        return self

    def stop_timer(self):
        self.end_time = time.perf_counter()

    def log_metric(self, content: List[Any], additional_metrics: Optional[Dict[str, Any]] = None):
        num_tokens = sum(self.tokenize_fn(item) for item in content)
        duration = self.end_time - self.start_time
        throughput = calculate_token_throughput(num_tokens, duration)
        self.metrics = {
            "num_tokens": num_tokens,
            "duration": duration,
            "throughput": throughput,
        }
        if additional_metrics:
            self.metrics.update(additional_metrics)

        log_metrics(
            total_elapsed_time=duration,
            total_gen_tokens=int(num_tokens),
            avg_speed=throughput,
            additional_metrics=additional_metrics,
        )

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Finalize the context manager and return the collected metrics.
        """
        self.end_time = time.perf_counter()
        self.metrics["duration"] = self.end_time - self.start_time
        return self.metrics


class LatencyStreamer(BaseStreamer):
    def __init__(self, device: Optional[torch.device] = None):
        self.device = torch.device(device) if device is not None else None
        self.start_wall_time: Optional[float] = None
        self.first_token_wall_time: Optional[float] = None
        self.end_wall_time: Optional[float] = None
        self.generated_tokens = 0
        self.token_wall_times: List[float] = []
        # Transformers sends the prompt to the streamer once before entering
        # the autoregressive loop.  With inputs_embeds, the synthetic
        # input_ids can have length one, so checking numel() alone cannot
        # distinguish that prompt callback from a generated token.
        self._prompt_callback_skipped = False

        self._use_cuda_events = self.device is not None and self.device.type == "cuda" and torch.cuda.is_available()
        self._start_event: Optional[torch.cuda.Event] = None
        self._first_token_event: Optional[torch.cuda.Event] = None
        self._last_token_event: Optional[torch.cuda.Event] = None
        self._end_event: Optional[torch.cuda.Event] = None

        if self._use_cuda_events:
            self._start_event = torch.cuda.Event(enable_timing=True)

    def start(self):
        self._prompt_callback_skipped = False
        self.generated_tokens = 0
        self.token_wall_times = []
        if self._use_cuda_events:
            torch.cuda.synchronize(self.device)
            self._start_event.record(torch.cuda.current_stream(self.device))
        self.start_wall_time = time.perf_counter()

    def put(self, value):
        if not self._prompt_callback_skipped:
            self._prompt_callback_skipped = True
            return
        if isinstance(value, torch.Tensor) and value.numel() > 1:
            return

        timestamp = time.perf_counter()
        self.generated_tokens += 1
        self.token_wall_times.append(timestamp)
        if self.first_token_wall_time is None:
            self.first_token_wall_time = timestamp

        if self._use_cuda_events:
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device))
            if self._first_token_event is None:
                self._first_token_event = event
            self._last_token_event = event

    def end(self):
        if self._use_cuda_events:
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._end_event.record(torch.cuda.current_stream(self.device))
            torch.cuda.synchronize(self.device)
        self.end_wall_time = time.perf_counter()

    def get_metrics(self) -> Optional[Dict[str, Any]]:
        if self.start_wall_time is None or self.end_wall_time is None:
            return None

        metrics: Dict[str, Any] = {
            "generated_tokens": int(self.generated_tokens),
            "e2e_wall_ms": (self.end_wall_time - self.start_wall_time) * 1000.0,
        }

        if self.first_token_wall_time is not None:
            metrics["ttft_wall_ms"] = (self.first_token_wall_time - self.start_wall_time) * 1000.0
        else:
            metrics["ttft_wall_ms"] = None

        if self.generated_tokens >= 2 and len(self.token_wall_times) >= 2:
            decode_wall = self.token_wall_times[-1] - self.token_wall_times[0]
            metrics["tpot_wall_ms"] = (decode_wall / (self.generated_tokens - 1)) * 1000.0 if decode_wall > 0 else 0.0
        else:
            metrics["tpot_wall_ms"] = 0.0

        if self._use_cuda_events and self._start_event is not None and self._end_event is not None:
            metrics["e2e_cuda_ms"] = float(self._start_event.elapsed_time(self._end_event))
            if self._first_token_event is not None:
                metrics["ttft_cuda_ms"] = float(self._start_event.elapsed_time(self._first_token_event))
            else:
                metrics["ttft_cuda_ms"] = None

            if self.generated_tokens >= 2 and self._first_token_event is not None and self._last_token_event is not None:
                decode_cuda_ms = float(self._first_token_event.elapsed_time(self._last_token_event))
                metrics["decode_cuda_ms"] = decode_cuda_ms
                metrics["tpot_cuda_ms"] = decode_cuda_ms / float(self.generated_tokens - 1) if self.generated_tokens > 1 else 0.0
            else:
                metrics["decode_cuda_ms"] = 0.0
                metrics["tpot_cuda_ms"] = 0.0
        else:
            metrics["e2e_cuda_ms"] = None
            metrics["ttft_cuda_ms"] = None
            metrics["decode_cuda_ms"] = None
            metrics["tpot_cuda_ms"] = None

        return metrics
