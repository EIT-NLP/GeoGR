from __future__ import annotations

import ast
import json
import re
from functools import lru_cache
from typing import Iterable, List, Sequence


def _is_number(value) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def normalize_box(box) -> list[float] | None:
    if box is None:
        return None
    if isinstance(box, str):
        box = parse_single_box(box)
    if not isinstance(box, (list, tuple)) or len(box) != 6:
        return None
    if not all(_is_number(item) for item in box):
        return None
    return [float(item) for item in box]


def normalize_boxes(boxes) -> list[list[float]]:
    if boxes is None:
        return []
    if isinstance(boxes, str):
        boxes = parse_box_list(boxes)
    if isinstance(boxes, (list, tuple)) and len(boxes) == 6 and all(_is_number(item) for item in boxes):
        single = normalize_box(boxes)
        return [single] if single is not None else []
    results = []
    if isinstance(boxes, (list, tuple)):
        for box in boxes:
            normalized = normalize_box(box)
            if normalized is not None:
                results.append(normalized)
    return results


def center_size_to_bounds(box: Sequence[float]) -> tuple[list[float], list[float]]:
    cx, cy, cz, dx, dy, dz = [float(value) for value in box]
    half = [dx / 2.0, dy / 2.0, dz / 2.0]
    mins = [cx - half[0], cy - half[1], cz - half[2]]
    maxs = [cx + half[0], cy + half[1], cz + half[2]]
    return mins, maxs


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    mins_a, maxs_a = center_size_to_bounds(box_a)
    mins_b, maxs_b = center_size_to_bounds(box_b)
    inter = 1.0
    for axis in range(3):
        overlap = max(0.0, min(maxs_a[axis], maxs_b[axis]) - max(mins_a[axis], mins_b[axis]))
        inter *= overlap
    if inter <= 0:
        return 0.0
    volume_a = max(0.0, (maxs_a[0] - mins_a[0])) * max(0.0, (maxs_a[1] - mins_a[1])) * max(0.0, (maxs_a[2] - mins_a[2]))
    volume_b = max(0.0, (maxs_b[0] - mins_b[0])) * max(0.0, (maxs_b[1] - mins_b[1])) * max(0.0, (maxs_b[2] - mins_b[2]))
    union = volume_a + volume_b - inter
    return inter / union if union > 0 else 0.0


def best_iou(box: Sequence[float], candidates: Iterable[Sequence[float]]) -> tuple[float, list[float] | None]:
    best_score = 0.0
    best_box = None
    for candidate in candidates:
        score = box_iou(box, candidate)
        if score > best_score:
            best_score = score
            best_box = list(candidate)
    return best_score, best_box


def _parse_pythonish(text: str):
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(text)
        except Exception:
            continue
    return None


def parse_single_box(value) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 6:
        return normalize_box(value)
    if not isinstance(value, str):
        return None
    parsed = _parse_pythonish(value)
    if parsed is not None:
        normalized = normalize_box(parsed)
        if normalized is not None:
            return normalized
        nested = normalize_boxes(parsed)
        return nested[0] if nested else None

    numbers = re.findall(r"-?\d+(?:\.\d+)?", value)
    if len(numbers) >= 6:
        return [float(item) for item in numbers[:6]]
    return None


def parse_box_list(value) -> list[list[float]]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return normalize_boxes(value)
    if not isinstance(value, str):
        return []
    parsed = _parse_pythonish(value)
    if parsed is not None:
        return normalize_boxes(parsed)

    blocks = re.findall(r"\[[^\[\]]+\]", value)
    parsed_boxes = []
    for block in blocks:
        parsed_box = parse_single_box(block)
        if parsed_box is not None:
            parsed_boxes.append(parsed_box)
    if parsed_boxes:
        return parsed_boxes

    single = parse_single_box(value)
    return [single] if single is not None else []


@lru_cache(maxsize=None)
def _all_assignments(num_rows: int, num_cols: int):
    if num_rows == 0 or num_cols == 0:
        return [tuple()]
    if num_rows > num_cols:
        raise ValueError("num_rows must be <= num_cols for assignment generation.")
    if num_rows > 8 or num_cols > 8:
        raise ValueError("Brute-force assignment generation is only allowed for matrices up to 8x8.")
    if num_rows == 1:
        return [(col,) for col in range(num_cols)]

    assignments = []
    for col in range(num_cols):
        for suffix in _all_assignments(num_rows - 1, num_cols - 1):
            adjusted = []
            for idx in suffix:
                adjusted.append(idx + 1 if idx >= col else idx)
            assignments.append((col, *adjusted))
    return assignments


def _hungarian_min_cost(cost_matrix: list[list[float]]) -> list[int]:
    """Return one minimum-cost column assignment per row."""
    num_rows = len(cost_matrix)
    num_cols = len(cost_matrix[0]) if cost_matrix else 0
    if num_rows == 0 or num_cols == 0:
        return []
    if num_rows > num_cols:
        raise ValueError("Hungarian solver expects num_rows <= num_cols.")

    u = [0.0] * (num_rows + 1)
    v = [0.0] * (num_cols + 1)
    p = [0] * (num_cols + 1)
    way = [0] * (num_cols + 1)

    for row in range(1, num_rows + 1):
        p[0] = row
        col0 = 0
        minv = [float("inf")] * (num_cols + 1)
        used = [False] * (num_cols + 1)

        while True:
            used[col0] = True
            row0 = p[col0]
            delta = float("inf")
            col1 = 0

            for col in range(1, num_cols + 1):
                if used[col]:
                    continue
                cur = cost_matrix[row0 - 1][col - 1] - u[row0] - v[col]
                if cur < minv[col]:
                    minv[col] = cur
                    way[col] = col0
                if minv[col] < delta:
                    delta = minv[col]
                    col1 = col

            for col in range(num_cols + 1):
                if used[col]:
                    u[p[col]] += delta
                    v[col] -= delta
                else:
                    minv[col] -= delta

            col0 = col1
            if p[col0] == 0:
                break

        while True:
            col1 = way[col0]
            p[col0] = p[col1]
            col0 = col1
            if col0 == 0:
                break

    assignment = [-1] * num_rows
    for col in range(1, num_cols + 1):
        if p[col] != 0:
            assignment[p[col] - 1] = col - 1
    return assignment


def max_weight_assignment(matrix: list[list[float]]) -> list[tuple[int, int]]:
    if not matrix or not matrix[0]:
        return []
    rows = len(matrix)
    cols = len(matrix[0])
    square_size = max(rows, cols)
    cost = [[0.0] * square_size for _ in range(square_size)]
    for row in range(rows):
        for col in range(cols):
            cost[row][col] = -float(matrix[row][col])

    assignment = _hungarian_min_cost(cost)
    pairs = []
    for row, col in enumerate(assignment):
        if row < rows and col < cols:
            pairs.append((row, col))
    return pairs
