"""Shared window-grid helpers for full-coverage tiled inference."""

from __future__ import annotations


def window_step(window_size: int, overlap: float) -> int:
    if window_size <= 0:
        raise ValueError("window size must be a positive integer")
    if not 0 <= overlap < 1:
        raise ValueError(f"overlap must satisfy 0 <= overlap < 1, got {overlap}")

    step = int(window_size * (1 - overlap))
    if step <= 0:
        raise ValueError(
            f"window size {window_size} and overlap {overlap} produce a non-positive step"
        )

    return step


def axis_window_starts(length: int, window_size: int, overlap: float) -> list[int]:
    """Return full-size window starts that cover an axis through its last pixel."""
    if length < window_size:
        raise ValueError(f"axis length {length} is smaller than window size {window_size}")
    step = window_step(window_size, overlap)
    last_start = length - window_size
    starts = list(range(0, last_start + 1, step))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def window_positions(
    width: int,
    height: int,
    window_size: int,
    overlap: float,
) -> list[tuple[int, int]]:
    """Return row-major ``(x, y)`` starts for a full-coverage window grid."""

    x_starts = axis_window_starts(width, window_size, overlap)
    y_starts = axis_window_starts(height, window_size, overlap)
    return [(x, y) for y in y_starts for x in x_starts]
