"""Turn selections on a dimension into a contiguous index range."""

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

from zarrextract._errors import EmptySelectionError, SelectionError
from zarrextract._probe import LazyAxis, not_monotonic_error

Scalar = int | float | np.number[Any] | str | date | np.datetime64
"""A single coordinate value: a number, an ISO string, a date or a datetime64."""

Selection = Scalar | tuple[Scalar, Scalar]
"""A scalar (nearest value) or a ``(start, stop)`` pair (inclusive range)."""

_UTC_SUFFIX = re.compile(r"(?:Z|[+-]00(?::?00)?)$", re.I)
_OFFSET_SUFFIX = re.compile(r"[+-]\d{2}(?::?\d{2})?$")

# Relative slack when comparing floating-point coordinates.
_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class Axis:
    """The coordinate values of one dimension."""

    name: str
    values: NDArray[Any]
    """Decoded values: ``datetime64`` for time, otherwise as stored."""
    is_time: bool = False
    is_longitude: bool = False

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.values.dtype

    @property
    def size(self) -> int:
        return len(self.values)

    def read(self, selected: slice) -> NDArray[Any]:
        """Return the decoded values in a slice."""
        return self.values[selected]


Indices = slice | tuple[slice, slice]
"""One slice, or two for a longitude range that wraps around the seam."""


def select(axis: Axis | LazyAxis, selection: Selection) -> Indices:
    """Return the slice of ``axis`` picked by ``selection``.

    A scalar picks the nearest coordinate value, as a slice of length 1. A
    ``(start, stop)`` pair picks every value from start to stop, inclusive.
    A longitude range that wraps around the edge of the axis gives two slices:
    first the one holding ``start``, then the one holding ``stop``.
    A ``LazyAxis`` is searched by bisection and gives the same answer.
    """
    if isinstance(axis, LazyAxis):
        return _select_lazy(axis, selection)
    coords = _numeric(axis)
    if coords.size == 0:
        raise EmptySelectionError(f"Dimension {axis.name!r} has no values.")
    increasing = _direction(axis.name, coords)
    if isinstance(selection, tuple | list):
        if len(selection) != 2:
            raise SelectionError(
                f"The selection for {axis.name!r} must be a single value or a "
                f"(start, stop) pair, not {len(selection)} values."
            )
        start, stop = (_to_number(axis, v) for v in selection)
        return _select_range(axis, coords, start, stop)
    return _select_nearest(axis, coords, _to_number(axis, selection), increasing)


def _numeric(axis: Axis) -> NDArray[Any]:
    values = axis.values
    if axis.is_time:
        if np.isnat(values).any():
            raise SelectionError(f"Time dimension {axis.name!r} contains NaT.")
        return values.astype(np.int64)
    if values.dtype.kind in "iu":
        return values.astype(np.int64)
    if values.dtype.kind == "f":
        return values.astype(np.float64)
    raise SelectionError(
        f"Cannot select on dimension {axis.name!r}: its values have dtype "
        f"{values.dtype}, which is not numeric or time."
    )


def _direction(name: str, coords: NDArray[Any]) -> bool:
    """Return True if increasing, False if decreasing; raise if neither."""
    if coords.size < 2:
        return True
    steps = np.diff(coords)
    if (steps > 0).all():
        return True
    if (steps < 0).all():
        return False
    raise not_monotonic_error(name)


def _parse_iso(text: str) -> np.datetime64:
    cleaned = text.strip()
    if "T" in cleaned.upper() or " " in cleaned:
        if _UTC_SUFFIX.search(cleaned):
            cleaned = _UTC_SUFFIX.sub("", cleaned).rstrip()
        elif _OFFSET_SUFFIX.search(cleaned):
            raise SelectionError(
                f"The time {text!r} has a UTC offset. Only UTC (Z or +00:00) is "
                "supported."
            )
    try:
        return np.datetime64(cleaned)
    except ValueError as exc:
        raise SelectionError(f"Cannot read {text!r} as a date or time: {exc}") from exc


def _to_ticks(axis: Axis | LazyAxis, value: object) -> int:
    """Convert a time selection to integer ticks in the axis resolution."""
    moment: np.datetime64
    if isinstance(value, np.datetime64):
        moment = value
    elif isinstance(value, str):
        moment = _parse_iso(value)
    elif isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(UTC).replace(tzinfo=None)
        moment = np.datetime64(value)
    elif isinstance(value, date):
        moment = np.datetime64(value)
    else:
        raise SelectionError(
            f"Dimension {axis.name!r} is a time dimension, so its selection must be "
            f"an ISO string, a datetime or a datetime64, not {value!r}."
        )
    if np.isnat(moment):
        raise SelectionError(f"The selection for {axis.name!r} is not a valid time.")
    unit, _ = np.datetime_data(axis.dtype)
    return int(moment.astype(f"datetime64[{unit}]").astype(np.int64))


def _to_number(axis: Axis | LazyAxis, value: object) -> float:
    if axis.is_time:
        return _to_ticks(axis, value)
    if isinstance(value, bool) or not isinstance(value, int | float | np.number):
        raise SelectionError(
            f"The selection for {axis.name!r} must be a number, not {value!r}."
        )
    return float(value)


def _tolerance(coords: NDArray[Any]) -> float:
    if coords.dtype.kind != "f" or coords.size < 2:
        return 0.0
    return _TOLERANCE * float(np.median(np.abs(np.diff(coords))))


def _to_store_lon(highest: Any, value: float) -> float:
    """Map a longitude to the convention of the store (0..360 or -180..180)."""
    if float(highest) > 180:
        return value if 0 <= value <= 360 else value % 360
    return value if -180 <= value <= 180 else (value + 180) % 360 - 180


def _numeric_longitude(axis: Axis | LazyAxis) -> bool:
    return axis.is_longitude and not axis.is_time


def _select_range(
    axis: Axis, coords: NDArray[Any], start: float, stop: float
) -> Indices:
    if _numeric_longitude(axis):
        if stop - start >= 360:
            return slice(0, coords.size)
        highest = coords.max()
        start, stop = _to_store_lon(highest, start), _to_store_lon(highest, stop)
        wraps = start > stop
    else:
        start, stop = min(start, stop), max(start, stop)
        wraps = False

    slack = _tolerance(coords)
    if wraps:
        # Values from start to the end of the axis, then from its other end to stop.
        masks = [coords >= start - slack, coords <= stop + slack]
    else:
        masks = [(coords >= start - slack) & (coords <= stop + slack)]
    runs: list[slice] = []
    for mask in masks:
        indices = np.flatnonzero(mask)
        if indices.size:
            runs.append(slice(int(indices[0]), int(indices[-1]) + 1))
    return _join_runs(axis, runs, start, stop, coords.min(), coords.max())


def _join_runs(
    axis: Axis | LazyAxis, runs: list[slice], start: Any, stop: Any, low: Any, high: Any
) -> Indices:
    """Return one slice, or two for a range across the longitude seam."""
    if not runs:
        raise EmptySelectionError(
            f"No values of {axis.name!r} lie between {_show(axis, start)} and "
            f"{_show(axis, stop)}. The values run from {_show(axis, low)} "
            f"to {_show(axis, high)}."
        )
    if len(runs) == 1:
        return runs[0]
    return runs[0], runs[1]


def _select_nearest(
    axis: Axis, coords: NDArray[Any], value: float, increasing: bool
) -> slice:
    longitude = _numeric_longitude(axis)
    if longitude:
        value = _to_store_lon(coords.max(), value)
    low, high = coords.min(), coords.max()

    if coords.size == 1:
        if value != coords[0]:
            raise SelectionError(
                f"{_show(axis, value)} is not within one grid step of "
                f"{axis.name!r}, which has the single value "
                f"{_show(axis, coords[0])}."
            )
        return slice(0, 1)

    step_at_start = abs(coords[1] - coords[0])
    step_at_end = abs(coords[-1] - coords[-2])
    step_low, step_high = (
        (step_at_start, step_at_end) if increasing else (step_at_end, step_at_start)
    )

    if longitude and high - low + step_high >= 360 * (1 - _TOLERANCE):
        # A global grid wraps around, so the nearest value may be across the seam.
        distance = np.abs((coords - value + 180.0) % 360.0 - 180.0)
    else:
        if value < low and low - value > step_low * (1 + _TOLERANCE):
            raise _outside(axis, value, low, high, step_low)
        if value > high and value - high > step_high * (1 + _TOLERANCE):
            raise _outside(axis, value, low, high, step_high)
        distance = np.abs(coords - value)
    nearest = int(np.argmin(distance))
    return slice(nearest, nearest + 1)


def _outside(
    axis: Axis | LazyAxis, value: float, low: Any, high: Any, step: Any
) -> SelectionError:
    step_text = f"{int(step)} {_time_unit(axis)}" if axis.is_time else str(step)
    return SelectionError(
        f"{_show(axis, value)} is outside the range of {axis.name!r} "
        f"({_show(axis, low)} to {_show(axis, high)}) by more than one grid "
        f"step ({step_text})."
    )


def _time_unit(axis: Axis | LazyAxis) -> str:
    return np.datetime_data(axis.dtype)[0]


def _show(axis: Axis | LazyAxis, value: Any) -> str:
    """Format a coordinate value for an error message."""
    if axis.is_time:
        ticks = np.array(int(value), dtype=np.int64)
        return str(ticks.astype(f"datetime64[{_time_unit(axis)}]"))
    return str(value)


# --- selection by bisection ---------------------------------------------------
# These mirror the whole-read functions above, but look at single values of a
# LazyAxis. Keep the two in step: the tests compare their answers.


def _first_true(test: Callable[[int], bool], low: int, high: int) -> int:
    """Return the first index in ``low..high`` where ``test`` is true.

    ``test`` must be false and then true along the axis. ``high`` is returned
    if it is never true.
    """
    while low < high:
        middle = (low + high) // 2
        if test(middle):
            high = middle
        else:
            low = middle + 1
    return low


def _select_lazy(axis: LazyAxis, selection: Selection) -> Indices:
    last = axis.size - 1
    first, second = axis.value(0), axis.value(1)
    before_last, end = axis.value(last - 1), axis.value(last)
    if first < second and before_last < end and first < end:
        increasing = True
    elif first > second and before_last > end and first > end:
        increasing = False
    else:
        raise not_monotonic_error(axis.name)
    low, high = (first, end) if increasing else (end, first)
    step_at_start, step_at_end = abs(second - first), abs(end - before_last)

    if isinstance(selection, tuple | list):
        if len(selection) != 2:
            raise SelectionError(
                f"The selection for {axis.name!r} must be a single value or a "
                f"(start, stop) pair, not {len(selection)} values."
            )
        start, stop = (_to_number(axis, v) for v in selection)
        tolerance = (
            _TOLERANCE * float(min(step_at_start, step_at_end))
            if low.dtype.kind == "f"
            else 0.0
        )
        return _bisect_range(axis, increasing, low, high, tolerance, start, stop)
    value = _to_number(axis, selection)
    steps = (step_at_start, step_at_end) if increasing else (step_at_end, step_at_start)
    return _bisect_nearest(axis, increasing, low, high, steps, value)


def _bisect_range(
    axis: LazyAxis,
    increasing: bool,
    low: Any,
    high: Any,
    tolerance: float,
    start: float,
    stop: float,
) -> Indices:
    if _numeric_longitude(axis):
        if stop - start >= 360:
            return slice(0, axis.size)
        start, stop = _to_store_lon(high, start), _to_store_lon(high, stop)
        wraps = start > stop
    else:
        start, stop = min(start, stop), max(start, stop)
        wraps = False

    bottom, top = start - tolerance, stop + tolerance
    bounds = [(bottom, math.inf), (-math.inf, top)] if wraps else [(bottom, top)]
    runs = [
        run
        for lower, upper in bounds
        if (run := _bisect_run(axis, increasing, lower, upper)) is not None
    ]
    return _join_runs(axis, runs, start, stop, low, high)


def _bisect_run(
    axis: LazyAxis, increasing: bool, bottom: float, top: float
) -> slice | None:
    """Return the slice of values from ``bottom`` to ``top``, or None if empty."""
    if increasing:
        begin = _first_true(lambda i: axis.value(i) >= bottom, 0, axis.size)
        end = _first_true(lambda i: axis.value(i) > top, begin, axis.size)
    else:
        begin = _first_true(lambda i: axis.value(i) <= top, 0, axis.size)
        end = _first_true(lambda i: axis.value(i) < bottom, begin, axis.size)
    return slice(begin, end) if begin < end else None


def _bisect_nearest(
    axis: LazyAxis,
    increasing: bool,
    low: Any,
    high: Any,
    steps: tuple[Any, Any],
    value: float,
) -> slice:
    longitude = _numeric_longitude(axis)
    if longitude:
        value = _to_store_lon(high, value)
    step_low, step_high = steps

    wraps = longitude and high - low + step_high >= 360 * (1 - _TOLERANCE)
    if not wraps:
        if value < low and low - value > step_low * (1 + _TOLERANCE):
            raise _outside(axis, value, low, high, step_low)
        if value > high and value - high > step_high * (1 + _TOLERANCE):
            raise _outside(axis, value, low, high, step_high)

    def distance(i: int) -> Any:
        if wraps:
            return np.abs((axis.value(i) - value + 180.0) % 360.0 - 180.0)
        return np.abs(axis.value(i) - value)

    if increasing:
        after = _first_true(lambda i: axis.value(i) >= value, 0, axis.size)
    else:
        after = _first_true(lambda i: axis.value(i) <= value, 0, axis.size)
    candidates = {max(after - 1, 0), min(after, axis.size - 1)}
    if wraps:
        candidates |= {0, axis.size - 1}
    # Lowest index wins a tie, as with argmin on the whole coordinate.
    nearest = min(sorted(candidates), key=distance)
    return slice(nearest, nearest + 1)
