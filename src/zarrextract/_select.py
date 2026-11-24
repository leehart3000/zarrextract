"""Turn selections on a dimension into a contiguous index range."""

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

from zarrextract._errors import (
    AntimeridianError,
    EmptySelectionError,
    SelectionError,
)

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


def select(axis: Axis, selection: Selection) -> slice:
    """Return the slice of ``axis`` picked by ``selection``.

    A scalar picks the nearest coordinate value, as a slice of length 1. A
    ``(start, stop)`` pair picks every value from start to stop, inclusive.
    """
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
    raise SelectionError(
        f"Cannot select on dimension {name!r}: its coordinate values are not "
        "strictly increasing or decreasing."
    )


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


def _to_ticks(axis: Axis, value: object) -> int:
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
    unit, _ = np.datetime_data(axis.values.dtype)
    return int(moment.astype(f"datetime64[{unit}]").astype(np.int64))


def _to_number(axis: Axis, value: object) -> float:
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


def _to_store_lon(coords: NDArray[Any], value: float) -> float:
    """Map a longitude to the convention of the store (0..360 or -180..180)."""
    if float(coords.max()) > 180:
        return value if 0 <= value <= 360 else value % 360
    return value if -180 <= value <= 180 else (value + 180) % 360 - 180


def _numeric_longitude(axis: Axis) -> bool:
    return axis.is_longitude and not axis.is_time


def _select_range(axis: Axis, coords: NDArray[Any], start: float, stop: float) -> slice:
    if _numeric_longitude(axis):
        if stop - start >= 360:
            return slice(0, coords.size)
        raw = (start, stop)
        start, stop = _to_store_lon(coords, start), _to_store_lon(coords, stop)
        if start > stop:
            raise AntimeridianError(
                f"The longitude range {raw[0]} to {raw[1]} wraps around the edge "
                f"of the store's longitude axis ({coords.min()} to {coords.max()}), "
                "so it cannot be read as one block. Request the two parts "
                "separately."
            )
    else:
        start, stop = min(start, stop), max(start, stop)

    slack = _tolerance(coords)
    indices = np.flatnonzero((coords >= start - slack) & (coords <= stop + slack))
    if indices.size == 0:
        raise EmptySelectionError(
            f"No values of {axis.name!r} lie between {_show(axis, start)} and "
            f"{_show(axis, stop)}. The values run from {_show(axis, coords.min())} "
            f"to {_show(axis, coords.max())}."
        )
    return slice(int(indices[0]), int(indices[-1]) + 1)


def _select_nearest(
    axis: Axis, coords: NDArray[Any], value: float, increasing: bool
) -> slice:
    longitude = _numeric_longitude(axis)
    if longitude:
        value = _to_store_lon(coords, value)
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
    axis: Axis, value: float, low: Any, high: Any, step: Any
) -> SelectionError:
    step_text = f"{int(step)} {_time_unit(axis)}" if axis.is_time else str(step)
    return SelectionError(
        f"{_show(axis, value)} is outside the range of {axis.name!r} "
        f"({_show(axis, low)} to {_show(axis, high)}) by more than one grid "
        f"step ({step_text})."
    )


def _time_unit(axis: Axis) -> str:
    return np.datetime_data(axis.values.dtype)[0]


def _show(axis: Axis, value: Any) -> str:
    """Format a coordinate value for an error message."""
    if axis.is_time:
        ticks = np.array(int(value), dtype=np.int64)
        return str(ticks.astype(f"datetime64[{_time_unit(axis)}]"))
    return str(value)
