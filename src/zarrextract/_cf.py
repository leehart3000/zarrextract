"""CF conventions: time decoding, packed data and fill values."""

import math
import re
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from zarrextract._errors import TimeDecodingError, UnsupportedCalendarError

# Each unit maps to (resolution of the decoded values, ticks per unit). The
# resolution follows the unit so that far-future dates still fit: datetime64[ns]
# stops at the year 2262, but CMIP6 scenarios run to 2300.
_UNIT_SPECS: dict[tuple[str, int], tuple[str, ...]] = {
    ("s", 86400): ("d", "day", "days"),
    ("s", 3600): ("h", "hr", "hrs", "hour", "hours"),
    ("s", 60): ("min", "mins", "minute", "minutes"),
    ("s", 1): ("s", "sec", "secs", "second", "seconds"),
    ("ms", 1): ("ms", "msec", "msecs", "millisecond", "milliseconds"),
    ("us", 1): ("us", "usec", "usecs", "microsecond", "microseconds"),
}
_UNITS: dict[str, tuple[str, int]] = {
    name: spec for spec, names in _UNIT_SPECS.items() for name in names
}

_UNITS_RE = re.compile(r"^\s*(?P<unit>[A-Za-z]+)\s+since\s+(?P<ref>.+?)\s*$", re.I)
_REFERENCE_RE = re.compile(
    r"""^(?P<year>\d{1,4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})
        (?:[ T]+(?P<hour>\d{1,2})
            (?::(?P<minute>\d{1,2})
                (?::(?P<second>\d{1,2})(?:\.(?P<fraction>\d+))?)?
            )?
        )?
        \s*(?P<tz>Z|[+-]\d{2}(?::?\d{2})?)?$""",
    re.I | re.VERBOSE,
)
_OFFSET_RE = re.compile(r"([+-])(\d{2})(?::?(\d{2}))?")

# Largest magnitude of a decoded offset, leaving room to add the reference date
# without overflowing int64.
_TICK_LIMIT = 2**62

_MIXED_CALENDARS = {"standard", "gregorian"}
_SUPPORTED_CALENDARS = _MIXED_CALENDARS | {"proleptic_gregorian"}
_GREGORIAN_START = np.datetime64("1582-10-15", "us")

_LONGITUDE_UNITS = {
    "degrees_east",
    "degree_east",
    "degrees_e",
    "degree_e",
    "degreese",
    "degreee",
}


def is_time_units(units: object) -> bool:
    """Return True if ``units`` looks like ``"<unit> since <date>"``."""
    return isinstance(units, str) and _UNITS_RE.match(units) is not None


def is_longitude(name: str, attrs: Mapping[str, Any]) -> bool:
    """Return True for a longitude coordinate, by attributes or common names."""
    if attrs.get("standard_name") == "longitude":
        return True
    units = attrs.get("units")
    if isinstance(units, str) and units.strip().lower() in _LONGITUDE_UNITS:
        return True
    return name.lower() in {"lon", "longitude"}


def _parse_reference(text: str) -> np.datetime64:
    match = _REFERENCE_RE.match(text)
    if match is None:
        raise TimeDecodingError(
            f"Cannot read the reference date {text!r}. Expected a date such as "
            "'1850-01-01' or '1900-01-01 00:00:00', optionally ending in Z or +00:00."
        )
    tz = match["tz"]
    if tz is not None and tz.upper() != "Z":
        offset = _OFFSET_RE.fullmatch(tz)
        if offset is None or any(int(part) for part in offset.groups("0")[1:]):
            raise TimeDecodingError(
                f"The reference date {text!r} has a UTC offset of {tz}. "
                "Only UTC (Z or +00:00) is supported."
            )
    fraction = (match["fraction"] or "")[:6].ljust(6, "0")
    iso = (
        f"{int(match['year']):04d}-{int(match['month']):02d}-{int(match['day']):02d}"
        f"T{int(match['hour'] or 0):02d}:{int(match['minute'] or 0):02d}"
        f":{int(match['second'] or 0):02d}.{fraction}"
    )
    try:
        return np.datetime64(iso, "us")
    except ValueError as exc:
        raise TimeDecodingError(f"Invalid reference date {text!r}: {exc}") from exc


def _parse_units(units: str) -> tuple[str, int, np.datetime64]:
    match = _UNITS_RE.match(units)
    if match is None:
        raise TimeDecodingError(
            f"Cannot read the time units {units!r}. Expected '<unit> since <date>'."
        )
    spec = _UNITS.get(match["unit"].lower())
    if spec is None:
        raise TimeDecodingError(
            f"Unsupported time unit {match['unit']!r} in {units!r}. Supported units "
            "are days, hours, minutes, seconds, milliseconds and microseconds."
        )
    return spec[0], spec[1], _parse_reference(match["ref"])


def _check_calendar(calendar: object) -> str:
    if calendar is None:
        return "standard"
    name = calendar.strip().lower() if isinstance(calendar, str) else None
    if name not in _SUPPORTED_CALENDARS:
        raise UnsupportedCalendarError(
            f"Unsupported calendar {calendar!r}. Supported calendars are standard, "
            "gregorian and proleptic_gregorian."
        )
    return name


def decode_time(
    values: NDArray[Any], units: str, calendar: object = None
) -> NDArray[np.datetime64]:
    """Decode CF time values to ``datetime64``.

    The resolution follows the unit: ``s`` for days, hours, minutes and seconds,
    ``ms`` for milliseconds and ``us`` for microseconds. Non-finite values
    become ``NaT``.
    """
    resolution, factor, reference = _parse_units(units)
    name = _check_calendar(calendar)
    mixed = name in _MIXED_CALENDARS
    if mixed and reference < _GREGORIAN_START:
        raise UnsupportedCalendarError(
            f"The reference date in {units!r} is before 1582-10-15, where the "
            f"{name!r} calendar differs from the proleptic Gregorian calendar. "
            "Use the proleptic_gregorian calendar, which is supported."
        )
    unit_type = f"datetime64[{resolution}]"
    reference_res = reference.astype(unit_type)
    if reference_res.astype("datetime64[us]") != reference:
        raise TimeDecodingError(
            f"The reference date in {units!r} is finer than the unit allows."
        )

    array = np.asarray(values)
    if array.dtype.kind not in "iuf":
        raise TimeDecodingError(
            f"Time values must be numbers, not dtype {array.dtype}."
        )
    finite: NDArray[np.bool_] | None = None
    if array.dtype.kind == "f":
        finite = np.isfinite(array)
        scaled = np.where(finite, array, 0.0).astype(np.float64) * factor
        if scaled.size and float(np.abs(scaled).max()) >= _TICK_LIMIT:
            raise TimeDecodingError(f"Time values in {units!r} are out of range.")
        ticks = np.rint(scaled).astype(np.int64)
    else:
        integers = array.astype(np.int64)
        if integers.size:
            largest = max(abs(int(integers.max())), abs(int(integers.min())))
            if largest * factor >= _TICK_LIMIT:
                raise TimeDecodingError(f"Time values in {units!r} are out of range.")
        ticks = integers * factor

    decoded: NDArray[np.datetime64] = reference_res + ticks.astype(
        f"timedelta64[{resolution}]"
    )
    if finite is not None:
        decoded[~finite] = np.array("NaT", dtype=unit_type)
    if mixed:
        valid = decoded[~np.isnat(decoded)]
        if valid.size and valid.min() < _GREGORIAN_START.astype(unit_type):
            raise UnsupportedCalendarError(
                f"Some times decoded from {units!r} fall before 1582-10-15, where "
                f"the {name!r} calendar differs from the proleptic Gregorian "
                "calendar."
            )
    return decoded


def _fill_values(attrs: Mapping[str, Any]) -> list[float]:
    found: list[float] = []
    for key in ("_FillValue", "missing_value"):
        value = attrs.get(key)
        if value is None:
            continue
        for item in value if isinstance(value, list | tuple) else [value]:
            try:
                found.append(float(item))
            except (TypeError, ValueError):
                continue
    return found


def unpack(
    raw: NDArray[Any], attrs: Mapping[str, Any]
) -> tuple[NDArray[Any], NDArray[np.bool_] | None]:
    """Apply CF ``scale_factor`` and ``add_offset``, and find missing values.

    Returns the values and a boolean mask that is True where the value is
    missing (equal to ``_FillValue`` or ``missing_value``), or ``None`` if no
    value is missing. The mask is computed on the stored values, before scaling.
    """
    mask: NDArray[np.bool_] | None = None
    for fill in _fill_values(attrs):
        if math.isnan(fill):
            hit = np.isnan(raw) if raw.dtype.kind == "f" else None
        else:
            hit = raw == fill
        if hit is not None:
            mask = hit if mask is None else mask | hit
    if mask is not None and not mask.any():
        mask = None

    scale = attrs.get("scale_factor")
    offset = attrs.get("add_offset")
    if scale is None and offset is None:
        return raw, mask
    values = raw.astype(np.float64)
    if scale is not None:
        values = values * float(scale)
    if offset is not None:
        values = values + float(offset)
    return values, mask
