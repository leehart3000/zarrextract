"""CF calendars whose dates are not all Gregorian dates: 360_day and all_leap.

Dates in these calendars, such as 30 February, cannot be held by ``datetime64``
or an Arrow timestamp. A time coordinate in one of them is therefore kept as
integer *ticks*: the number of seconds, milliseconds or microseconds (the
resolution that follows from the time units) since 0000-01-01T00:00:00 in that
calendar. Ticks increase with time, so selection works on them as on any other
integer coordinate. A ``Clock`` converts between ticks, dates and ISO strings.
"""

import re
from collections.abc import Sequence
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from zarrextract._errors import SelectionError, ZarrExtractError

CLOCK_CALENDARS = frozenset({"360_day", "all_leap", "366_day"})
"""The calendars handled by a ``Clock``, as lowercase names."""

MISSING = np.iinfo(np.int64).min
"""The tick that stands for a missing time (the value numpy uses for NaT)."""

_MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_LEAP_MONTHS = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
_MONTH_LENGTHS: dict[str, tuple[int, ...]] = {
    "360_day": (30,) * 12,
    "all_leap": _LEAP_MONTHS,
    "366_day": _LEAP_MONTHS,
}
_MONTH_STARTS: dict[str, NDArray[np.int64]] = {
    name: np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    for name, lengths in _MONTH_LENGTHS.items()
}
_PER_SECOND = {"s": 1, "ms": 1000, "us": 1_000_000}

_LAST_YEAR = 9999
_ISO = re.compile(
    r"""^(?P<year>\d{4})
        (?:-(?P<month>\d{1,2})
            (?:-(?P<day>\d{1,2})
                (?:[T\ ]+(?P<hour>\d{1,2})
                    (?::(?P<minute>\d{1,2})
                        (?::(?P<second>\d{1,2})(?:\.(?P<fraction>\d+))?)?
                    )?
                    \s*(?P<tz>Z|[+-]\d{2}(?::?\d{2})?)?
                )?
            )?
        )?$""",
    re.I | re.VERBOSE,
)
_UTC = re.compile(r"Z|[+-]00(?::?00)?", re.I)

Fields = tuple[int, int, int, int, int, int, int]
"""Year, month, day, hour, minute, second and microsecond."""

_PRECISIONS = ("year", "month", "day", "hour", "minute", "second", "fraction")
_WHOLE_PERIODS = ("year", "month", "day")
_EXAMPLE = "Expected an ISO date such as '2000-02-28' or '2000-02-28T12:00:00'."


def parse_iso_fields(text: str) -> tuple[Fields, str] | None:
    """Split an ISO 8601 date or time into fields, without checking them.

    Returns the fields and the finest part that was given (``"year"``,
    ``"month"``, ``"day"``, ``"hour"``, ``"minute"``, ``"second"`` or
    ``"fraction"``), or ``None`` if the text is not of the form
    ``YYYY[-MM[-DD[THH[:MM[:SS[.f]]]]]]``, optionally ending in Z or +00:00.
    A UTC offset other than zero raises ``SelectionError``.
    """
    match = _ISO.match(text.strip())
    if match is None:
        return None
    tz = match["tz"]
    if tz is not None and _UTC.fullmatch(tz) is None:
        raise SelectionError(
            f"The time {text!r} has a UTC offset. Only UTC (Z or +00:00) is supported."
        )
    parts = [match[name] for name in _PRECISIONS]
    given = [part is not None for part in parts]
    precision = _PRECISIONS[sum(given) - 1]
    fraction = (match["fraction"] or "")[:6].ljust(6, "0")
    fields = (
        int(match["year"]),
        int(match["month"] or 1),
        int(match["day"] or 1),
        int(match["hour"] or 0),
        int(match["minute"] or 0),
        int(match["second"] or 0),
        int(fraction),
    )
    return fields, precision


class Clock:
    """Dates, ISO strings and ticks in a 360_day or all_leap calendar.

    ``calendar`` is the name the store gives, in lowercase (``366_day`` is
    ``all_leap`` under another name). ``resolution`` is ``s``, ``ms`` or ``us``,
    the unit of one tick.
    """

    __slots__ = (
        "calendar",
        "limit",
        "month_lengths",
        "month_starts",
        "per_day",
        "per_second",
        "resolution",
        "year_days",
    )

    def __init__(self, calendar: str, resolution: str) -> None:
        self.calendar = calendar
        self.resolution = resolution
        self.month_lengths = _MONTH_LENGTHS[calendar]
        self.month_starts = _MONTH_STARTS[calendar]
        self.year_days = sum(self.month_lengths)
        self.per_second = _PER_SECOND[resolution]
        self.per_day = 86400 * self.per_second
        self.limit = (_LAST_YEAR + 1) * self.year_days * self.per_day
        """The first tick that is past the year 9999."""

    def __repr__(self) -> str:
        return f"Clock({self.calendar!r}, {self.resolution!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Clock) and (self.calendar, self.resolution) == (
            other.calendar,
            other.resolution,
        )

    def __hash__(self) -> int:
        return hash((self.calendar, self.resolution))

    # --- fields to ticks -------------------------------------------------------

    def _check(
        self,
        fields: Fields,
        label: str,
        error: type[ZarrExtractError],
        hint: bool = False,
    ) -> None:
        year, month, day, hour, minute, second, _ = fields
        problem = None
        if not 0 <= year <= _LAST_YEAR:
            problem = f"the year must be 0 to {_LAST_YEAR}"
        elif not 1 <= month <= 12:
            problem = "the month must be 1 to 12"
        elif not 0 <= hour <= 23:
            problem = "the hour must be 0 to 23"
        elif not 0 <= minute <= 59:
            problem = "the minute must be 0 to 59"
        elif not 0 <= second <= 59:
            problem = "the second must be 0 to 59"
        elif day < 1:
            problem = "the day must be at least 1"
        if problem is not None:
            raise error(
                f"Cannot read {label!r} as a date or time in the {self.calendar} "
                f"calendar: {problem}. {_EXAMPLE}"
            )
        length = self.month_lengths[month - 1]
        if day > length:
            if self.calendar == "360_day":
                reason = "where every month has 30 days"
            else:
                reason = f"where {_MONTH_NAMES[month - 1]} has {length} days"
            last = f"{year:04d}-{month:02d}-{length:02d}"
            extra = (
                f" Use {last!r} for the last day, or {last[:7]!r} for the whole month."
                if hint
                else ""
            )
            raise error(
                f"{label!r} does not exist in the {self.calendar} calendar, "
                f"{reason}.{extra}"
            )

    def _ticks(self, fields: Fields) -> int:
        year, month, day, hour, minute, second, micro = fields
        days = year * self.year_days + int(self.month_starts[month - 1]) + day - 1
        seconds = (hour * 60 + minute) * 60 + second
        # Whole ticks: a fraction finer than the resolution is dropped.
        sub = micro // (1_000_000 // self.per_second)
        return days * self.per_day + seconds * self.per_second + sub

    def ticks(
        self,
        fields: Fields,
        label: str,
        error: type[ZarrExtractError] = SelectionError,
    ) -> int:
        """Return the tick of a date, or raise ``error`` if it does not exist."""
        self._check(fields, label, error)
        return self._ticks(fields)

    def loses_precision(self, fields: Fields) -> bool:
        """Return True if the fraction of a second is finer than one tick."""
        return fields[6] % (1_000_000 // self.per_second) != 0

    # --- selections ------------------------------------------------------------

    def _read(self, value: object, name: str) -> tuple[Fields, str, str]:
        """Return the fields, the finest part given and a label for messages."""
        if isinstance(value, str):
            parsed = parse_iso_fields(value)
            if parsed is None:
                raise SelectionError(
                    f"Cannot read {value!r} as a date or time in the "
                    f"{self.calendar} calendar. {_EXAMPLE}"
                )
            fields, precision = parsed
            return fields, precision, value.strip()
        if isinstance(value, np.datetime64):
            if np.isnat(value):
                raise SelectionError(f"The selection for {name!r} is not a valid time.")
            text = str(np.datetime_as_string(value.astype("datetime64[us]"), unit="us"))
            parsed = parse_iso_fields(text)
            if parsed is None:
                raise SelectionError(
                    f"Cannot read {text!r} as a date or time in the "
                    f"{self.calendar} calendar. {_EXAMPLE}"
                )
            return parsed[0], "fraction", text
        if isinstance(value, datetime):
            if value.tzinfo is not None:
                value = value.astimezone(UTC).replace(tzinfo=None)
            fields = (
                value.year,
                value.month,
                value.day,
                value.hour,
                value.minute,
                value.second,
                value.microsecond,
            )
            return fields, "fraction", value.isoformat()
        if isinstance(value, date):
            return (
                (value.year, value.month, value.day, 0, 0, 0, 0),
                "day",
                (value.isoformat()),
            )
        raise SelectionError(
            f"Dimension {name!r} is a time dimension, so its selection must be "
            f"an ISO string, a datetime or a datetime64, not {value!r}."
        )

    def moment(self, value: object, name: str) -> int:
        """Return the tick that a time selection names, read in this calendar."""
        fields, _, label = self._read(value, name)
        return self.ticks(fields, label)

    def bounds(self, value: object, name: str) -> tuple[int, int]:
        """Return the first and last tick that a range end covers.

        An ISO string given only to the year, month or day covers that whole
        period of this calendar. Anything else covers the single tick it names.
        """
        fields, precision, label = self._read(value, name)
        self._check(fields, label, SelectionError, hint=True)
        first = self._ticks(fields)
        if isinstance(value, str) and precision in _WHOLE_PERIODS:
            if precision == "year":
                days = self.year_days
            elif precision == "month":
                days = self.month_lengths[fields[1] - 1]
            else:
                days = 1
            return first, first + days * self.per_day - 1
        return first, first

    # --- ticks to dates --------------------------------------------------------

    def fields(self, ticks: NDArray[np.int64]) -> tuple[NDArray[np.int64], ...]:
        """Split ticks into year, month, day, hour, minute, second and fraction.

        The fraction is in ticks of the resolution (0 for seconds). The ticks
        must not be ``MISSING``.
        """
        days, in_day = np.divmod(ticks, self.per_day)
        years, day_of_year = np.divmod(days, self.year_days)
        months = np.searchsorted(self.month_starts, day_of_year, side="right") - 1
        day = day_of_year - self.month_starts[months] + 1
        seconds, fraction = np.divmod(in_day, self.per_second)
        hours, rest = np.divmod(seconds, 3600)
        minutes, second = np.divmod(rest, 60)
        return years, months + 1, day, hours, minutes, second, fraction

    def format(self, ticks: Sequence[int] | NDArray[Any]) -> "pa.StringArray":
        """Write ticks as ISO 8601 strings of one fixed width, null if missing.

        ``YYYY-MM-DDTHH:MM:SS``, followed by ``.fff`` for milliseconds or
        ``.ffffff`` for microseconds. The width does not change, so the strings
        sort in the order of the times.
        """
        values = np.asarray(ticks, dtype=np.int64)
        missing = values == MISSING
        parts = self.fields(np.where(missing, 0, values))
        width = {"s": 0, "ms": 3, "us": 6}[self.resolution]
        suffix = f".{{:0{width}d}}" if width else ""
        pattern = "{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:02d}" + suffix
        rows = zip(*(part.tolist() for part in parts), strict=True)
        strings: list[str | None] = [
            pattern.format(*row[: 7 if width else 6]) for row in rows
        ]
        for index in np.flatnonzero(missing).tolist():
            strings[index] = None
        return pa.array(strings, type=pa.string())

    def show(self, tick: int) -> str:
        """Write one tick for an error message."""
        text = self.format([tick])[0].as_py()
        return str(text)
