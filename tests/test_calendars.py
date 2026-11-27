"""The 360_day and all_leap (366_day) calendars, whose dates are not timestamps."""

import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq
import pytest

import zarrextract as zx
from helpers import ZarrFormat, make_grid
from test_bisect import assert_same, counted_line, make_line, run
from zarrextract import (
    EmptySelectionError,
    SelectionError,
    TimeDecodingError,
    UnsupportedCalendarError,
)
from zarrextract._cf import decode_calendar_ticks, decode_time, parse_time_encoding

CALENDARS = ["360_day", "all_leap", "366_day"]
LEAP_MONTHS = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


@pytest.fixture(params=[2, 3], ids=["zarr-v2", "zarr-v3"])
def zarr_format(request: pytest.FixtureRequest) -> ZarrFormat:
    fmt: ZarrFormat = request.param
    return fmt


def month_lengths(calendar: str) -> list[int]:
    return [30] * 12 if calendar == "360_day" else LEAP_MONTHS


def day_labels(calendar: str, count: int, start_year: int = 2000) -> list[str]:
    """Midnight of each day from 1 January, counted one day at a time."""
    labels: list[str] = []
    year = start_year
    while len(labels) < count:
        for month, length in enumerate(month_lengths(calendar), 1):
            for day in range(1, length + 1):
                labels.append(f"{year:04d}-{month:02d}-{day:02d}T00:00:00")
        year += 1
    return labels[:count]


def decode(units: str, values: Any, calendar: str) -> list[str | None]:
    encoding = parse_time_encoding(units, calendar)
    assert encoding.clock is not None
    ticks = decode_calendar_ticks(np.asarray(values), encoding)
    return list(encoding.clock.format(ticks).to_pylist())


# --- decoding -----------------------------------------------------------------


def test_decode_360_day() -> None:
    assert decode(
        "days since 2000-01-01", [0, 29, 30, 59, 60, 359, 360], "360_day"
    ) == [
        "2000-01-01T00:00:00",
        "2000-01-30T00:00:00",
        "2000-02-01T00:00:00",
        "2000-02-30T00:00:00",
        "2000-03-01T00:00:00",
        "2000-12-30T00:00:00",
        "2001-01-01T00:00:00",
    ]


@pytest.mark.parametrize("calendar", ["all_leap", "366_day", "ALL_LEAP"])
def test_decode_all_leap(calendar: str) -> None:
    assert decode(
        "days since 2000-01-01", [0, 31, 59, 60, 365, 366, 366 + 59], calendar
    ) == [
        "2000-01-01T00:00:00",
        "2000-02-01T00:00:00",
        "2000-02-29T00:00:00",
        "2000-03-01T00:00:00",
        "2000-12-31T00:00:00",
        "2001-01-01T00:00:00",
        "2001-02-29T00:00:00",
    ]


@pytest.mark.parametrize("calendar", CALENDARS)
def test_decode_matches_a_day_by_day_count(calendar: str) -> None:
    years = 120
    days = years * (360 if calendar == "360_day" else 366)

    decoded = decode("days since 0000-01-01", np.arange(days), calendar)

    assert decoded == day_labels(calendar, days, start_year=0)


def test_decode_other_units_with_fixed_width_strings() -> None:
    assert decode("hours since 2000-02-30 12:00:00", [0, 12, 36], "360_day") == [
        "2000-02-30T12:00:00",
        "2000-03-01T00:00:00",
        "2000-03-02T00:00:00",
    ]
    assert decode("minutes since 2000-01-30", [0, 1440, 1441], "360_day") == [
        "2000-01-30T00:00:00",
        "2000-02-01T00:00:00",
        "2000-02-01T00:01:00",
    ]
    assert decode("seconds since 2000-02-29 23:59:59", [0, 1], "all_leap") == [
        "2000-02-29T23:59:59",
        "2000-03-01T00:00:00",
    ]
    assert decode("milliseconds since 2000-01-01", [1500, 86_400_000], "360_day") == [
        "2000-01-01T00:00:01.500",
        "2000-01-02T00:00:00.000",
    ]
    assert decode("microseconds since 2000-02-30", [1500], "360_day") == [
        "2000-02-30T00:00:00.001500"
    ]


def test_decode_floats_and_missing_values() -> None:
    values = np.array([0.5, np.nan, np.inf, 1.25])

    assert decode("days since 2000-02-30", values, "360_day") == [
        "2000-02-30T12:00:00",
        None,
        None,
        "2000-03-01T06:00:00",
    ]


def test_decode_empty() -> None:
    assert decode("days since 2000-01-01", np.array([], dtype="f8"), "360_day") == []


def test_the_reference_date_is_read_in_the_calendar() -> None:
    assert decode("days since 2000-02-30", [0], "360_day") == ["2000-02-30T00:00:00"]
    assert decode("days since 2000-02-29T06:00Z", [0], "all_leap") == [
        "2000-02-29T06:00:00"
    ]


@pytest.mark.parametrize(
    ("units", "calendar", "message"),
    [
        ("days since 2000-01-31", "360_day", "does not exist in the 360_day"),
        ("days since 2001-04-31", "all_leap", "April has 30 days"),
        ("days since 2000-13-01", "360_day", "month must be 1 to 12"),
        ("months since 2000-01-01", "360_day", "Months and years"),
        ("years since 2000-01-01", "all_leap", "Months and years"),
        ("seconds since 2000-01-01 00:00:00.5", "360_day", "finer"),
        ("days since 2000-01-01 00:00:00 +05:00", "360_day", "UTC offset"),
    ],
)
def test_bad_encodings(units: str, calendar: str, message: str) -> None:
    with pytest.raises(TimeDecodingError, match=message):
        parse_time_encoding(units, calendar)


@pytest.mark.parametrize(
    "values", [[3_700_000], [-1]], ids=["after-year-9999", "before-year-0"]
)
def test_times_outside_the_years_0_to_9999_raise(values: list[int]) -> None:
    with pytest.raises(TimeDecodingError, match="0 to 9999"):
        decode("days since 0000-01-01", values, "360_day")


def test_decode_overflow_and_non_numbers() -> None:
    with pytest.raises(TimeDecodingError, match="out of range"):
        decode("days since 2000-01-01", [1e30], "360_day")
    with pytest.raises(TimeDecodingError, match="numbers"):
        decode("days since 2000-01-01", np.array(["a"]), "360_day")


def test_datetime64_decoding_is_not_offered_for_these_calendars() -> None:
    with pytest.raises(TimeDecodingError, match="decode_calendar_ticks"):
        decode_time(np.array([0]), "days since 2000-01-01", "360_day")


def test_other_calendars_are_still_unsupported() -> None:
    for calendar in ("julian", "none"):
        with pytest.raises(UnsupportedCalendarError, match="360_day, all_leap"):
            parse_time_encoding("days since 2000-01-01", calendar)


# --- the column ---------------------------------------------------------------


@pytest.mark.parametrize("calendar", CALENDARS)
def test_the_time_column_is_strings_with_calendar_metadata(
    tmp_path: Path, zarr_format: ZarrFormat, calendar: str
) -> None:
    grid = make_grid(
        tmp_path / "g.zarr",
        zarr_format,
        time=np.arange(0, 4),
        time_attrs={
            "units": "days since 2000-02-28",
            "calendar": calendar,
            "long_name": "Time",
        },
    )

    table = zx.extract(grid, "temp", lat=(0, 45), lon=(0, 90))

    field = table.schema.field("time")
    assert field.type == pa.string()
    # The units describe the stored numbers, so they are left out as ever.
    assert field.metadata == {b"long_name": b"Time", b"calendar": calendar.encode()}
    rows = table.num_rows
    assert rows == 4 * 2 * 2
    times = table["time"].to_pylist()
    assert times[:: rows // 4] == [
        "2000-02-28T00:00:00",
        "2000-02-29T00:00:00",
        "2000-02-30T00:00:00" if calendar == "360_day" else "2000-03-01T00:00:00",
        "2000-03-01T00:00:00" if calendar == "360_day" else "2000-03-02T00:00:00",
    ]
    assert all(time == times[i // 4 * 4] for i, time in enumerate(times))
    # Other columns are as before.
    assert table.schema.field("lat").type == pa.float64()


def test_the_noleap_column_is_still_a_timestamp_without_a_calendar(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    grid = make_grid(
        tmp_path / "g.zarr",
        zarr_format,
        time_attrs={"units": "days since 2000-01-01", "calendar": "noleap"},
    )

    table = zx.extract(grid, "temp", lat=0, lon=0)

    assert table.schema.field("time").type == pa.timestamp("s")
    assert table.schema.field("time").metadata is None


def test_missing_times_are_nulls(tmp_path: Path, zarr_format: ZarrFormat) -> None:
    grid = make_grid(
        tmp_path / "g.zarr",
        zarr_format,
        time=np.array([0.0, np.nan, 2.0, 3.0]),
        time_attrs={"units": "days since 2000-01-29", "calendar": "360_day"},
    )

    table = zx.extract(grid, "temp", lat=0, lon=0)

    assert table["time"].to_pylist() == [
        "2000-01-29T00:00:00",
        None,
        "2000-02-01T00:00:00",
        "2000-02-02T00:00:00",
    ]
    with pytest.raises(SelectionError, match="NaT"):
        zx.extract(grid, "temp", lat=0, lon=0, time="2000-02-01")


# --- selections ---------------------------------------------------------------


def daily(
    tmp_path: Path,
    zarr_format: ZarrFormat,
    calendar: str,
    *,
    count: int = 800,
    chunk: int = 5,
    descending: bool = False,
) -> Path:
    """A daily axis from 2000-01-01, with ``v`` the index of each day."""
    days = np.arange(count)
    values = days[::-1] if descending else days
    return make_line(
        tmp_path / f"{calendar}.zarr",
        zarr_format,
        "time",
        values,
        {"units": "days since 2000-01-01", "calendar": calendar},
        chunk=chunk,
    )


def table_of(result: pa.Table | tuple[type, str]) -> pa.Table:
    assert isinstance(result, pa.Table), result
    return result


def check(path: Path, selection: dict[str, Any]) -> pa.Table:
    """Select, on both the bisection and the whole-read path, and compare."""
    assert_same(path, selection)
    return table_of(run(path, selection))


def fails(path: Path, selection: dict[str, Any]) -> tuple[type, str]:
    assert_same(path, selection)
    result = run(path, selection)
    assert isinstance(result, tuple), result
    return result


@pytest.mark.parametrize("calendar", CALENDARS)
@pytest.mark.parametrize("chunk", [5, 800], ids=["bisection", "whole-read"])
def test_a_range_covers_whole_months_and_years(
    tmp_path: Path, zarr_format: ZarrFormat, calendar: str, chunk: int
) -> None:
    path = daily(tmp_path, zarr_format, calendar, chunk=chunk)
    labels = day_labels(calendar, 800)
    lengths = month_lengths(calendar)

    february = check(path, {"time": ("2000-02", "2000-02")})
    assert february.num_rows == lengths[1]
    assert february["time"].to_pylist() == labels[lengths[0] : lengths[0] + lengths[1]]

    year = check(path, {"time": ("2001", "2001")})
    assert year.num_rows == sum(lengths)
    assert year["v"].to_pylist() == [
        float(i) for i in range(sum(lengths), 2 * sum(lengths))
    ]

    both = check(path, {"time": ("2000-02", "2000-03-15")})
    first = labels.index("2000-02-01T00:00:00")
    last = labels.index("2000-03-15T00:00:00")
    assert both["time"].to_pylist() == labels[first : last + 1]


@pytest.mark.parametrize("chunk", [5, 800], ids=["bisection", "whole-read"])
def test_30_february_in_360_day(
    tmp_path: Path, zarr_format: ZarrFormat, chunk: int
) -> None:
    path = daily(tmp_path, zarr_format, "360_day", chunk=chunk)
    labels = day_labels("360_day", 800)
    day = labels.index("2000-02-30T00:00:00")

    point = check(path, {"time": "2000-02-30"})
    assert point["v"].to_pylist() == [float(day)]
    assert point["time"].to_pylist() == ["2000-02-30T00:00:00"]
    # The nearest value to a time of day is the nearer midnight.
    assert check(path, {"time": "2000-02-30T11:00"})["v"].to_pylist() == [float(day)]
    assert check(path, {"time": "2000-02-30T13:00"})["v"].to_pylist() == [day + 1.0]

    ranged = check(path, {"time": ("2000-02-28", "2000-03-02")})
    assert ranged["time"].to_pylist() == [
        "2000-02-28T00:00:00",
        "2000-02-29T00:00:00",
        "2000-02-30T00:00:00",
        "2000-03-01T00:00:00",
        "2000-03-02T00:00:00",
    ]
    # Given the wrong way round, the range still runs from low to high.
    assert check(path, {"time": ("2000-03-02", "2000-02-28")}).equals(ranged)


@pytest.mark.parametrize("chunk", [5, 800], ids=["bisection", "whole-read"])
def test_29_february_in_every_year_of_all_leap(
    tmp_path: Path, zarr_format: ZarrFormat, chunk: int
) -> None:
    path = daily(tmp_path, zarr_format, "all_leap", chunk=chunk)

    for year in ("2000", "2001"):
        point = check(path, {"time": f"{year}-02-29"})
        assert point["time"].to_pylist() == [f"{year}-02-29T00:00:00"]
    assert check(path, {"time": ("2001-02", "2001-02")}).num_rows == 29
    # 29 February is valid, and so is its last second.
    last = check(path, {"time": ("2001-02-29T23:59:59", "2001-03-01")})
    assert last["time"].to_pylist() == ["2001-03-01T00:00:00"]


def test_a_partial_day_covers_every_time_on_that_day(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    path = make_line(
        tmp_path / "h.zarr",
        zarr_format,
        "time",
        np.arange(0, 24 * 90),
        {"units": "hours since 2000-01-01", "calendar": "360_day"},
    )

    day = check(path, {"time": ("2000-02-30", "2000-02-30")})

    assert day.num_rows == 24
    assert day["time"][0].as_py() == "2000-02-30T00:00:00"
    assert day["time"][23].as_py() == "2000-02-30T23:00:00"
    # With a time of day, a range end means exactly that moment.
    hours = check(path, {"time": ("2000-02-30T06:00", "2000-02-30T08:00")})
    assert hours["time"].to_pylist() == [
        "2000-02-30T06:00:00",
        "2000-02-30T07:00:00",
        "2000-02-30T08:00:00",
    ]


@pytest.mark.parametrize("calendar", ["360_day", "all_leap"])
def test_dates_and_datetimes_are_read_by_their_fields(
    tmp_path: Path, zarr_format: ZarrFormat, calendar: str
) -> None:
    path = daily(tmp_path, zarr_format, calendar)
    labels = day_labels(calendar, 800)
    first = labels.index("2000-02-28T00:00:00")
    last = labels.index("2000-03-02T00:00:00")
    expected = labels[first : last + 1]
    assert len(expected) == (5 if calendar == "360_day" else 4)

    for selection in (
        (dt.date(2000, 2, 28), dt.date(2000, 3, 2)),
        (dt.datetime(2000, 2, 28), dt.datetime(2000, 3, 2)),
        (
            dt.datetime(2000, 2, 28, tzinfo=dt.UTC),
            dt.datetime(2000, 3, 2, tzinfo=dt.UTC),
        ),
        (np.datetime64("2000-02-28"), np.datetime64("2000-03-02T00:00:00")),
        ("2000-02-28T00:00:00Z", "2000-03-02 00:00:00"),
    ):
        assert check(path, {"time": selection})["time"].to_pylist() == expected

    # 13:00 on 29 February is nearer to midnight of the day after.
    after = labels[labels.index("2000-02-29T00:00:00") + 1]
    for value in (
        dt.datetime(2000, 2, 29, 13),
        dt.datetime(2000, 2, 29, 13, tzinfo=dt.UTC),
        np.datetime64("2000-02-29T13:00"),
    ):
        assert check(path, {"time": value})["time"].to_pylist() == [after]


def test_a_descending_axis(tmp_path: Path, zarr_format: ZarrFormat) -> None:
    path = daily(tmp_path, zarr_format, "360_day", descending=True)
    labels = day_labels("360_day", 800)
    first = labels.index("2000-02-28T00:00:00")

    table = check(path, {"time": ("2000-02-28", "2000-03-02")})

    assert table["time"].to_pylist() == list(reversed(labels[first : first + 5]))
    assert check(path, {"time": "2000-02-30"})["time"].to_pylist() == [
        "2000-02-30T00:00:00"
    ]


# --- dates that do not exist --------------------------------------------------


@pytest.mark.parametrize("chunk", [5, 800], ids=["bisection", "whole-read"])
def test_dates_that_do_not_exist_in_360_day(
    tmp_path: Path, zarr_format: ZarrFormat, chunk: int
) -> None:
    path = daily(tmp_path, zarr_format, "360_day", chunk=chunk)

    for selection in (
        "2000-02-31",
        "2000-01-31",
        dt.date(2000, 1, 31),
        np.datetime64("2000-12-31"),
        dt.datetime(2000, 5, 31, 12),
    ):
        kind, message = fails(path, {"time": selection})
        assert kind is SelectionError
        assert "does not exist in the 360_day calendar" in message
        assert "every month has 30 days" in message
        assert "Use '" not in message  # the hint is for range ends

    kind, message = fails(path, {"time": ("2000-01-01", "2000-12-31")})
    assert kind is SelectionError
    assert "'2000-12-31' does not exist in the 360_day calendar" in message
    assert "Use '2000-12-30' for the last day, or '2000-12' for the whole month." in (
        message
    )
    kind, message = fails(path, {"time": ("2000-01-31", "2000-02")})
    assert "'2000-01-31'" in message


@pytest.mark.parametrize("calendar", ["all_leap", "366_day"])
def test_dates_that_do_not_exist_in_all_leap(
    tmp_path: Path, zarr_format: ZarrFormat, calendar: str
) -> None:
    path = daily(tmp_path, zarr_format, calendar)

    kind, message = fails(path, {"time": "2001-04-31"})
    assert kind is SelectionError
    assert f"'2001-04-31' does not exist in the {calendar} calendar" in message
    assert "April has 30 days" in message
    kind, message = fails(path, {"time": ("2001-02-30", "2001-03")})
    assert "February has 29 days" in message
    # 29 February exists in every year, and 31 January does too.
    check(path, {"time": ("2001-02-29", "2001-01-31")})


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        ("2000-13-01", "month must be 1 to 12"),
        ("2000-00-10", "month must be 1 to 12"),
        ("2000-02-00", "day must be at least 1"),
        ("2000-02-01T24:00:00", "hour must be 0 to 23"),
        ("2000-02-01T10:60", "minute must be 0 to 59"),
        ("2000-02-01T10:00:61", "second must be 0 to 59"),
        ("yesterday", "Cannot read 'yesterday' as a date or time in the 360_day"),
        ("2000/02/01", "Cannot read"),
        ("2000-02-30T00:00:00+05:00", "UTC offset"),
        (5, "must be an ISO string"),
        (5.5, "must be an ISO string"),
        (np.datetime64("NaT", "D"), "not a valid time"),
    ],
)
def test_other_selections_that_cannot_be_read(
    tmp_path: Path, zarr_format: ZarrFormat, selection: Any, message: str
) -> None:
    path = daily(tmp_path, zarr_format, "360_day")

    kind, text = fails(path, {"time": selection})

    assert kind is SelectionError
    assert message in text


def test_a_range_of_three_values_is_still_rejected(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    path = daily(tmp_path, zarr_format, "360_day")

    kind, message = fails(path, {"time": ("2000-01", "2000-02", "2000-03")})

    assert kind is SelectionError
    assert "3 values" in message


def test_messages_about_the_axis_show_dates_of_the_calendar(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    path = daily(tmp_path, zarr_format, "360_day", count=720)

    kind, message = fails(path, {"time": "2005-01-01"})
    assert kind is SelectionError
    assert "2005-01-01T00:00:00 is outside the range" in message
    assert "(2000-01-01T00:00:00 to 2001-12-30T00:00:00)" in message
    assert "by more than one grid step (86400 s)" in message

    kind, message = fails(path, {"time": ("2005-01", "2005-02")})
    assert kind is EmptySelectionError
    assert "between 2005-01-01T00:00:00 and 2005-02-30T23:59:59" in message
    assert "run from 2000-01-01T00:00:00 to 2001-12-30T00:00:00" in message


def test_impossible_gregorian_dates_give_a_clear_message(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    for calendar in ("noleap", "365_day", "standard", "proleptic_gregorian"):
        path = daily(tmp_path, zarr_format, calendar)

        kind, message = fails(path, {"time": "2001-02-29"})
        assert kind is SelectionError
        assert (
            message == "'2001-02-29' is not a valid date: February 2001 has no day 29."
        )
        kind, message = fails(path, {"time": ("2000-01", "2000-02-30")})
        assert "February 2000 has no day 30" in message
        kind, message = fails(path, {"time": "2000-13-01"})
        assert message.startswith("Cannot read '2000-13-01' as a date or time:")

    # A date that exists in the Gregorian calendar still picks the nearest value.
    path = daily(tmp_path, zarr_format, "noleap")
    assert check(path, {"time": "2000-02-29"}).num_rows == 1


# --- bisection ----------------------------------------------------------------


def random_selection(rng: np.random.Generator) -> Any:
    def one() -> str:
        year = int(rng.integers(1999, 2016))
        month = int(rng.integers(1, 14))  # 13 does not exist
        day = int(rng.integers(1, 33))  # 31 and 32 do not exist in 360_day
        kind = int(rng.integers(4))
        if kind == 0:
            return f"{year:04d}"
        if kind == 1:
            return f"{year:04d}-{month:02d}"
        text = f"{year:04d}-{month:02d}-{day:02d}"
        if kind == 3:
            text += f"T{int(rng.integers(24)):02d}:00:00Z"
        return text

    return one() if rng.integers(2) else (one(), one())


@pytest.mark.parametrize("calendar", ["360_day", "all_leap"])
def test_bisection_matches_the_whole_read(
    tmp_path: Path, zarr_format: ZarrFormat, calendar: str
) -> None:
    # About 14 years in 100 chunks of 50, so that it is searched by bisection.
    path = daily(tmp_path, zarr_format, calendar, count=5000, chunk=50)
    rng = np.random.default_rng(20261127)
    outcomes: set[str] = set()

    for _ in range(30):
        selection = random_selection(rng)
        assert_same(path, {"time": selection})
        result = run(path, {"time": selection})
        outcomes.add("table" if isinstance(result, pa.Table) else result[0].__name__)

    # The selections cover rows, empty ranges and impossible dates.
    assert {"table", "SelectionError"} <= outcomes


@pytest.mark.parametrize("calendar", ["360_day", "all_leap"])
def test_a_point_and_a_range_on_a_long_axis_read_few_chunks(
    tmp_path: Path, zarr_format: ZarrFormat, calendar: str
) -> None:
    path = daily(tmp_path, zarr_format, calendar, count=5000, chunk=5)
    labels = day_labels(calendar, 5000)

    table, store = counted_line(path, {"time": "2008-06-29"})
    assert table["time"].to_pylist() == ["2008-06-29T00:00:00"]
    assert 0 < len(store.chunk_reads("time")) <= 30

    table, store = counted_line(path, {"time": ("2008-06-27", "2008-06-30")})
    first = labels.index("2008-06-27T00:00:00")
    assert table["time"].to_pylist() == labels[first : first + 4]
    assert 0 < len(store.chunk_reads("time")) <= 30


def test_a_missing_time_found_by_bisection_raises(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    values = np.arange(0.0, 2000.0)
    values[1000] = np.nan
    path = make_line(
        tmp_path / "n.zarr",
        zarr_format,
        "time",
        values,
        {"units": "days since 2000-01-01", "calendar": "360_day"},
        chunk=50,
    )

    kind, message = fails(path, {"time": "2001-01-01"})

    assert kind is SelectionError
    assert "NaT" in message


# --- saving and provenance ----------------------------------------------------


@pytest.fixture
def table(tmp_path: Path, zarr_format: ZarrFormat) -> pa.Table:
    grid = make_grid(
        tmp_path / "g.zarr",
        zarr_format,
        time=np.arange(55, 65),
        time_attrs={
            "units": "days since 2000-01-01",
            "calendar": "360_day",
            "long_name": "Time",
        },
    )
    return zx.extract(grid, "temp", lat=0, lon=0, time=("2000-02-26", "2000-03-05"))


EXPECTED_TIMES = [
    "2000-02-26T00:00:00",
    "2000-02-27T00:00:00",
    "2000-02-28T00:00:00",
    "2000-02-29T00:00:00",
    "2000-02-30T00:00:00",
    "2000-03-01T00:00:00",
    "2000-03-02T00:00:00",
    "2000-03-03T00:00:00",
    "2000-03-04T00:00:00",
    "2000-03-05T00:00:00",
]


def test_the_table_has_the_days_of_the_calendar(table: pa.Table) -> None:
    assert table["time"].to_pylist() == EXPECTED_TIMES


@pytest.mark.parametrize("suffix", [".parquet", ".arrow", ".feather"])
def test_binary_files_keep_the_strings_and_the_calendar(
    table: pa.Table, tmp_path: Path, suffix: str
) -> None:
    (path,) = zx.write(table, tmp_path / f"t{suffix}")

    if suffix == ".parquet":
        back = pq.read_table(path)
    else:
        with pa.OSFile(str(path), "rb") as file:
            back = pa.ipc.open_file(file).read_all()
    assert back.schema.field("time").type == pa.string()
    assert back["time"].to_pylist() == EXPECTED_TIMES
    assert back.schema.field("time").metadata[b"calendar"] == b"360_day"
    info = zx.provenance(path)
    assert info is not None
    assert info.columns["time"] == {"long_name": "Time", "calendar": "360_day"}


def test_csv_has_the_strings_and_the_sidecar_has_the_calendar(
    table: pa.Table, tmp_path: Path
) -> None:
    path, sidecar = zx.write(table, tmp_path / "t.csv")

    assert "2000-02-30T00:00:00" in path.read_text()
    payload = json.loads(sidecar.read_text())
    time_column = payload["columns"][0]
    assert time_column["name"] == "time"
    assert time_column["type"] == "string"
    assert time_column["metadata"] == {"calendar": "360_day", "long_name": "Time"}
    # A reader that takes the types from the sidecar gets the strings back.
    back = pa_csv.read_csv(
        path,
        convert_options=pa_csv.ConvertOptions(column_types={"time": pa.string()}),
    )
    assert back["time"].to_pylist() == EXPECTED_TIMES
    info = zx.provenance(path)
    assert info is not None
    assert info.columns["time"]["calendar"] == "360_day"


def test_the_selections_are_recorded_as_given(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    grid = make_grid(
        tmp_path / "g.zarr",
        zarr_format,
        time=np.arange(55, 65),
        time_attrs={"units": "days since 2000-01-01", "calendar": "360_day"},
    )

    by_text = zx.provenance(zx.extract(grid, "temp", lat=0, lon=0, time="2000-02-30"))
    by_range = zx.provenance(
        zx.extract(grid, "temp", lat=0, lon=0, time=(dt.date(2000, 2, 28), "2000-03"))
    )

    assert by_text is not None
    assert by_text.selections["time"] == "2000-02-30"
    assert by_range is not None
    assert by_range.selections["time"] == ["2000-02-28", "2000-03"]
    assert by_range.columns["time"] == {"calendar": "360_day"}


def test_a_handle_gives_the_same_tables_on_every_call(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    path = daily(tmp_path, zarr_format, "360_day", count=2000, chunk=5)
    selections: list[dict[str, Any]] = [
        {"time": "2000-02-30"},
        {"time": ("2001", "2001")},
        {"time": ("2000-02", "2000-03-15")},
    ]
    expected = [zx.extract(path, "v", **selection) for selection in selections]

    with zx.open(path) as source:
        for _ in range(2):
            for selection, want in zip(selections, expected, strict=True):
                got = source.extract("v", **selection)
                assert got.to_pydict() == want.to_pydict()
                assert got.schema.field("time").metadata == (
                    want.schema.field("time").metadata
                )
