from typing import Any

import numpy as np
import pytest

from zarrextract import (
    TimeDecodingError,
    UnsupportedCalendarError,
    ZarrExtractError,
)
from zarrextract._cf import decode_time, is_longitude, is_time_units, unpack


def seconds(*moments: str) -> Any:
    return np.array(moments, dtype="datetime64[s]")


@pytest.mark.parametrize(
    ("units", "values", "expected"),
    [
        ("days since 1850-01-01", [0, 1], ["1850-01-01", "1850-01-02"]),
        (
            "hours since 1900-01-01 00:00:00",
            [0, 25],
            ["1900-01-01T00:00:00", "1900-01-02T01:00:00"],
        ),
        (
            "hours since 1900-1-1 0:0:0",
            [0, 25],
            ["1900-01-01T00:00:00", "1900-01-02T01:00:00"],
        ),
        ("days since 1850-01-01Z", [1], ["1850-01-02"]),
        ("days since 1850-01-01 00:00:00Z", [1], ["1850-01-02"]),
        ("days since 1850-01-01 00:00:00 +00:00", [1], ["1850-01-02"]),
        ("days since 1850-01-01T00:00:00+0000", [1], ["1850-01-02"]),
        ("days since 1850-01-01T00:00:00-00:00", [1], ["1850-01-02"]),
        ("days since 1850-01-01 00:00:00 +00", [1], ["1850-01-02"]),
        ("minutes since 2000-01-01 12:30", [30], ["2000-01-01T13:00"]),
        ("hours since 2000-01-01 06", [1], ["2000-01-01T07:00"]),
        ("DAYS SINCE 2000-01-01", [1], ["2000-01-02"]),
        ("hours since 1-1-1", [0], ["0001-01-01"]),
    ],
)
def test_decode_reference_date_formats(
    units: str, values: list[int], expected: list[str]
) -> None:
    decoded = decode_time(np.array(values), units, "proleptic_gregorian")

    assert decoded.dtype == np.dtype("datetime64[s]")
    assert np.array_equal(decoded, seconds(*expected))


@pytest.mark.parametrize("offset", ["+05:00", "-03:30", "+0100", "+00:30", "-08"])
def test_decode_rejects_other_utc_offsets(offset: str) -> None:
    with pytest.raises(TimeDecodingError, match="UTC offset"):
        decode_time(np.array([0]), f"days since 2000-01-01 00:00:00 {offset}")


@pytest.mark.parametrize(
    "units",
    [
        "days since yesterday",
        "days since 2000-13-01",
        "days since 2000-02-30",
        "days since 2000/01/01",
        "days since",
        "months since 2000-01-01",
        "years since 2000-01-01",
        "degrees_east",
        "",
    ],
)
def test_decode_rejects_bad_units(units: str) -> None:
    with pytest.raises(TimeDecodingError):
        decode_time(np.array([0]), units)


def test_resolution_follows_the_unit() -> None:
    day = decode_time(np.array([1]), "days since 2000-01-01")
    milli = decode_time(np.array([1500]), "milliseconds since 2000-01-01")
    micro = decode_time(np.array([1500]), "microseconds since 2000-01-01")

    assert day.dtype == np.dtype("datetime64[s]")
    assert milli.dtype == np.dtype("datetime64[ms]")
    assert micro.dtype == np.dtype("datetime64[us]")
    assert milli[0] == np.datetime64("2000-01-01T00:00:01.500", "ms")
    assert micro[0] == np.datetime64("2000-01-01T00:00:00.001500", "us")


def test_decode_dates_beyond_2262() -> None:
    target = np.datetime64("2300-06-15")
    days = int((target - np.datetime64("1850-01-01")) / np.timedelta64(1, "D"))

    decoded = decode_time(np.array([days]), "days since 1850-01-01")

    assert decoded.dtype == np.dtype("datetime64[s]")
    assert decoded[0] == target
    assert decoded[0].astype("datetime64[Y]") == np.datetime64("2300")


def test_decode_floats() -> None:
    decoded = decode_time(np.array([0.5, 1.25]), "days since 2000-01-01")

    assert np.array_equal(decoded, seconds("2000-01-01T12:00:00", "2000-01-02T06:00"))


def test_decode_non_finite_becomes_nat() -> None:
    decoded = decode_time(np.array([0.0, np.nan, np.inf]), "days since 2000-01-01")

    assert decoded[0] == np.datetime64("2000-01-01")
    assert np.isnat(decoded[1:]).all()


def test_decode_empty() -> None:
    assert decode_time(np.array([], dtype="f8"), "days since 2000-01-01").size == 0


@pytest.mark.parametrize("values", [np.array([10**17]), np.array([1e30])])
def test_decode_overflow(values: Any) -> None:
    with pytest.raises(TimeDecodingError, match="out of range"):
        decode_time(values, "days since 2000-01-01")


def test_decode_requires_numbers() -> None:
    with pytest.raises(TimeDecodingError, match="numbers"):
        decode_time(np.array(["a", "b"]), "days since 2000-01-01")


def test_decode_reference_finer_than_unit() -> None:
    with pytest.raises(TimeDecodingError, match="finer"):
        decode_time(np.array([0]), "seconds since 2000-01-01 00:00:00.5")

    decoded = decode_time(np.array([0]), "milliseconds since 2000-01-01 00:00:00.5")
    assert decoded[0] == np.datetime64("2000-01-01T00:00:00.500", "ms")


@pytest.mark.parametrize(
    "calendar", [None, "standard", "Gregorian", "proleptic_gregorian"]
)
def test_supported_calendars(calendar: str | None) -> None:
    decoded = decode_time(np.array([1]), "days since 2000-01-01", calendar)

    assert decoded[0] == np.datetime64("2000-01-02")


@pytest.mark.parametrize(
    "calendar",
    ["all_leap", "366_day", "360_day", "julian", "none", "", 5],
)
def test_unsupported_calendars(calendar: object) -> None:
    with pytest.raises(UnsupportedCalendarError, match="noleap and 365_day"):
        decode_time(np.array([1]), "days since 2000-01-01", calendar)


@pytest.mark.parametrize("calendar", ["noleap", "365_day", "NoLeap"])
@pytest.mark.parametrize(
    ("units", "values", "expected"),
    [
        # 28 February to 1 March, in a leap year of the Gregorian calendar too.
        (
            "days since 2000-02-27",
            [0, 1, 2, 3],
            ["2000-02-27", "2000-02-28", "2000-03-01", "2000-03-02"],
        ),
        # Year ends, and a reference that is not at the start of the day.
        (
            "days since 2001-12-31",
            [0, 1, 365],
            ["2001-12-31", "2002-01-01", "2002-12-31"],
        ),
        (
            "hours since 2001-12-31 18:00:00",
            [0, 6, 7, 24 * 365],
            [
                "2001-12-31T18:00",
                "2002-01-01T00:00",
                "2002-01-01T01:00",
                "2002-12-31T18:00",
            ],
        ),
        # CESM2: the first day of year 1 and the end of year 1.
        (
            "days since 0001-01-01",
            [0, 31, 59, 364, 365],
            [
                "0001-01-01",
                "0001-02-01",
                "0001-03-01",
                "0001-12-31",
                "0002-01-01",
            ],
        ),
        # 2000 would have a 29 February: day 59 of 2000 is 1 March.
        (
            "days since 2000-01-01",
            [58, 59, 365],
            ["2000-02-28", "2000-03-01", "2001-01-01"],
        ),
        # Years before 1582 are fine, and so are negative offsets.
        ("days since 1500-03-01", [-1, 0], ["1500-02-28", "1500-03-01"]),
        ("days since 1850-01-01", [-1, 150 * 365], ["1849-12-31", "2000-01-01"]),
        (
            "minutes since 2000-03-01 00:00:00",
            [-1, 1440],
            ["2000-02-28T23:59", "2000-03-02"],
        ),
        (
            "seconds since 2000-01-01",
            [86399, 86400],
            ["2000-01-01T23:59:59", "2000-01-02"],
        ),
    ],
)
def test_decode_noleap(
    calendar: str, units: str, values: list[int], expected: list[str]
) -> None:
    decoded = decode_time(np.array(values), units, calendar)

    assert decoded.dtype == np.dtype("datetime64[s]")
    assert np.array_equal(decoded, seconds(*expected))


def test_decode_noleap_floats_and_nan() -> None:
    decoded = decode_time(
        np.array([0.5, 59.25, np.nan, np.inf]), "days since 2000-01-01", "noleap"
    )

    assert decoded[0] == np.datetime64("2000-01-01T12:00:00")
    assert decoded[1] == np.datetime64("2000-03-01T06:00:00")
    assert np.isnat(decoded[2:]).all()


def test_decode_noleap_resolutions() -> None:
    milli = decode_time(
        np.array([1500, 86400000]), "milliseconds since 2000-02-28", "noleap"
    )
    micro = decode_time(np.array([1500]), "microseconds since 2000-02-28", "365_day")

    assert milli.dtype == np.dtype("datetime64[ms]")
    assert milli[0] == np.datetime64("2000-02-28T00:00:01.500", "ms")
    assert milli[1] == np.datetime64("2000-03-01T00:00:00.000", "ms")
    assert micro.dtype == np.dtype("datetime64[us]")
    assert micro[0] == np.datetime64("2000-02-28T00:00:00.001500", "us")


def test_decode_noleap_matches_a_day_by_day_count() -> None:
    # Count the days of 300 years without 29 February, starting at year 1.
    days = np.arange(300 * 365)
    decoded = decode_time(days, "days since 0001-01-01", "noleap")

    gregorian = np.arange(
        np.datetime64("0001-01-01"), np.datetime64("0301-01-01"), np.timedelta64(1, "D")
    )
    month_start = gregorian.astype("datetime64[M]")
    is_leap_day = (month_start.astype(int) % 12 == 1) & (
        (gregorian - month_start.astype("datetime64[D]")).astype(int) == 28
    )
    expected = gregorian[~is_leap_day].astype("datetime64[s]")
    assert expected.size == days.size
    assert np.array_equal(decoded, expected)
    assert (np.diff(decoded) > np.timedelta64(0, "s")).all()


def test_decode_noleap_is_ordered_across_days_and_units() -> None:
    values = np.arange(-1000, 5000, 7)
    decoded = decode_time(values, "hours since 1999-12-31 12:00", "noleap")

    assert (np.diff(decoded) > np.timedelta64(0, "s")).all()


@pytest.mark.parametrize(
    "units", ["days since 2000-02-29", "days since 1996-02-29 06:00"]
)
@pytest.mark.parametrize("calendar", ["noleap", "365_day"])
def test_decode_noleap_rejects_a_reference_on_29_february(
    units: str, calendar: str
) -> None:
    with pytest.raises(TimeDecodingError, match="29 February"):
        decode_time(np.array([0]), units, calendar)


def test_noleap_does_not_have_the_gregorian_cutover() -> None:
    decoded = decode_time(np.array([0]), "days since 0001-01-01", "noleap")

    assert decoded[0] == np.datetime64("0001-01-01")


def test_unsupported_calendar_is_a_time_decoding_error() -> None:
    assert issubclass(UnsupportedCalendarError, TimeDecodingError)
    assert issubclass(UnsupportedCalendarError, ZarrExtractError)


def test_gregorian_switch_is_not_guessed() -> None:
    # Before 1582-10-15 the standard calendar is Julian, which is not supported.
    with pytest.raises(UnsupportedCalendarError, match="1582"):
        decode_time(np.array([0]), "days since 0001-01-01", "standard")
    with pytest.raises(UnsupportedCalendarError, match="1582"):
        decode_time(np.array([-100000]), "days since 1600-01-01", "gregorian")

    decoded = decode_time(np.array([0]), "days since 0001-01-01", "proleptic_gregorian")
    assert decoded[0] == np.datetime64("0001-01-01")


def test_is_time_units() -> None:
    assert is_time_units("days since 2000-01-01")
    assert is_time_units("Hours Since 1900-1-1 0:0:0")
    assert not is_time_units("degrees_east")
    assert not is_time_units("K")
    assert not is_time_units(None)
    assert not is_time_units(5)


def test_is_longitude() -> None:
    assert is_longitude("lon", {})
    assert is_longitude("Longitude", {})
    assert is_longitude("x", {"standard_name": "longitude"})
    assert is_longitude("x", {"units": "degrees_east"})
    assert is_longitude("x", {"units": "degree_E"})
    assert not is_longitude("lat", {"units": "degrees_north"})
    assert not is_longitude("x", {})


def test_unpack_without_attributes_returns_the_data() -> None:
    raw = np.array([1, 2, 3], dtype="i2")

    values, mask = unpack(raw, {})

    assert values is raw
    assert mask is None


def test_unpack_scale_and_offset() -> None:
    raw = np.array([0, 10, 20], dtype="i2")

    values, mask = unpack(raw, {"scale_factor": 0.5, "add_offset": 100.0})

    assert values.dtype == np.dtype("f8")
    assert values.tolist() == [100.0, 105.0, 110.0]
    assert mask is None


def test_unpack_only_offset_or_only_scale() -> None:
    raw = np.array([2], dtype="i2")

    assert unpack(raw, {"add_offset": 1.5})[0].tolist() == [3.5]
    assert unpack(raw, {"scale_factor": 3})[0].tolist() == [6.0]


def test_unpack_fill_value_masks_before_scaling() -> None:
    raw = np.array([-32768, 0, 4], dtype="i2")

    values, mask = unpack(
        raw, {"_FillValue": -32768, "scale_factor": 0.5, "add_offset": 10.0}
    )

    assert mask is not None
    assert mask.tolist() == [True, False, False]
    assert values[1:].tolist() == [10.0, 12.0]


def test_unpack_missing_value_scalar_and_list_combine_with_fill_value() -> None:
    raw = np.array([-1, -2, -3, 4], dtype="i2")

    _, mask = unpack(raw, {"_FillValue": -1, "missing_value": [-2, -3]})

    assert mask is not None
    assert mask.tolist() == [True, True, True, False]


def test_unpack_nan_fill_value() -> None:
    raw = np.array([1.0, np.nan, 3.0])

    for fill in (np.nan, "NaN"):
        _, mask = unpack(raw, {"_FillValue": fill})
        assert mask is not None
        assert mask.tolist() == [False, True, False]


def test_unpack_nan_is_not_missing_unless_declared() -> None:
    _, mask = unpack(np.array([1.0, np.nan]), {})

    assert mask is None


def test_unpack_no_matching_value_gives_no_mask() -> None:
    _, mask = unpack(np.array([1, 2], dtype="i2"), {"_FillValue": -999})

    assert mask is None


def test_unpack_unsigned_fill_value() -> None:
    raw = np.array([255, 3], dtype="u1")

    _, mask = unpack(raw, {"_FillValue": 255})

    assert mask is not None
    assert mask.tolist() == [True, False]


def test_unpack_ignores_unusable_fill_values() -> None:
    _, mask = unpack(np.array([1, 2], dtype="i2"), {"_FillValue": "not a number"})

    assert mask is None
