import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
import zarr
from zarr.abc.store import ByteRequest
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.storage import LocalStore, WrapperStore

import zarrextract
from helpers import ZarrFormat, make_grid, make_store, write_array
from zarrextract import extract
from zarrextract._extract import _extract_from_group
from zarrextract._select import Axis, select

LATS = [90.0, 45.0, 0.0, -45.0, -90.0]
LONS = [0.0, 90.0, 180.0, 270.0]


def expected_temp(t: int, y: int, x: int) -> float:
    return float(100 * t + 10 * y + x)


@pytest.fixture(params=[2, 3], ids=["zarr-v2", "zarr-v3"])
def zarr_format(request: pytest.FixtureRequest) -> ZarrFormat:
    fmt: ZarrFormat = request.param
    return fmt


@pytest.fixture
def grid(tmp_path: Path, zarr_format: ZarrFormat) -> Path:
    return make_grid(tmp_path / "grid.zarr", zarr_format)


# --- the three kinds of extraction -------------------------------------------


def test_nearest_point(grid: Path) -> None:
    table = extract(grid, "temp", time="2000-01-02", lat=44, lon=100)

    assert table.column_names == ["time", "lat", "lon", "temp"]
    assert table.to_pydict() == {
        "time": [dt.datetime(2000, 1, 2)],
        "lat": [45.0],
        "lon": [90.0],
        "temp": [expected_temp(1, 1, 1)],
    }


def test_time_series(grid: Path) -> None:
    table = extract(grid, "temp", lat=0, lon=270)

    assert table.to_pydict() == {
        "time": [dt.datetime(2000, 1, d) for d in (1, 2, 3, 4)],
        "lat": [0.0] * 4,
        "lon": [270.0] * 4,
        "temp": [expected_temp(t, 2, 3) for t in range(4)],
    }


def test_bounding_box_is_in_c_order(grid: Path) -> None:
    table = extract(
        grid, "temp", time=("2000-01-02", "2000-01-03"), lat=(-50, 50), lon=(80, 200)
    )

    cells = [(t, y, x) for t in (1, 2) for y in (1, 2, 3) for x in (1, 2)]
    assert table.num_rows == 12
    assert table.to_pydict() == {
        "time": [dt.datetime(2000, 1, 1 + t) for t, _, _ in cells],
        "lat": [LATS[y] for _, y, _ in cells],
        "lon": [LONS[x] for _, _, x in cells],
        "temp": [expected_temp(t, y, x) for t, y, x in cells],
    }


def test_unselected_dimensions_are_kept_whole(grid: Path) -> None:
    table = extract(grid, "temp")

    assert table.num_rows == 4 * 5 * 4
    assert table["temp"].to_pylist() == [
        expected_temp(t, y, x) for t in range(4) for y in range(5) for x in range(4)
    ]
    assert table["lat"].to_pylist()[:5] == [90.0] * 4 + [45.0]


def test_column_types(grid: Path) -> None:
    schema = extract(grid, "temp", lat=0).schema

    assert schema.field("time").type == pa.timestamp("s")
    assert schema.field("lat").type == pa.float64()
    assert schema.field("lon").type == pa.float64()
    assert schema.field("temp").type == pa.float64()


def test_selection_keeps_dimension_of_length_one(grid: Path) -> None:
    table = extract(grid, "temp", time="2000-01-01", lat=0, lon=0)

    assert table.num_rows == 1
    assert table.column_names == ["time", "lat", "lon", "temp"]


def test_inclusive_range_includes_both_ends(grid: Path) -> None:
    table = extract(grid, "temp", lat=(45, 0), lon=0, time="2000-01-01")

    assert table["lat"].to_pylist() == [45.0, 0.0]


def test_float_coordinates_tolerate_rounding(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    lat = np.arange(0, 1.0, 0.1)  # 0.30000000000000004 and friends
    store = make_grid(tmp_path / "g.zarr", zarr_format, lat=lat)

    table = extract(store, "temp", time="2000-01-01", lon=0, lat=(0.1, 0.3))

    assert table.num_rows == 3


def test_extract_accepts_str_and_path(grid: Path) -> None:
    assert extract(grid, "temp", lat=0) == extract(str(grid), "temp", lat=0)


# --- descending coordinates --------------------------------------------------


def test_descending_latitude_range(grid: Path) -> None:
    table = extract(grid, "temp", time="2000-01-01", lon=0, lat=(-50, 50))

    assert table["lat"].to_pylist() == [45.0, 0.0, -45.0]
    assert table["temp"].to_pylist() == [expected_temp(0, y, 0) for y in (1, 2, 3)]


def test_range_ends_can_be_given_in_either_order(grid: Path) -> None:
    low_high = extract(grid, "temp", lat=(-50, 50))
    high_low = extract(grid, "temp", lat=(50, -50))

    assert low_high == high_low


def test_descending_latitude_nearest(grid: Path) -> None:
    far_south = extract(grid, "temp", time="2000-01-01", lon=0, lat=-80)
    north = extract(grid, "temp", time="2000-01-01", lon=0, lat=60)

    assert far_south["lat"].to_pylist() == [-90.0]
    assert north["lat"].to_pylist() == [45.0]


def test_descending_time_and_ascending_latitude(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    store = make_grid(
        tmp_path / "g.zarr",
        zarr_format,
        time=[3, 2, 1, 0],
        lat=[-45.0, 0.0, 45.0],
    )

    table = extract(store, "temp", time=("2000-01-02", "2000-01-03"), lat=10, lon=0)

    assert table["time"].to_pylist() == [
        dt.datetime(2000, 1, 3),
        dt.datetime(2000, 1, 2),
    ]
    assert table["lat"].to_pylist() == [0.0, 0.0]


# --- longitude conventions ---------------------------------------------------


def test_longitude_in_minus_180_to_180_on_a_0_to_360_store(grid: Path) -> None:
    point = extract(grid, "temp", time="2000-01-01", lat=0, lon=-90)
    box = extract(grid, "temp", time="2000-01-01", lat=0, lon=(-100, -80))

    assert point["lon"].to_pylist() == [270.0]
    assert box["lon"].to_pylist() == [270.0]


def test_longitude_in_0_to_360_on_a_minus_180_to_180_store(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    store = make_grid(tmp_path / "g.zarr", zarr_format, lon=[-180.0, -90.0, 0.0, 90.0])

    point = extract(store, "temp", time="2000-01-01", lat=0, lon=270)
    box = extract(store, "temp", time="2000-01-01", lat=0, lon=(190, 280))

    assert point["lon"].to_pylist() == [-90.0]
    assert box["lon"].to_pylist() == [-90.0]


def test_longitude_range_in_the_stores_own_convention(grid: Path) -> None:
    table = extract(grid, "temp", time="2000-01-01", lat=0, lon=(80, 200))

    assert table["lon"].to_pylist() == [90.0, 180.0]


def test_full_circle_of_longitudes_selects_everything(grid: Path) -> None:
    table = extract(grid, "temp", time="2000-01-01", lat=0, lon=(-180, 180))

    assert table["lon"].to_pylist() == LONS


def test_nearest_longitude_wraps_on_a_global_grid(grid: Path) -> None:
    # 340 degrees is 20 from 0 across the seam, but 70 from 270.
    east = extract(grid, "temp", time="2000-01-01", lat=0, lon=340)
    west = extract(grid, "temp", time="2000-01-01", lat=0, lon=-20)

    assert east["lon"].to_pylist() == [0.0]
    assert west["lon"].to_pylist() == [0.0]


@pytest.mark.parametrize(
    ("lon_values", "box"),
    [
        ([-180.0, -90.0, 0.0, 90.0], (170, -170)),
        ([0.0, 90.0, 180.0, 270.0], (-10, 10)),
        ([0.0, 90.0, 180.0, 270.0], (350, 10)),
    ],
)
def test_box_crossing_the_antimeridian_raises(
    tmp_path: Path,
    zarr_format: ZarrFormat,
    lon_values: list[float],
    box: tuple[float, float],
) -> None:
    store = make_grid(tmp_path / "g.zarr", zarr_format, lon=lon_values)

    with pytest.raises(zarrextract.AntimeridianError, match="wraps around"):
        extract(store, "temp", lon=box)


def test_longitude_found_by_name_when_there_are_no_attributes(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    group = zarr.open_group(tmp_path / "g.zarr", mode="w", zarr_format=zarr_format)
    write_array(group, "lon", [0.0, 90.0, 180.0, 270.0], ("lon",))
    write_array(group, "v", [1.0, 2.0, 3.0, 4.0], ("lon",))

    table = extract(tmp_path / "g.zarr", "v", lon=-90)

    assert table["lon"].to_pylist() == [270.0]


# --- packed data and fill values ---------------------------------------------


def test_packed_integers_with_fill_values(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    raw = np.arange(4 * 5 * 4, dtype="i2").reshape(4, 5, 4)
    raw[0, 0, 0] = -32768  # _FillValue
    raw[1, 1, 1] = -32767  # missing_value
    store = make_grid(
        tmp_path / "g.zarr",
        zarr_format,
        data=raw,
        data_attrs={
            "scale_factor": 0.5,
            "add_offset": 10.0,
            "_FillValue": -32768,
            "missing_value": -32767,
        },
    )

    table = extract(store, "temp")

    temp = table["temp"]
    assert temp.type == pa.float64()
    assert temp.null_count == 2
    values = temp.to_pylist()
    assert values[0] is None
    assert values[1 * 20 + 1 * 4 + 1] is None
    assert values[1] == 1 * 0.5 + 10.0
    assert values[79] == 79 * 0.5 + 10.0


def test_fill_value_without_packing_keeps_integers(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    raw = np.arange(4 * 5 * 4, dtype="i2").reshape(4, 5, 4)
    raw[0, 0, 1] = -1
    store = make_grid(
        tmp_path / "g.zarr", zarr_format, data=raw, data_attrs={"_FillValue": -1}
    )

    table = extract(store, "temp", time="2000-01-01", lat=90, lon=(0, 90))

    assert table["temp"].type == pa.int16()
    assert table["temp"].to_pylist() == [0, None]


def test_nan_is_null_only_when_declared_as_a_fill_value(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    raw = np.zeros((4, 5, 4))
    raw[0, 0, 0] = np.nan
    plain = make_grid(tmp_path / "a.zarr", zarr_format, data=raw)
    declared = make_grid(
        tmp_path / "b.zarr", zarr_format, data=raw, data_attrs={"_FillValue": "NaN"}
    )
    selection: dict[str, Any] = {"time": "2000-01-01", "lat": 90, "lon": 0}

    assert extract(plain, "temp", **selection)["temp"].null_count == 0
    assert extract(declared, "temp", **selection)["temp"].null_count == 1


# --- time --------------------------------------------------------------------


@pytest.mark.parametrize(
    "selection",
    [
        "2000-01-02",
        "2000-01-02T00:00:00",
        "2000-01-02 00:00:00",
        "2000-01-02T00:00:00Z",
        "2000-01-02T00:00:00+00:00",
        "2000-01-02T10:00",  # nearest day
        np.datetime64("2000-01-02"),
        np.datetime64("2000-01-02T00:00:00"),
        np.datetime64("2000-01-02T00:00:00.000000000"),
        dt.datetime(2000, 1, 2),
        dt.datetime(2000, 1, 2, tzinfo=dt.UTC),
        dt.datetime(2000, 1, 2, 5, tzinfo=dt.timezone(dt.timedelta(hours=5))),
        dt.date(2000, 1, 2),
    ],
)
def test_time_selection_types(grid: Path, selection: Any) -> None:
    table = extract(grid, "temp", time=selection, lat=0, lon=0)

    assert table["time"].to_pylist() == [dt.datetime(2000, 1, 2)]


def test_time_range_with_mixed_types(grid: Path) -> None:
    table = extract(
        grid,
        "temp",
        time=("2000-01-02T12:00", np.datetime64("2000-01-04")),
        lat=0,
        lon=0,
    )

    assert table["time"].to_pylist() == [
        dt.datetime(2000, 1, 3),
        dt.datetime(2000, 1, 4),
    ]


def test_cf_time_with_hours_and_unpadded_reference(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    store = make_grid(
        tmp_path / "g.zarr",
        zarr_format,
        time=[0, 6, 12, 18],
        time_attrs={"units": "hours since 1900-1-1 0:0:0", "calendar": "gregorian"},
    )

    table = extract(
        store, "temp", time=("1900-01-01T05:00Z", "1900-01-01T13:00"), lat=0, lon=0
    )

    assert table["time"].to_pylist() == [
        dt.datetime(1900, 1, 1, 6),
        dt.datetime(1900, 1, 1, 12),
    ]


@pytest.mark.parametrize(
    ("units", "unit", "one_step"),
    [
        ("seconds since 2000-01-01", "s", dt.timedelta(seconds=1)),
        ("milliseconds since 2000-01-01", "ms", dt.timedelta(milliseconds=1)),
        ("microseconds since 2000-01-01", "us", dt.timedelta(microseconds=1)),
    ],
)
def test_timestamp_unit_matches_the_time_unit(
    tmp_path: Path,
    zarr_format: ZarrFormat,
    units: str,
    unit: str,
    one_step: dt.timedelta,
) -> None:
    store = make_grid(
        tmp_path / "g.zarr",
        zarr_format,
        time=[0, 1, 2, 3],
        time_attrs={"units": units},
    )

    table = extract(store, "temp", lat=0, lon=0)

    assert table.schema.field("time").type == pa.timestamp(unit)
    assert table["time"].to_pylist()[1] == dt.datetime(2000, 1, 1) + one_step


def test_time_axis_running_to_the_year_2300(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    first = int((np.datetime64("2299-12-30") - np.datetime64("1850-01-01")).astype(int))
    store = make_grid(
        tmp_path / "g.zarr",
        zarr_format,
        time=[first, first + 1, first + 2, first + 3],
        time_attrs={"units": "days since 1850-01-01", "calendar": "standard"},
    )

    table = extract(store, "temp", time=("2299-12-31", "2300-01-01"), lat=0, lon=0)

    assert table.schema.field("time").type == pa.timestamp("s")
    assert table["time"].to_pylist() == [
        dt.datetime(2299, 12, 31),
        dt.datetime(2300, 1, 1),
    ]


def test_time_without_units_is_numeric(tmp_path: Path, zarr_format: ZarrFormat) -> None:
    store = make_grid(
        tmp_path / "g.zarr", zarr_format, time=[10, 20, 30, 40], time_attrs={}
    )

    table = extract(store, "temp", time=(15, 35), lat=0, lon=0)

    assert table["time"].to_pylist() == [20, 30]


def test_unsupported_calendar_raises(tmp_path: Path, zarr_format: ZarrFormat) -> None:
    store = make_grid(
        tmp_path / "g.zarr",
        zarr_format,
        time_attrs={"units": "days since 2000-01-01", "calendar": "noleap"},
    )

    with pytest.raises(zarrextract.UnsupportedCalendarError, match="noleap"):
        extract(store, "temp", lat=0, lon=0)


def test_bad_time_units_raise(tmp_path: Path, zarr_format: ZarrFormat) -> None:
    store = make_grid(
        tmp_path / "g.zarr",
        zarr_format,
        time_attrs={"units": "days since 2000-01-01 00:00:00 +05:00"},
    )

    with pytest.raises(zarrextract.TimeDecodingError, match="UTC offset"):
        extract(store, "temp", lat=0, lon=0)


def test_dimension_without_a_coordinate_uses_its_index(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    group = zarr.open_group(tmp_path / "g.zarr", mode="w", zarr_format=zarr_format)
    write_array(group, "v", np.arange(12.0).reshape(3, 4), ("y", "x"))

    table = extract(tmp_path / "g.zarr", "v", y=(1, 2), x=3)

    assert table.to_pydict() == {"y": [1, 2], "x": [3, 3], "v": [7.0, 11.0]}


def test_scalar_variable(tmp_path: Path, zarr_format: ZarrFormat) -> None:
    group = zarr.open_group(tmp_path / "g.zarr", mode="w", zarr_format=zarr_format)
    write_array(group, "v", np.array(5.0), ())

    assert extract(tmp_path / "g.zarr", "v").to_pydict() == {"v": [5.0]}


# --- scalar selections outside the coordinates -------------------------------


@pytest.mark.parametrize(
    ("selection", "expected_lat"),
    [
        ({"lat": 134}, 90.0),  # 44 above the top, step is 45
        ({"lat": -134}, -90.0),
    ],
)
def test_scalar_within_one_step_outside_is_accepted(
    grid: Path, selection: dict[str, float], expected_lat: float
) -> None:
    table = extract(grid, "temp", time="2000-01-01", lon=0, **selection)

    assert table["lat"].to_pylist() == [expected_lat]


@pytest.mark.parametrize("lat", [136, -136, 1000])
def test_scalar_beyond_one_step_outside_raises(grid: Path, lat: float) -> None:
    with pytest.raises(zarrextract.SelectionError, match="one grid step"):
        extract(grid, "temp", lat=lat)


@pytest.mark.parametrize("lon", [-10, 30])
def test_longitude_scalar_just_outside_a_regional_grid(
    tmp_path: Path, zarr_format: ZarrFormat, lon: float
) -> None:
    store = make_grid(tmp_path / "g.zarr", zarr_format, lon=[0.0, 10.0, 20.0, 30.0])

    table = extract(store, "temp", time="2000-01-01", lat=0, lon=lon)

    assert table["lon"].to_pylist() == [min(max(lon, 0.0), 30.0)]


@pytest.mark.parametrize("lon", [-10.5, 40.5, 100])
def test_longitude_scalar_far_outside_a_regional_grid(
    tmp_path: Path, zarr_format: ZarrFormat, lon: float
) -> None:
    store = make_grid(tmp_path / "g.zarr", zarr_format, lon=[0.0, 10.0, 20.0, 30.0])

    with pytest.raises(zarrextract.SelectionError, match="one grid step"):
        extract(store, "temp", lon=lon)


@pytest.mark.parametrize(
    ("time", "accepted"),
    [
        ("2000-01-05", True),
        ("1999-12-31", True),
        ("2000-01-05T00:00:01", False),
        ("1999-12-30", False),
        ("2000-01-06", False),
    ],
)
def test_time_scalar_outside_the_axis(grid: Path, time: str, accepted: bool) -> None:
    if accepted:
        assert extract(grid, "temp", time=time, lat=0, lon=0).num_rows == 1
    else:
        with pytest.raises(zarrextract.SelectionError, match="one grid step"):
            extract(grid, "temp", time=time, lat=0, lon=0)


def test_single_value_axis_accepts_only_that_value(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    store = make_grid(tmp_path / "g.zarr", zarr_format, lat=[10.0])

    assert extract(store, "temp", time="2000-01-01", lon=0, lat=10)[
        "lat"
    ].to_pylist() == [10.0]
    with pytest.raises(zarrextract.SelectionError, match="single value"):
        extract(store, "temp", lat=11)


# --- errors ------------------------------------------------------------------


def test_unknown_variable(grid: Path) -> None:
    with pytest.raises(zarrextract.VariableNotFoundError, match="temp") as info:
        extract(grid, "nope")

    assert isinstance(info.value, KeyError)
    assert isinstance(info.value, zarrextract.ZarrExtractError)


def test_group_is_not_a_variable(grid: Path) -> None:
    zarr.open_group(grid, mode="a").create_group("sub")

    with pytest.raises(zarrextract.VariableNotFoundError, match="group"):
        extract(grid, "sub")


def test_unknown_dimension(grid: Path) -> None:
    with pytest.raises(zarrextract.UnknownDimensionError, match="depth") as info:
        extract(grid, "temp", depth=1)

    assert "time, lat, lon" in str(info.value)


def test_missing_dimension_names(tmp_path: Path, zarr_format: ZarrFormat) -> None:
    store = make_store(tmp_path / "s.zarr", zarr_format, with_dims=False)

    with pytest.raises(zarrextract.MissingDimensionNamesError):
        extract(store, "temp")


def test_coordinate_cannot_be_extracted(grid: Path) -> None:
    with pytest.raises(zarrextract.ZarrExtractError, match="coordinate"):
        extract(grid, "lat")


def test_time_errors_show_dates_not_raw_numbers(grid: Path) -> None:
    with pytest.raises(zarrextract.EmptySelectionError) as empty:
        extract(grid, "temp", time=("2001-01-01", "2001-02-01"))
    with pytest.raises(zarrextract.SelectionError) as outside:
        extract(grid, "temp", time="2000-01-06")

    assert "2001-01-01T00:00:00" in str(empty.value)
    assert "2000-01-04T00:00:00" in str(empty.value)
    assert "2000-01-06T00:00:00" in str(outside.value)
    assert "(86400 s)" in str(outside.value)


@pytest.mark.parametrize(
    "selection",
    [
        {"lat": (100, 200)},
        {"lat": (20, 30)},
        {"time": ("2001-01-01", "2001-02-01")},
        {"lon": (10, 20)},
    ],
)
def test_empty_selection(grid: Path, selection: dict[str, Any]) -> None:
    with pytest.raises(zarrextract.EmptySelectionError):
        extract(grid, "temp", **selection)


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        ({"lat": (1, 2, 3)}, "pair"),
        ({"lat": ()}, "pair"),
        ({"lat": "north"}, "number"),
        ({"lat": True}, "number"),
        ({"lat": None}, "number"),
        ({"lat": (0, "north")}, "number"),
        ({"time": 3}, "ISO string"),
        ({"time": "not a date"}, "date or time"),
        ({"time": "2000-01-01T00:00:00+05:00"}, "UTC offset"),
        ({"time": np.datetime64("NaT", "s")}, "not a valid time"),
        ({"time": ("2000-01-01", 2)}, "ISO string"),
    ],
)
def test_invalid_selections(
    grid: Path, selection: dict[str, Any], message: str
) -> None:
    with pytest.raises(zarrextract.SelectionError, match=message):
        extract(grid, "temp", **selection)


def test_selection_error_hierarchy_is_reachable(grid: Path) -> None:
    with pytest.raises(zarrextract.ZarrExtractError):
        extract(grid, "temp", lat=(1, 2, 3))  # type: ignore[arg-type]


def test_coordinate_that_is_not_monotonic(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    store = make_grid(tmp_path / "g.zarr", zarr_format, lat=[0.0, 10.0, 5.0])

    with pytest.raises(zarrextract.SelectionError, match="strictly"):
        extract(store, "temp", lat=5)
    # Dimensions that are not selected on are not checked.
    assert extract(store, "temp", time="2000-01-01", lon=0).num_rows == 3


def test_unreadable_store(tmp_path: Path) -> None:
    with pytest.raises(zarrextract.StoreOpenError):
        extract(tmp_path / "missing.zarr", "temp")


def test_select_on_non_numeric_axis() -> None:
    axis = Axis("station", np.array(["a", "b"]))

    with pytest.raises(zarrextract.SelectionError, match="not numeric"):
        select(axis, 1)


def test_select_on_empty_axis() -> None:
    with pytest.raises(zarrextract.EmptySelectionError):
        select(Axis("x", np.array([], dtype="f8")), 1)


def test_select_on_time_axis_with_nat() -> None:
    values = np.array(["2000-01-01", "NaT"], dtype="datetime64[s]")

    with pytest.raises(zarrextract.SelectionError, match="NaT"):
        select(Axis("time", values, is_time=True), "2000-01-01")


# --- reading only what is needed ---------------------------------------------


class CountingStore(WrapperStore[LocalStore]):
    """A store that records every key that is read."""

    def __init__(self, store: LocalStore) -> None:
        super().__init__(store)
        self.reads: list[str] = []

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        self.reads.append(key)
        return await super().get(key, prototype, byte_range)

    def chunk_reads(self, array: str) -> set[str]:
        metadata = {"zarr.json", ".zarray", ".zattrs", ".zgroup", ".zmetadata"}
        return {
            key
            for key in self.reads
            if key.startswith(f"{array}/") and key.rsplit("/", 1)[-1] not in metadata
        }


def counted_extract(path: Path, **selections: Any) -> tuple[pa.Table, CountingStore]:
    store = CountingStore(LocalStore(path, read_only=True))
    group = zarr.open_group(store=store, mode="r")
    return _extract_from_group(group, "temp", selections), store


def test_a_single_chunk_is_read_for_a_small_box(grid: Path) -> None:
    # Chunks are (1, 2, 2): this box lies inside the chunk at index (2, 0, 0).
    table, store = counted_extract(grid, time="2000-01-03", lat=(90, 45), lon=(0, 90))

    assert table["temp"].to_pylist() == [
        expected_temp(2, y, x) for y in (0, 1) for x in (0, 1)
    ]
    assert len(store.chunk_reads("temp")) == 1


def test_only_chunks_overlapping_the_box_are_read(grid: Path) -> None:
    # 2 time chunks x 1 latitude chunk (rows 2 and 3) x 2 longitude chunks.
    _, store = counted_extract(
        grid, time=("2000-01-02", "2000-01-03"), lat=(0, -45), lon=(0, 180)
    )

    assert len(store.chunk_reads("temp")) == 4


def test_a_point_reads_one_chunk(grid: Path) -> None:
    _, store = counted_extract(grid, time="2000-01-04", lat=-90, lon=270)

    assert len(store.chunk_reads("temp")) == 1


def test_everything_is_read_without_selections(grid: Path) -> None:
    _, store = counted_extract(grid)

    # 4 time steps x 3 latitude chunks x 2 longitude chunks.
    assert len(store.chunk_reads("temp")) == 24


def test_coordinates_are_read_whole(grid: Path) -> None:
    _, store = counted_extract(grid, time="2000-01-04", lat=-90, lon=270)

    for name in ("time", "lat", "lon"):
        assert len(store.chunk_reads(name)) == 1
