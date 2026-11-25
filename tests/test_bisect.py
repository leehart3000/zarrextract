"""Selection by bisection must match selection on the whole coordinate."""

import datetime as dt
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
import zarr
from zarr.storage import LocalStore

from helpers import ZarrFormat, write_array
from test_extract import CountingStore
from zarrextract import ZarrExtractError
from zarrextract._extract import _extract_from_group

CHUNK = 5


@pytest.fixture(params=[2, 3], ids=["zarr-v2", "zarr-v3"])
def zarr_format(request: pytest.FixtureRequest) -> ZarrFormat:
    fmt: ZarrFormat = request.param
    return fmt


def make_line(
    path: Path,
    zarr_format: ZarrFormat,
    name: str,
    values: Any,
    attrs: dict[str, Any],
    chunk: int = CHUNK,
) -> Path:
    """Write a coordinate called ``name`` and a variable ``v`` on it."""
    coords = np.asarray(values)
    group = zarr.open_group(path, mode="w", zarr_format=zarr_format)
    write_array(group, name, coords, (name,), chunks=(chunk,), attrs=attrs)
    write_array(group, "v", np.arange(coords.size, dtype="f8"), (name,))
    return path


def run(path: Path, selection: dict[str, Any]) -> pa.Table | tuple[type, str]:
    group = zarr.open_group(path, mode="r")
    try:
        return _extract_from_group(group, "v", selection)
    except ZarrExtractError as exc:
        return type(exc), str(exc)


def assert_same(path: Path, selection: dict[str, Any]) -> None:
    """Compare the bisection path with the whole-read path."""
    patch = pytest.MonkeyPatch()
    try:
        bisected = run(path, selection)
        patch.setattr("zarrextract._extract.BISECT_MIN_CHUNKS", 10**9)
        expected = run(path, selection)
    finally:
        patch.undo()
    if isinstance(expected, tuple):
        assert bisected == expected, selection
    else:
        assert isinstance(bisected, pa.Table), (selection, bisected)
        assert bisected.equals(expected), selection


Case = tuple[str, Any, dict[str, Any], Callable[[np.random.Generator], Any]]


def _floats(low: float, high: float) -> Callable[[np.random.Generator], Any]:
    return lambda rng: float(rng.uniform(low, high))


def _grid_or_off(values: Any, pad: float) -> Callable[[np.random.Generator], Any]:
    """A value on the grid, just off the grid, or anywhere near the grid."""
    grid = np.asarray(values, dtype="f8")
    span = float(grid.max() - grid.min())

    def pick(rng: np.random.Generator) -> float:
        kind = int(rng.integers(3))
        on = float(rng.choice(grid))
        if kind == 0:
            return on
        if kind == 1:
            return on + float(rng.choice([-1e-9, 1e-9, 0.5, -0.5])) * span / grid.size
        return float(rng.uniform(grid.min() - pad, grid.max() + pad))

    return pick


def _days(rng: np.random.Generator) -> str:
    day = np.datetime64("2000-01-01") + np.timedelta64(int(rng.integers(-15, 90)), "D")
    if rng.integers(2):
        return str(day)
    return f"{day}T{int(rng.integers(24)):02d}:00:00Z"


CASES: dict[str, Case] = {
    "ascending": (
        "y",
        np.linspace(-20.0, 80.0, 61),
        {},
        _grid_or_off(np.linspace(-20.0, 80.0, 61), 10),
    ),
    "descending": (
        "lat",
        np.linspace(90.0, -90.0, 73),
        {"units": "degrees_north"},
        _grid_or_off(np.linspace(90.0, -90.0, 73), 10),
    ),
    "integers": (
        "level",
        np.arange(100, 400, 5),
        {},
        _grid_or_off(np.arange(100, 400, 5), 20),
    ),
    "time-days": (
        "time",
        np.arange(0, 60),
        {"units": "days since 2000-01-01"},
        _days,
    ),
    "time-hours-float": (
        "time",
        np.arange(0.0, 24 * 60, 6.0),
        {"units": "hours since 2000-01-01 00:00:00"},
        _days,
    ),
    "time-descending": (
        "time",
        np.arange(60, 0, -1),
        {"units": "days since 2000-01-01"},
        _days,
    ),
    "lon-0-360-global": (
        "lon",
        np.arange(0.0, 360.0, 5.0),
        {"units": "degrees_east"},
        _floats(-400, 760),
    ),
    "lon-180-global": (
        "lon",
        np.arange(-180.0, 180.0, 5.0),
        {"units": "degrees_east"},
        _floats(-400, 760),
    ),
    "lon-0-360-regional": (
        "lon",
        np.arange(100.0, 250.0, 2.5),
        {"units": "degrees_east"},
        _floats(-300, 600),
    ),
    "lon-180-regional": (
        "lon",
        np.arange(-120.0, 30.0, 2.5),
        {"units": "degrees_east"},
        _floats(-300, 600),
    ),
    "lon-descending": (
        "lon",
        np.arange(175.0, -180.0, -5.0),
        {"units": "degrees_east"},
        _floats(-400, 760),
    ),
}


@pytest.mark.parametrize("case", CASES)
def test_bisection_matches_the_whole_read(
    tmp_path: Path, zarr_format: ZarrFormat, case: str
) -> None:
    name, values, attrs, draw = CASES[case]
    path = make_line(tmp_path / "line.zarr", zarr_format, name, values, attrs)
    rng = np.random.default_rng(20240607)

    for _ in range(60):
        assert_same(path, {name: draw(rng)})
    for _ in range(60):
        assert_same(path, {name: (draw(rng), draw(rng))})


@pytest.mark.parametrize("case", [c for c in CASES if c.startswith("lon-")])
def test_bisection_matches_the_whole_read_for_wrapping_ranges(
    tmp_path: Path, zarr_format: ZarrFormat, case: str
) -> None:
    name, values, attrs, draw = CASES[case]
    path = make_line(tmp_path / "line.zarr", zarr_format, name, values, attrs)
    rng = np.random.default_rng(7)

    tables = 0
    for _ in range(80):
        first, second = draw(rng), draw(rng)
        for selection in ((first, second), (second, first)):
            assert_same(path, {name: selection})
            tables += isinstance(run(path, {name: selection}), pa.Table)
    assert tables > 20


def test_wrapping_ranges_on_both_paths_give_the_same_longitudes(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    path = make_line(
        tmp_path / "line.zarr",
        zarr_format,
        "lon",
        np.arange(0.0, 360.0, 5.0),
        {"units": "degrees_east"},
    )
    expected = [345.0, 350.0, 355.0, 0.0, 5.0, 10.0]

    for selection in ((-15, 10), (345, 10), (345, 370)):
        assert_same(path, {"lon": selection})
        table = run(path, {"lon": selection})
        assert isinstance(table, pa.Table)
        assert table["lon"].to_pylist() == expected


def test_bisection_matches_for_edge_selections(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    path = make_line(
        tmp_path / "line.zarr",
        zarr_format,
        "lon",
        np.arange(0.0, 360.0, 5.0),
        {"units": "degrees_east"},
    )
    selections: list[Any] = [
        (0, 360),
        (-10, 10),
        (350, 370),
        (10, 350),
        (350, 10),
        (359.9, 359.95),
        (1, 2),
        (0, 0),
        "text",
        (1, 2, 3),
        (np.float32(5), np.int64(40)),
    ]
    for selection in selections:
        assert_same(path, {"lon": selection})


def test_bisection_matches_for_other_dimensions_and_zarr_formats(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    # The longitude axis has too few chunks to be bisected, time has enough.
    group = zarr.open_group(tmp_path / "g.zarr", mode="w", zarr_format=zarr_format)
    time = np.arange(0, 80)
    lon = np.arange(0.0, 360.0, 90.0)
    write_array(
        group,
        "time",
        time,
        ("time",),
        chunks=(5,),
        attrs={"units": "days since 2000-01-01"},
    )
    write_array(group, "lon", lon, ("lon",), attrs={"units": "degrees_east"})
    data = np.arange(time.size * lon.size, dtype="f8").reshape(time.size, lon.size)
    write_array(group, "v", data, ("time", "lon"), chunks=(5, 2))

    for day in ("2000-01-01", "2000-02-20", "2000-03-20", "2000-04-01"):
        for selection in ({"time": day, "lon": 100}, {"time": (day, "2000-03-01")}):
            assert_same(tmp_path / "g.zarr", selection)


def test_a_selected_range_of_dates_gives_the_right_rows(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    path = make_line(
        tmp_path / "t.zarr",
        zarr_format,
        "time",
        np.arange(0, 60),
        {"units": "days since 2000-01-01"},
    )

    table = run(path, {"time": (dt.date(2000, 1, 11), dt.date(2000, 1, 13))})

    assert isinstance(table, pa.Table)
    assert table["v"].to_pylist() == [10.0, 11.0, 12.0]


# --- reading only what is needed ---------------------------------------------


def counted_line(
    path: Path, selection: dict[str, Any]
) -> tuple[pa.Table, CountingStore]:
    store = CountingStore(LocalStore(path, read_only=True))
    group = zarr.open_group(store=store, mode="r")
    return _extract_from_group(group, "v", selection), store


@pytest.fixture
def long_line(tmp_path: Path, zarr_format: ZarrFormat) -> Path:
    # 1,000 chunks of 5 values.
    return make_line(
        tmp_path / "long.zarr",
        zarr_format,
        "time",
        np.arange(5000),
        {"units": "days since 2000-01-01"},
    )


def test_a_point_on_a_long_coordinate_reads_few_chunks(long_line: Path) -> None:
    table, store = counted_line(long_line, {"time": "2008-06-01"})

    assert table["v"].to_pylist() == [
        float((dt.date(2008, 6, 1) - dt.date(2000, 1, 1)).days)
    ]
    assert 0 < len(store.chunk_reads("time")) <= 30


def test_a_range_on_a_long_coordinate_reads_few_chunks(long_line: Path) -> None:
    table, store = counted_line(long_line, {"time": ("2008-06-01", "2008-06-04")})

    assert table.num_rows == 4
    assert 0 < len(store.chunk_reads("time")) <= 30


def test_a_range_across_the_seam_reads_only_its_two_ends(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    # 720 longitudes in 144 chunks of 5, and the variable has the same chunks.
    lon = np.arange(0.0, 360.0, 0.5)
    group = zarr.open_group(tmp_path / "w.zarr", mode="w", zarr_format=zarr_format)
    write_array(
        group, "lon", lon, ("lon",), chunks=(5,), attrs={"units": "degrees_east"}
    )
    write_array(group, "v", np.arange(720.0), ("lon",), chunks=(5,))

    table, store = counted_line(tmp_path / "w.zarr", {"lon": (-10, 10)})

    assert table["lon"].to_pylist() == [*lon[700:], *lon[:21]]
    assert table["v"].to_pylist() == [*range(700, 720), *range(21)]
    # Indices 700..719 are 4 chunks and 0..20 are 5 chunks.
    assert len(store.chunk_reads("v")) == 9
    assert len(store.chunk_reads("lon")) <= 30


def test_a_coordinate_with_few_chunks_is_read_whole(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    path = make_line(
        tmp_path / "short.zarr", zarr_format, "x", np.arange(40.0), {}, chunk=5
    )

    _, store = counted_line(path, {"x": 20.0})

    assert len(store.chunk_reads("x")) == 8
