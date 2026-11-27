"""Searching a long coordinate takes few rounds and gives the same answers.

A round is one array read of the coordinate, which zarr fetches concurrently.
The reference is the search that these tests replaced: a bisection by index,
which makes one read after another.
"""

import asyncio
import contextlib
import math
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
import zarr
from zarr.abc.store import ByteRequest
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.storage import LocalStore, WrapperStore

import zarrextract._select as select_module
from helpers import ZarrFormat
from test_bisect import make_line, run
from zarrextract import ZarrExtractError
from zarrextract import open as open_store
from zarrextract._probe import LazyAxis
from zarrextract._select import (
    _TOLERANCE,
    Indices,
    _join_runs,
    _numeric_longitude,
    _outside,
    _to_store_lon,
)

DAYS = {"units": "days since 2000-01-01"}


@pytest.fixture(params=[2, 3], ids=["zarr-v2", "zarr-v3"])
def zarr_format(request: pytest.FixtureRequest) -> ZarrFormat:
    fmt: ZarrFormat = request.param
    return fmt


# --- the reference: the bisection by index that was used before --------------


def first_true(test: Callable[[int], bool], low: int, high: int) -> int:
    while low < high:
        middle = (low + high) // 2
        if test(middle):
            high = middle
        else:
            low = middle + 1
    return low


def reference_range(
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

    def run_between(lower: float, upper: float) -> slice | None:
        if increasing:
            begin = first_true(lambda i: axis.value(i) >= lower, 0, axis.size)
            end = first_true(lambda i: axis.value(i) > upper, begin, axis.size)
        else:
            begin = first_true(lambda i: axis.value(i) <= upper, 0, axis.size)
            end = first_true(lambda i: axis.value(i) < lower, begin, axis.size)
        return slice(begin, end) if begin < end else None

    runs = [run for lower, upper in bounds if (run := run_between(lower, upper))]
    return _join_runs(axis, runs, start, stop, low, high)


def reference_nearest(
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
        after = first_true(lambda i: axis.value(i) >= value, 0, axis.size)
    else:
        after = first_true(lambda i: axis.value(i) <= value, 0, axis.size)
    candidates = {max(after - 1, 0), min(after, axis.size - 1)}
    if wraps:
        candidates |= {0, axis.size - 1}
    nearest = min(sorted(candidates), key=distance)
    return slice(nearest, nearest + 1)


# --- counting rounds ----------------------------------------------------------


class Rounds:
    """The chunks that each round of a ``LazyAxis`` read for the first time."""

    def __init__(self) -> None:
        self.reads: list[list[int]] = []

    def __len__(self) -> int:
        return len(self.reads)

    @property
    def chunks(self) -> int:
        return sum(len(read) for read in self.reads)


class observed:
    """Count the rounds of everything done inside the ``with`` block.

    With ``reference=True`` the lines are searched by the reference instead.
    """

    def __init__(self, *, reference: bool = False) -> None:
        self.reference = reference
        self.rounds = Rounds()
        self.patch = pytest.MonkeyPatch()

    def __enter__(self) -> Rounds:
        original = LazyAxis._fetch
        rounds = self.rounds

        def counted(axis: LazyAxis, chunks: Iterable[int]) -> None:
            new = sorted({k for k in chunks if k not in axis._chunks})
            if new:
                rounds.reads.append(new)
            original(axis, new)

        self.patch.setattr(LazyAxis, "_fetch", counted)
        if self.reference:
            self.patch.setattr(select_module, "_bisect_range", reference_range)
            self.patch.setattr(select_module, "_bisect_nearest", reference_nearest)
        return rounds

    def __exit__(self, *exc: object) -> None:
        self.patch.undo()


def result_of(path: Path, selection: dict[str, Any], *, reference: bool = False) -> Any:
    with observed(reference=reference) as rounds:
        result = run(path, selection)
    return result, rounds


# --- the lines ----------------------------------------------------------------


def _months() -> np.ndarray[Any, Any]:
    starts = np.arange("2000-01", "2125-01", dtype="datetime64[M]")
    days: np.ndarray[Any, Any] = (
        starts.astype("datetime64[D]") - np.datetime64("2000-01-01")
    ).astype(int)
    return days


_RNG = np.random.default_rng(5)
_STEPS = np.concatenate([[0.0], np.cumsum(_RNG.uniform(0.5, 1.5, 5999))])
_I = np.arange(6000)

# name: (values, attrs, days the line spans)
LINES: dict[str, tuple[Any, dict[str, Any], int]] = {
    "even": (_I, DAYS, 6000),
    "even-stamped": (
        (24 * _I + 9).astype("f8"),
        {"units": "hours since 2000-01-01"},
        6000,
    ),
    "descending": (_I[::-1], DAYS, 6000),
    "monthly": (_months(), DAYS, 45000),
    "random-steps": (_STEPS, DAYS, 6000),
    "quadratic": (_I + _I**2 / 8000, DAYS, 10500),
    "gap": (np.concatenate([np.arange(3000), np.arange(6650, 9650)]), DAYS, 9650),
}
EVEN = ["even", "even-stamped", "descending"]


class Lines(dict[str, Path]):
    """The stores of the lines, each written the first time it is asked for."""

    def __init__(self, tmp_path: Path, zarr_format: ZarrFormat) -> None:
        super().__init__()
        self.tmp_path = tmp_path
        self.zarr_format = zarr_format

    def __missing__(self, name: str) -> Path:
        values, attrs, _ = LINES[name]
        path = make_line(
            self.tmp_path / f"{name}.zarr", self.zarr_format, "time", values, attrs
        )
        self[name] = path
        return path


@pytest.fixture
def lines(tmp_path: Path, zarr_format: ZarrFormat) -> dict[str, Path]:
    return Lines(tmp_path, zarr_format)


def draw(rng: np.random.Generator, offset: int) -> Any:
    """The day ``offset`` days from the start, as a day, a month, a time of day
    or a datetime64."""
    day = np.datetime64("2000-01-01") + np.timedelta64(offset, "D")
    kind = int(rng.integers(4))
    if kind == 0:
        return str(day)
    if kind == 1:
        return str(day)[:7]
    if kind == 2:
        return f"{day}T{int(rng.integers(24)):02d}:30:00Z"
    return day


def selections(name: str, count: int) -> list[dict[str, Any]]:
    """Points and short ranges in or just off the line, and one range of it all.

    The ranges are short because every chunk of a range is read, and the tests
    run each selection twice.
    """
    span = LINES[name][2]
    rng = np.random.default_rng(len(name))
    chosen: list[dict[str, Any]] = []
    for i in range(count):
        first = int(rng.integers(-40, span + 40))
        if i % 2:
            chosen.append({"time": draw(rng, first)})
        else:
            second = first + int(rng.integers(-60, 60))
            chosen.append({"time": (draw(rng, first), draw(rng, second))})
    chosen.append({"time": ("1999-12-01", "2125-01-01")})
    return chosen


def same(new: Any, expected: Any, selection: dict[str, Any]) -> None:
    if isinstance(expected, tuple):  # the type and the text of an error
        assert new == expected, selection
    else:
        assert isinstance(new, pa.Table), (selection, new)
        assert new.equals(expected), selection


# --- the same answers ---------------------------------------------------------


@pytest.mark.parametrize("name", LINES)
def test_search_gives_the_same_tables_and_errors_as_the_reference(
    lines: dict[str, Path], name: str
) -> None:
    for selection in selections(name, 80):
        new, _ = result_of(lines[name], selection)
        expected, _ = result_of(lines[name], selection, reference=True)
        same(new, expected, selection)


def test_search_gives_the_same_longitudes_across_the_seam(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    lon = np.arange(0.0, 360.0, 0.1)  # 3,600 values in 720 chunks
    path = make_line(
        tmp_path / "w.zarr", zarr_format, "lon", lon, {"units": "degrees_east"}
    )
    rng = np.random.default_rng(3)
    chosen: list[dict[str, Any]] = []
    for _ in range(20):
        # Ranges are short, as a wide one reads most of the 720 chunks.
        first = float(rng.uniform(-400, 760))
        second = first + float(rng.uniform(-40, 40))
        chosen += [{"lon": first}, {"lon": (first, second)}]
    for selection in chosen:
        new, _ = result_of(path, selection)
        expected, _ = result_of(path, selection, reference=True)
        same(new, expected, selection)
    across: list[dict[str, Any]] = [
        {"lon": (-10, 10)},
        {"lon": (350, 5)},
        {"lon": (10, 350)},
    ]
    for selection in across:
        new, _ = result_of(path, selection)
        expected, _ = result_of(path, selection, reference=True)
        same(new, expected, selection)
        assert isinstance(new, pa.Table)


# --- few rounds ---------------------------------------------------------------


@pytest.mark.parametrize("name", EVEN)
def test_a_range_and_a_point_on_an_even_axis_take_few_rounds(
    lines: dict[str, Path], name: str
) -> None:
    ranges: list[dict[str, Any]] = [
        {"time": ("2008-06", "2008-06")},
        {"time": ("2004-02-03", "2011-09-30")},
        {"time": ("2010-01-01T10:00", "2010-01-20T00:00")},
    ]
    for selection in ranges:
        result, rounds = result_of(lines[name], selection)
        assert isinstance(result, pa.Table)
        # The first and last chunks, the search, and the middle of the range.
        assert len(rounds) <= 3, (selection, rounds.reads)
    points: list[dict[str, Any]] = [
        {"time": "2008-06-01"},
        {"time": "2012-12-12T05:00"},
    ]
    for selection in points:
        result, rounds = result_of(lines[name], selection)
        assert isinstance(result, pa.Table)
        assert len(rounds) <= 2, (selection, rounds.reads)


@pytest.mark.parametrize("name", EVEN)
def test_further_selections_on_an_open_handle_take_fewer_rounds(
    lines: dict[str, Path], name: str
) -> None:
    with open_store(lines[name]) as source, observed() as rounds:
        source.extract("v", time=("2008-06", "2008-06"))

        rounds.reads.clear()
        source.extract("v", time=("2008-07", "2008-07"))
        assert len(rounds) <= 2, rounds.reads

        rounds.reads.clear()
        source.extract("v", time="2009-03-15")
        assert len(rounds) <= 1, rounds.reads

        rounds.reads.clear()
        source.extract("v", time=("2008-07", "2008-07"))
        source.extract("v", time="2009-03-15")
        assert rounds.reads == []


@pytest.mark.parametrize(
    "name", [n for n in LINES if n not in ("even", "even-stamped")]
)
def test_other_lines_take_no_more_rounds_than_the_reference(
    lines: dict[str, Path], name: str
) -> None:
    worst = 0
    for selection in selections(name, 40):
        _, rounds = result_of(lines[name], selection)
        _, reference = result_of(lines[name], selection, reference=True)
        assert len(rounds) <= len(reference), (selection, rounds.reads, reference.reads)
        worst = max(worst, len(rounds))
    print(f"{name}: at most {worst} rounds")


# --- what is kept -------------------------------------------------------------


def array_of(path: Path, name: str, **open_options: Any) -> "zarr.Array[Any]":
    node = zarr.open_group(path, mode="r", **open_options)[name]
    assert isinstance(node, zarr.Array)
    return node


def axes_of(source: Any) -> list[LazyAxis]:
    return [
        axis for axis, _ in source._cache.axes.values() if isinstance(axis, LazyAxis)
    ]


@pytest.mark.parametrize("name", LINES)
def test_the_chunks_that_are_kept_are_the_real_chunks(
    lines: dict[str, Path], name: str
) -> None:
    stored = np.asarray(array_of(lines[name], "time")[:])
    with open_store(lines[name]) as source:
        for selection in selections(name, 30):
            # Some selections are off the line and raise.
            with contextlib.suppress(ZarrExtractError):
                source.extract("v", **selection)
        (axis,) = axes_of(source)
        size = axis._chunk_size
        assert axis._cached == sorted(axis._chunks)
        assert len(axis._cached) > 3
        for k, chunk in axis._chunks.items():
            expected = stored[k * size : (k + 1) * size]
            assert chunk.dtype == expected.dtype
            assert chunk.shape == expected.shape
            np.testing.assert_array_equal(chunk, expected)


def test_a_read_of_chunks_that_are_not_next_to_each_other_keeps_whole_chunks(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    # 5,003 values: the last of 1,001 chunks holds 3.
    values = np.arange(5003, dtype="f8") * 2
    path = make_line(tmp_path / "s.zarr", zarr_format, "x", values, {})
    node = array_of(path, "x")
    axis = LazyAxis(
        node, "x", units=None, calendar=None, is_time=False, is_longitude=False
    )
    assert sorted(axis._chunks) == [0, 1000]  # the first and the last chunk

    axis._fetch([7, 3, 1000, 500, 4])  # gaps, and the short chunk at the end
    axis._fetch([3, 8, 9, 501])  # some kept, some new
    axis.read(slice(2, 60))  # a range across chunks that are kept and not

    assert axis._cached == sorted(axis._chunks)
    for k, chunk in axis._chunks.items():
        np.testing.assert_array_equal(chunk, values[k * 5 : (k + 1) * 5], str(k))
    assert axis._chunks[1000].shape == (3,)
    np.testing.assert_array_equal(axis.read(slice(2, 60)), values[2:60])


# --- concurrency --------------------------------------------------------------


class SlowStore(WrapperStore[LocalStore]):
    """A store that is slow to answer and records how many reads overlap."""

    def __init__(self, store: LocalStore) -> None:
        super().__init__(store)
        self.active = 0
        self.most = 0

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        self.active += 1
        self.most = max(self.most, self.active)
        try:
            await asyncio.sleep(0.02)
            return await super().get(key, prototype, byte_range)
        finally:
            self.active -= 1


def test_the_chunks_of_one_round_are_fetched_together(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    path = make_line(tmp_path / "c.zarr", zarr_format, "x", np.arange(5000.0), {})
    store = SlowStore(LocalStore(path, read_only=True))
    group = zarr.open_group(store=store, mode="r")
    node = group["x"]
    assert isinstance(node, zarr.Array)
    axis = LazyAxis(
        node, "x", units=None, calendar=None, is_time=False, is_longitude=False
    )
    assert store.most >= 2  # the first and last chunk, in the first round

    store.most = 0
    axis._fetch([100, 300, 500, 700])

    assert store.most >= 4


@pytest.mark.parametrize("increasing", [True, False], ids=["up", "down"])
def test_search_finds_every_edge_and_reads_nothing_it_knows(
    tmp_path: Path, zarr_format: ZarrFormat, increasing: bool
) -> None:
    values = np.arange(5000.0)[:: 1 if increasing else -1]
    path = make_line(tmp_path / "f.zarr", zarr_format, "x", values, {})
    node = array_of(path, "x")
    axis = LazyAxis(
        node, "x", units=None, calendar=None, is_time=False, is_longitude=False
    )
    edges = [(10.0, False), (10.0, True), (-5.0, False), (1e9, False), (3333.5, True)]

    with observed() as rounds:
        found = axis.search(edges, increasing)
        again = axis.search(edges, increasing)

    # The first index where the test is true, as numpy finds it.
    def expected(threshold: float, strict: bool) -> int:
        if increasing:
            test = values > threshold if strict else values >= threshold
        else:
            test = values < threshold if strict else values <= threshold
        return int(np.argmax(test)) if test.any() else values.size

    assert found == [expected(*edge) for edge in edges]
    assert again == found
    # Five edges on an even axis are found in one round, and the second search
    # of the same edges reads nothing.
    assert len(rounds) == 1, rounds.reads
