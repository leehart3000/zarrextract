"""zarrextract.open() gives a Source that reuses what it has read."""

import datetime as dt
import pickle
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Self

import fsspec
import numpy as np
import pyarrow as pa
import pytest
import zarr
from fsspec.implementations.memory import MemoryFileSystem
from fsspec.registry import register_implementation
from zarr.storage import LocalStore

import zarrextract as zx
from helpers import ZarrFormat, make_grid, make_store
from test_bisect import make_line
from test_extract import CountingStore
from zarrextract import Source

METADATA = {"zarr.json", ".zarray", ".zattrs", ".zgroup", ".zmetadata"}

reads: list[str] = []


class CountingFileSystem(MemoryFileSystem):  # type: ignore[misc]
    """An in-memory filesystem that records every key that is read."""

    protocol = "countingmem"

    @classmethod
    def _strip_protocol(cls, path: str) -> str:
        return str(super()._strip_protocol(path.removeprefix("countingmem://")))

    def cat_file(
        self, path: str, start: int | None = None, end: int | None = None, **kw: Any
    ) -> Any:
        reads.append(path)
        return super().cat_file(path, start, end, **kw)


@pytest.fixture(autouse=True)
def counting_filesystem() -> Iterator[None]:
    reads.clear()
    register_implementation("countingmem", CountingFileSystem, clobber=True)
    CountingFileSystem.clear_instance_cache()
    yield
    CountingFileSystem.clear_instance_cache()
    MemoryFileSystem.store.clear()
    MemoryFileSystem.pseudo_dirs[:] = [""]


@pytest.fixture(params=[2, 3], ids=["zarr-v2", "zarr-v3"])
def zarr_format(request: pytest.FixtureRequest) -> ZarrFormat:
    fmt: ZarrFormat = request.param
    return fmt


@pytest.fixture
def store(tmp_path: Path, zarr_format: ZarrFormat) -> Path:
    return make_store(tmp_path / "s.zarr", zarr_format)


def to_memory(local: Path) -> str:
    """Copy a local store into the memory filesystem and return its URL."""
    fsspec.filesystem("memory").put(str(local), f"/{local.name}", recursive=True)
    return f"countingmem://{local.name}"


def metadata_reads() -> list[str]:
    return [key for key in reads if key.rsplit("/", 1)[-1] in METADATA]


def chunk_reads(root: str, array: str) -> list[str]:
    return [
        key
        for key in reads
        if key.startswith(f"/{root}/{array}/")
        and key.rsplit("/", 1)[-1] not in METADATA
    ]


class FrozenDatetime(dt.datetime):
    @classmethod
    def now(cls, tz: dt.tzinfo | None = None) -> Self:
        return cls(2026, 1, 2, 3, 4, 5, tzinfo=tz)


@pytest.fixture
def frozen_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("zarrextract._provenance.datetime", FrozenDatetime)


# --- reading only once ----------------------------------------------------------


def test_the_second_identical_call_reads_only_the_variable(store: Path) -> None:
    with zx.open(to_memory(store)) as src:
        src.extract("temp", time=0, lat=0)
        reads.clear()

        src.extract("temp", time=0, lat=0)

    assert reads
    assert metadata_reads() == []
    assert chunk_reads("s.zarr", "time") == []
    assert chunk_reads("s.zarr", "lat") == []
    assert chunk_reads("s.zarr", "lon") == []


def test_extract_after_describe_reads_no_metadata(store: Path) -> None:
    with zx.open(to_memory(store)) as src:
        src.describe()
        reads.clear()

        assert src.extract("temp", time=1).num_rows == 8

    assert metadata_reads() == []


def test_the_open_itself_reads_no_arrays(store: Path) -> None:
    zx.open(to_memory(store))

    assert all(key.rsplit("/", 1)[-1] in METADATA for key in reads)
    assert not any("/temp/" in key for key in reads)


def test_a_coordinate_read_whole_is_not_read_again(store: Path) -> None:
    with zx.open(to_memory(store)) as src:
        src.extract("temp", lat=0)  # time and lon are read whole, unselected
        reads.clear()

        src.extract("temp", time=1, lon=90)  # now they are selected

    assert chunk_reads("s.zarr", "time") == []
    assert chunk_reads("s.zarr", "lon") == []

    with zx.open(to_memory(store)) as src:
        reads.clear()
        src.extract("temp", time=1)  # selected, but short, so read whole
        reads.clear()

        src.extract("temp")  # unselected

    assert chunk_reads("s.zarr", "time") == []


@pytest.fixture
def long_line(tmp_path: Path, zarr_format: ZarrFormat) -> Path:
    # 1,000 chunks of 5 days.
    return make_line(
        tmp_path / "long.zarr",
        zarr_format,
        "time",
        np.arange(5000),
        {"units": "days since 2000-01-01"},
    )


def test_a_later_point_on_a_long_coordinate_reads_few_new_chunks(
    long_line: Path,
) -> None:
    with zx.open(to_memory(long_line)) as src:
        first = src.extract("v", time="2008-06-01")
        first_chunks = set(chunk_reads("long.zarr", "time"))
        reads.clear()

        src.extract("v", time="2008-06-02")
        second_chunks = set(chunk_reads("long.zarr", "time"))
        reads.clear()

        again = src.extract("v", time="2008-06-02")

    assert 0 < len(first_chunks) <= 30
    assert len(second_chunks) <= 2
    assert chunk_reads("long.zarr", "time") == []
    assert metadata_reads() == []
    assert again.num_rows == 1
    assert first.num_rows == 1


def test_a_repeated_range_on_a_long_coordinate_reads_no_coordinate_chunks(
    long_line: Path,
) -> None:
    # 60 days are 12 chunks of 5 values.
    selection = ("2008-06-01", "2008-07-30")
    with zx.open(to_memory(long_line)) as src:
        first = src.extract("v", time=selection)
        first_chunks = set(chunk_reads("long.zarr", "time"))
        reads.clear()

        again = src.extract("v", time=selection)

    assert first.num_rows == 60
    assert len(first_chunks) >= 12
    assert chunk_reads("long.zarr", "time") == []
    assert again.equals(first)


def test_an_overlapping_range_reads_only_the_new_coordinate_chunks(
    long_line: Path,
) -> None:
    with zx.open(to_memory(long_line)) as src:
        src.extract("v", time=("2008-06-01", "2008-06-30"))
        first = {
            int(key.rsplit("/", 1)[-1]) for key in chunk_reads("long.zarr", "time")
        }
        reads.clear()

        # Days 3074 to 3103 are chunks 614 to 620, and this range runs on to
        # day 3133, which is chunk 626. Which of the new chunks the first
        # search already read depends on where it probed, so the test checks
        # the rule rather than chunk numbers.
        table = src.extract("v", time=("2008-06-15", "2008-07-30"))
        new_chunks = set(chunk_reads("long.zarr", "time"))

    start = (dt.date(2008, 6, 15) - dt.date(2000, 1, 1)).days
    assert table["v"].to_pylist() == [float(i) for i in range(start, start + 46)]
    numbers = {int(key.rsplit("/", 1)[-1]) for key in new_chunks}
    # No chunk read for the first range is read again.
    assert not numbers & first
    # Between them, the two calls read every chunk of the second range.
    assert set(range(617, 627)) <= first | numbers
    # The search for the end of the range also probes a few chunks beyond it.
    assert len(numbers) <= 12


def test_the_same_counts_with_a_counting_store(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    path = make_store(tmp_path / "s.zarr", zarr_format)
    counting = CountingStore(LocalStore(path, read_only=True))
    src = Source._from_group(zarr.open_group(store=counting, mode="r"), "s.zarr")
    src.extract("temp", time=0)
    counting.reads.clear()

    src.extract("temp", time=0)

    assert counting.reads
    assert set(counting.reads) == counting.chunk_reads("temp")


# --- the same results as extract() -----------------------------------------------

GRID_SELECTIONS: list[dict[str, Any]] = [
    {},
    {"time": "2000-01-03"},
    {"lat": 0, "lon": 270},
    {"time": "2000-01-02", "lat": (45, -45)},
    {"lon": (-100, 100)},
    {"lon": (350, 10)},
    {"time": ("2000-01", "2000-01-02"), "lon": 90},
    {"time": "2000-01-03", "lat": (float("-inf"), 10)},
]


def assert_identical(a: pa.Table, b: pa.Table) -> None:
    assert a.equals(b, check_metadata=True)
    assert a.schema.metadata == b.schema.metadata


@pytest.mark.usefixtures("frozen_time")
def test_results_are_identical_to_extract(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    path = make_grid(tmp_path / "g.zarr", zarr_format)
    expected = [zx.extract(path, "temp", **sel) for sel in GRID_SELECTIONS]

    with zx.open(path) as src:
        # Forwards, backwards, and each twice, so that every call meets a
        # cache that earlier calls filled differently.
        for index in [
            *range(len(GRID_SELECTIONS)),
            *reversed(range(len(GRID_SELECTIONS))),
        ]:
            selection = GRID_SELECTIONS[index]
            assert_identical(src.extract("temp", **selection), expected[index])
            assert_identical(src.extract("temp", **selection), expected[index])


@pytest.mark.usefixtures("frozen_time")
def test_results_are_identical_on_a_noleap_calendar(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    path = make_grid(
        tmp_path / "n.zarr",
        zarr_format,
        time=np.arange(0, 800, 30),
        time_attrs={"units": "days since 2000-01-01", "calendar": "noleap"},
    )
    selections: list[dict[str, Any]] = [
        {"time": "2000-03-01"},
        {"time": ("2001", "2001")},
        {"time": ("2000-02", "2000-03-15"), "lon": 90},
    ]
    expected = [zx.extract(path, "temp", **sel) for sel in selections]

    with zx.open(path) as src:
        for _ in range(2):
            for sel, want in zip(selections, expected, strict=True):
                assert_identical(src.extract("temp", **sel), want)


@pytest.mark.usefixtures("frozen_time")
def test_results_are_identical_on_a_long_coordinate(long_line: Path) -> None:
    selections: list[dict[str, Any]] = [
        {"time": "2008-06-01"},
        {"time": ("2008-06-01", "2008-06-04")},
        {"time": ("2012-02", "2012-02")},
        {"time": "2008-06-02"},
        {"time": ("2008-06-01", "2008-06-04")},
    ]
    expected = [zx.extract(long_line, "v", **sel) for sel in selections]

    with zx.open(long_line) as src:
        for sel, want in zip(selections, expected, strict=True):
            assert_identical(src.extract("v", **sel), want)


def test_the_provenance_is_taken_at_the_time_of_each_call(
    store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with zx.open(store) as src:
        first = zx.provenance(src.extract("temp", time=0))
        monkeypatch.setattr("zarrextract._provenance.datetime", FrozenDatetime)
        second = zx.provenance(src.extract("temp", time=0))

    assert first is not None
    assert second is not None
    assert second.extracted_at == dt.datetime(2026, 1, 2, 3, 4, 5, tzinfo=dt.UTC)
    assert second.source == first.source == str(store)
    assert second.selections == {"time": 0}


def test_describe_is_the_same_and_is_kept(store: Path) -> None:
    with zx.open(store) as src:
        found = src.describe()

        assert found == zx.describe(store)
        assert src.describe() is found
        assert src.describe(max_depth=0) == zx.describe(store, max_depth=0)

        with pytest.raises(ValueError, match="max_depth"):
            src.describe(max_depth=-1)


ERROR_CASES: list[tuple[str, dict[str, Any]]] = [
    ("nothing", {}),
    ("temp", {"nodim": 1}),
    ("temp", {"lat": 1000}),
    ("temp", {"lat": (1000, 2000)}),
    ("temp", {"lat": float("nan")}),
    ("lat", {}),
    ("temp", {"time": (1, 2, 3)}),
]


@pytest.mark.parametrize(("variable", "selection"), ERROR_CASES)
def test_errors_are_the_same_and_leave_the_source_usable(
    store: Path, variable: str, selection: dict[str, Any]
) -> None:
    with pytest.raises(zx.ZarrExtractError) as expected:
        zx.extract(store, variable, **selection)

    with zx.open(store) as src:
        for _ in range(2):
            with pytest.raises(zx.ZarrExtractError) as found:
                src.extract(variable, **selection)

            assert type(found.value) is type(expected.value)
            assert str(found.value) == str(expected.value)
            assert src.extract("temp", time=1).num_rows == 8


def test_storage_options_is_not_accepted_by_extract(store: Path) -> None:
    with (
        zx.open(store) as src,
        pytest.raises(TypeError, match=r"pass storage_options to zx\.open"),
    ):
        src.extract("temp", storage_options={"anon": True})  # type: ignore[arg-type]


def test_a_store_that_cannot_be_opened_fails_in_open(tmp_path: Path) -> None:
    with pytest.raises(zx.StoreOpenError):
        zx.open(tmp_path / "missing.zarr")


# --- lifecycle -----------------------------------------------------------------


def test_with_closes_the_source(store: Path) -> None:
    with zx.open(store) as src:
        assert not src.closed
        table = src.extract("temp", time=0)

    assert src.closed
    assert table.num_rows == 8


def test_a_source_works_without_with(store: Path) -> None:
    src = zx.open(store)

    assert src.extract("temp", time=0).num_rows == 8
    assert src.extract("temp", time=1).num_rows == 8
    assert not src.closed

    src.close()

    assert src.closed


def test_after_close_a_source_raises_and_close_is_repeatable(store: Path) -> None:
    src = zx.open(store)
    src.describe()
    src.close()
    src.close()

    for call in (
        lambda: src.extract("temp"),
        lambda: src.describe(),
        src.__enter__,
    ):
        with pytest.raises(zx.SourceClosedError) as info:
            call()

        assert isinstance(info.value, ValueError)
        assert isinstance(info.value, zx.ZarrExtractError)
    assert src.store == str(store)
    assert repr(src) == f"Source({str(store)!r}, closed)"


def test_close_drops_what_was_kept(store: Path) -> None:
    src = zx.open(store)
    src.extract("temp", time=0)
    src.describe()
    assert src._cache is not None

    src.close()

    assert src._cache is None


def test_repr(store: Path) -> None:
    with zx.open(store) as src:
        assert repr(src) == f"Source({str(store)!r})"


def test_a_source_cannot_be_pickled(store: Path) -> None:
    with zx.open(store) as src, pytest.raises(TypeError, match="open the store again"):
        pickle.dumps(src)


def test_open_is_exported_but_not_star_imported() -> None:
    namespace: dict[str, Any] = {}
    exec("from zarrextract import *", namespace)

    assert "open" not in namespace
    assert "open" not in zx.__all__
    assert namespace["Source"] is Source
    assert namespace["SourceClosedError"] is zx.SourceClosedError
    assert zx.open.__module__ == "zarrextract"
    assert Source.__module__ == "zarrextract"


def test_threads_give_the_same_tables(tmp_path: Path, zarr_format: ZarrFormat) -> None:
    # A smoke test on a local store. Thread safety is not promised.
    path = make_grid(tmp_path / "g.zarr", zarr_format)
    selections = [sel for sel in GRID_SELECTIONS for _ in range(4)]
    expected = [zx.extract(path, "temp", **sel) for sel in selections]

    with zx.open(path) as src, ThreadPoolExecutor(max_workers=8) as pool:
        found = list(pool.map(lambda sel: src.extract("temp", **sel), selections))

    for got, want in zip(found, expected, strict=True):
        assert got.equals(want)
