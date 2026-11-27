"""Provenance metadata on extracted tables, and reading it back."""

import json
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pyarrow as pa
import pytest
import zarr
from fsspec.implementations.memory import MemoryFileSystem
from fsspec.registry import register_implementation

import zarrextract as zx
from helpers import ZarrFormat, make_grid, make_store, write_array
from zarrextract._extract import _extract_from_group
from zarrextract._provenance import (
    provenance_metadata,
    sanitize_store,
    selections_json,
)

FORMATS: list[ZarrFormat] = [2, 3]
KEYS = {
    "zarrextract.provenance_version",
    "zarrextract.source",
    "zarrextract.variable",
    "zarrextract.selections",
    "zarrextract.version",
    "zarrextract.extracted_at",
}


def schema_metadata(table: pa.Table) -> dict[str, str]:
    return {k.decode(): v.decode() for k, v in table.schema.metadata.items()}


def field_metadata(table: pa.Table) -> dict[str, dict[str, str] | None]:
    return {
        field.name: None
        if field.metadata is None
        else {k.decode(): v.decode() for k, v in field.metadata.items()}
        for field in table.schema
    }


@pytest.mark.parametrize("zarr_format", FORMATS)
def test_extract_attaches_provenance(tmp_path: Path, zarr_format: ZarrFormat) -> None:
    grid = make_grid(tmp_path / "g.zarr", zarr_format)
    before = datetime.now(UTC).replace(microsecond=0)

    table = extract_example(grid)

    after = datetime.now(UTC)
    meta = schema_metadata(table)
    assert set(meta) == KEYS
    assert meta["zarrextract.provenance_version"] == "1"
    assert meta["zarrextract.source"] == str(grid)
    assert meta["zarrextract.variable"] == "temp"
    selections = json.loads(meta["zarrextract.selections"])
    assert selections == {"time": "2000-01-02", "lat": [0, 45], "lon": 90}
    assert list(selections) == ["time", "lat", "lon"]  # as given
    assert meta["zarrextract.version"] == zx.__version__
    stamp = meta["zarrextract.extracted_at"]
    assert stamp.endswith("Z")
    assert before <= datetime.fromisoformat(stamp) <= after


def extract_example(grid: Path) -> pa.Table:
    return zx.extract(grid, "temp", time="2000-01-02", lat=(0, 45), lon=90)


def test_extract_without_selections_records_an_empty_object(tmp_path: Path) -> None:
    table = zx.extract(make_grid(tmp_path / "g.zarr", 3), "temp")

    assert schema_metadata(table)["zarrextract.selections"] == "{}"


def test_selections_are_recorded_as_given() -> None:
    selections: dict[str, Any] = {
        "a": 1,
        "b": 2.5,
        "c": np.float64(1.5),
        "d": np.int32(3),
        "e": ("2000-01-01", "2000-02-01"),
        "f": datetime(2000, 1, 2, 3, 4, 5),
        "g": date(2000, 1, 2),
        "h": np.datetime64("2000-01-02T03"),
        "i": (np.datetime64("2000-01-01"), date(2001, 1, 1)),
        "j": [1, 2],
        "k": -10,
    }

    assert json.loads(selections_json(selections)) == {
        "a": 1,
        "b": 2.5,
        "c": 1.5,
        "d": 3,
        "e": ["2000-01-01", "2000-02-01"],
        "f": "2000-01-02T03:04:05",
        "g": "2000-01-02",
        "h": "2000-01-02T03",
        "i": ["2000-01-01", "2001-01-01"],
        "j": [1, 2],
        "k": -10,  # not converted to the store's longitude convention
    }


def test_open_ended_ranges_are_valid_json(tmp_path: Path) -> None:
    grid = make_grid(tmp_path / "g.zarr", 3)

    table = zx.extract(grid, "temp", lat=(float("-inf"), 10), lon=0, time="2000-01-01")

    text = schema_metadata(table)["zarrextract.selections"]
    assert "Infinity" not in text
    assert json.loads(text)["lat"] == ["-inf", 10]
    info = zx.provenance(table)
    assert info is not None
    assert info.selections["lat"] == ["-inf", 10]


@pytest.mark.parametrize("zarr_format", FORMATS)
def test_units_and_long_names_are_field_metadata(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    grid = make_grid(
        tmp_path / "g.zarr",
        zarr_format,
        time_attrs={"units": "days since 2000-01-01", "long_name": "Time"},
        data_attrs={"units": "K", "long_name": "Air temperature", "_FillValue": -1.0},
    )
    group = zarr.open_group(grid, mode="r+")
    group["lat"].attrs["long_name"] = "Latitude"

    table = zx.extract(grid, "temp", lat=0)

    assert field_metadata(table) == {
        # The units of a decoded time column describe the stored numbers.
        "time": {"long_name": "Time"},
        "lat": {"units": "degrees_north", "long_name": "Latitude"},
        "lon": {"units": "degrees_east"},
        "temp": {"units": "K", "long_name": "Air temperature"},
    }


def test_time_without_a_long_name_has_no_metadata(tmp_path: Path) -> None:
    table = zx.extract(make_grid(tmp_path / "g.zarr", 3), "temp")

    assert field_metadata(table)["time"] is None


def test_dimensions_without_coordinates_have_no_metadata(tmp_path: Path) -> None:
    group = zarr.open_group(tmp_path / "g.zarr", mode="w", zarr_format=3)
    write_array(
        group, "v", np.arange(6.0).reshape(2, 3), ("y", "x"), attrs={"units": "m"}
    )

    table = zx.extract(tmp_path / "g.zarr", "v")

    assert field_metadata(table) == {"y": None, "x": None, "v": {"units": "m"}}


@pytest.mark.parametrize("selected", [True, False])
def test_units_of_a_coordinate_read_by_bisection(
    tmp_path: Path, selected: bool
) -> None:
    # 20 chunks of one value are more than a selected coordinate reads whole.
    group = zarr.open_group(tmp_path / "g.zarr", mode="w", zarr_format=3)
    attrs = {"units": "m", "long_name": "Distance"}
    write_array(group, "x", np.arange(20.0), ("x",), chunks=(1,), attrs=attrs)
    write_array(group, "v", np.arange(20.0), ("x",))
    selections: dict[str, Any] = {"x": (3, 5)} if selected else {}

    table = zx.extract(tmp_path / "g.zarr", "v", **selections)

    assert field_metadata(table)["x"] == attrs


@pytest.mark.parametrize("zarr_format", FORMATS)
def test_the_table_is_unchanged_apart_from_metadata(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    data = np.arange(60.0).reshape(4, 5, 3)
    data[1, 2, 0] = -999.0
    grid = make_grid(
        tmp_path / "g.zarr",
        zarr_format,
        lon=(0.0, 90.0, 180.0),
        data=data,
        data_attrs={"units": "K", "_FillValue": -999.0},
    )
    selections: dict[str, Any] = {"lat": (0, 45), "time": ("2000-01-01", "2000-01-03")}

    table = zx.extract(grid, "temp", **selections)
    plain = _extract_from_group(zarr.open_group(grid, mode="r"), "temp", selections)

    assert table.equals(plain)  # names, types, values and nulls
    assert table.schema.equals(
        pa.schema(
            [
                ("time", pa.timestamp("s")),
                ("lat", pa.float64()),
                ("lon", pa.float64()),
                ("temp", pa.float64()),
            ]
        )
    )
    assert table.column("temp").null_count == plain.column("temp").null_count == 1
    assert table.to_pydict() == plain.to_pydict()
    assert plain.schema.metadata is None  # the schema metadata is added by extract()


@pytest.mark.parametrize(
    ("store", "expected"),
    [
        (
            "https://user:secret@example.com/data/s.zarr?token=abc#frag",
            "https://example.com/data/s.zarr",
        ),
        ("https://user@example.com:8443/s.zarr", "https://example.com:8443/s.zarr"),
        ("https://user:p@ss@example.com/x", "https://example.com/x"),
        ("s3://bucket/key/s.zarr", "s3://bucket/key/s.zarr"),
        ("s3://bucket/s.zarr?versionId=1", "s3://bucket/s.zarr"),
        (
            "simplecache::s3://key:secret@bucket/s.zarr",
            "simplecache::s3://bucket/s.zarr",
        ),
        (
            "zip://inner.zarr::https://u:p@host/a.zip?sig=1",
            "zip://inner.zarr::https://host/a.zip",
        ),
        ("http://[::1]:8080/s.zarr?x=1", "http://[::1]:8080/s.zarr"),
        ("file:///data/s.zarr", "file:///data/s.zarr"),
        ("https://[::1/s.zarr?token=abc", "https://<removed>"),  # not a valid URL
        ("/data/s.zarr", "/data/s.zarr"),
        ("relative/s.zarr", "relative/s.zarr"),
        ("s.zarr", "s.zarr"),
        ("C:\\data\\s.zarr", "C:\\data\\s.zarr"),
    ],
)
def test_sanitize_store(store: str, expected: str) -> None:
    assert sanitize_store(store) == expected


def test_the_source_has_no_credentials() -> None:
    meta = provenance_metadata(
        "https://user:secret@example.com/s.zarr?token=abc", "temp", {}
    )

    assert meta["zarrextract.source"] == "https://example.com/s.zarr"
    assert "secret" not in repr(meta)
    assert "token" not in repr(meta)


class ProvenanceFileSystem(MemoryFileSystem):  # type: ignore[misc]
    """An in-memory filesystem with its own protocol, which takes any options."""

    protocol = "provmem"

    @classmethod
    def _strip_protocol(cls, path: str) -> str:
        return str(super()._strip_protocol(path.removeprefix("provmem://")))


@pytest.fixture
def memory_store(tmp_path: Path) -> Iterator[str]:
    register_implementation("provmem", ProvenanceFileSystem, clobber=True)
    ProvenanceFileSystem.clear_instance_cache()
    local = make_store(tmp_path / "s.zarr", 3)
    fsspec.filesystem("memory").put(str(local), "/s.zarr", recursive=True)
    yield "provmem://s.zarr"
    ProvenanceFileSystem.clear_instance_cache()
    MemoryFileSystem.store.clear()
    MemoryFileSystem.pseudo_dirs[:] = [""]


def test_storage_options_are_not_recorded(memory_store: str) -> None:
    table = zx.extract(
        memory_store, "temp", storage_options={"key": "hunter2-marker"}, time=0
    )

    assert schema_metadata(table)["zarrextract.source"] == memory_store
    assert "hunter2-marker" not in repr(table.schema)
    assert "storage_options" not in repr(table.schema)


@pytest.mark.parametrize("zarr_format", FORMATS)
def test_provenance_of_a_table(tmp_path: Path, zarr_format: ZarrFormat) -> None:
    grid = make_grid(
        tmp_path / "g.zarr", zarr_format, data_attrs={"units": "K", "long_name": "T"}
    )
    before = datetime.now(UTC).replace(microsecond=0)

    info = zx.provenance(extract_example(grid))

    assert info is not None
    assert info.provenance_version == "1"
    assert info.source == str(grid)
    assert info.variable == "temp"
    assert info.selections == {"time": "2000-01-02", "lat": [0, 45], "lon": 90}
    assert info.version == zx.__version__
    assert info.columns == {
        "lat": {"units": "degrees_north"},
        "lon": {"units": "degrees_east"},
        "temp": {"units": "K", "long_name": "T"},
    }
    assert info.extracted_at.utcoffset() == timedelta(0)  # aware, in UTC
    assert before <= info.extracted_at <= datetime.now(UTC)


def test_provenance_of_a_plain_table_is_none() -> None:
    assert zx.provenance(pa.table({"a": [1, 2]})) is None
    other = pa.table({"a": [1]}).replace_schema_metadata({"x": "y"})
    assert zx.provenance(other) is None


@pytest.fixture
def table(tmp_path: Path) -> pa.Table:
    return extract_example(make_grid(tmp_path / "g.zarr", 3))


def with_metadata(table: pa.Table, **changes: str | None) -> pa.Table:
    meta = schema_metadata(table)
    for key, value in changes.items():
        name = f"zarrextract.{key}"
        if value is None:
            del meta[name]
        else:
            meta[name] = value
    return table.replace_schema_metadata(meta)


def test_an_unknown_provenance_version_is_an_error(table: pa.Table) -> None:
    changed = with_metadata(table, provenance_version="2")

    with pytest.raises(zx.ZarrExtractError, match="Unsupported provenance version '2'"):
        zx.provenance(changed)


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        ({"selections": "{not json"}, "zarrextract.selections is not valid JSON"),
        ({"selections": "[1, 2]"}, "zarrextract.selections is missing or is not"),
        ({"source": None}, "zarrextract.source is missing"),
        ({"extracted_at": "yesterday"}, "zarrextract.extracted_at is not an ISO"),
        ({"provenance_version": None}, "zarrextract.provenance_version is missing"),
    ],
)
def test_malformed_provenance_is_an_error(
    table: pa.Table, changes: dict[str, str | None], problem: str
) -> None:
    changed = with_metadata(table, **changes)

    with pytest.raises(zx.ZarrExtractError, match="Malformed provenance in") as info:
        zx.provenance(changed)

    assert problem in str(info.value)


def test_provenance_of_a_file_with_an_unknown_suffix(tmp_path: Path) -> None:
    with pytest.raises(zx.ZarrExtractError, match=r"unknown suffix '\.xlsx'"):
        zx.provenance(tmp_path / "x.xlsx")
