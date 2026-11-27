"""Write tables to Parquet, Arrow IPC and CSV files, and read them back."""

import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq
import pytest

import zarrextract as zx
from helpers import make_grid

TYPES = {"timestamp[s]": pa.timestamp("s"), "double": pa.float64()}


@pytest.fixture
def table(tmp_path: Path) -> pa.Table:
    grid = make_grid(
        tmp_path / "g.zarr",
        3,
        data_attrs={"units": "K", "long_name": "Air temperature"},
    )
    return zx.extract(grid, "temp", lat=(0, 45), lon=90)


@pytest.fixture
def out(tmp_path: Path) -> Path:
    directory = tmp_path / "out"
    directory.mkdir()
    return directory


def read_arrow(path: Path) -> pa.Table:
    with pa.OSFile(str(path), "rb") as file:
        return pa.ipc.open_file(file).read_all()


def test_parquet_round_trip(table: pa.Table, out: Path) -> None:
    written = zx.write(table, out / "t.parquet")

    assert written == (out / "t.parquet",)
    back = pq.read_table(out / "t.parquet")
    # Parquet has no unit of seconds: the times come back in milliseconds.
    assert back.schema.field("time").type == pa.timestamp("ms")
    assert back.to_pydict() == table.to_pydict()
    assert back.cast(table.schema).equals(table, check_metadata=True)
    assert back.schema.metadata == table.schema.metadata
    assert [f.metadata for f in back.schema] == [f.metadata for f in table.schema]
    assert zx.provenance(out / "t.parquet") == zx.provenance(table)


def test_parquet_is_compressed_with_zstd(table: pa.Table, out: Path) -> None:
    zx.write(table, out / "t.parquet")

    row_group = pq.ParquetFile(out / "t.parquet").metadata.row_group(0)
    assert {row_group.column(i).compression for i in range(4)} == {"ZSTD"}


@pytest.mark.parametrize("suffix", [".arrow", ".feather"])
def test_arrow_round_trip(table: pa.Table, out: Path, suffix: str) -> None:
    path = out / f"t{suffix}"

    written = zx.write(table, path)

    assert written == (path,)
    assert read_arrow(path).equals(table, check_metadata=True)
    assert read_arrow(path).schema.metadata == table.schema.metadata
    assert zx.provenance(path) == zx.provenance(table)


def test_csv_round_trip_with_a_sidecar(table: pa.Table, out: Path) -> None:
    written = zx.write(table, out / "t.csv")

    assert written == (out / "t.csv", out / "t.csv.json")
    sidecar = json.loads((out / "t.csv.json").read_text(encoding="utf-8"))
    assert set(sidecar) == {"columns", "metadata"}
    assert [c["name"] for c in sidecar["columns"]] == table.column_names
    assert [c["type"] for c in sidecar["columns"]] == ["timestamp[s]"] + ["double"] * 3
    assert sidecar["columns"][3]["metadata"] == {
        "units": "K",
        "long_name": "Air temperature",
    }
    assert "metadata" not in sidecar["columns"][0]
    assert sidecar["metadata"]["zarrextract.selections"] == {
        "lat": [0, 45],
        "lon": 90,
    }
    assert sidecar["metadata"]["zarrextract.variable"] == "temp"
    # The sidecar types bring back what CSV loses.
    types = {c["name"]: TYPES[c["type"]] for c in sidecar["columns"]}
    back = pa_csv.read_csv(
        out / "t.csv", convert_options=pa_csv.ConvertOptions(column_types=types)
    )
    assert back.equals(table)
    assert zx.provenance(out / "t.csv") == zx.provenance(table)


def test_csv_of_a_table_without_provenance(out: Path) -> None:
    plain = pa.table({"a": [1, 2]})

    assert zx.write(plain, out / "p.csv") == (out / "p.csv", out / "p.csv.json")

    sidecar = json.loads((out / "p.csv.json").read_text(encoding="utf-8"))
    assert sidecar == {"columns": [{"name": "a", "type": "int64"}], "metadata": {}}
    assert zx.provenance(out / "p.csv") is None


def test_csv_without_a_sidecar_has_no_provenance(out: Path) -> None:
    (out / "x.csv").write_text("a\n1\n")

    assert zx.provenance(out / "x.csv") is None


@pytest.mark.parametrize("name", ["T.PARQUET", "t.Csv", "t.ARROW", "t.Feather"])
def test_the_suffix_is_matched_without_case(
    table: pa.Table, out: Path, name: str
) -> None:
    written = zx.write(table, out / name)

    assert written[0] == out / name
    assert all(path.exists() for path in written)
    assert zx.provenance(out / name) == zx.provenance(table)


def test_the_path_can_be_a_string(table: pa.Table, out: Path) -> None:
    assert zx.write(table, str(out / "t.parquet")) == (out / "t.parquet",)


@pytest.mark.parametrize(
    ("name", "problem"),
    [
        ("t.xlsx", "unknown suffix '.xlsx'. Use .parquet, .csv, .arrow or .feather."),
        (
            "t.csv.json",
            "unknown suffix '.json'. Use .parquet, .csv, .arrow or .feather.",
        ),
        ("t", "the path has no suffix. Use .parquet, .csv, .arrow or .feather."),
    ],
)
def test_unsupported_suffix(
    table: pa.Table, out: Path, name: str, problem: str
) -> None:
    with pytest.raises(zx.UnsupportedFormatError) as info:
        zx.write(table, out / name)

    assert str(info.value) == f"Cannot write '{out / name}': {problem}"
    assert isinstance(info.value, zx.WriteError | zx.ZarrExtractError | ValueError)
    assert list(out.iterdir()) == []


def test_urls_are_not_supported_and_their_credentials_are_not_shown(
    table: pa.Table,
) -> None:
    with pytest.raises(zx.UnsupportedFormatError) as info:
        zx.write(table, "s3://user:secret@bucket/t.parquet?token=abc")

    assert str(info.value) == (
        "Cannot write 's3://bucket/t.parquet': write() takes local paths only."
    )


def test_a_file_that_exists_is_an_error(table: pa.Table, out: Path) -> None:
    (out / "t.parquet").write_bytes(b"keep")

    with pytest.raises(zx.OutputExistsError) as info:
        zx.write(table, out / "t.parquet")

    assert str(info.value) == (
        f"Cannot write '{out / 't.parquet'}': the file already exists. "
        "Pass overwrite=True to replace it."
    )
    assert isinstance(info.value, zx.WriteError | FileExistsError)
    assert (out / "t.parquet").read_bytes() == b"keep"


def test_a_csv_sidecar_that_exists_is_an_error(table: pa.Table, out: Path) -> None:
    (out / "t.csv.json").write_text("keep")

    with pytest.raises(zx.OutputExistsError) as info:
        zx.write(table, out / "t.csv")

    assert str(info.value) == (
        f"Cannot write '{out / 't.csv'}': its metadata file "
        f"'{out / 't.csv.json'}' already exists. Pass overwrite=True to replace it."
    )
    assert not (out / "t.csv").exists()  # nothing was written
    assert (out / "t.csv.json").read_text() == "keep"


def test_a_csv_that_exists_is_an_error(table: pa.Table, out: Path) -> None:
    (out / "t.csv").write_text("keep")

    with pytest.raises(zx.OutputExistsError, match="the file already exists"):
        zx.write(table, out / "t.csv")

    assert not (out / "t.csv.json").exists()
    assert (out / "t.csv").read_text() == "keep"


def test_overwrite_replaces_a_parquet_file(table: pa.Table, out: Path) -> None:
    (out / "t.parquet").write_bytes(b"old")

    written = zx.write(table.slice(0, 1), out / "t.parquet", overwrite=True)

    assert written == (out / "t.parquet",)
    assert pq.read_table(out / "t.parquet").num_rows == 1


def test_overwrite_replaces_an_arrow_file(table: pa.Table, out: Path) -> None:
    (out / "t.arrow").write_bytes(b"old")

    zx.write(table.slice(0, 1), out / "t.arrow", overwrite=True)

    assert read_arrow(out / "t.arrow").num_rows == 1


def test_overwrite_replaces_both_csv_files(table: pa.Table, out: Path) -> None:
    (out / "t.csv").write_text("old")
    (out / "t.csv.json").write_text("old")

    written = zx.write(table.slice(0, 1), out / "t.csv", overwrite=True)

    assert written == (out / "t.csv", out / "t.csv.json")
    assert len((out / "t.csv").read_text().splitlines()) == 2  # header and one row
    assert zx.provenance(out / "t.csv") == zx.provenance(table)


def test_a_json_file_next_to_parquet_is_not_a_conflict(
    table: pa.Table, out: Path
) -> None:
    (out / "t.parquet.json").write_text("mine")

    zx.write(table, out / "t.parquet")

    assert (out / "t.parquet.json").read_text() == "mine"


def test_a_failed_csv_write_leaves_no_files(
    table: pa.Table, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(table: pa.Table, file: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("zarrextract._write._write_sidecar", fail)

    with pytest.raises(RuntimeError, match="boom"):
        zx.write(table, out / "t.csv")

    assert list(out.iterdir()) == []


def test_a_failed_parquet_write_leaves_no_file(
    table: pa.Table, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(table: pa.Table, file: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("zarrextract._write._write_parquet", fail)

    with pytest.raises(RuntimeError, match="boom"):
        zx.write(table, out / "t.parquet")

    assert list(out.iterdir()) == []


def boom(*args: object, **kwargs: object) -> None:
    raise RuntimeError("boom")


@pytest.mark.parametrize(
    ("name", "failing"),
    [
        ("t.parquet", "_write_parquet"),
        ("t.arrow", "_write_ipc"),
        ("t.csv", "_write_csv"),
        ("t.csv", "_write_sidecar"),
    ],
)
def test_a_failed_overwrite_leaves_the_old_files_as_they_were(
    table: pa.Table,
    out: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    failing: str,
) -> None:
    old = {name: b"old data"}
    if name.endswith(".csv"):
        old[f"{name}.json"] = b"old sidecar"
    for file_name, content in old.items():
        (out / file_name).write_bytes(content)
    monkeypatch.setattr(f"zarrextract._write.{failing}", boom)

    with pytest.raises(RuntimeError, match="boom"):
        zx.write(table, out / name, overwrite=True)

    assert {p.name: p.read_bytes() for p in out.iterdir()} == old  # and no temporaries


def test_a_failure_while_moving_into_place_leaves_the_old_files(
    table: pa.Table, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (out / "t.csv").write_bytes(b"old data")
    (out / "t.csv.json").write_bytes(b"old sidecar")
    monkeypatch.setattr("os.replace", boom)

    with pytest.raises(RuntimeError, match="boom"):
        zx.write(table, out / "t.csv", overwrite=True)

    assert {p.name: p.read_bytes() for p in out.iterdir()} == {
        "t.csv": b"old data",
        "t.csv.json": b"old sidecar",
    }


def test_overwrite_keeps_the_mode_of_the_old_file(table: pa.Table, out: Path) -> None:
    (out / "t.parquet").write_bytes(b"old")
    (out / "t.parquet").chmod(0o640)

    zx.write(table, out / "t.parquet", overwrite=True)

    assert (out / "t.parquet").stat().st_mode & 0o777 == 0o640


def test_a_new_file_gets_the_usual_permissions(table: pa.Table, out: Path) -> None:
    umask = os.umask(0o022)
    try:
        zx.write(table, out / "t.parquet")
    finally:
        os.umask(umask)

    assert (out / "t.parquet").stat().st_mode & 0o777 == 0o644


def test_no_temporary_files_are_left_after_success(table: pa.Table, out: Path) -> None:
    zx.write(table, out / "t.csv")
    zx.write(table, out / "t.csv", overwrite=True)

    assert sorted(p.name for p in out.iterdir()) == ["t.csv", "t.csv.json"]


def test_a_missing_directory_is_an_os_error(table: pa.Table, out: Path) -> None:
    with pytest.raises(FileNotFoundError):
        zx.write(table, out / "missing" / "t.parquet")

    assert list(out.iterdir()) == []


def test_a_csv_that_cannot_be_written_leaves_no_files(out: Path) -> None:
    nested = pa.table({"a": [[1, 2]]})  # CSV has no nested values

    with pytest.raises(pa.ArrowInvalid):
        zx.write(nested, out / "n.csv")

    assert list(out.iterdir()) == []


def test_only_tables_can_be_written(out: Path) -> None:
    with pytest.raises(TypeError, match=r"write\(\) takes a pyarrow.Table, not list"):
        zx.write([1, 2], out / "t.parquet")

    assert list(out.iterdir()) == []
