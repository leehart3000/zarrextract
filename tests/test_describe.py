import dataclasses
from pathlib import Path

import pytest
import zarr

import zarrextract
from helpers import ZarrFormat, make_store, write_array
from zarrextract import ArrayInfo, StoreDescription, describe

FORMATS: list[ZarrFormat] = [2, 3]


@pytest.mark.parametrize("zarr_format", FORMATS)
def test_describe_lists_arrays(tmp_path: Path, zarr_format: ZarrFormat) -> None:
    store = make_store(tmp_path / "s.zarr", zarr_format)

    result = describe(store)

    assert [a.name for a in result.arrays] == ["lat", "lon", "temp", "time"]
    assert result["temp"] == ArrayInfo(
        name="temp",
        dimensions=("time", "lat", "lon"),
        shape=(3, 2, 4),
        chunks=(1, 2, 2),
        dtype="float32",
        is_coordinate=False,
    )
    assert result["lon"].dimensions == ("lon",)
    assert result["lon"].shape == (4,)
    assert result["lon"].dtype == "float64"


@pytest.mark.parametrize("zarr_format", FORMATS)
def test_describe_identifies_coordinates(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    store = make_store(tmp_path / "s.zarr", zarr_format)

    result = describe(store)

    assert [a.name for a in result.coordinates] == ["lat", "lon", "time"]
    assert [a.name for a in result.data_variables] == ["temp"]


@pytest.mark.parametrize("zarr_format", FORMATS)
def test_describe_accepts_path_objects_and_strings(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    store = make_store(tmp_path / "s.zarr", zarr_format)

    assert describe(store) == describe(str(store))


@pytest.mark.parametrize("zarr_format", FORMATS)
def test_describe_with_consolidated_metadata(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    plain = make_store(tmp_path / "plain.zarr", zarr_format)
    consolidated = make_store(tmp_path / "cons.zarr", zarr_format, consolidate=True)

    assert describe(consolidated) == describe(plain)


@pytest.mark.parametrize(
    ("zarr_format", "array_metadata_file"),
    [(2, ".zarray"), (3, "zarr.json")],
)
def test_describe_uses_consolidated_metadata_when_present(
    tmp_path: Path, zarr_format: ZarrFormat, array_metadata_file: str
) -> None:
    store = make_store(tmp_path / "s.zarr", zarr_format, consolidate=True)
    # Without the array's own metadata file, only the consolidated copy remains.
    (store / "temp" / array_metadata_file).unlink()

    assert describe(store)["temp"].shape == (3, 2, 4)


@pytest.mark.parametrize("zarr_format", FORMATS)
def test_describe_without_dimension_names(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    store = make_store(tmp_path / "s.zarr", zarr_format, with_dims=False)

    result = describe(store)

    assert all(a.dimensions is None for a in result.arrays)
    assert result.coordinates == ()
    assert len(result.data_variables) == 4


@pytest.mark.parametrize("zarr_format", FORMATS)
def test_coordinate_must_be_named_after_its_own_dimension(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    group = zarr.open_group(tmp_path / "s.zarr", mode="w", zarr_format=zarr_format)
    write_array(group, "time", [1.0, 2.0], ("time",))
    write_array(group, "other", [1.0, 2.0], ("time",))  # 1-D, different name
    write_array(group, "lat", [[1.0, 2.0], [3.0, 4.0]], ("lat", "lon"))  # 2-D

    result = describe(tmp_path / "s.zarr")

    assert [a.name for a in result.coordinates] == ["time"]


@pytest.mark.parametrize("zarr_format", FORMATS)
def test_describe_zero_dimensional_array(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    group = zarr.open_group(tmp_path / "s.zarr", mode="w", zarr_format=zarr_format)
    write_array(group, "scalar", 1.0, ())

    info = describe(tmp_path / "s.zarr")["scalar"]

    assert info.dimensions == ()
    assert info.shape == ()
    assert not info.is_coordinate


@pytest.mark.parametrize("zarr_format", FORMATS)
def test_describe_ignores_nested_groups(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    group = zarr.open_group(tmp_path / "s.zarr", mode="w", zarr_format=zarr_format)
    write_array(group, "a", [1.0], ("a",))
    write_array(group.create_group("sub"), "b", [1.0], ("b",))

    assert [a.name for a in describe(tmp_path / "s.zarr").arrays] == ["a"]


def test_results_are_frozen(tmp_path: Path) -> None:
    result = describe(make_store(tmp_path / "s.zarr", 3))

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.arrays = ()  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result["temp"].name = "x"  # type: ignore[misc]


def test_description_is_a_collection_of_arrays(tmp_path: Path) -> None:
    result = describe(make_store(tmp_path / "s.zarr", 3))

    assert [info.name for info in result] == ["lat", "lon", "temp", "time"]
    assert all(isinstance(info, ArrayInfo) for info in result)
    assert list(result) == list(result.arrays)
    assert len(result) == 4
    assert "temp" in result
    assert "nope" not in result
    assert 0 not in result
    assert result["temp"] not in result  # membership is by name, not by value


def test_empty_description_is_falsy(tmp_path: Path) -> None:
    zarr.open_group(tmp_path / "s.zarr", mode="w", zarr_format=3)

    result = describe(tmp_path / "s.zarr")

    assert len(result) == 0
    assert not result
    assert list(result) == []


def test_getitem_unknown_name(tmp_path: Path) -> None:
    result = describe(make_store(tmp_path / "s.zarr", 3))

    with pytest.raises(zarrextract.VariableNotFoundError, match="nope"):
        result["nope"]


def test_describe_missing_path(tmp_path: Path) -> None:
    with pytest.raises(zarrextract.StoreOpenError, match="missing"):
        describe(tmp_path / "missing.zarr")


def test_describe_directory_that_is_not_a_store(tmp_path: Path) -> None:
    with pytest.raises(zarrextract.StoreOpenError):
        describe(tmp_path)


def test_describe_url_without_backend() -> None:
    # The scheme is made up so that no installed fsspec backend can handle it.
    with pytest.raises(zarrextract.StoreOpenError, match="nosuchscheme"):
        describe("nosuchscheme://bucket/store.zarr")


def test_describe_url_with_missing_backend_package() -> None:
    # fsspec knows gs:// but needs gcsfs, which is not a dependency.
    pytest.importorskip("fsspec")
    with pytest.raises(zarrextract.StoreOpenError, match="gcsfs"):
        describe("gs://bucket/store.zarr")


def test_exception_hierarchy() -> None:
    base = zarrextract.ZarrExtractError
    assert issubclass(zarrextract.StoreOpenError, base)
    assert issubclass(zarrextract.MissingDimensionNamesError, base)
    assert issubclass(zarrextract.VariableNotFoundError, base)
    assert issubclass(zarrextract.VariableNotFoundError, KeyError)
    assert issubclass(zarrextract.SelectionError, base)
    for sub in (
        zarrextract.UnknownDimensionError,
        zarrextract.EmptySelectionError,
        zarrextract.AntimeridianError,
    ):
        assert issubclass(sub, zarrextract.SelectionError)
    assert issubclass(zarrextract.TimeDecodingError, base)
    assert issubclass(
        zarrextract.UnsupportedCalendarError, zarrextract.TimeDecodingError
    )


def test_description_type(tmp_path: Path) -> None:
    result = describe(make_store(tmp_path / "s.zarr", 2))

    assert isinstance(result, StoreDescription)
