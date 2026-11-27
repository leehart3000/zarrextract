"""describe() on OME-Zarr (OME-NGFF 0.4 and 0.5) stores."""

import copy
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import zarr

import zarrextract
from helpers import (
    OME_VERSIONS,
    OmeVersion,
    make_ome_image,
    make_ome_labels,
    make_ome_plate,
    make_store,
    multiscales_entry,
    ome_root,
    set_ome,
)
from test_describe import break_listing
from zarrextract import (
    OmeAxis,
    OmeContainer,
    OmeImage,
    OmeLabels,
    OmeLevel,
    OmeMetadataError,
    OmeMetadataWarning,
    OmeMultiscale,
    OmePlate,
    OmeWell,
    StoreDescription,
    describe,
)


def only_image(description: StoreDescription) -> OmeImage:
    (image,) = description.images
    return image


def add_ome_key(group: zarr.Group, version: OmeVersion, key: str, value: Any) -> None:
    """Add one key to the OME metadata a group already has."""
    if version == "0.4":
        group.attrs[key] = value
    else:
        group.attrs["ome"] = ome_of(group) | {key: value}


def ome_of(group: zarr.Group) -> dict[str, Any]:
    """A copy of a v0.5 group's ``ome`` attribute, to change and write back."""
    ome: Any = group.attrs["ome"]
    return copy.deepcopy(dict(ome))


def consolidate(path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # v3 consolidated metadata is not in the spec
        zarr.consolidate_metadata(path)


# Images


@pytest.mark.parametrize("version", OME_VERSIONS)
def test_image_axes_levels_and_transforms(tmp_path: Path, version: OmeVersion) -> None:
    axes = [
        {"name": "c", "type": "channel"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x"},
    ]
    make_ome_image(
        ome_root(tmp_path / "s.zarr", version),
        version,
        multiscales=[multiscales_entry(version, axes=axes)],
    )

    result = describe(tmp_path / "s.zarr")

    image = only_image(result)
    assert result.is_ome
    assert image.name == ""
    assert image.version == version
    assert image.supported
    assert not image.is_label
    (entry,) = image.multiscales
    assert entry.name is None
    assert entry.axes == (
        OmeAxis("c", "channel", None),
        OmeAxis("y", "space", "micrometer"),
        OmeAxis("x", None, None),
    )
    assert entry.levels == (
        OmeLevel(
            path="0",
            name="0",
            shape=(2, 4, 4),
            chunks=(1, 4, 4),
            dtype="uint16",
            scale=(1.0, 1.0, 1.0),
            translation=None,
        ),
        OmeLevel(
            path="1",
            name="1",
            shape=(2, 2, 2),
            chunks=(1, 2, 2),
            dtype="uint16",
            scale=(1.0, 2.0, 2.0),
            translation=(0.0, 0.5, 0.5),
        ),
    )
    assert image.levels == entry.levels
    assert entry.scale is None
    assert entry.translation is None
    # The arrays are listed as before, and the level names are their keys.
    assert [a.name for a in result] == ["0", "1"]
    assert all(level.name in result for level in image.levels)


@pytest.mark.parametrize("version", OME_VERSIONS)
def test_nested_image_level_names_are_extract_paths(
    tmp_path: Path, version: OmeVersion
) -> None:
    root = ome_root(tmp_path / "s.zarr", version)
    make_ome_image(root.create_group("0"), version)

    result = describe(tmp_path / "s.zarr")
    level = only_image(result).levels[1]
    table = zarrextract.extract(tmp_path / "s.zarr", level.name, c=0, y=0, x=0)

    assert only_image(result).name == "0"
    assert level.name == "0/1"
    assert table.column_names == ["c", "y", "x", "0/1"]


@pytest.mark.parametrize("version", OME_VERSIONS)
def test_every_multiscales_entry_is_kept_with_its_name(
    tmp_path: Path, version: OmeVersion
) -> None:
    make_ome_image(
        ome_root(tmp_path / "s.zarr", version),
        version,
        multiscales=[
            multiscales_entry(version, ("a0", "a1"), name="first"),
            multiscales_entry(version, ("b0",), name="second"),
        ],
    )

    image = only_image(describe(tmp_path / "s.zarr"))

    assert [entry.name for entry in image.multiscales] == ["first", "second"]
    assert [level.name for level in image.levels] == ["a0", "a1", "b0"]


@pytest.mark.parametrize("version", OME_VERSIONS)
def test_multiscales_transforms_are_kept_apart_from_the_levels(
    tmp_path: Path, version: OmeVersion
) -> None:
    entry = multiscales_entry(version)
    entry["coordinateTransformations"] = [
        {"type": "scale", "scale": [1.0, 0.5, 0.5]},
        {"type": "translation", "translation": [0.0, 10.0, 20.0]},
    ]
    make_ome_image(ome_root(tmp_path / "s.zarr", version), version, multiscales=[entry])

    (result,) = only_image(describe(tmp_path / "s.zarr")).multiscales

    assert result.scale == (1.0, 0.5, 0.5)
    assert result.translation == (0.0, 10.0, 20.0)
    assert result.levels[0].scale == (1.0, 1.0, 1.0)
    assert result.levels[1].scale == (1.0, 2.0, 2.0)


def test_v04_without_a_version_gives_none(tmp_path: Path) -> None:
    entry = multiscales_entry("0.4")
    del entry["version"]
    make_ome_image(ome_root(tmp_path / "s.zarr", "0.4"), "0.4", multiscales=[entry])

    image = only_image(describe(tmp_path / "s.zarr"))

    assert image.version is None
    assert image.supported


def test_v05_image_inherits_the_nearest_ancestor_version(tmp_path: Path) -> None:
    root = ome_root(tmp_path / "s.zarr", "0.5")
    root.attrs["ome"] = {"version": "0.5"}
    middle = root.create_group("a")
    middle.attrs["ome"] = {"version": "0.4"}  # nearer than the root
    make_ome_image(middle.create_group("b"), "0.5", with_version=False)
    make_ome_image(root.create_group("c"), "0.5", with_version=False)

    images = describe(tmp_path / "s.zarr").images

    assert [(i.name, i.version) for i in images] == [("a/b", "0.4"), ("c", "0.5")]


def test_v05_without_any_version_gives_none(tmp_path: Path) -> None:
    root = ome_root(tmp_path / "s.zarr", "0.5")
    make_ome_image(root.create_group("0"), "0.5", with_version=False)

    image = only_image(describe(tmp_path / "s.zarr"))

    assert image.version is None
    assert image.supported


@pytest.mark.parametrize("version", OME_VERSIONS)
def test_channel_labels_from_omero(tmp_path: Path, version: OmeVersion) -> None:
    omero = {"channels": [{"label": "DAPI", "color": "0000FF"}, {"color": "FF0000"}]}
    make_ome_image(
        ome_root(tmp_path / "s.zarr", version), version, extra={"omero": omero}
    )

    assert only_image(describe(tmp_path / "s.zarr")).channel_labels == ("DAPI", None)


def test_v05_omero_at_the_top_level_is_read(tmp_path: Path) -> None:
    root = make_ome_image(ome_root(tmp_path / "s.zarr", "0.5"), "0.5")
    root.attrs["omero"] = {"channels": [{"label": "GFP"}]}

    assert only_image(describe(tmp_path / "s.zarr")).channel_labels == ("GFP",)


@pytest.mark.parametrize("version", OME_VERSIONS)
def test_no_omero_gives_no_channel_labels(tmp_path: Path, version: OmeVersion) -> None:
    make_ome_image(ome_root(tmp_path / "s.zarr", version), version)

    assert only_image(describe(tmp_path / "s.zarr")).channel_labels == ()


# Labels


@pytest.mark.parametrize("version", OME_VERSIONS)
def test_labels_group_and_label_images(tmp_path: Path, version: OmeVersion) -> None:
    image = make_ome_image(ome_root(tmp_path / "s.zarr", version), version)
    make_ome_labels(image, version, ("cells", "nuclei"))

    result = describe(tmp_path / "s.zarr")

    assert [(type(g).__name__, g.name) for g in result.ome] == [
        ("OmeImage", ""),
        ("OmeLabels", "labels"),
        ("OmeImage", "labels/cells"),
        ("OmeImage", "labels/nuclei"),
    ]
    root, labels, cells, nuclei = result.ome
    assert isinstance(root, OmeImage)
    assert root.labels == ("cells", "nuclei")
    # A v0.4 labels group has no version of its own.
    expected_version = "0.5" if version == "0.5" else None
    assert labels == OmeLabels("labels", expected_version, ("cells", "nuclei"))
    assert isinstance(cells, OmeImage)
    assert isinstance(nuclei, OmeImage)
    assert cells.is_label
    assert not nuclei.is_label  # image-label is optional
    assert cells.levels[0].name == "labels/cells/0"
    assert "labels/cells/0" in result


# Plates


@pytest.mark.parametrize("version", OME_VERSIONS)
def test_plate_and_wells(tmp_path: Path, version: OmeVersion) -> None:
    make_ome_plate(
        ome_root(tmp_path / "s.zarr", version), version, ("A/1", "A/2", "B/1")
    )

    result = describe(tmp_path / "s.zarr")

    plate, *rest = result.ome
    assert plate == OmePlate("", version, ("A", "B"), ("1", "2"), ("A/1", "A/2", "B/1"))
    wells = [g for g in rest if isinstance(g, OmeWell)]
    assert wells == [
        OmeWell("A/1", version, ("0",)),
        OmeWell("A/2", version, ("0",)),
        OmeWell("B/1", version, ("0",)),
    ]
    assert [i.name for i in result.images] == ["A/1/0", "A/2/0", "B/1/0"]


# bioformats2raw containers


def make_container(path: Path, version: OmeVersion, n_images: int) -> zarr.Group:
    root = ome_root(path, version)
    set_ome(root, version, {"bioformats2raw.layout": 3})
    for i in range(n_images):
        make_ome_image(root.create_group(str(i)), version)
    return root


@pytest.mark.parametrize("version", OME_VERSIONS)
@pytest.mark.parametrize("listable", [True, False])
def test_container_with_numbered_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: OmeVersion, listable: bool
) -> None:
    root = make_container(tmp_path / "s.zarr", version, 2)
    make_ome_image(root.create_group("3"), version)  # after the gap: not found
    if not listable:
        break_listing(monkeypatch, ValueError("409 Conflict"))

    result = describe(tmp_path / "s.zarr")

    container = result.ome[0]
    assert container == OmeContainer(
        "", "0.5" if version == "0.5" else None, 3, ("0", "1"), "numbered"
    )
    if listable:
        # A listing finds every image group, numbered or not.
        assert [i.name for i in result.images] == ["0", "1", "3"]
    else:
        assert [i.name for i in result.images] == ["0", "1"]
        assert [a.name for a in result] == ["0/0", "0/1", "1/0", "1/1"]


@pytest.mark.parametrize("version", OME_VERSIONS)
def test_container_series_beats_numbering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: OmeVersion
) -> None:
    root = make_container(tmp_path / "s.zarr", version, 3)
    set_ome(root.create_group("OME"), version, {"series": ["0", "2"]})
    break_listing(monkeypatch, ValueError("409 Conflict"))

    result = describe(tmp_path / "s.zarr")

    assert result.ome[0] == OmeContainer(
        "", "0.5" if version == "0.5" else None, 3, ("0", "2"), "series"
    )
    assert [i.name for i in result.images] == ["0", "2"]


def test_container_series_at_the_top_level_in_v05(tmp_path: Path) -> None:
    root = make_container(tmp_path / "s.zarr", "0.5", 2)
    root.create_group("OME").attrs["series"] = ["1"]

    container = describe(tmp_path / "s.zarr").ome[0]

    assert isinstance(container, OmeContainer)
    assert container.images == ("1",)


@pytest.mark.parametrize("version", OME_VERSIONS)
def test_container_plate_beats_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: OmeVersion
) -> None:
    root = ome_root(tmp_path / "s.zarr", version)
    make_ome_plate(root, version, ("A/1", "B/1"))
    add_ome_key(root, version, "bioformats2raw.layout", 3)
    set_ome(root.create_group("OME"), version, {"series": ["0"]})
    break_listing(monkeypatch, ValueError("409 Conflict"))

    result = describe(tmp_path / "s.zarr")

    container, plate = result.ome[:2]
    assert isinstance(plate, OmePlate)
    assert container == OmeContainer("", plate.version, 3, ("A/1/0", "B/1/0"), "plate")
    assert [i.name for i in result.images] == ["A/1/0", "B/1/0"]
    assert "A/1/0/0" in result


# Stores that can't be listed


@pytest.mark.parametrize("version", OME_VERSIONS)
def test_unlistable_image_found_from_its_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: OmeVersion
) -> None:
    image = make_ome_image(ome_root(tmp_path / "s.zarr", version), version)
    make_ome_labels(image, version, ("cells",))
    expected = describe(tmp_path / "s.zarr")
    break_listing(monkeypatch, ValueError("409 Conflict"))

    result = describe(tmp_path / "s.zarr")

    assert result == expected
    assert [a.name for a in result] == ["0", "1", "labels/cells/0", "labels/cells/1"]
    assert [a.name for a in describe(tmp_path / "s.zarr", max_depth=0)] == ["0", "1"]


def test_unlistable_idr_like_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The shape of the IDR store in the notebook: a v0.5 container whose
    # image 0 holds the levels 0/0 and 0/1.
    make_container(tmp_path / "s.zarr", "0.5", 1)
    break_listing(monkeypatch, ValueError("409 Conflict"))

    result = describe(tmp_path / "s.zarr")

    assert [type(g).__name__ for g in result.ome] == ["OmeContainer", "OmeImage"]
    assert [a.name for a in result] == ["0/0", "0/1"]
    assert result["0/0"].dimensions == ("c", "y", "x")


@pytest.mark.parametrize("version", OME_VERSIONS)
def test_consolidated_ome_store_describes_the_same(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: OmeVersion
) -> None:
    image = make_ome_image(ome_root(tmp_path / "s.zarr", version), version)
    make_ome_labels(image, version, ("cells",))
    expected = describe(tmp_path / "s.zarr")
    consolidate(tmp_path / "s.zarr")
    break_listing(monkeypatch, ValueError("409 Conflict"))

    assert describe(tmp_path / "s.zarr") == expected


@pytest.mark.parametrize("version", OME_VERSIONS)
def test_max_depth_does_not_limit_the_metadata(
    tmp_path: Path, version: OmeVersion
) -> None:
    make_container(tmp_path / "s.zarr", version, 2)

    result = describe(tmp_path / "s.zarr", max_depth=0)

    assert [a.name for a in result] == []  # no arrays at the top level
    assert [i.name for i in result.images] == ["0", "1"]


# Unsupported versions


@pytest.mark.parametrize("listable", [True, False])
def test_unsupported_version_reads_only_the_dataset_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, listable: bool
) -> None:
    root = ome_root(tmp_path / "s.zarr", "0.5")
    entry = multiscales_entry("0.5")
    make_ome_image(root, "0.5", multiscales=[entry], extra={"omero": 1})
    # Metadata a 0.4/0.5 reader would reject: none of it is read.
    entry["axes"] = "whatever 0.6 says"
    entry["datasets"][0]["coordinateTransformations"] = [{"type": "affine"}]
    root.attrs["ome"] = {"version": "0.6", "multiscales": [entry], "omero": 1}
    if not listable:
        break_listing(monkeypatch, ValueError("409 Conflict"))

    image = only_image(describe(tmp_path / "s.zarr"))

    assert image.version == "0.6"
    assert not image.supported
    (result,) = image.multiscales
    assert result.axes == ()
    assert [(level.name, level.shape) for level in result.levels] == [
        ("0", (2, 4, 4)),
        ("1", (2, 2, 2)),
    ]
    assert all(level.scale is None for level in result.levels)
    assert image.channel_labels == ()
    assert "not supported: only the dataset paths were read" in str(image)


# Malformed metadata


def image_attrs(
    change: Callable[[dict[str, Any]], None],
) -> Callable[[zarr.Group], None]:
    """Make a function that writes a v0.5 image and changes its ome attributes."""

    def write(root: zarr.Group) -> None:
        make_ome_image(root, "0.5")
        ome = ome_of(root)
        change(ome)
        root.attrs["ome"] = ome

    return write


def entry0(ome: dict[str, Any]) -> dict[str, Any]:
    entry: dict[str, Any] = ome["multiscales"][0]
    return entry


def transforms0(ome: dict[str, Any]) -> list[dict[str, Any]]:
    transforms: list[dict[str, Any]] = entry0(ome)["datasets"][0][
        "coordinateTransformations"
    ]
    return transforms


MALFORMED_IMAGES: list[tuple[Callable[[dict[str, Any]], None], str]] = [
    (lambda o: o.update(multiscales=[]), '"multiscales" must be a non-empty list'),
    (lambda o: entry0(o).pop("axes"), 'multiscales 0: "axes" must be a list'),
    (lambda o: entry0(o)["axes"][2].pop("name"), 'axis 2 has no "name"'),
    (
        lambda o: entry0(o)["axes"][2].update(unit=1),
        "axis 'x' has a \"unit\" that is not a string",
    ),
    (
        lambda o: entry0(o)["axes"][2].update(name="y"),
        "duplicate axis name 'y'",
    ),
    (
        lambda o: entry0(o).update(datasets=[]),
        'multiscales 0: "datasets" must be a non-empty list',
    ),
    (lambda o: entry0(o)["datasets"][1].pop("path"), 'dataset 1 has no "path"'),
    (
        # Dataset 1 has a scale and a translation: put the translation first.
        lambda o: entry0(o)["datasets"][1]["coordinateTransformations"].reverse(),
        "dataset '1': \"coordinateTransformations\" must be one scale, "
        "optionally followed by one translation",
    ),
    (
        lambda o: transforms0(o).append(transforms0(o)[0]),
        "must be one scale, optionally followed by one translation",
    ),
    (
        lambda o: entry0(o)["datasets"][0].pop("coordinateTransformations"),
        "must be one scale, optionally followed by one translation",
    ),
    (
        lambda o: transforms0(o).append({"type": "affine"}),
        "unknown transform type 'affine'",
    ),
    (
        lambda o: transforms0(o)[0].update(scale=["1", 1, 1]),
        "dataset '0': \"scale\" must be a list of numbers",
    ),
    (
        lambda o: transforms0(o)[0].update(scale=[1.0, 1.0]),
        "dataset '0': \"scale\" has 2 values for 3 axes",
    ),
    (
        lambda o: entry0(o).update(coordinateTransformations=[{"type": "translation"}]),
        'multiscales 0: "coordinateTransformations" must be one scale',
    ),
    (
        lambda o: entry0(o)["datasets"][1].update(path="9"),
        "dataset '9' names the array '9', which cannot be opened (KeyError",
    ),
    (
        lambda o: entry0(o).update(
            axes=entry0(o)["axes"][1:],
            datasets=[
                {
                    "path": "0",
                    "coordinateTransformations": [{"type": "scale", "scale": [1, 1]}],
                }
            ],
        ),
        "array '0' has 3 dimensions for 2 axes",
    ),
    (lambda o: entry0(o).update(name=3), '"name" must be a string'),
    (lambda o: o.update(omero={"channels": {}}), '"omero.channels" must be a list'),
    (lambda o: o.update(omero=[]), '"omero" must be an object'),
    (lambda o: o.update(version=5), '"version" must be a string'),
    (lambda o: o.update({"image-label": []}), '"image-label" must be an object'),
    (
        lambda o: o.update({"version": None, "image-label": {}}),
        '"ome.version" is required for a label image',
    ),
]


@pytest.mark.parametrize(("change", "problem"), MALFORMED_IMAGES)
def test_malformed_image_warns_and_keeps_the_arrays(
    tmp_path: Path, change: Callable[[dict[str, Any]], None], problem: str
) -> None:
    root = ome_root(tmp_path / "s.zarr", "0.5")
    image_attrs(change)(root)

    with pytest.warns(OmeMetadataWarning) as record:
        result = describe(tmp_path / "s.zarr")

    (warning,) = record
    message = str(warning.message)
    assert message.startswith("Invalid OME-Zarr metadata in group '': ")
    assert problem in message
    assert warning.filename == __file__  # points at the call of describe()
    assert result.ome == ()
    assert [a.name for a in result] == ["0", "1"]


@pytest.mark.parametrize(("change", "problem"), MALFORMED_IMAGES[:4])
def test_malformed_image_in_an_unlistable_store_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: Callable[[dict[str, Any]], None],
    problem: str,
) -> None:
    image_attrs(change)(ome_root(tmp_path / "s.zarr", "0.5"))
    break_listing(monkeypatch, ValueError("409 Conflict"))

    with pytest.raises(OmeMetadataError, match="Invalid OME-Zarr metadata") as info:
        describe(tmp_path / "s.zarr")

    assert problem in str(info.value)
    assert "extract() still works" in str(info.value)


def test_ome_key_must_be_an_object(tmp_path: Path) -> None:
    root = ome_root(tmp_path / "s.zarr", "0.5")
    root.attrs["ome"] = "0.5"

    with pytest.warns(OmeMetadataWarning, match='"ome" must be an object'):
        assert describe(tmp_path / "s.zarr").ome == ()


@pytest.mark.parametrize("version", OME_VERSIONS)
def test_malformed_label_group_leaves_out_only_that_group(
    tmp_path: Path, version: OmeVersion
) -> None:
    image = make_ome_image(ome_root(tmp_path / "s.zarr", version), version)
    labels = make_ome_labels(image, version, ("cells",))
    set_ome(labels, version, {"labels": "cells"})

    with pytest.warns(OmeMetadataWarning, match="'labels'.*\"labels\" must be a list"):
        result = describe(tmp_path / "s.zarr")

    # The label image is still found by the listing, but not linked.
    assert [g.name for g in result.ome] == ["", "labels/cells"]
    assert only_image_named(result, "").labels == ()


def only_image_named(result: StoreDescription, name: str) -> OmeImage:
    (image,) = [i for i in result.images if i.name == name]
    return image


def plate_keys(**changes: Any) -> dict[str, Any]:
    plate = {
        "rows": [{"name": "A"}],
        "columns": [{"name": "1"}],
        "wells": [{"path": "A/1", "rowIndex": 0, "columnIndex": 0}],
    }
    return {"plate": plate | changes}


@pytest.mark.parametrize(
    ("keys", "problem"),
    [
        (
            plate_keys(wells=[{"row": "A"}]),
            '"wells" must be a list of objects with a "path"',
        ),
        (plate_keys(rows=["A"]), '"rows" must be a list of objects with a "name"'),
        (plate_keys(columns=None), '"columns" must be a list of objects with a "name"'),
        ({"plate": []}, '"plate" must be an object'),
        (
            {"well": {"images": ["0"]}},
            '"images" must be a list of objects with a "path"',
        ),
        ({"bioformats2raw.layout": 2}, '"bioformats2raw.layout" is 2, only 3 is known'),
    ],
)
def test_malformed_plate_well_and_container(
    tmp_path: Path, keys: dict[str, Any], problem: str
) -> None:
    root = ome_root(tmp_path / "s.zarr", "0.5")
    set_ome(root, "0.5", keys)

    with pytest.warns(OmeMetadataWarning, match="Invalid OME-Zarr metadata") as record:
        result = describe(tmp_path / "s.zarr")

    assert problem in str(record[0].message)
    assert result.ome == ()


def test_v05_plate_needs_a_version(tmp_path: Path) -> None:
    root = ome_root(tmp_path / "s.zarr", "0.5")
    set_ome(root, "0.5", plate_keys(wells=[]), with_version=False)

    with pytest.warns(
        OmeMetadataWarning, match='"ome.version" is required for a plate'
    ):
        describe(tmp_path / "s.zarr")


def test_v05_plate_well_inherits_the_version(tmp_path: Path) -> None:
    root = ome_root(tmp_path / "s.zarr", "0.5")
    make_ome_plate(root, "0.5", ("A/1",))
    well = root["A/1"]
    assert isinstance(well, zarr.Group)
    set_ome(well, "0.5", {"well": {"images": [{"path": "0"}]}}, with_version=False)

    result = describe(tmp_path / "s.zarr")

    assert OmeWell("A/1", "0.5", ("0",)) in result.ome


def test_malformed_series(tmp_path: Path) -> None:
    root = make_container(tmp_path / "s.zarr", "0.5", 1)
    set_ome(root.create_group("OME"), "0.5", {"series": "0"})

    with pytest.warns(
        OmeMetadataWarning, match='"OME" group: "series" must be a list of paths'
    ):
        result = describe(tmp_path / "s.zarr")

    assert [g.name for g in result.ome] == ["0"]  # the image, found by listing


def test_missing_well_and_image_groups(tmp_path: Path) -> None:
    root = ome_root(tmp_path / "s.zarr", "0.5")
    make_ome_plate(root, "0.5", ("A/1",))
    plate = ome_of(root)
    plate["plate"]["wells"].append({"path": "B/2", "rowIndex": 1, "columnIndex": 1})
    root.attrs["ome"] = plate
    well = root["A/1"]
    assert isinstance(well, zarr.Group)
    set_ome(well, "0.5", {"well": {"images": [{"path": "0"}, {"path": "1"}]}})

    with pytest.warns(OmeMetadataWarning) as record:
        describe(tmp_path / "s.zarr")

    messages = sorted(str(w.message) for w in record)
    assert messages == [
        "Invalid OME-Zarr metadata in group 'A/1/1': the well lists this group "
        "as an image, but the group is missing.",
        "Invalid OME-Zarr metadata in group 'B/2': the plate lists this group "
        "as a well, but the group is missing.",
    ]


def test_container_image_without_multiscales(tmp_path: Path) -> None:
    root = make_container(tmp_path / "s.zarr", "0.5", 1)
    root.create_group("1")

    with pytest.warns(
        OmeMetadataWarning,
        match="group '1': the bioformats2raw container lists this group as an "
        'image, but it has no "multiscales"',
    ):
        describe(tmp_path / "s.zarr")


# Stores that can't be listed: what is an error and what is a warning


def break_multiscales(group: zarr.Group, version: OmeVersion) -> None:
    """Make the multiscales metadata of an image invalid."""
    set_ome(group, version, {"multiscales": []})


def messages(record: pytest.WarningsRecorder) -> list[str]:
    return sorted(str(w.message) for w in record)


@pytest.mark.parametrize("version", OME_VERSIONS)
def test_unlistable_malformed_plate_at_the_root_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: OmeVersion
) -> None:
    root = ome_root(tmp_path / "s.zarr", version)
    make_ome_plate(root, version, ("A/1",))
    set_ome(root, version, {"plate": []})
    break_listing(monkeypatch, ValueError("409 Conflict"))

    with pytest.raises(OmeMetadataError, match='"plate" must be an object'):
        describe(tmp_path / "s.zarr")


@pytest.mark.parametrize("version", OME_VERSIONS)
def test_unlistable_malformed_series_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: OmeVersion
) -> None:
    root = make_container(tmp_path / "s.zarr", version, 2)
    set_ome(root.create_group("OME"), version, {"series": "0"})
    break_listing(monkeypatch, ValueError("409 Conflict"))

    with pytest.raises(OmeMetadataError, match='"series" must be a list of paths'):
        describe(tmp_path / "s.zarr")


@pytest.mark.parametrize("version", OME_VERSIONS)
def test_unlistable_plate_skips_a_bad_well_and_a_bad_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: OmeVersion
) -> None:
    root = ome_root(tmp_path / "s.zarr", version)
    make_ome_plate(root, version, ("A/1", "A/2", "B/1", "B/2"))
    bad_well = root["A/2"]
    bad_image = root["B/1/0"]
    assert isinstance(bad_well, zarr.Group)
    assert isinstance(bad_image, zarr.Group)
    set_ome(bad_well, version, {"well": {"images": "0"}})
    break_multiscales(bad_image, version)
    del root["B/2"]  # named by the plate, but missing
    break_listing(monkeypatch, ValueError("409 Conflict"))

    with pytest.warns(OmeMetadataWarning) as record:
        result = describe(tmp_path / "s.zarr")

    found = messages(record)
    assert len(found) == 3
    assert "group 'A/2': " in found[0]
    assert '"images" must be a list of objects' in found[0]
    assert "group 'B/1/0': " in found[1]
    assert '"multiscales" must be a non-empty list' in found[1]
    assert "group 'B/2': the plate lists this group as a well" in found[2]
    # The rest is described: the plate, the good well and its image.
    assert [i.name for i in result.images] == ["A/1/0"]
    assert [w.name for w in result.ome if isinstance(w, OmeWell)] == ["A/1", "B/1"]
    assert [a.name for a in result] == ["A/1/0/0", "A/1/0/1"]


@pytest.mark.parametrize("version", OME_VERSIONS)
def test_unlistable_container_skips_a_bad_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: OmeVersion
) -> None:
    root = make_container(tmp_path / "s.zarr", version, 2)
    bad = root["1"]
    assert isinstance(bad, zarr.Group)
    break_multiscales(bad, version)
    break_listing(monkeypatch, ValueError("409 Conflict"))

    with pytest.warns(OmeMetadataWarning, match="group '1': ") as record:
        result = describe(tmp_path / "s.zarr")

    assert len(record) == 1
    assert [i.name for i in result.images] == ["0"]
    assert [a.name for a in result] == ["0/0", "0/1"]


@pytest.mark.parametrize("version", OME_VERSIONS)
def test_unlistable_image_skips_a_bad_labels_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: OmeVersion
) -> None:
    root = make_ome_image(ome_root(tmp_path / "s.zarr", version), version)
    labels = make_ome_labels(root, version, ("cells",))
    set_ome(labels, version, {"labels": "cells"})
    break_listing(monkeypatch, ValueError("409 Conflict"))

    with pytest.warns(OmeMetadataWarning, match="group 'labels': ") as record:
        result = describe(tmp_path / "s.zarr")

    assert len(record) == 1
    assert [i.name for i in result.images] == [""]
    assert only_image_named(result, "").labels == ()
    assert [a.name for a in result] == ["0", "1"]


@pytest.mark.parametrize("version", OME_VERSIONS)
def test_unlistable_image_skips_a_missing_and_a_bad_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: OmeVersion
) -> None:
    root = make_ome_image(ome_root(tmp_path / "s.zarr", version), version)
    labels = make_ome_labels(root, version, ("cells", "nuclei", "gone"))
    bad = labels["nuclei"]
    assert isinstance(bad, zarr.Group)
    break_multiscales(bad, version)
    del labels["gone"]
    break_listing(monkeypatch, ValueError("409 Conflict"))

    with pytest.warns(OmeMetadataWarning) as record:
        result = describe(tmp_path / "s.zarr")

    found = messages(record)
    assert len(found) == 2
    assert "group 'labels/gone': the labels group lists this group" in found[0]
    assert "group 'labels/nuclei': " in found[1]
    assert [i.name for i in result.images] == ["", "labels/cells"]
    # The labels group still lists all three paths, as written.
    assert only_image_named(result, "").labels == ("cells", "nuclei", "gone")
    assert "labels/cells/0" in result


@pytest.mark.parametrize("version", OME_VERSIONS)
@pytest.mark.parametrize("listable", [True, False])
def test_image_without_a_labels_group_gives_no_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: OmeVersion, listable: bool
) -> None:
    make_ome_image(ome_root(tmp_path / "s.zarr", version), version)
    if not listable:
        break_listing(monkeypatch, ValueError("409 Conflict"))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = describe(tmp_path / "s.zarr")

    assert only_image_named(result, "").labels == ()
    assert [a.name for a in result] == ["0", "1"]


# Stores that are not OME-Zarr


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_non_ome_store_is_described_as_before(tmp_path: Path, zarr_format: Any) -> None:
    result = describe(make_store(tmp_path / "s.zarr", zarr_format))

    assert result.ome == ()
    assert not result.is_ome
    assert result.images == ()
    assert result == StoreDescription(arrays=result.arrays)


def test_non_ome_repr_is_unchanged(tmp_path: Path) -> None:
    result = describe(make_store(tmp_path / "s.zarr", 3))

    assert repr(result) == (
        "StoreDescription(arrays=("
        "ArrayInfo(name='lat', dimensions=('lat',), shape=(2,), chunks=(2,), "
        "dtype='float64', is_coordinate=True), "
        "ArrayInfo(name='lon', dimensions=('lon',), shape=(4,), chunks=(4,), "
        "dtype='float64', is_coordinate=True), "
        "ArrayInfo(name='temp', dimensions=('time', 'lat', 'lon'), "
        "shape=(3, 2, 4), chunks=(1, 2, 2), dtype='float32', "
        "is_coordinate=False), "
        "ArrayInfo(name='time', dimensions=('time',), shape=(3,), chunks=(3,), "
        "dtype='float64', is_coordinate=True)))"
    )


def test_ome_repr_adds_the_groups(tmp_path: Path) -> None:
    make_ome_image(ome_root(tmp_path / "s.zarr", "0.5"), "0.5")

    result = describe(tmp_path / "s.zarr")

    assert repr(result) == (
        f"StoreDescription(arrays={result.arrays!r}, ome={result.ome!r})"
    )
    assert repr(result).endswith("labels=()),))")


def test_unrelated_attributes_are_not_ome(tmp_path: Path) -> None:
    root = zarr.open_group(tmp_path / "s.zarr", mode="w", zarr_format=3)
    root.attrs.update({"title": "x", "omero": {"channels": []}, "series": ["0"]})
    root.create_group("0").attrs["ome"] = {"version": "0.5"}

    assert describe(tmp_path / "s.zarr").ome == ()


def test_unlistable_non_ome_store_still_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_store(tmp_path / "s.zarr", 3)
    break_listing(monkeypatch, ValueError("409 Conflict"))

    with pytest.raises(zarrextract.StoreOpenError, match="cannot be listed"):
        describe(tmp_path / "s.zarr")


# __str__


def test_str_of_an_image(tmp_path: Path) -> None:
    image = make_ome_image(
        ome_root(tmp_path / "s.zarr", "0.5"),
        "0.5",
        extra={"omero": {"channels": [{"label": "DAPI"}, {}]}},
    )
    entry = ome_of(image)
    entry["multiscales"][0]["coordinateTransformations"] = [
        {"type": "scale", "scale": [1, 1, 1]}
    ]
    image.attrs["ome"] = entry
    make_ome_labels(image, "0.5", ("cells",))

    result = describe(tmp_path / "s.zarr")

    assert str(only_image_named(result, "")) == (
        "OME-Zarr image '' (version 0.5)\n"
        "  multiscales 1 of 1 (no name)\n"
        "    axes: c (channel), y (space, micrometer), x (space, micrometer)\n"
        "    after the level transforms: scale (1.0, 1.0, 1.0)\n"
        "    levels:\n"
        "      0: shape (2, 4, 4), uint16, chunks (1, 4, 4)\n"
        "        scale (1.0, 1.0, 1.0)\n"
        "      1: shape (2, 2, 2), uint16, chunks (1, 2, 2)\n"
        "        scale (1.0, 2.0, 2.0)\n"
        "        translation (0.0, 0.5, 0.5)\n"
        "  channels: DAPI, (no label)\n"
        "  labels: cells"
    )
    assert str(only_image_named(result, "labels/cells")).startswith(
        "OME-Zarr label image 'labels/cells' (version 0.5)\n"
    )
    assert str(result.ome[1]) == (
        "OME-Zarr labels group 'labels' (version 0.5): 1 label: cells"
    )


def test_str_of_a_multiscale_axis_and_level() -> None:
    level = OmeLevel("0", "0/0", (4,), (2,), "uint8", (1.0,), None)
    entry = OmeMultiscale("raw", (OmeAxis("x", None, None),), (level,), None, None)

    assert str(OmeAxis("z", None, "micrometer")) == "z (micrometer)"
    assert str(entry) == (
        "multiscales 'raw'\n"
        "  axes: x\n"
        "  levels:\n"
        "    0/0: shape (4), uint8, chunks (2)\n"
        "      scale (1.0)"
    )


def test_str_of_a_container_plate_and_well() -> None:
    wells = tuple(f"A/{i}" for i in range(1, 13))

    assert str(OmeContainer("", "0.5", 3, ("0", "1"), "numbered")) == (
        "OME-Zarr bioformats2raw container '' (layout 3, version 0.5): "
        "2 images, numbered: 0, 1"
    )
    assert str(OmeContainer("", None, 3, ("A/1/0",), "plate")) == (
        "OME-Zarr bioformats2raw container '' (layout 3, version not given): "
        "1 image, from the plate: A/1/0"
    )
    assert str(OmePlate("", "0.4", ("A",), ("1", "2"), wells)) == (
        "OME-Zarr plate '' (version 0.4): 1 row x 2 columns, 12 wells: "
        "A/1, A/2, A/3, A/4, A/5, A/6, A/7, A/8, A/9, A/10, ... (2 more)"
    )
    assert str(OmeWell("A/1", "0.4", ())) == (
        "OME-Zarr well 'A/1' (version 0.4): 0 images: (none)"
    )


def test_ome_classes_are_public_and_frozen() -> None:
    for cls in (OmeAxis, OmeContainer, OmeImage, OmeLabels, OmeLevel, OmeMultiscale):
        assert cls.__module__ == "zarrextract"
    assert OmeMetadataWarning.__module__ == "zarrextract"
    assert issubclass(OmeMetadataError, zarrextract.ZarrExtractError)
    assert issubclass(OmeMetadataWarning, UserWarning)
    with pytest.raises(AttributeError):
        OmeAxis("x", None, None).name = "y"  # type: ignore[misc]
