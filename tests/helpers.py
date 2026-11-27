"""Helpers for building tiny Zarr stores in a temporary directory."""

import copy
import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import zarr

ZarrFormat = Literal[2, 3]


def write_array(
    group: zarr.Group,
    name: str,
    data: npt.ArrayLike,
    dims: tuple[str, ...] | None,
    *,
    chunks: tuple[int, ...] | None = None,
    attrs: dict[str, Any] | None = None,
    fill_value: Any = None,
) -> None:
    """Write an array with dimension names in the style of the store's format."""
    values = np.asarray(data)
    v3 = group.metadata.zarr_format == 3
    array = group.create_array(
        name,
        shape=values.shape,
        chunks=chunks if chunks is not None else values.shape,
        dtype=values.dtype,
        dimension_names=dims if v3 else None,
        fill_value=fill_value,
    )
    array[...] = values
    if not v3 and dims is not None:
        array.attrs["_ARRAY_DIMENSIONS"] = list(dims)
    if attrs:
        array.attrs.update(attrs)


def make_grid(
    path: Path,
    zarr_format: ZarrFormat,
    *,
    time: npt.ArrayLike = (0, 1, 2, 3),
    time_attrs: dict[str, Any] | None = None,
    lat: npt.ArrayLike = (90.0, 45.0, 0.0, -45.0, -90.0),
    lon: npt.ArrayLike = (0.0, 90.0, 180.0, 270.0),
    data: npt.ArrayLike | None = None,
    data_attrs: dict[str, Any] | None = None,
    chunks: tuple[int, int, int] = (1, 2, 2),
    fill_value: Any = None,
) -> Path:
    """Write a (time, lat, lon) grid with a variable called ``temp``.

    By default ``temp[t, y, x]`` is ``100 * t + 10 * y + x`` (indices, not
    coordinate values), which makes expected values easy to work out.
    """
    time_values = np.asarray(time)
    lat_values = np.asarray(lat)
    lon_values = np.asarray(lon)
    if data is None:
        t = np.arange(time_values.size)[:, None, None]
        y = np.arange(lat_values.size)[None, :, None]
        x = np.arange(lon_values.size)[None, None, :]
        data = (100 * t + 10 * y + x).astype("f8")
    group = zarr.open_group(path, mode="w", zarr_format=zarr_format)
    write_array(
        group,
        "time",
        time_values,
        ("time",),
        attrs=time_attrs
        if time_attrs is not None
        else {"units": "days since 2000-01-01"},
    )
    write_array(group, "lat", lat_values, ("lat",), attrs={"units": "degrees_north"})
    write_array(group, "lon", lon_values, ("lon",), attrs={"units": "degrees_east"})
    write_array(
        group,
        "temp",
        data,
        ("time", "lat", "lon"),
        chunks=chunks,
        attrs=data_attrs,
        fill_value=fill_value,
    )
    return path


def make_store(
    path: Path,
    zarr_format: ZarrFormat,
    *,
    consolidate: bool = False,
    with_dims: bool = True,
) -> Path:
    """Write a small gridded store: time, lat and lon coordinates plus data."""
    group = zarr.open_group(path, mode="w", zarr_format=zarr_format)
    dims = ("time", "lat", "lon") if with_dims else None
    write_array(
        group, "time", np.arange(3, dtype="f8"), ("time",) if with_dims else None
    )
    write_array(group, "lat", np.array([10.0, 0.0]), ("lat",) if with_dims else None)
    write_array(
        group,
        "lon",
        np.array([0.0, 90.0, 180.0, 270.0]),
        ("lon",) if with_dims else None,
    )
    write_array(
        group,
        "temp",
        np.arange(24, dtype="f4").reshape(3, 2, 4),
        dims,
        chunks=(1, 2, 2),
    )
    if consolidate:
        with warnings.catch_warnings():
            # Zarr warns that v3 consolidated metadata is not in the spec.
            warnings.simplefilter("ignore")
            zarr.consolidate_metadata(path)
    return path


# OME-Zarr fixtures. v0.4 stores are Zarr v2 with the metadata at the top
# level of the attributes; v0.5 stores are Zarr v3 with it under "ome".

OmeVersion = Literal["0.4", "0.5"]
OME_VERSIONS: list[OmeVersion] = ["0.4", "0.5"]

DEFAULT_AXES: list[dict[str, str]] = [
    {"name": "c", "type": "channel"},
    {"name": "y", "type": "space", "unit": "micrometer"},
    {"name": "x", "type": "space", "unit": "micrometer"},
]


def ome_root(path: Path, version: OmeVersion) -> zarr.Group:
    """Create an empty group in the Zarr format of the OME version."""
    return zarr.open_group(path, mode="w", zarr_format=2 if version == "0.4" else 3)


def set_ome(
    group: zarr.Group,
    version: OmeVersion,
    keys: dict[str, Any],
    *,
    with_version: bool = True,
) -> None:
    """Write OME keys in the layout of ``version``.

    In v0.5 the version goes in ``ome.version``. In v0.4 it goes into the
    multiscales entries, the plate, the well and the image-label, so
    ``keys`` should already hold it there.
    """
    if version == "0.4":
        group.attrs.update(keys)
    else:
        group.attrs["ome"] = ({"version": "0.5"} if with_version else {}) | keys


def multiscales_entry(
    version: OmeVersion,
    paths: tuple[str, ...] = ("0", "1"),
    *,
    axes: list[dict[str, str]] | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """A multiscales entry: level i has scale 2**i in y and x.

    Every level but the first also has a translation of half a pixel.
    """
    datasets = []
    for i, path in enumerate(paths):
        transforms: list[dict[str, Any]] = [
            {"type": "scale", "scale": [1.0, 2.0**i, 2.0**i]}
        ]
        if i:
            transforms.append(
                {"type": "translation", "translation": [0.0, 0.5 * i, 0.5 * i]}
            )
        datasets.append({"path": path, "coordinateTransformations": transforms})
    entry: dict[str, Any] = {
        # A copy, so that a test changing the axes doesn't change the default.
        "axes": copy.deepcopy(DEFAULT_AXES if axes is None else axes),
        "datasets": datasets,
    }
    if name is not None:
        entry["name"] = name
    if version == "0.4":
        entry["version"] = "0.4"
    return entry


def make_ome_image(
    group: zarr.Group,
    version: OmeVersion,
    *,
    multiscales: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
    with_version: bool = True,
) -> zarr.Group:
    """Write an image into ``group``: its metadata and one array per level.

    Level i of each entry has the shape ``(2, 4 // 2**i, 4 // 2**i)``, chunks
    of 1 along c, and the dimension names of the entry's axes.
    """
    entries = multiscales if multiscales is not None else [multiscales_entry(version)]
    for entry in entries:
        dims = tuple(axis["name"] for axis in entry["axes"])
        for i, dataset in enumerate(entry["datasets"]):
            side = 4 // 2**i
            write_array(
                group,
                dataset["path"],
                np.zeros((2, side, side), dtype="u2"),
                dims,
                chunks=(1, side, side),
            )
    set_ome(
        group,
        version,
        {"multiscales": entries} | (extra or {}),
        with_version=with_version,
    )
    return group


def make_ome_labels(
    image: zarr.Group, version: OmeVersion, names: tuple[str, ...]
) -> zarr.Group:
    """Add a labels group beside the levels of ``image``, with label images.

    The first label image has image-label metadata and the others don't.
    """
    labels = image.create_group("labels")
    set_ome(labels, version, {"labels": list(names)})
    for i, name in enumerate(names):
        extra: dict[str, Any] = {}
        if i == 0:
            extra["image-label"] = {"colors": [{"label-value": 1, "rgba": [255] * 4}]}
            if version == "0.4":
                extra["image-label"]["version"] = "0.4"
        make_ome_image(labels.create_group(name), version, extra=extra)
    return labels


def make_ome_plate(
    group: zarr.Group, version: OmeVersion, wells: tuple[str, ...]
) -> zarr.Group:
    """Write a plate with rows A, B and columns 1, 2, and one image per well."""
    plate: dict[str, Any] = {
        "rows": [{"name": "A"}, {"name": "B"}],
        "columns": [{"name": "1"}, {"name": "2"}],
        "wells": [
            {"path": well, "rowIndex": "AB".index(well[0]), "columnIndex": 0}
            for well in wells
        ],
    }
    if version == "0.4":
        plate["version"] = "0.4"
    set_ome(group, version, {"plate": plate})
    for well_path in wells:
        well_group = group.require_group(well_path)
        well: dict[str, Any] = {"images": [{"path": "0"}]}
        if version == "0.4":
            well["version"] = "0.4"
        set_ome(well_group, version, {"well": well})
        make_ome_image(well_group.create_group("0"), version)
    return group
