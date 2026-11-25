"""Helpers for building tiny Zarr stores in a temporary directory."""

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
