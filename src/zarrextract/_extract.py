"""Extract points, time series and boxes from a Zarr store into Arrow."""

import math
import os
from collections.abc import Mapping
from typing import Any

import numpy as np
import pyarrow as pa
import zarr
from numpy.typing import NDArray

from zarrextract._cf import decode_time, is_longitude, is_time_units, unpack
from zarrextract._errors import (
    MissingDimensionNamesError,
    UnknownDimensionError,
    VariableNotFoundError,
    ZarrExtractError,
)
from zarrextract._select import Axis, Selection, select
from zarrextract._store import dimension_names, open_group


def extract(
    store: str | os.PathLike[str],
    variable: str,
    /,
    *,
    storage_options: Mapping[str, Any] | None = None,
    **selections: Selection,
) -> pa.Table:
    """Extract part of a variable from a Zarr store as a long-format table.

    The table has one column per dimension of ``variable``, holding the
    coordinate values, followed by a column for the variable itself. There is
    one row per value, in C order (the last dimension changes fastest).

    Each keyword names a dimension and selects along it:

    * a scalar picks the nearest coordinate value, so the dimension has length
      1. A scalar more than one grid step outside the coordinates is an error.
    * a ``(start, stop)`` pair picks every value from start to stop, inclusive.
    * a dimension without a keyword is kept whole.

    Coordinates may be ascending or descending. Longitudes are converted to the
    store's convention (0 to 360 or -180 to 180), but a range that wraps around
    the edge of the store's longitude axis raises ``AntimeridianError``. Time
    selections can be ISO strings, ``datetime`` or ``numpy.datetime64``.

    CF time coordinates (``"<unit> since <date>"`` with the standard,
    gregorian or proleptic_gregorian calendar) become Arrow timestamps whose
    unit is ``s``, ``ms`` or ``us`` to match the time units. ``scale_factor``
    and ``add_offset`` are applied, and ``_FillValue`` and ``missing_value``
    become nulls. Only the chunks that overlap the selection are read.

    ``storage_options`` is not a selection. It is passed to the fsspec backend,
    for example ``{"anon": True}`` to read a public S3 bucket without
    credentials, so a dimension cannot be called ``storage_options``.

    Raises:
        StoreOpenError: The store cannot be opened.
        VariableNotFoundError: ``variable`` is not an array in the store.
        MissingDimensionNamesError: ``variable`` has no dimension names.
        UnknownDimensionError: A keyword is not a dimension of ``variable``.
        SelectionError: A selection cannot be applied. Its subclasses are
            ``EmptySelectionError`` and ``AntimeridianError``.
        TimeDecodingError: Time units or values cannot be decoded.
            ``UnsupportedCalendarError`` is a subclass.
    """
    return _extract_from_group(open_group(store, storage_options), variable, selections)


def _find_array(group: zarr.Group, variable: str) -> "zarr.Array[Any]":
    try:
        node = group[variable]
    except KeyError:
        available = ", ".join(sorted(name for name, _ in group.arrays())) or "none"
        raise VariableNotFoundError(
            f"No array named {variable!r} in the store. Available arrays: {available}."
        ) from None
    if not isinstance(node, zarr.Array):
        raise VariableNotFoundError(f"{variable!r} is a group, not an array.")
    return node


def _read_axis(group: zarr.Group, dim: str, length: int) -> Axis:
    """Read the whole coordinate for a dimension, or use its index if it has none."""
    try:
        node = group[dim]
    except KeyError:
        node = None
    if (
        isinstance(node, zarr.Array)
        and node.shape == (length,)
        and dimension_names(node) in (None, (dim,))
    ):
        values = np.asarray(node[...])
        attrs = dict(node.attrs)
        units = attrs.get("units")
        if values.dtype.kind == "M":
            return Axis(dim, values, is_time=True)
        if is_time_units(units):
            assert isinstance(units, str)
            decoded = decode_time(values, units, attrs.get("calendar"))
            return Axis(dim, decoded, is_time=True)
        if values.dtype.kind == "T":
            values = values.astype(object)
        return Axis(dim, values, is_longitude=is_longitude(dim, attrs))
    return Axis(dim, np.arange(length))


def _column(values: NDArray[Any], inner: int, outer: int) -> pa.Array:
    repeated = np.tile(np.repeat(values, inner), outer)
    if repeated.dtype.kind == "M":
        unit, _ = np.datetime_data(repeated.dtype)
        return pa.array(repeated, type=pa.timestamp(unit))
    return pa.array(repeated)


def _extract_from_group(
    group: zarr.Group, variable: str, selections: dict[str, Selection]
) -> pa.Table:
    array = _find_array(group, variable)
    dims = dimension_names(array)
    if dims is None:
        raise MissingDimensionNamesError(
            f"The array {variable!r} has no dimension names, so it cannot be "
            "selected on. Zarr v3 arrays need dimension_names and Zarr v2 arrays "
            "need the _ARRAY_DIMENSIONS attribute."
        )
    if variable in dims:
        raise ZarrExtractError(
            f"{variable!r} is a coordinate. Its values appear as a column when you "
            "extract any variable that uses it."
        )
    unknown = [name for name in selections if name not in dims]
    if unknown:
        raise UnknownDimensionError(
            f"{variable!r} has no dimension named {', '.join(map(repr, unknown))}. "
            f"Its dimensions are {', '.join(dims)}."
        )

    axes = [
        _read_axis(group, dim, length)
        for dim, length in zip(dims, array.shape, strict=True)
    ]
    slices = [
        select(axis, selections[axis.name])
        if axis.name in selections
        else slice(0, len(axis.values))
        for axis in axes
    ]
    sizes = [s.stop - s.start for s in slices]

    # Index with contiguous slices so that zarr reads only the chunks it needs.
    raw = np.asarray(array[tuple(slices)]).reshape(sizes)
    values, mask = unpack(raw, array.attrs)

    columns = [
        _column(
            axis.values[sl],
            inner=math.prod(sizes[k + 1 :]),
            outer=math.prod(sizes[:k]),
        )
        for k, (axis, sl) in enumerate(zip(axes, slices, strict=True))
    ]
    columns.append(
        pa.array(values.ravel(), mask=None if mask is None else mask.ravel())
    )
    return pa.Table.from_arrays(columns, names=[*dims, variable])
