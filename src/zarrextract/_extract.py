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
from zarrextract._probe import LazyAxis
from zarrextract._select import Axis, Selection, select
from zarrextract._store import dimension_names, open_group, wrap_client_errors

BISECT_MIN_CHUNKS = 8
"""Selected coordinates with more chunks than this are searched by bisection."""


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
    store's convention (0 to 360 or -180 to 180). A longitude range may wrap
    around the edge of the store's longitude axis, for example ``(-10, 10)`` on
    a 0 to 360 store: the longitude column then runs from ``start`` to ``stop``
    across the seam (350 to 359.75, then 0 to 10). Time selections can be ISO
    strings, ``datetime`` or ``numpy.datetime64``.

    In a time ``(start, stop)`` range, an ISO string given only to the year,
    month or day covers that whole period, as in pandas and xarray:
    ``("2019-06", "2019-06")`` is all of June 2019, and a stop of
    ``"2019-06-30"`` includes every time on that day. A start of ``"2019-06"``
    means the start of June. Strings with a time of day, ``datetime`` and
    ``datetime64`` values mean exactly that moment, and so does a scalar.

    CF time coordinates (``"<unit> since <date>"`` with the standard,
    gregorian, proleptic_gregorian, noleap or 365_day calendar) become Arrow
    timestamps whose unit is ``s``, ``ms`` or ``us`` to match the time units.
    ``scale_factor`` and ``add_offset`` are applied, and ``_FillValue`` and
    ``missing_value`` become nulls. In Zarr v2 so does the fill value in the
    array metadata, as in xarray; in Zarr v3 only the attributes count. Only
    the chunks that overlap the selection are read. A selected coordinate with
    more than 8 chunks is searched by bisection, which reads about one chunk
    per halving instead of the whole coordinate.

    ``storage_options`` is not a selection. It is passed to the fsspec backend,
    for example ``{"anon": True}`` to read a public S3 bucket without
    credentials, so a dimension cannot be called ``storage_options``.

    Raises:
        StoreOpenError: The store cannot be opened, or the server or the
            connection fails while it is read.
        VariableNotFoundError: ``variable`` is not an array in the store.
        MissingDimensionNamesError: ``variable`` has no dimension names.
        UnknownDimensionError: A keyword is not a dimension of ``variable``.
        SelectionError: A selection cannot be applied. ``EmptySelectionError``
            is a subclass.
        TimeDecodingError: Time units or values cannot be decoded.
            ``UnsupportedCalendarError`` is a subclass.
    """
    with wrap_client_errors(os.fspath(store)):
        return _extract_from_group(
            open_group(store, storage_options), variable, selections
        )


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


def _read_axis(
    group: zarr.Group, dim: str, length: int, *, selected: bool = False
) -> Axis | LazyAxis:
    """Read the coordinate for a dimension, or use its index if it has none.

    A selected coordinate with more than ``BISECT_MIN_CHUNKS`` chunks is not read
    here. It comes back as a ``LazyAxis`` that is searched by bisection.
    """
    try:
        node = group[dim]
    except KeyError:
        node = None
    if (
        isinstance(node, zarr.Array)
        and node.shape == (length,)
        and dimension_names(node) in (None, (dim,))
    ):
        attrs = dict(node.attrs)
        units = attrs.get("units")
        kind = node.dtype.kind
        if (
            selected
            and kind in "iufM"
            and -(-length // node.chunks[0]) > BISECT_MIN_CHUNKS
        ):
            time_units = kind != "M" and is_time_units(units)
            return LazyAxis(
                node,
                dim,
                units=units if time_units and isinstance(units, str) else None,
                calendar=attrs.get("calendar"),
                is_time=kind == "M" or time_units,
                is_longitude=kind != "M"
                and not time_units
                and is_longitude(dim, attrs),
            )
        values = np.asarray(node[...])
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


def _join_indices(parts: tuple[slice, ...]) -> NDArray[np.intp]:
    return np.concatenate([np.arange(part.start, part.stop) for part in parts])


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
        _read_axis(group, dim, length, selected=dim in selections)
        for dim, length in zip(dims, array.shape, strict=True)
    ]
    slices = [
        select(axis, selections[axis.name])
        if axis.name in selections
        else slice(0, axis.size)
        for axis in axes
    ]
    runs = [(sl,) if isinstance(sl, slice) else sl for sl in slices]
    sizes = [sum(s.stop - s.start for s in parts) for parts in runs]

    if all(len(parts) == 1 for parts in runs):
        # Index with contiguous slices so that zarr reads only the chunks it needs.
        raw = np.asarray(array[tuple(parts[0] for parts in runs)])
    else:
        # A range across the longitude seam is two runs of one dimension, which
        # orthogonal indexing reads together.
        raw = np.asarray(
            array.oindex[
                tuple(
                    parts[0] if len(parts) == 1 else _join_indices(parts)
                    for parts in runs
                )
            ]
        )
    raw = raw.reshape(sizes)
    # In Zarr v2 the fill value in the array metadata marks missing data, as in
    # xarray. In v3 it is always set (often 0), so only the attributes count.
    metadata = array.metadata
    values, mask = unpack(
        raw,
        array.attrs,
        metadata.fill_value if metadata.zarr_format == 2 else None,
    )

    columns = [
        _column(
            np.concatenate([axis.read(part) for part in parts]),
            inner=math.prod(sizes[k + 1 :]),
            outer=math.prod(sizes[:k]),
        )
        for k, (axis, parts) in enumerate(zip(axes, runs, strict=True))
    ]
    columns.append(
        pa.array(values.ravel(), mask=None if mask is None else mask.ravel())
    )
    return pa.Table.from_arrays(columns, names=[*dims, variable])
