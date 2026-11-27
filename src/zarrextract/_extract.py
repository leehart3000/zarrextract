"""Extract points, time series and boxes from a Zarr store into Arrow."""

import math
import os
from collections.abc import Mapping
from typing import Any

import numpy as np
import pyarrow as pa
import zarr
from numpy.typing import NDArray

from zarrextract._cache import AxisEntry, Cache
from zarrextract._calendar import Clock
from zarrextract._cf import (
    decode_calendar_ticks,
    decode_time_with,
    is_longitude,
    is_time_units,
    parse_time_encoding,
    unpack,
)
from zarrextract._errors import (
    MissingDimensionNamesError,
    UnknownDimensionError,
    VariableNotFoundError,
    ZarrExtractError,
)
from zarrextract._probe import LazyAxis
from zarrextract._provenance import field_metadata
from zarrextract._select import Axis, Selection, check_finite, select
from zarrextract._store import dimension_names

BISECT_MIN_CHUNKS = 8
"""Selected coordinates with more chunks than this are searched, not read whole."""


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
      An infinite end, such as ``(-inf, 10)``, leaves the range open.
      NaN, and an infinite scalar, raise ``SelectionError``.
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
    In the 360_day and all_leap (366_day) calendars some dates, such as 30
    February, are not timestamps, so the time column holds ISO 8601 strings of
    one fixed width (``"2000-02-30T00:00:00"``) and its field metadata has
    ``calendar``. Selections on such a coordinate are read in its own calendar:
    ``"2000-02"`` is 30 days in 360_day, ``"2000-02-30"`` is valid there, and a
    date that does not exist in the calendar raises ``SelectionError``.
    ``scale_factor`` and ``add_offset`` are applied, and ``_FillValue`` and
    ``missing_value`` become nulls. In Zarr v2 so does the fill value in the
    array metadata, as in xarray; in Zarr v3 only the attributes count. Only
    the chunks that overlap the selection are read. A selected coordinate with
    more than 8 chunks is searched instead of read whole: it estimates where
    a value lies from the values already read and reads the chunks around the
    estimate, a few at a time.

    The table carries its provenance as schema metadata under keys that start
    with ``zarrextract.``: the store (without any user name, password, query or
    fragment), the variable, the selections as given, the zarrextract version
    and the time of extraction in UTC. The ``units`` and ``long_name``
    attributes of the variable and the coordinates are the metadata of their
    fields. ``storage_options`` is never recorded. Use ``provenance()`` to read
    it back, and ``write()`` to save a table with it.

    ``storage_options`` is not a selection. It is passed to the fsspec backend,
    for example ``{"anon": True}`` to read a public S3 bucket without
    credentials, so a dimension cannot be called ``storage_options``.

    Each call opens the store again. To make many selections from one store,
    open it once with ``open()`` and call ``Source.extract()``, which gives the
    same tables.

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
    check_finite(selections)
    # Imported here because _source imports this module.
    from zarrextract._source import open as open_source

    with open_source(store, storage_options=storage_options) as source:
        return source.extract(variable, **selections)


def _find_array(cache: Cache, variable: str) -> "zarr.Array[Any]":
    node = cache.node(variable)
    if node is None:
        available = (
            ", ".join(sorted(name for name, _ in cache.group.arrays())) or "none"
        )
        raise VariableNotFoundError(
            f"No array named {variable!r} in the store. Available arrays: {available}."
        )
    if not isinstance(node, zarr.Array):
        raise VariableNotFoundError(f"{variable!r} is a group, not an array.")
    return node


def _read_axis(
    cache: Cache, dim: str, length: int, *, selected: bool = False
) -> AxisEntry:
    """Read the coordinate for a dimension, or use its index if it has none.

    A selected coordinate with more than ``BISECT_MIN_CHUNKS`` chunks is not read
    here. It comes back as a ``LazyAxis`` that is searched, reading few chunks.
    The second value is the metadata (units and long name) of its Arrow field.

    The result is kept in the cache. A coordinate that is read whole is the
    same entry whether or not a later call selects on it.
    """
    node = cache.node(dim)
    array = (
        node
        if isinstance(node, zarr.Array)
        and node.shape == (length,)
        and dimension_names(node) in (None, (dim,))
        else None
    )
    lazy = (
        selected
        and array is not None
        and array.dtype.kind in "iufM"
        and -(-length // array.chunks[0]) > BISECT_MIN_CHUNKS
    )
    key = (dim, length, lazy)
    found = cache.axes.get(key)
    if found is None:
        found = cache.axes.setdefault(key, _build_axis(array, dim, length, lazy=lazy))
    return found


def _build_axis(
    node: "zarr.Array[Any] | None", dim: str, length: int, *, lazy: bool
) -> AxisEntry:
    if node is not None:
        attrs = dict(node.attrs)
        units = attrs.get("units")
        kind = node.dtype.kind
        if lazy:
            time_units = kind != "M" and is_time_units(units)
            is_time = kind == "M" or time_units
            axis = LazyAxis(
                node,
                dim,
                units=units if time_units and isinstance(units, str) else None,
                calendar=attrs.get("calendar"),
                is_time=is_time,
                is_longitude=kind != "M"
                and not time_units
                and is_longitude(dim, attrs),
            )
            return axis, field_metadata(
                attrs,
                decoded_time=is_time,
                calendar=axis.clock.calendar if axis.clock else None,
            )
        values = np.asarray(node[...])
        if values.dtype.kind == "M":
            return Axis(dim, _frozen(values), is_time=True), field_metadata(
                attrs, decoded_time=True
            )
        if is_time_units(units):
            assert isinstance(units, str)
            encoding = parse_time_encoding(units, attrs.get("calendar"))
            if encoding.clock is not None:
                ticks = decode_calendar_ticks(values, encoding)
                return Axis(
                    dim, _frozen(ticks), is_time=True, clock=encoding.clock
                ), field_metadata(
                    attrs, decoded_time=True, calendar=encoding.clock.calendar
                )
            decoded = decode_time_with(values, encoding)
            return Axis(dim, _frozen(decoded), is_time=True), field_metadata(
                attrs, decoded_time=True
            )
        if values.dtype.kind == "T":
            values = values.astype(object)
        return Axis(
            dim, _frozen(values), is_longitude=is_longitude(dim, attrs)
        ), field_metadata(attrs, decoded_time=False)
    return Axis(dim, _frozen(np.arange(length))), None


def _frozen(values: NDArray[Any]) -> NDArray[Any]:
    """Make an array read-only: a cached coordinate is shared by later calls."""
    values.flags.writeable = False
    return values


def _join_indices(parts: tuple[slice, ...]) -> NDArray[np.intp]:
    return np.concatenate([np.arange(part.start, part.stop) for part in parts])


def _column(
    values: NDArray[Any], inner: int, outer: int, clock: Clock | None = None
) -> pa.Array:
    if clock is not None:
        # Write each distinct time once, then repeat the strings.
        strings = clock.format(values)
        positions = np.tile(np.repeat(np.arange(len(values)), inner), outer)
        return strings.take(pa.array(positions))
    repeated = np.tile(np.repeat(values, inner), outer)
    if repeated.dtype.kind == "M":
        unit, _ = np.datetime_data(repeated.dtype)
        return pa.array(repeated, type=pa.timestamp(unit))
    return pa.array(repeated)


def _extract_from_group(
    group: zarr.Group, variable: str, selections: dict[str, Selection]
) -> pa.Table:
    """Extract from a group with nothing remembered, as one call used to."""
    return extract_from_cache(Cache(group), variable, selections)


def extract_from_cache(
    cache: Cache, variable: str, selections: dict[str, Selection]
) -> pa.Table:
    """Extract from an open store, reading only what ``cache`` does not hold."""
    array = _find_array(cache, variable)
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

    read = [
        _read_axis(cache, dim, length, selected=dim in selections)
        for dim, length in zip(dims, array.shape, strict=True)
    ]
    axes = [axis for axis, _ in read]
    field_metadata_by_column = [meta for _, meta in read]
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
            clock=axis.clock,
        )
        for k, (axis, parts) in enumerate(zip(axes, runs, strict=True))
    ]
    columns.append(
        pa.array(values.ravel(), mask=None if mask is None else mask.ravel())
    )
    field_metadata_by_column.append(field_metadata(array.attrs, decoded_time=False))
    schema = pa.schema(
        [
            pa.field(name, column.type, metadata=meta)
            for name, column, meta in zip(
                [*dims, variable], columns, field_metadata_by_column, strict=True
            )
        ]
    )
    return pa.Table.from_arrays(columns, schema=schema)
