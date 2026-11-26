"""Describe the arrays in a Zarr store."""

import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import zarr

from zarrextract._errors import StoreOpenError, VariableNotFoundError
from zarrextract._store import dimension_names, open_group


@dataclass(frozen=True, slots=True)
class ArrayInfo:
    """Metadata for one array in a store.

    Attributes:
        name: The array's path from the store.
        dimensions: Dimension names, or ``None`` if the array does not record
            them.
        shape: The length of each dimension.
        chunks: The chunk length along each dimension.
        dtype: The data type, as a string.
        is_coordinate: True for a 1-D array named after its own single
            dimension.
    """

    name: str
    dimensions: tuple[str, ...] | None
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: str
    is_coordinate: bool


@dataclass(frozen=True, slots=True)
class StoreDescription:
    """The arrays of a store, sorted by name (their path from the store).

    Attributes:
        arrays: Every array in the store.
        coordinates: The 1-D coordinate arrays.
        data_variables: Every array that is not a coordinate.
    """

    arrays: tuple[ArrayInfo, ...]

    @property
    def coordinates(self) -> tuple[ArrayInfo, ...]:
        return tuple(a for a in self.arrays if a.is_coordinate)

    @property
    def data_variables(self) -> tuple[ArrayInfo, ...]:
        return tuple(a for a in self.arrays if not a.is_coordinate)

    def __iter__(self) -> Iterator[ArrayInfo]:
        return iter(self.arrays)

    def __len__(self) -> int:
        return len(self.arrays)

    def __contains__(self, name: object) -> bool:
        return any(info.name == name for info in self.arrays)

    def __getitem__(self, name: str) -> ArrayInfo:
        for info in self.arrays:
            if info.name == name:
                return info
        raise VariableNotFoundError(f"No array named {name!r} in the store")


def _list_arrays(
    group: zarr.Group, max_depth: int | None, target: str
) -> list[tuple[str, "zarr.Array[Any]"]]:
    """List the arrays under a group, by listing or else from consolidated metadata."""
    try:
        return [
            (path, node)
            for path, node in group.members(max_depth=max_depth)
            if isinstance(node, zarr.Array)
        ]
    except Exception as exc:
        consolidated = group.metadata.consolidated_metadata
        if consolidated is None:
            raise StoreOpenError(
                f"Cannot list the arrays in {target!r}: the store cannot be listed "
                f"({type(exc).__name__}: {exc}) and it has no consolidated "
                "metadata. extract() still works if you know the array's path."
            ) from exc
        arrays: list[tuple[str, zarr.Array[Any]]] = []
        for path in consolidated.flattened_metadata:
            if max_depth is not None and path.count("/") > max_depth:
                continue
            node = group[path]
            if isinstance(node, zarr.Array):
                arrays.append((path, node))
        return arrays


def describe(
    store: str | os.PathLike[str],
    *,
    max_depth: int | None = None,
    storage_options: Mapping[str, Any] | None = None,
) -> StoreDescription:
    """List the arrays in a Zarr store without reading any data.

    ``store`` is a local path or a URL string. URLs work if the matching
    fsspec backend is installed. Consolidated metadata is used when present.

    Arrays in nested groups are listed too, named by their path from the
    store, such as ``"0/0"`` in an OME-Zarr image. That path is what
    ``extract()`` takes. ``max_depth`` limits how many levels of groups are
    searched: 0 lists only the top-level arrays, 1 also the arrays in their
    child groups, and so on. By default there is no limit.

    Some servers cannot list a group, for example plain HTTPS servers that
    answer a directory URL with an error. Then the group's consolidated
    metadata is used instead, and if there is none, ``StoreOpenError`` says so.

    ``storage_options`` is passed to the fsspec backend, for example
    ``{"anon": True}`` to read a public S3 bucket without credentials.

    Raises:
        StoreOpenError: The store cannot be opened or listed, or the server
            or the connection fails.
        ValueError: ``max_depth`` is negative.
    """
    if max_depth is not None and max_depth < 0:
        raise ValueError(f"max_depth must be None or >= 0, not {max_depth}")
    target = os.fspath(store)
    group = open_group(store, storage_options)
    infos: list[ArrayInfo] = []
    for name, array in sorted(
        _list_arrays(group, max_depth, target), key=lambda item: item[0]
    ):
        dims = dimension_names(array)
        infos.append(
            ArrayInfo(
                name=name,
                dimensions=dims,
                shape=tuple(array.shape),
                chunks=tuple(array.chunks),
                dtype=str(array.dtype),
                is_coordinate=dims is not None and dims == (name,),
            )
        )
    return StoreDescription(arrays=tuple(infos))
