"""Describe the arrays in a Zarr store."""

import os
from dataclasses import dataclass

from zarrextract._errors import VariableNotFoundError
from zarrextract._store import dimension_names, open_group


@dataclass(frozen=True, slots=True)
class ArrayInfo:
    """Metadata for one array in a store."""

    name: str
    dimensions: tuple[str, ...] | None
    """Dimension names, or ``None`` if the array does not record them."""
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: str
    is_coordinate: bool
    """True for a 1-D array named after its own single dimension."""


@dataclass(frozen=True, slots=True)
class StoreDescription:
    """The top-level arrays of a store, sorted by name."""

    arrays: tuple[ArrayInfo, ...]

    @property
    def coordinates(self) -> tuple[ArrayInfo, ...]:
        """The 1-D coordinate arrays."""
        return tuple(a for a in self.arrays if a.is_coordinate)

    @property
    def data_variables(self) -> tuple[ArrayInfo, ...]:
        """Every array that is not a coordinate."""
        return tuple(a for a in self.arrays if not a.is_coordinate)

    def __getitem__(self, name: str) -> ArrayInfo:
        for info in self.arrays:
            if info.name == name:
                return info
        raise VariableNotFoundError(f"No array named {name!r} in the store")


def describe(store: str | os.PathLike[str]) -> StoreDescription:
    """List the arrays in a Zarr store without reading any data.

    ``store`` is a local path or a URL string. URLs work if the matching
    fsspec backend is installed. Consolidated metadata is used when present.
    Only arrays at the top level of the store are listed.

    Raises:
        StoreOpenError: The store cannot be opened.
    """
    group = open_group(store)
    infos: list[ArrayInfo] = []
    for name, array in sorted(group.arrays(), key=lambda item: item[0]):
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
