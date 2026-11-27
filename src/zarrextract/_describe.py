"""Describe the arrays in a Zarr store."""

import os
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import zarr

from zarrextract._errors import StoreOpenError, VariableNotFoundError
from zarrextract._ome import OmeFinder, OmeGroup, OmeImage
from zarrextract._provenance import redact_urls
from zarrextract._store import call_wrapped, dimension_names, open_group


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


@dataclass(frozen=True, slots=True, repr=False)
class StoreDescription:
    """The arrays of a store, sorted by name (their path from the store).

    Attributes:
        arrays: Every array in the store.
        ome: The OME-Zarr groups found in the store, sorted by name: images,
            label images, labels groups, plates, wells and bioformats2raw
            containers. Empty for a store that is not OME-Zarr.
        coordinates: The 1-D coordinate arrays.
        data_variables: Every array that is not a coordinate.
        images: The OME-Zarr images, including label images.
        is_ome: True if the store has any OME-Zarr groups.
    """

    arrays: tuple[ArrayInfo, ...]
    ome: tuple[OmeGroup, ...] = ()

    def __repr__(self) -> str:
        # Without OME groups, the same text as the generated dataclass repr.
        ome = f", ome={self.ome!r}" if self.ome else ""
        return f"{type(self).__qualname__}(arrays={self.arrays!r}{ome})"

    @property
    def images(self) -> tuple[OmeImage, ...]:
        return tuple(group for group in self.ome if isinstance(group, OmeImage))

    @property
    def is_ome(self) -> bool:
        return bool(self.ome)

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


Node = zarr.Group | zarr.Array[Any]


def _list_members(
    group: zarr.Group, max_depth: int | None
) -> tuple[dict[str, Node] | None, Exception | None]:
    """List the groups and arrays under a group.

    Uses the listing, or else the consolidated metadata. If neither works,
    returns ``None`` and the listing's exception.
    """
    try:
        return dict(group.members(max_depth=max_depth)), None
    except Exception as exc:
        consolidated = group.metadata.consolidated_metadata
        if consolidated is None:
            return None, exc
        return {
            path: group[path]
            for path in consolidated.flattened_metadata
            if max_depth is None or path.count("/") <= max_depth
        }, None


def _node_opener(
    group: zarr.Group, members: dict[str, Node] | None, complete: bool
) -> Callable[[str], Node | None]:
    """Return a function that opens the node at a path, or gives ``None``.

    Nodes already listed are reused. If the listing is ``complete``, a path
    that is not in it does not exist; otherwise it is opened by path.
    """
    cache: dict[str, Node | None] = dict(members or {})

    def open_node(path: str) -> Node | None:
        if path == "":
            return group
        if path not in cache:
            if complete:
                return None
            try:
                cache[path] = group[path]
            except (KeyError, FileNotFoundError):
                cache[path] = None
        return cache[path]

    return open_node


def _array_info(name: str, array: "zarr.Array[Any]") -> ArrayInfo:
    dims = dimension_names(array)
    return ArrayInfo(
        name=name,
        dimensions=dims,
        shape=tuple(array.shape),
        chunks=tuple(array.chunks),
        dtype=str(array.dtype),
        is_coordinate=dims is not None and dims == (name,),
    )


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

    OME-Zarr (OME-NGFF 0.4 and 0.5) groups are described in
    ``StoreDescription.ome``: images with their axes, resolution levels,
    transforms and channel labels, label images, labels groups, plates,
    wells and bioformats2raw containers. They are found from the root and
    by following the links in their metadata, so ``max_depth`` does not
    limit them; groups found only by listing are limited by it. Malformed
    OME metadata gives an ``OmeMetadataWarning`` and the group is left out.

    Some servers cannot list a group, for example plain HTTPS servers that
    answer a directory URL with an error. Then the group's consolidated
    metadata is used instead. If there is none, the arrays of an OME-Zarr
    store are found from its metadata, without a listing; for any other
    store, ``StoreOpenError`` says so.

    ``storage_options`` is passed to the fsspec backend, for example
    ``{"anon": True}`` to read a public S3 bucket without credentials.

    Raises:
        StoreOpenError: The store cannot be opened or listed, or the server
            or the connection fails.
        OmeMetadataError: The store cannot be listed and the OME-Zarr
            metadata needed to find any arrays is not valid: that of the
            root group, or the ``series`` list of a container. Problems
            further down give an ``OmeMetadataWarning``.
        ValueError: ``max_depth`` is negative.
    """
    if max_depth is not None and max_depth < 0:
        raise ValueError(f"max_depth must be None or >= 0, not {max_depth}")
    target = os.fspath(store)
    group = open_group(store, storage_options)
    members, list_error = _list_members(group, max_depth)
    listed = members is not None
    finder = OmeFinder(
        _node_opener(group, members, complete=listed and max_depth is None),
        strict=not listed,
    )
    call_wrapped(
        target,
        lambda: finder.run(
            path
            for path, node in (members or {}).items()
            if isinstance(node, zarr.Group)
        ),
    )

    arrays: dict[str, zarr.Array[Any]]
    if members is not None:
        arrays = {p: n for p, n in members.items() if isinstance(n, zarr.Array)}
    elif finder.arrays:
        arrays = {
            path: array
            for path, array in finder.arrays.items()
            if max_depth is None or path.count("/") <= max_depth
        }
    else:
        # From None: the listing error's text and traceback can show credentials.
        raise StoreOpenError(
            redact_urls(
                f"Cannot list the arrays in {target!r}: the store cannot be listed "
                f"({type(list_error).__name__}: {list_error}) and it has no "
                "consolidated metadata. extract() still works if you know the "
                "array's path."
            )
        ) from None
    infos = tuple(_array_info(name, arrays[name]) for name in sorted(arrays))
    return StoreDescription(arrays=infos, ome=finder.entries)
