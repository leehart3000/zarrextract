"""What an open store remembers between calls."""

from typing import TYPE_CHECKING, Any

import zarr

from zarrextract._probe import LazyAxis
from zarrextract._select import Axis

if TYPE_CHECKING:
    from zarrextract._describe import StoreDescription

Node = zarr.Group | zarr.Array[Any]

AxisEntry = tuple[Axis | LazyAxis, dict[str, str] | None]
"""A coordinate and the metadata (units and long name) of its Arrow field."""


class Cache:
    """The opened group and everything read from it so far.

    Every table is only ever added to, and holds values that are not changed
    afterwards, so a call that finds an entry can use it as it is. The data of
    the variables is not kept, so the size of the cache is set by the
    coordinates and the metadata of the store, not by the number of calls.
    """

    def __init__(self, group: zarr.Group) -> None:
        self.group = group
        self.nodes: dict[str, Node | None] = {}
        """Arrays and groups by path. ``None`` means there is nothing there."""
        self.axes: dict[tuple[str, int, bool], AxisEntry] = {}
        """Coordinates by dimension, length and whether they are searched lazily."""
        self.descriptions: dict[int | None, StoreDescription] = {}
        """Results of ``describe()`` by ``max_depth``."""

    def node(self, path: str) -> Node | None:
        """Return the array or group at ``path``, or ``None`` if there is none."""
        if path not in self.nodes:
            found: Node | None
            try:
                found = self.group[path]
            except KeyError:
                found = None
            return self.nodes.setdefault(path, found)
        return self.nodes[path]
