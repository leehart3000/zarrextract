"""A Zarr store that is opened once and used for many calls."""

import os
from collections.abc import Mapping
from typing import Any, NoReturn, Self

import pyarrow as pa
import zarr

from zarrextract._cache import Cache
from zarrextract._describe import (
    StoreDescription,
    check_max_depth,
    describe_from_cache,
)
from zarrextract._errors import SourceClosedError
from zarrextract._extract import extract_from_cache
from zarrextract._provenance import provenance_metadata, sanitize_store
from zarrextract._select import Selection, check_finite
from zarrextract._store import call_wrapped, open_group


class Source:
    """A Zarr store that has been opened, for making many selections from it.

    Create one with ``open()``. ``Source.extract()`` and ``Source.describe()``
    give the same results as ``extract()`` and ``describe()``, which open the
    store again on every call. A ``Source`` opens the store once and keeps what
    it has read:

    * the group and the metadata of the arrays it has looked at;
    * each coordinate that was read whole, already decoded;
    * the coordinate chunks that were read to search a long coordinate, and
      the values found in them;
    * the result of ``describe()``, for each ``max_depth``.

    The chunks of the variables are not kept, so the memory used is set by the
    size of the store's coordinates, not by how many calls are made. A
    coordinate with a hundred million values can take about 1.6 GB if every
    one of its chunks ends up being searched. There is no limit on the cache.

    The store is assumed not to change while the ``Source`` is open. If it does,
    a result can mix kept metadata and coordinates with new data. Open the store
    again to see changes.

    A ``Source`` can be used as a context manager, which closes it on exit, or
    on its own until ``close()`` is called. It has not been tested for use from
    several threads except on local and in-memory stores, and it is not
    guaranteed to be safe. It cannot be pickled: open the store again in the
    other process.

    A second ``describe()`` call with the same ``max_depth`` returns the first
    result, so an ``OmeMetadataWarning`` is given only the first time.
    """

    def __init__(self, cache: Cache, target: str) -> None:
        self._cache: Cache | None = cache
        # Without any user name, password, query or fragment.
        self._target = sanitize_store(target)

    @classmethod
    def _from_group(cls, group: zarr.Group, target: str) -> Self:
        """Wrap a group that is already open (for tests that count reads)."""
        return cls(Cache(group), target)

    @property
    def store(self) -> str:
        """The store, without any user name, password, query or fragment."""
        return self._target

    @property
    def closed(self) -> bool:
        """True after ``close()``."""
        return self._cache is None

    def __repr__(self) -> str:
        state = ", closed" if self.closed else ""
        return f"{type(self).__qualname__}({self._target!r}{state})"

    def __reduce__(self) -> NoReturn:
        raise TypeError(
            f"Cannot pickle {type(self).__qualname__}: open the store again in "
            "the other process."
        )

    def _live(self) -> Cache:
        cache = self._cache
        if cache is None:
            raise SourceClosedError(f"The source {self._target!r} is closed.")
        return cache

    def __enter__(self) -> Self:
        self._live()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Forget everything that was read, and make the source unusable.

        This drops the group and the cache, which frees their memory. It does
        not close the filesystem, which fsspec shares between users of the same
        store. Calling it again does nothing. Afterwards ``describe()`` and
        ``extract()`` raise ``SourceClosedError``. ``store``, ``closed`` and
        ``repr()`` still work, and tables that were returned earlier are not
        affected.
        """
        self._cache = None

    def describe(self, *, max_depth: int | None = None) -> StoreDescription:
        """List the arrays in the store without reading any data.

        The same as ``describe()`` for this store. The result is kept, so a
        second call with the same ``max_depth`` does not read the store, and
        the arrays it found are available to ``extract()`` without reading
        their metadata again.

        Raises:
            SourceClosedError: The source is closed.
            StoreOpenError: The store cannot be listed, or the server or the
                connection fails.
            OmeMetadataError: As for ``describe()``.
            ValueError: ``max_depth`` is negative.
        """
        cache = self._live()
        check_max_depth(max_depth)
        return describe_from_cache(cache, self._target, max_depth)

    def extract(self, variable: str, /, **selections: Selection) -> pa.Table:
        """Extract part of a variable as a long-format table.

        The same as ``extract()`` for this store, with the same selections and
        the same table, including its provenance metadata (``extracted_at`` is
        the time of this call). Coordinates and metadata that an earlier call
        has read are not read again.

        ``storage_options`` is not accepted here: pass it to ``open()``.

        Raises:
            SourceClosedError: The source is closed.
            TypeError: ``storage_options`` was given.

        It raises the same errors as ``extract()`` for the store, the variable
        and the selections.
        """
        if "storage_options" in selections:
            raise TypeError("pass storage_options to zx.open, not to extract")
        cache = self._live()
        check_finite(selections)

        def run() -> pa.Table:
            table = extract_from_cache(cache, variable, selections)
            return table.replace_schema_metadata(
                provenance_metadata(self._target, variable, selections)
            )

        return call_wrapped(self._target, run)


def open(
    store: str | os.PathLike[str],
    /,
    *,
    storage_options: Mapping[str, Any] | None = None,
) -> Source:
    """Open a Zarr store once, to make many selections from it.

    ``store`` is a local path or a URL string, and ``storage_options`` is passed
    to the fsspec backend, as for ``extract()``. The store is opened here, so a
    store that cannot be opened raises ``StoreOpenError`` now. Nothing else is
    read until it is used.

    ``zarrextract.open`` is not in ``zarrextract.__all__``, so that
    ``from zarrextract import *`` does not replace the built-in ``open``.

    ::

        with zx.open("s3://bucket/sst.zarr", storage_options={"anon": True}) as src:
            src.describe()
            src.extract("analysed_sst", lat=51.5, lon=-0.1, time="2019-06-01")

    Returns:
        A ``Source``. Close it with ``close()`` or by using it in a ``with``
        block.

    Raises:
        StoreOpenError: The store cannot be opened.
    """
    return Source(Cache(open_group(store, storage_options)), os.fspath(store))
