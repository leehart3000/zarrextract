"""Opening Zarr stores and reading array metadata."""

import os
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import zarr
import zarr.errors

from zarrextract._errors import StoreOpenError


def is_client_error(exc: BaseException) -> bool:
    """Say whether ``exc`` is an HTTP client or fsspec error from a remote store.

    aiohttp is an optional dependency, so it is only consulted if something
    has already imported it, as any remote store using it will have done.
    """
    aiohttp = sys.modules.get("aiohttp")
    if aiohttp is not None and isinstance(exc, aiohttp.ClientError):
        return True
    fsspec_exceptions = sys.modules.get("fsspec.exceptions")
    if fsspec_exceptions is not None:
        return isinstance(
            exc,
            fsspec_exceptions.FSTimeoutError | fsspec_exceptions.BlocksizeMismatchError,
        )
    return False


@contextmanager
def wrap_client_errors(target: str) -> Iterator[None]:
    """Raise ``StoreOpenError`` for HTTP client and fsspec errors."""
    try:
        yield
    except Exception as exc:
        if is_client_error(exc):
            raise StoreOpenError(
                f"Cannot read {target!r}: the server or the connection failed "
                f"({type(exc).__name__}: {exc})."
            ) from exc
        raise


def open_group(
    store: str | os.PathLike[str],
    storage_options: Mapping[str, Any] | None = None,
) -> zarr.Group:
    """Open a store read-only, using consolidated metadata when present.

    ``store`` is a local path or a URL. URLs need the matching fsspec backend.
    ``storage_options`` is passed to ``zarr.open_group``, which hands it to
    the fsspec backend.
    """
    target = os.fspath(store)
    try:
        with wrap_client_errors(target):
            return zarr.open_group(
                target,
                mode="r",
                use_consolidated=None,
                storage_options=None
                if storage_options is None
                else dict(storage_options),
            )
    except ImportError as exc:
        raise StoreOpenError(
            f"Cannot open {target!r}: a required package is missing ({exc}). "
            "URLs need fsspec and the backend for their protocol, for example "
            "s3fs for s3:// or gcsfs for gs://."
        ) from exc
    except (OSError, ValueError, zarr.errors.BaseZarrError) as exc:
        raise StoreOpenError(f"Cannot open {target!r} as a Zarr group: {exc}") from exc


def dimension_names(array: zarr.Array[Any]) -> tuple[str, ...] | None:
    """Return the dimension names of an array, or ``None`` if it has none.

    Zarr v3 stores them in ``dimension_names``. Zarr v2 stores them in the
    ``_ARRAY_DIMENSIONS`` attribute.
    """
    if array.ndim == 0:
        # Zarr v3 stores no names for a 0-d array, but it trivially has none to name.
        return ()
    metadata = array.metadata
    names: object
    if metadata.zarr_format == 3:
        names = metadata.dimension_names
    elif metadata.zarr_format == 2:
        names = array.attrs.get("_ARRAY_DIMENSIONS")
    else:  # pragma: no cover - only reachable if zarr adds a new format
        return None
    if not isinstance(names, list | tuple):
        return None
    if len(names) != array.ndim or not all(isinstance(n, str) for n in names):
        return None
    return tuple(str(n) for n in names)
