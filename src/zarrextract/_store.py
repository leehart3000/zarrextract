"""Opening Zarr stores and reading array metadata."""

import os
from typing import Any

import zarr
import zarr.errors
from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata

from zarrextract._errors import StoreOpenError


def open_group(store: str | os.PathLike[str]) -> zarr.Group:
    """Open a store read-only, using consolidated metadata when present.

    ``store`` is a local path or a URL. URLs need the matching fsspec backend.
    """
    target = os.fspath(store)
    try:
        return zarr.open_group(target, mode="r", use_consolidated=None)
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
    metadata = array.metadata
    names: object
    if isinstance(metadata, ArrayV3Metadata):
        names = metadata.dimension_names
    elif isinstance(metadata, ArrayV2Metadata):
        names = array.attrs.get("_ARRAY_DIMENSIONS")
    else:  # pragma: no cover - only reachable if zarr adds a new format
        return None
    if not isinstance(names, list | tuple):
        return None
    if len(names) != array.ndim or not all(isinstance(n, str) for n in names):
        return None
    return tuple(str(n) for n in names)
