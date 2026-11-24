"""Extract points, time series and bounding boxes from Zarr stores."""

from importlib.metadata import version

from zarrextract._describe import ArrayInfo, StoreDescription, describe
from zarrextract._errors import (
    AntimeridianError,
    EmptySelectionError,
    MissingDimensionNamesError,
    SelectionError,
    StoreOpenError,
    TimeDecodingError,
    UnknownDimensionError,
    UnsupportedCalendarError,
    VariableNotFoundError,
    ZarrExtractError,
)

__version__ = version("zarrextract")

__all__ = [
    "AntimeridianError",
    "ArrayInfo",
    "EmptySelectionError",
    "MissingDimensionNamesError",
    "SelectionError",
    "StoreDescription",
    "StoreOpenError",
    "TimeDecodingError",
    "UnknownDimensionError",
    "UnsupportedCalendarError",
    "VariableNotFoundError",
    "ZarrExtractError",
    "__version__",
    "describe",
]
