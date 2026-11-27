"""Extract points, time series and bounding boxes from Zarr stores."""

from importlib.metadata import version

from zarrextract._describe import ArrayInfo, StoreDescription, describe
from zarrextract._errors import (
    AntimeridianError,
    EmptySelectionError,
    MissingDimensionNamesError,
    OmeMetadataError,
    OmeMetadataWarning,
    SelectionError,
    StoreOpenError,
    TimeDecodingError,
    UnknownDimensionError,
    UnsupportedCalendarError,
    VariableNotFoundError,
    ZarrExtractError,
)
from zarrextract._extract import extract
from zarrextract._ome import (
    OmeAxis,
    OmeContainer,
    OmeImage,
    OmeLabels,
    OmeLevel,
    OmeMultiscale,
    OmePlate,
    OmeWell,
)

__version__ = version("zarrextract")

__all__ = [
    "AntimeridianError",
    "ArrayInfo",
    "EmptySelectionError",
    "MissingDimensionNamesError",
    "OmeAxis",
    "OmeContainer",
    "OmeImage",
    "OmeLabels",
    "OmeLevel",
    "OmeMetadataError",
    "OmeMetadataWarning",
    "OmeMultiscale",
    "OmePlate",
    "OmeWell",
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
    "extract",
]

# Show the public names as zarrextract.X, not by their private module paths.
for _name in __all__:
    if _name != "__version__":
        globals()[_name].__module__ = __name__
del _name
