"""Extract points, time series and bounding boxes from Zarr stores."""

from importlib.metadata import version

from zarrextract._describe import ArrayInfo, StoreDescription, describe
from zarrextract._errors import (
    AntimeridianError,
    EmptySelectionError,
    MissingDimensionNamesError,
    OmeMetadataError,
    OmeMetadataWarning,
    OutputExistsError,
    SelectionError,
    StoreOpenError,
    TimeDecodingError,
    UnknownDimensionError,
    UnsupportedCalendarError,
    UnsupportedFormatError,
    VariableNotFoundError,
    WriteError,
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
from zarrextract._provenance import Provenance, provenance
from zarrextract._write import write

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
    "OutputExistsError",
    "Provenance",
    "SelectionError",
    "StoreDescription",
    "StoreOpenError",
    "TimeDecodingError",
    "UnknownDimensionError",
    "UnsupportedCalendarError",
    "UnsupportedFormatError",
    "VariableNotFoundError",
    "WriteError",
    "ZarrExtractError",
    "__version__",
    "describe",
    "extract",
    "provenance",
    "write",
]

# Show the public names as zarrextract.X, not by their private module paths.
for _name in __all__:
    if _name != "__version__":
        globals()[_name].__module__ = __name__
del _name
