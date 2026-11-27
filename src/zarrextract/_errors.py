"""Exceptions raised by zarrextract."""


class ZarrExtractError(Exception):
    """Base class for every error raised by zarrextract."""


class StoreOpenError(ZarrExtractError):
    """The store could not be opened as a Zarr group."""


class VariableNotFoundError(ZarrExtractError, KeyError):
    """The requested array does not exist in the store."""

    def __str__(self) -> str:
        # KeyError.__str__ would wrap the message in quotes.
        return Exception.__str__(self)


class MissingDimensionNamesError(ZarrExtractError):
    """An array has no dimension names, so it cannot be selected on."""


class SelectionError(ZarrExtractError):
    """A selection could not be applied."""


class UnknownDimensionError(SelectionError):
    """A selection names a dimension the variable does not have."""


class EmptySelectionError(SelectionError):
    """A selection matches no values."""


class AntimeridianError(SelectionError):
    """A longitude range wraps around the edge of the longitude axis.

    No longer raised: such ranges are read across the seam since 0.2. The class
    stays for code that catches it.
    """


class OmeMetadataError(ZarrExtractError):
    """OME-Zarr metadata is not valid, and the store cannot be listed.

    Raised by ``describe()`` only when the metadata needed to find any arrays
    is unusable, so there is nothing to describe: that of the root group, or
    the ``series`` list of a container. For a malformed or missing well, image
    in a well, labels group or label, and in any store that can be listed,
    ``OmeMetadataWarning`` is given instead.
    """


class OmeMetadataWarning(UserWarning):
    """The OME-Zarr metadata of a group is not valid.

    The message names the group and the problem. ``describe()`` leaves that
    group out of ``StoreDescription.ome``, and lists its arrays as usual.
    """


class WriteError(ZarrExtractError):
    """A table could not be written to a file by ``write()``."""


class UnsupportedFormatError(WriteError, ValueError):
    """The path has no suffix, an unknown suffix, or is not a local path."""


class OutputExistsError(WriteError, FileExistsError):
    """The file exists, and ``overwrite`` is not set."""


class TimeDecodingError(ZarrExtractError):
    """Time values or units could not be decoded."""


class UnsupportedCalendarError(TimeDecodingError):
    """The CF calendar is not supported."""
