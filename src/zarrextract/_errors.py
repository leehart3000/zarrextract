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
    """A longitude range crosses the antimeridian, which is not supported yet."""


class TimeDecodingError(ZarrExtractError):
    """Time values or units could not be decoded."""


class UnsupportedCalendarError(TimeDecodingError):
    """The CF calendar is not supported."""
