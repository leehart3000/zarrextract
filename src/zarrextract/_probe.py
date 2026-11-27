"""Read single values of a long coordinate, one chunk at a time."""

from typing import Any

import numpy as np
import zarr
from numpy.typing import NDArray

from zarrextract._calendar import MISSING, Clock
from zarrextract._cf import decode_calendar_ticks, decode_time_with, parse_time_encoding
from zarrextract._errors import SelectionError


def not_monotonic_error(name: str) -> SelectionError:
    return SelectionError(
        f"Cannot select on dimension {name!r}: its coordinate values are not "
        "strictly increasing or decreasing."
    )


class LazyAxis:
    """A 1-D coordinate that is read by probing values instead of whole.

    Chunks that have been read are kept, so probing values in the same chunk
    costs one read. Values are numeric as in ``_select``: ``int64`` ticks for
    time, ``int64`` for integers and ``float64`` for floats.
    """

    def __init__(
        self,
        node: "zarr.Array[Any]",
        name: str,
        *,
        units: str | None,
        calendar: object,
        is_time: bool,
        is_longitude: bool,
    ) -> None:
        self.name = name
        self.is_time = is_time
        self.is_longitude = is_longitude
        self.size = node.shape[0]
        self._node = node
        # Parsed once here, not on every probe of a value.
        self._encoding = None if units is None else parse_time_encoding(units, calendar)
        self.clock: Clock | None = (
            None if self._encoding is None else self._encoding.clock
        )
        """Set for 360_day and all_leap, whose values are ticks, not datetime64."""
        self._chunk_size = node.chunks[0]
        self._chunks: dict[int, NDArray[Any]] = {}
        self._values: dict[int, Any] = {}
        self.dtype: np.dtype[Any] = self._decode(self._chunk(0)[:1]).dtype

    def _decode(self, raw: NDArray[Any]) -> NDArray[Any]:
        if self._encoding is None:
            return raw
        if self._encoding.clock is not None:
            return decode_calendar_ticks(raw, self._encoding)
        return decode_time_with(raw, self._encoding)

    def _chunk(self, k: int) -> NDArray[Any]:
        if k not in self._chunks:
            start = k * self._chunk_size
            stop = min(start + self._chunk_size, self.size)
            self._chunks[k] = np.asarray(self._node[start:stop])
        return self._chunks[k]

    def value(self, i: int) -> Any:
        """Return the numeric value at index ``i``."""
        if i not in self._values:
            k, offset = divmod(i, self._chunk_size)
            decoded = self._decode(self._chunk(k)[offset : offset + 1])[0]
            number: Any
            if self.is_time:
                missing = decoded == MISSING if self.clock else np.isnat(decoded)
                if missing:
                    raise SelectionError(f"Time dimension {self.name!r} contains NaT.")
                number = decoded.astype(np.int64)
            elif decoded.dtype.kind == "f":
                number = np.float64(decoded)
                if np.isnan(number):
                    raise not_monotonic_error(self.name)
            else:
                number = np.int64(decoded)
            self._values[i] = number
        return self._values[i]

    def read(self, selected: slice) -> NDArray[Any]:
        """Return the decoded values in a slice.

        The chunks the slice covers that have not been read yet are read with
        one array read, so that zarr fetches them together, and are kept.
        """
        size = self._chunk_size
        first, last = selected.start // size, (selected.stop - 1) // size
        missing = [k for k in range(first, last + 1) if k not in self._chunks]
        if missing:
            # One read from the first to the last missing chunk. Chunks in
            # between that are already cached are read again, which is rare
            # because ranges are contiguous and cached chunks lie at the ends.
            start = missing[0] * size
            stop = min((missing[-1] + 1) * size, self.size)
            block = np.asarray(self._node[start:stop])
            for k in range(missing[0], missing[-1] + 1):
                if k not in self._chunks:
                    offset = k * size - start
                    self._chunks[k] = block[offset : offset + size]
        joined = np.concatenate([self._chunks[k] for k in range(first, last + 1)])
        raw = joined[selected.start - first * size :][: selected.stop - selected.start]
        return self._decode(raw)
