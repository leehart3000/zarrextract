"""Search a long coordinate by reading only a few of its chunks."""

import math
from bisect import insort
from collections.abc import Callable, Sequence
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
    """A 1-D coordinate that is searched by reading a few chunks, not the whole.

    ``search()`` finds where a test on the values starts to be true. Each round
    it guesses the index from the values it already has, as for evenly spaced
    values, and reads the chunks around the guess. It also reads the middle
    chunk of what is still unknown, so that the unknown part at least halves
    each round and uneven values cost no more rounds than a bisection. The
    chunks of every end that is searched are read with one array read, so that
    zarr fetches them together.

    Chunks that have been read are kept. Values are numeric as in ``_select``:
    ``int64`` ticks for time, ``int64`` for integers and ``float64`` for floats.
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
        """The raw values of each chunk that was read, whole chunks only."""
        self._cached: list[int] = []
        """The numbers of the chunks in ``_chunks``, sorted."""
        self._values: dict[int, Any] = {}
        # The first and last two values are always needed to check the direction
        # and the steps, so they are read together, in one round.
        last = self.size - 1
        self._fetch(
            sorted({0, max(last - 1, 0) // self._chunk_size, last // self._chunk_size})
        )
        self.dtype: np.dtype[Any] = self._decode(self._chunks[0][:1]).dtype

    def _decode(self, raw: NDArray[Any]) -> NDArray[Any]:
        if self._encoding is None:
            return raw
        if self._encoding.clock is not None:
            return decode_calendar_ticks(raw, self._encoding)
        return decode_time_with(raw, self._encoding)

    def _span(self, k: int) -> tuple[int, int]:
        """Return the start and stop index of chunk ``k``."""
        start = k * self._chunk_size
        return start, min(start + self._chunk_size, self.size)

    def _fetch(self, chunks: Sequence[int]) -> None:
        """Read the chunks that are not kept yet with one array read, and keep them.

        This is the only place that reads the coordinate, and one call is one
        round trip. Every index of each wanted chunk is requested, so what is
        kept is always a whole chunk, and the last chunk may be short.
        """
        wanted = sorted({k for k in chunks if k not in self._chunks})
        if not wanted:
            return
        spans = [self._span(k) for k in wanted]
        if wanted[-1] - wanted[0] + 1 == len(wanted):
            block = np.asarray(self._node[spans[0][0] : spans[-1][1]])
        else:
            indices = np.concatenate([np.arange(start, stop) for start, stop in spans])
            block = np.asarray(self._node.get_orthogonal_selection(indices))
        offset = 0
        for k, (start, stop) in zip(wanted, spans, strict=True):
            self._chunks[k] = block[offset : offset + stop - start]
            offset += stop - start
            insort(self._cached, k)

    def _chunk(self, k: int) -> NDArray[Any]:
        self._fetch([k])
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
        self._fetch(range(first, last + 1))
        joined = np.concatenate([self._chunks[k] for k in range(first, last + 1)])
        raw = joined[selected.start - first * size :][: selected.stop - selected.start]
        return self._decode(raw)

    def _last(self, k: int) -> int:
        """Return the index of the last value of chunk ``k``."""
        return self._span(k)[1] - 1

    def _bracket(self, test: Callable[[int], bool]) -> tuple[int, int]:
        """Return ``(low, high)`` from the chunks that are kept, without reading.

        ``test`` is false and then true along the axis. ``low`` is the last
        index known to be false (-1 if none is) and ``high`` is the first index
        known to be true (``size`` if none is). The first true index is in
        ``low + 1..high``. Unless ``high == low + 1``, the indices between them
        are whole chunks that are not kept.
        """
        cached = self._cached
        first, stop = 0, len(cached)
        while first < stop:  # the first kept chunk that ends with a true value
            middle = (first + stop) // 2
            if test(self._last(cached[middle])):
                stop = middle
            else:
                first = middle + 1
        if first == len(cached):
            return self._last(cached[-1]), self.size
        begin, end = self._span(cached[first])
        found, end = begin, end - 1
        while found < end:
            middle = (found + end) // 2
            if test(middle):
                end = middle
            else:
                found = middle + 1
        if found == begin:
            return (self._last(cached[first - 1]) if first else -1), found
        return found - 1, found

    def _wanted(self, low: int, high: int, threshold: float) -> set[int]:
        """Return the chunks to read next to narrow the bracket ``low..high``."""
        size = self._chunk_size
        unknown_first, unknown_last = (low + 1) // size, (high - 1) // size
        # The middle one at least halves what is unknown, whatever the spacing.
        wanted = {(unknown_first + unknown_last) // 2}
        if low >= 0 and high < self.size:
            # Where the threshold falls between the two known values if the
            # values in between are evenly spaced, and the indices on both sides.
            below, above = float(self.value(low)), float(self.value(high))
            if above != below:
                guess = low + (threshold - below) / (above - below) * (high - low)
                if math.isfinite(guess):
                    guess = min(max(guess, low), high)
                    start = max(math.floor(guess) - 1, low + 1)
                    stop = min(math.ceil(guess) + 1, high - 1)
                    wanted |= {k for k in range(start // size, stop // size + 1)}
        return {k for k in wanted if unknown_first <= k <= unknown_last}

    def search(
        self, edges: Sequence[tuple[float, bool]], increasing: bool
    ) -> list[int]:
        """Return the first index where each edge's test is true, or ``size``.

        An edge is ``(threshold, strict)``. Its test is ``value >= threshold``,
        or ``value > threshold`` if strict, on an increasing axis, and ``<=``
        or ``<`` on a decreasing one. All edges are searched together, and the
        chunks every one of them needs in a round are read with one array read.
        """

        def make(threshold: float, strict: bool) -> Callable[[int], bool]:
            def test(i: int) -> bool:
                value = self.value(i)
                if increasing:
                    return bool(value > threshold if strict else value >= threshold)
                return bool(value < threshold if strict else value <= threshold)

            return test

        tests = [make(threshold, strict) for threshold, strict in edges]
        found: dict[int, int] = {}
        while len(found) < len(edges):
            wanted: set[int] = set()
            for n, (threshold, _) in enumerate(edges):
                if n not in found:
                    low, high = self._bracket(tests[n])
                    if high == low + 1:
                        found[n] = high
                    else:
                        wanted |= self._wanted(low, high, float(threshold))
            self._fetch(sorted(wanted))
        return [found[n] for n in range(len(edges))]
