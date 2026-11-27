# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Support for the CF calendars `360_day`, `all_leap` and its alias `366_day`.
  Some of their dates, such as 30 February, are not timestamps, so the time
  column of these calendars holds ISO 8601 strings of one fixed width, such as
  `"2000-02-30T00:00:00"` (with `.fff` or `.ffffff` for milliseconds or
  microseconds), and a missing time is a null. The field metadata of the
  column has `calendar`, and so has `Provenance.columns`. The other calendars
  keep their `timestamp` columns.
- Selections on these calendars are read in the store's own calendar:
  `("2000-02", "2000-02")` covers 30 days in `360_day`, and `"2000-02-30"` is
  valid there. `date`, `datetime` and `numpy.datetime64` selections are read by
  their fields. A date that does not exist in the calendar, such as
  `"2000-02-31"` in `360_day`, raises `SelectionError`, also as the end of a
  range. Bisection on a long coordinate works as for other calendars.
- The reference date of the time units is read in the calendar, so
  `days since 2000-02-30` is valid in `360_day`. Years from 0000 to 9999 are
  supported. Months and years as units are still not supported, and the error
  says why.

### Changed

- A long coordinate is now searched in fewer rounds of reads, where a round is
  one array read of the coordinate. Each round estimates where the value lies
  from the values already read and reads the chunks around the estimate,
  together with the middle of what is still unknown, so uneven, descending
  and gappy coordinates need no more rounds than the bisection used before.
  Both ends of a range are searched together, and the chunks of a round are
  read with one array read, so that zarr fetches them at the same time. On an
  evenly spaced time axis a range takes about 3 rounds the first time
  (including the middle of the range) and about 2 on an open handle, instead of
  about 22. The results, the error messages and the cached chunks do not
  change; the chunks are still kept whole.
- A selection such as `"2001-02-29"` on a time coordinate in any other
  calendar now says that the date is not valid and which month has no such
  day, instead of repeating NumPy's message.
- `UnsupportedCalendarError` for `julian`, `none` and other calendars now
  lists `360_day`, `all_leap` and `366_day` as supported.
- Documentation and notebooks: the Performance page says "about the same"
  where the two times are within 10% (case 4). The README, the documentation
  home page and the contributor guide mention saving tables with `zx.write()`.
  The benchmarks notebook no longer ends with an empty cell. The OME-Zarr
  notebook now uses `zx.open()` and `describe()` on the IDR store, reads the
  scale of each level from the `OmeImage`, and ends with a section that saves
  the time series to Parquet and reads its provenance back. The guide and the
  README describe the new calendars.

### Fixed

- On a handle from `zx.open()`, a time range that covers several chunks of a
  long coordinate was read again on every call, because the cache was used
  only when it held all of the chunks. The missing chunks are now read with
  one array read, so that they are fetched together, and kept. A repeated
  range reads no coordinate chunks, and an overlapping range reads only the
  new ones. The results do not change.

## [0.4.0] - 2026-11-27

### Added

- `extract()` attaches its provenance to the table as schema metadata, under
  keys that start with `zarrextract.`: the store, the variable, the selections
  as given (JSON), the zarrextract version, the time of extraction in UTC and
  a `provenance_version`. The `units` and `long_name` attributes of the
  variable and the coordinates become the metadata of their fields. The
  columns and values of the table do not change. `storage_options` is never
  recorded, and the user name, password, query string and fragment are
  removed from the store URL.
- `write(table, path, *, overwrite=False)` writes a table to a local file in
  the format of its suffix: `.parquet` (zstd), `.arrow` or `.feather` (Arrow
  IPC), or `.csv`. Parquet and Arrow files keep the metadata. A CSV file gets
  a `<name>.csv.json` file next to it with the metadata and the column types.
  It writes to temporary files and moves them into place, so a failed write
  never changes an existing file. It returns the paths written, and raises
  the new `UnsupportedFormatError`
  and `OutputExistsError`, which derive from the new `WriteError`.
- `provenance()` reads the provenance back from a table, or from a Parquet,
  Arrow or CSV file written by `write()`, as the new `Provenance` class.
- `describe()` recognises OME-Zarr stores (OME-NGFF 0.4 and 0.5). The new
  `StoreDescription.ome` holds the images, label images, labels groups,
  plates, wells and bioformats2raw containers it finds, as the new classes
  `OmeImage`, `OmeMultiscale`, `OmeLevel`, `OmeAxis`, `OmeLabels`, `OmePlate`,
  `OmeWell` and `OmeContainer`. An image reports its version, its axes, each
  resolution level with its shape, dtype, chunks, scale and translation, and
  its channel labels. Printing one gives a readable summary. Stores that are
  not OME-Zarr are described exactly as before.
- `describe()` on an OME-Zarr store that can't be listed and has no
  consolidated metadata, such as the IDR stores over HTTPS, now finds the
  arrays from the OME metadata instead of raising `StoreOpenError`.
- `OmeMetadataWarning` for malformed OME metadata, which leaves the group out
  of `StoreDescription.ome`, and `OmeMetadataError` when the store can't be
  listed and the metadata needed to find any arrays (the root group's, or a
  container's `series`) is unusable. Problems further down, such as a
  malformed well or label, give the warning and the rest is described.
- A Performance page in the docs, comparing extraction times with xarray, and
  the saved results of `notebooks/benchmarks.ipynb`.
- `open(store, *, storage_options=None)` opens a store once and returns a
  `Source`, for making many selections from it. `Source.extract()` and
  `Source.describe()` give the same results as the functions, including the
  provenance, but keep what they have read: the metadata of the arrays, the
  coordinates read whole (already decoded), the coordinate chunks read while
  searching a long coordinate by bisection, and the result of `describe()`.
  The chunks of the variables are not kept, and there is no limit on the
  cache. The store is assumed not to change while it is open. `Source` is a
  context manager, and `close()` drops everything it kept. `open` is not in
  `__all__`, so `from zarrextract import *` leaves the built-in `open` alone.
  A `Source` cannot be pickled, and thread safety is not promised.
- `SourceClosedError`, which derives from `ZarrExtractError` and `ValueError`,
  for a `Source` used after `close()`.

### Changed

- `extract()` and `describe()` now open a `Source` for the call. Their
  signatures, results and errors do not change.
- The Performance page has new results, including repeated selections on an
  open store with `zx.open()`.

### Fixed

- `extract()` raises `SelectionError` for a NaN selection, as a single value
  or as an end of a range, and for an infinite single value, before reading
  anything. A NaN selection used to return the first row silently. An
  infinite end of a range still leaves it open, and is recorded in the
  selections as the string `"inf"` or `"-inf"`.

### Security

- The messages of `StoreOpenError`, `OmeMetadataError` and
  `OmeMetadataWarning` no longer show the user name, password, query string or
  fragment of a URL, in the store URL or in the text of the underlying error.
  These errors are now raised without the original error as their `__cause__`
  or `__context__`, so a traceback doesn't show the raw URL either. This also
  hides the original traceback.

## [0.3.0] - 2026-11-26

### Added

- The ZarrExtract logo and favicon in the documentation and the README, and
  the brand colours in the documentation theme.
- Documentation built with Sphinx for Read the Docs, with a user guide, an API
  reference and the two example notebooks.
- A third demo notebook, `notebooks/ome_zarr_idr0051.ipynb`, that reads a
  pixel time series and a 2D slice of an OME-Zarr light-sheet image from the
  Image Data Resource over HTTPS.
- The `noleap` and `365_day` calendars (the same calendar), which most CMIP6
  models use, for example CESM2 with `days since 0001-01-01`. The times decode
  to ordinary `datetime64` dates, and selections use ordinary dates. A
  selection on 29 February picks the nearest value. A reference date on 29
  February raises `TimeDecodingError`.

### Changed

- In a time `(start, stop)` range, an ISO string given only to the year,
  month or day now covers that whole period, as in pandas and xarray. For
  example a stop of `"2019-06-30"` includes every time on that day, where it
  used to mean midnight and dropped values stamped later in the day. Strings
  with a time of day, `datetime` and `datetime64` values, and scalar
  selections, are unchanged.
- The API reference shows the public classes and exceptions as
  `zarrextract.X`, not by their private module paths.
- The `UnsupportedCalendarError` message now names the supported calendars.
  Other calendars, such as `360_day`, `all_leap` and `julian`, still raise it.

## [0.2.0]

### Added

- `storage_options` argument for `describe()` and `extract()`, and the `s3`
  and `http` extras that install the matching fsspec backends.
- Bisection on long coordinates, so a selection reads about one chunk per
  halving instead of the whole coordinate.
- Longitude ranges that cross the seam of the longitude axis, such as
  `lon=(-10, 10)` on a 0 to 360 store.
- `describe()` lists arrays in nested groups by their path, such as `0/0`,
  and has a `max_depth` argument.
- `describe()` works on stores that cannot be listed, such as some HTTPS
  servers, if they have consolidated metadata. Otherwise it raises
  `StoreOpenError`. HTTP client and fsspec errors from remote stores are now
  raised as `StoreOpenError`.

### Fixed

- The fill value in Zarr v2 array metadata now becomes null, as in xarray.

## [0.1.1]

First release.

[Unreleased]: https://gitlab.com/leehart3000/zarrextract/-/compare/v0.4.0...main
[0.4.0]: https://gitlab.com/leehart3000/zarrextract/-/compare/v0.3.0...v0.4.0
[0.3.0]: https://gitlab.com/leehart3000/zarrextract/-/compare/v0.2.0...v0.3.0
[0.2.0]: https://gitlab.com/leehart3000/zarrextract/-/compare/v0.1.1...v0.2.0
[0.1.1]: https://gitlab.com/leehart3000/zarrextract/-/tags/v0.1.1
