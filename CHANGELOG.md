# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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

[Unreleased]: https://gitlab.com/leehart3000/zarrextract/-/compare/v0.3.0...main
[0.3.0]: https://gitlab.com/leehart3000/zarrextract/-/compare/v0.2.0...v0.3.0
[0.2.0]: https://gitlab.com/leehart3000/zarrextract/-/compare/v0.1.1...v0.2.0
[0.1.1]: https://gitlab.com/leehart3000/zarrextract/-/tags/v0.1.1
