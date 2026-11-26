# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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

[0.2.0]: https://gitlab.com/leehart3000/zarrextract/-/compare/v0.1.1...v0.2.0
[0.1.1]: https://gitlab.com/leehart3000/zarrextract/-/tags/v0.1.1
