# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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
