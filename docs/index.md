# zarrextract

zarrextract extracts points, time series and bounding boxes from Zarr stores
into Apache Arrow tables. It is a lighter alternative to
`xarray.open_zarr(...).sel(...)` for the common case where you only want a
small piece of a large store, and it needs only `zarr`, `pyarrow` and `numpy`.

It requires Python 3.12 or newer.

## Install

```bash
pip install zarrextract
```

URLs need `fsspec` and the backend for their protocol. Two extras install the
common ones:

```bash
pip install "zarrextract[s3]"    # s3fs, for s3://
pip install "zarrextract[http]"  # fsspec and aiohttp, for http:// and https://
```

## Quick start

```python
import zarrextract as zx

description = zx.describe("era5.zarr")
print(description["t2m"].dimensions)  # ('time', 'lat', 'lon')

point = zx.extract("era5.zarr", "t2m", lat=51.5, lon=-0.1, time="2020-06-01")
series = zx.extract(
    "era5.zarr", "t2m", lat=51.5, lon=-0.1, time=("2020-01-01", "2020-12-31")
)
box = zx.extract("era5.zarr", "t2m", lat=(50, 52), lon=(-1, 1), time="2020-06-01")

import pyarrow.parquet as pq

pq.write_table(series, "series.parquet")
```

```{toctree}
:maxdepth: 2
:caption: Contents

guide
api
examples
changelog
```
