# zarrextract

zarrextract extracts points, time series and bounding boxes from Zarr stores
into Apache Arrow tables. It is a lighter alternative to
`xarray.open_zarr(...).sel(...)` for the common case where you only want a
small piece of a large store, and it needs only `zarr`, `pyarrow` and `numpy`.

It requires Python 3.12 or newer.

```bash
pip install zarrextract
```

## Usage

The examples assume a store with `time`, `lat` and `lon` coordinates and a
variable `t2m` with those dimensions.

### Describe a store

`describe()` lists the arrays in a store without reading any data.

```python
from zarrextract import describe

description = describe("era5.zarr")
for array in description:
    print(array.name, array.dimensions, array.shape, array.dtype)

description["t2m"].chunks
description.coordinates  # the 1-D coordinate arrays
description.data_variables  # everything else
```

### Extract data

`extract(store, variable, **selections)` returns a `pyarrow.Table` in long
format: one column per dimension of the variable, then a column for the
variable itself, with one row per value. Each keyword names a dimension:

- a single value picks the nearest coordinate;
- a `(start, stop)` pair picks every value from start to stop, inclusive;
- a dimension without a keyword is kept whole.

```python
from zarrextract import extract

# A point
point = extract("era5.zarr", "t2m", lat=51.5, lon=-0.1, time="2020-06-01")

# A time series at one location
series = extract(
    "era5.zarr", "t2m", lat=51.5, lon=-0.1, time=("2020-01-01", "2020-12-31")
)

# A bounding box at one time
box = extract("era5.zarr", "t2m", lat=(50, 52), lon=(-1, 1), time="2020-06-01")
```

Longitudes are converted to the store's convention (0 to 360 or -180 to 180).
CF time coordinates become Arrow timestamps, `scale_factor` and `add_offset`
are applied, and `_FillValue` and `missing_value` become nulls. Only the
chunks that overlap the selection are read.

### Write to Parquet

```python
import pyarrow.parquet as pq

pq.write_table(series, "series.parquet")
```

## Stores

Local paths work as they are. URLs need `fsspec` and the backend for their
protocol, for example `s3fs` for `s3://` or `gcsfs` for `gs://`.

```bash
pip install fsspec s3fs
```

```python
table = extract("s3://bucket/era5.zarr", "t2m", lat=51.5, lon=-0.1)
```

The backend finds credentials in its usual way, for example from
environment variables or a configuration file. Passing storage options to
the backend, such as anonymous access to a public bucket, is not supported
yet.

## Limitations

- A box that crosses the edge of the store's longitude axis raises
  `AntimeridianError`. This includes the prime meridian on a store that uses
  0 to 360. Request the two parts and join them with `pyarrow.concat_tables`:

  ```python
  import pyarrow as pa

  west = extract("era5.zarr", "t2m", lat=(50, 52), lon=(350, 359.75))
  east = extract("era5.zarr", "t2m", lat=(50, 52), lon=(0, 10))
  box = pa.concat_tables([west, east])
  ```

  This example is for a store that uses 0 to 360, where `lon=(-10, 10)`
  would raise the error.
- Only the `standard`, `gregorian` and `proleptic_gregorian` calendars are
  supported, and `standard` or `gregorian` dates before 1582-10-15 are not.
- Only arrays at the top level of the store are supported.
- Dask is not supported.

## Licence

BSD-3-Clause. See
[LICENSE](https://gitlab.com/leehart3000/zarrextract/-/blob/main/LICENSE).
