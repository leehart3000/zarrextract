<p align="center">
  <img src="https://gitlab.com/leehart3000/zarrextract/-/raw/main/docs/_static/brand/zarrextract-lockup.svg" alt="ZarrExtract" width="320">
</p>

ZarrExtract extracts points, time series and bounding boxes from Zarr stores
into Apache Arrow tables, and saves them to Parquet, CSV and Arrow files with
`zx.write()`. It is a lighter alternative to
`xarray.open_zarr(...).sel(...)` for the common case where you only want a
small piece of a large store, and it needs only `zarr`, `pyarrow` and `numpy`.

It requires Python 3.12 or newer. Read the
[documentation](https://zarrextract.readthedocs.io) for the full guide.

```bash
pip install zarrextract
```

## Usage

The examples assume a store with `time`, `lat` and `lon` coordinates and a
variable `t2m` with those dimensions.

### Describe a store

`describe()` lists the arrays in a store without reading any data.

```python
import zarrextract as zx

description = zx.describe("era5.zarr")
for array in description:
    print(array.name, array.dimensions, array.shape, array.dtype)

description["t2m"].chunks
description.coordinates  # the 1-D coordinate arrays
description.data_variables  # everything else
```

Arrays in nested groups are listed by their path, which is what `extract()`
takes. For example, an OME-Zarr image keeps its data in `0/0`:

```python
description = zx.describe("image.zarr")
print(description["0/0"].dimensions)  # ('t', 'c', 'z', 'y', 'x')
series = zx.extract("image.zarr", "0/0", c=0, z=0, y=100, x=100)

zx.describe("image.zarr", max_depth=0)  # top-level arrays only
```

OME-Zarr stores (OME-NGFF 0.4 and 0.5) are recognised. `description.ome`
holds the images, label images, plates, wells and bioformats2raw containers,
with the axes, resolution levels, transforms and channel labels of each
image:

```python
image = description.images[0]
print(image)  # a readable summary
level = image.levels[1]
level.name, level.scale  # ('0/1', (1.0, 1.0, 1.0, 2.0, 2.0))
```

Some servers cannot list a group. `describe()` then uses the consolidated
metadata. If there is none, it finds the arrays of an OME-Zarr store from its
metadata, and raises `StoreOpenError` for any other store. `extract()` still
works there if you know the array's path.

### Extract data

`zx.extract(store, variable, **selections)` returns a `pyarrow.Table` in long
format: one column per dimension of the variable, then a column for the
variable itself, with one row per value. Each keyword names a dimension:

- a single value picks the nearest coordinate;
- a `(start, stop)` pair picks every value from start to stop, inclusive;
- a dimension without a keyword is kept whole.

```python
import zarrextract as zx

# A point
point = zx.extract("era5.zarr", "t2m", lat=51.5, lon=-0.1, time="2020-06-01")

# A time series at one location
series = zx.extract(
    "era5.zarr", "t2m", lat=51.5, lon=-0.1, time=("2020-01-01", "2020-12-31")
)

# A bounding box at one time
box = zx.extract("era5.zarr", "t2m", lat=(50, 52), lon=(-1, 1), time="2020-06-01")
```

Coordinates are assumed to be sorted, ascending or descending. Long
coordinates are searched without being read whole, which gives wrong results
on unsorted values.

Longitudes are converted to the store's convention (0 to 360 or -180 to 180).
CF time coordinates become Arrow timestamps, `scale_factor` and `add_offset`
are applied, and `_FillValue` and `missing_value` become nulls. In Zarr v2
stores the fill value in the array metadata also becomes null, as in xarray.
Only the chunks that overlap the selection are read.

A longitude range may cross the edge of the store's longitude axis. On a store
that uses 0 to 360, `lon=(-10, 10)` gives the longitudes 350 to 359.75 and then
0 to 10, in that order, and reads only the chunks it needs:

```python
box = zx.extract("era5.zarr", "t2m", lat=(50, 52), lon=(-10, 10), time="2020-06-01")
```

### Many selections from one store

`zx.extract()` opens the store on every call. To make several selections from
the same store, open it once with `zx.open()`. It keeps the metadata and the
coordinates it has read, and gives the same tables:

```python
with zx.open("era5.zarr") as src:
    src.describe()
    london = src.extract("t2m", lat=51.5, lon=-0.1, time="2020-06-01")
    paris = src.extract("t2m", lat=48.9, lon=2.4, time="2020-06-01")
```

The store is assumed not to change while it is open. See the user guide for
what is kept, and `close()`.

### Try it in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leehart3000/zarrextract/blob/main/notebooks/mur_sst.ipynb) A demo notebook that reads points, time series and boxes of sea surface temperature from a public S3 store.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leehart3000/zarrextract/blob/main/notebooks/cmip6_tas.ipynb) A second notebook that reads a monthly climate model time series and a box across the prime meridian of a 0 to 360 longitude store (CMIP6, CC BY 4.0).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leehart3000/zarrextract/blob/main/notebooks/ome_zarr_idr0051.ipynb) A third notebook that reads a pixel time series and a 2D slice of a light-sheet OME-Zarr image over HTTPS, in a store that cannot be listed (IDR idr0051, Steventon et al., CC BY 4.0; [data](https://doi.org/10.17867/10000117), [paper](https://doi.org/10.1242/dev.166728), [Image Data Resource](https://doi.org/10.1038/nmeth.4326)).

### Save a table

Every table records its provenance as schema metadata: the store (without
credentials), the variable, the selections, the zarrextract version and the
time of extraction, and the units of each column. `zx.write()` saves it to a
local file, choosing the format from the suffix: `.parquet` (zstd), `.arrow` or
`.feather` (Arrow IPC), or `.csv`. Parquet and Arrow files keep the metadata.
CSV can't, so `series.csv` comes with `series.csv.json`.

```python
zx.write(series, "series.parquet")  # raises OutputExistsError if it exists

info = zx.provenance("series.parquet")
print(info.source, info.selections, info.extracted_at)
```

## Stores

Local paths work as they are. URLs need `fsspec` and the backend for their
protocol, for example `s3fs` for `s3://` or `gcsfs` for `gs://`.

```bash
pip install "zarrextract[s3]"    # s3fs, for s3://
pip install "zarrextract[http]"  # fsspec and aiohttp, for http:// and https://
```

```python
table = zx.extract("s3://bucket/era5.zarr", "t2m", lat=51.5, lon=-0.1)
```

The backend finds credentials in its usual way, for example from
environment variables or a configuration file. To pass options to the
backend, use the keyword-only `storage_options` argument of `describe()` and
`extract()`. For example, to read a public bucket without credentials:

```python
table = zx.extract(
    "s3://bucket/era5.zarr",
    "t2m",
    lat=51.5,
    lon=-0.1,
    storage_options={"anon": True},
)
```

`storage_options` is not a selection, so a dimension cannot be called
`storage_options`.

## Limitations

- Only the `standard`, `gregorian`, `proleptic_gregorian`, `noleap`,
  `365_day`, `360_day`, `all_leap` and `366_day` calendars are supported, and
  `standard` or `gregorian` dates before 1582-10-15 are not. Other calendars,
  such as `julian` and `none`, raise `UnsupportedCalendarError`. In `360_day`
  and `all_leap` the time column holds ISO 8601 strings, such as
  `"2000-02-30T00:00:00"`, because some of their dates are not timestamps.
- Coordinates are looked up at the top level of the store, so arrays in nested
  groups are selected by index unless the store has a top-level coordinate
  of the same name.
- Dask is not supported.

## Licence

BSD-3-Clause. See
[LICENSE](https://gitlab.com/leehart3000/zarrextract/-/blob/main/LICENSE).
