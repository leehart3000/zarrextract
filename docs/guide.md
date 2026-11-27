# User guide

The examples assume a store with `time`, `lat` and `lon` coordinates and a
variable `t2m` with those dimensions.

`zx.extract(store, variable, **selections)` returns a `pyarrow.Table` in long
format: one column per dimension of the variable, then a column for the
variable itself, with one row per value, in C order.

## Selections

Each keyword names a dimension of the variable and selects along it.

- A single value picks the **nearest** coordinate value, so the dimension has
  length 1.
- A `(start, stop)` pair picks every value from start to stop, **inclusive**.
  The two ends can be given in either order, except for longitudes (see
  {ref}`longitudes`), where `start > stop` after conversion means the range
  crosses the seam.
- A dimension without a keyword is kept whole.

```python
point = zx.extract("era5.zarr", "t2m", lat=51.5, lon=-0.1, time="2020-06-01")
box = zx.extract("era5.zarr", "t2m", lat=(52, 50), lon=(-1, 1))  # lat in either order
```

A single value more than one grid step outside the coordinates is an error
(`SelectionError`), so a typing mistake such as `lat=510` does not quietly
return the edge of the grid. A range that matches no values raises
`EmptySelectionError`. Naming a dimension the variable does not have raises
`UnknownDimensionError`.

`storage_options` is not a selection, so a dimension cannot be called
`storage_options`.

## Coordinates

A coordinate is a 1-D array in the top level of the store that is named after
its dimension and has the dimension's length.

- Coordinates must be sorted, ascending or descending. Long coordinates are
  searched by bisection, which gives wrong results on unsorted values.
  Descending coordinates, such as a latitude running from 90 to -90, work the
  same way as ascending ones.
- A dimension with no coordinate uses index positions `0, 1, 2, ...`, so
  `y=(10, 20)` picks the rows 10 to 20.
- Coordinates are looked up at the top level of the store, so arrays in nested
  groups are selected by index unless the store has a top-level coordinate of
  the same name.
- Arrays in nested groups are named by their path, which is what `extract()`
  takes. In an OME-Zarr image the data are in `0/0` (see {ref}`ome-zarr`):

```python
description = zx.describe("image.zarr")
print(description["0/0"].dimensions)  # ('t', 'c', 'z', 'y', 'x')
series = zx.extract("image.zarr", "0/0", c=0, z=0, y=100, x=100)

zx.describe("image.zarr", max_depth=0)  # top-level arrays only
```

(ome-zarr)=
## OME-Zarr images

`describe()` recognises OME-Zarr (OME-NGFF) stores, version 0.4 (Zarr v2,
metadata in `.zattrs`) and version 0.5 (Zarr v3, metadata under the `ome` key
of `zarr.json`). The groups it finds are in `description.ome`, sorted by path:

- `OmeImage`: an image, with its version, its `multiscales` entries (each
  with its name, axes and resolution levels), and the channel labels from
  `omero` metadata. A label image is an `OmeImage` with `is_label` set.
- `OmeLabels`: the `labels` group beside an image's resolution levels, which
  lists its label images.
- `OmePlate` and `OmeWell`: a plate with its rows, columns and well paths,
  and a well with its image paths.
- `OmeContainer`: a bioformats2raw container, a group that holds images
  `0`, `1`, ... (as in the stores of the Image Data Resource).

A store that is not OME-Zarr has `description.ome == ()` and is described
exactly as before. Printing an OME class gives a readable summary:

```python
description = zx.describe("image.zarr")
image = description.images[0]
print(image)
```

```text
OME-Zarr image '0' (version 0.5)
  multiscales 1 of 1 (no name)
    axes: t (time), c (channel), z (space), y (space), x (space)
    levels:
      0/0: shape (79, 1, 201, 333, 333), uint16, chunks (1, 1, 1, 333, 333)
        scale (1.0, 1.0, 1.0, 1.0, 1.0)
      0/1: shape (79, 1, 201, 166, 166), uint16, chunks (1, 1, 1, 166, 166)
        scale (1.0, 1.0, 1.0, 2.0, 2.0)
  channels: (none in the metadata)
  labels: (none)
```

Each level's `name` is the array path that `extract()` takes:

```python
level = image.levels[1]  # half resolution
level.scale  # (1.0, 1.0, 1.0, 2.0, 2.0): one value per axis
plane = zx.extract("image.zarr", level.name, t=0, c=0, z=100)
```

**Transforms** are reported as the store gives them. Each level has its
`scale` and, if present, its `translation`. A `multiscales` entry can have its
own `scale` and `translation`, which apply after the level's. They are kept
on the entry, not combined into the levels. Axis types and units are `None`
when the metadata leaves them out.

**Versions.** A v0.4 group may leave out its version, and then `version` is
`None`. A v0.5 group without `ome.version` takes the version of the nearest
group above it that has one. For a version other than 0.4 or 0.5, only the
dataset paths are read: the image has `supported=False` and no axes,
transforms or channel labels.

**Finding the images.** OME groups are found from the root of the store and
by following the paths in their metadata: a container's images, a plate's
wells, a well's images, an image's `labels` group and its label images.
`max_depth` does not limit these. Other OME groups are found by listing, and
`max_depth` limits those as it limits the arrays.

**Stores that can't be listed.** An OME-Zarr store does not need a listing:
`describe()` finds the arrays from the `multiscales` dataset paths. This
reads only metadata, but costs one request per group and array it opens: for
a container, one per image plus one for the first missing number, and for a
plate, one per well and image.

**Malformed metadata.** If a group's OME metadata is not valid, `describe()`
gives an `OmeMetadataWarning` that names the group and the problem, leaves
that group out of `description.ome`, and lists its arrays as usual. Only when
the store can't be listed and the metadata needed to find any arrays is
unusable, so that there is nothing to describe, it raises `OmeMetadataError`
instead. That is the case for the root group's metadata and for the `series`
list of a container. A malformed or missing well, image in a well, labels
group or label still gives a warning, and the rest is described. An image
without a `labels` group is normal and gives no warning.

(longitudes)=
## Longitudes

A coordinate is a longitude if its `standard_name` is `longitude`, its units
are degrees east, or it is called `lon` or `longitude`.

**Conventions.** Longitudes you pass in are converted to the store's
convention, which is 0 to 360 or -180 to 180. On a 0 to 360 store,
`lon=-10` is read as 350.

**The seam.** A range may cross the edge of the store's longitude axis. On a
store that uses 0 to 360, `lon=(-10, 10)` gives the longitudes 350 to 359.75
and then 0 to 10, in that order, and reads only the chunks it needs:

```python
box = zx.extract("era5.zarr", "t2m", lat=(50, 52), lon=(-10, 10), time="2020-06-01")
```

**The global wrap.** On a global grid, the nearest value to a single longitude
may be across the seam. For example, on a grid that runs from 0 to 359.75 in
steps of 0.25, `lon=359.9` and `lon=-0.1` both give 0, which is nearer than
359.75. On a grid that does not cover the globe, a longitude more than one
grid step outside the edge is an error, as for any other coordinate.

## Time

CF time coordinates (`"<unit> since <date>"`) become Arrow timestamps. Time
selections can be ISO strings, `datetime` objects or `numpy.datetime64`
values.

- **Partial dates:** in a `(start, stop)` range, an ISO string given only to
  the year, month or day covers that whole period, as in pandas and xarray.
  `("2019-06", "2019-06")` is all of June, and a stop of `"2019-06-30"`
  includes every time on that day, even if the values are stamped at 09:00. A
  start of `"2019-06"` means the start of June. Strings with a time of day,
  `datetime` and `datetime64` values mean exactly that moment, and so does a
  scalar, which picks the nearest value.
- **Units:** days, hours, minutes, seconds, milliseconds and microseconds,
  and their usual abbreviations. The reference date must be in UTC.
- **Calendars:** `standard`, `gregorian`, `proleptic_gregorian`, `noleap` and
  `365_day`. Dates in the `standard` and `gregorian` calendars before
  1582-10-15 are not supported. Other calendars, such as `360_day`, `all_leap`
  and `julian`, raise `UnsupportedCalendarError`. In the `noleap` calendar the
  times decode to ordinary dates, so a selection on 29 February picks the
  nearest value.
- **Resolution:** the timestamp unit follows the time units: `s` for days,
  hours, minutes and seconds, `ms` for milliseconds and `us` for microseconds.
  Far-future dates, such as the year 2300 in CMIP6 scenarios, therefore fit.

## Missing values

- `scale_factor` and `add_offset` are applied, and the values become floats.
- `_FillValue` and `missing_value` become nulls. They are compared with the
  stored values, before scaling.
- In Zarr v2 stores, the `fill_value` in the array metadata also becomes null,
  as in xarray. In Zarr v3 the fill value is always set (often 0), so only
  the attributes count.

## Remote stores

Local paths work as they are. URLs need `fsspec` and the backend for their
protocol, for example `s3fs` for `s3://` or `gcsfs` for `gs://`.

```python
table = zx.extract("s3://bucket/era5.zarr", "t2m", lat=51.5, lon=-0.1)
```

**Storage options.** The backend finds credentials in its usual way, for
example from environment variables or a configuration file. To pass options to
the backend, use the keyword-only `storage_options` argument of `describe()`
and `extract()`.

**Anonymous buckets.** To read a public bucket without credentials, pass
`{"anon": True}`:

```python
table = zx.extract(
    "s3://bucket/era5.zarr",
    "t2m",
    lat=51.5,
    lon=-0.1,
    storage_options={"anon": True},
)
```

**Stores that can't be listed.** Some servers cannot list a group, for example
plain HTTPS servers that answer a directory URL with an error. `describe()`
then uses the consolidated metadata. If there is none, it finds the arrays of
an OME-Zarr store from its metadata (see {ref}`ome-zarr`), and raises
`StoreOpenError` for any other store. `extract()` still works if you know the
array's path.

## Performance

- Only the chunks that overlap the selection are read.
- A selected coordinate with more than 8 chunks is searched by bisection,
  which reads about one chunk per halving instead of the whole coordinate.
  Coordinates you do not select on are read whole, because the table needs
  their values.
- Dask is not supported.
