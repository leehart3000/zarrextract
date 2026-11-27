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

`NaN` is never a valid selection, and neither is an infinite single value, as
it has no nearest coordinate: both raise `SelectionError` before anything is
read. An infinite end of a range leaves it open, so `lat=(float("-inf"), 10)`
picks every latitude up to 10.

`storage_options` is not a selection, so a dimension cannot be called
`storage_options`.

## Coordinates

A coordinate is a 1-D array in the top level of the store that is named after
its dimension and has the dimension's length.

- Coordinates must be sorted, ascending or descending. Long coordinates are
  searched without being read whole, which gives wrong results on unsorted
  values.
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
- **Calendars:** `standard`, `gregorian`, `proleptic_gregorian`, `noleap`,
  `365_day`, `360_day`, `all_leap` and `366_day`. Dates in the `standard` and
  `gregorian` calendars before 1582-10-15 are not supported. Other calendars,
  such as `julian` and `none`, raise `UnsupportedCalendarError`. In the
  `noleap` calendar the times decode to ordinary dates, so a selection on 29
  February picks the nearest value. The `360_day` and `all_leap` calendars
  have dates that are not timestamps, so their time column is different: see
  [below](#day-and-all-leap).
- **Resolution:** the timestamp unit follows the time units: `s` for days,
  hours, minutes and seconds, `ms` for milliseconds and `us` for microseconds.
  Far-future dates, such as the year 2300 in CMIP6 scenarios, therefore fit.

### 360_day and all_leap

The `360_day` calendar has 12 months of 30 days, so it has 30 February. The
`all_leap` calendar (also called `366_day`) has 29 February in every year.
Neither can be held by a timestamp, which is why the Met Office models in
CMIP6 (HadGEM3, UKESM1) could not be read before. For these calendars:

- **The time column holds ISO 8601 strings** of one fixed width, such as
  `"2000-02-30T00:00:00"`. A time in milliseconds or microseconds ends in
  `.fff` or `.ffffff`. The width never changes, so the strings sort in the
  order of the times. The field metadata of the column has `calendar`, such as
  `"360_day"`, because the strings do not say it, and `Provenance.columns`
  has it too. A time that is missing in the store is a null.
- **Selections are read in the store's own calendar.** `("2000-02",
  "2000-02")` is all 30 days of February in `360_day` and 29 days in
  `all_leap`, `("2000", "2000")` is the 360 or 366 days of the year, and
  `"2000-02-30"` is a valid date in `360_day`. A scalar picks the nearest
  value, as in any calendar.
- **`date`, `datetime` and `numpy.datetime64`** are read by their year, month,
  day and time of day, as a label in the store's calendar: `date(2000, 6, 15)`
  is 15 June in `360_day`. An aware `datetime` is converted to UTC first.
- **A date that does not exist is an error**, never the nearest value:
  `"2000-02-31"` or `date(2000, 1, 31)` in `360_day`, and `"2001-04-31"` in
  `all_leap`, raise `SelectionError`. This includes the end of a range, so
  `("2000-01-01", "2000-12-31")` fails in `360_day`. The message suggests
  `"2000-12-30"`, or `"2000-12"` for the whole month.
- **Units** are days, hours, minutes, seconds, milliseconds and microseconds,
  and the reference date is read in the calendar, so `days since 2000-02-30`
  is valid in `360_day`. Months and years are not supported: in CF a month is
  a fixed fraction of a year, not a calendar month, even in `360_day`.
- **Years** must be from 0000 to 9999, so that every string has four digits
  for the year.

Parquet and Arrow files keep the strings and the `calendar` metadata. A CSV
file has the strings, and its `.csv.json` file has the column type `string`
and the calendar. Some CSV readers turn a column of ISO strings into
timestamps when all of the dates happen to exist in the Gregorian calendar.
Read the column as a string to avoid that, for example with
`pyarrow.csv.ConvertOptions(column_types={"time": pyarrow.string()})`. To do
arithmetic on the dates, convert them with cftime, for example
`cftime.datetime(2000, 2, 30, calendar="360_day")`.

## Missing values

- `scale_factor` and `add_offset` are applied, and the values become floats.
- `_FillValue` and `missing_value` become nulls. They are compared with the
  stored values, before scaling.
- In Zarr v2 stores, the `fill_value` in the array metadata also becomes null,
  as in xarray. In Zarr v3 the fill value is always set (often 0), so only
  the attributes count.

## Provenance and saving

Every table from `extract()` records where it came from, as schema metadata
under keys that start with `zarrextract.`:

- `zarrextract.provenance_version`: the layout of this metadata, currently `1`.
- `zarrextract.source`: the store, as given, without credentials (see below).
- `zarrextract.variable`: the variable.
- `zarrextract.selections`: the selections as given, as JSON. Dates are ISO
  strings and a `(start, stop)` pair is a list. Values are not converted, so
  `lon=-10` stays `-10`. An infinite end of a range is written as the string
  `"inf"` or `"-inf"`.
- `zarrextract.version`: the zarrextract version.
- `zarrextract.extracted_at`: the time of extraction, in UTC, to the second.

The `units` and `long_name` attributes of the variable and of the coordinates
are the metadata of their columns (the Arrow fields). A decoded time column
has no `units`, because `days since ...` describes the stored numbers, not the
timestamps. The columns and values of the table are the same as without the
metadata, and comparing tables with `Table.equals()` ignores it unless you pass
`check_metadata=True`.

`storage_options` is never recorded. The user name, password, query string and
fragment are removed from the store URL, so `https://me:pw@host/s.zarr?token=x`
is recorded as `https://host/s.zarr`. A secret that is part of the URL path
can't be recognised, and a local path is recorded as you gave it.

The same rule applies to the errors and warnings of zarrextract: a URL in a
message has its credentials removed. Such an error is raised without the
original error as its cause or context, so a traceback can't show them either.
That also hides the original traceback, which makes some failures harder to
debug.

### Saving a table

`zx.write(table, path, overwrite=False)` writes to a local file. The suffix
selects the format, whatever its case:

| Suffix | Format |
|---|---|
| `.parquet` | Parquet, compressed with zstd |
| `.arrow`, `.feather` | The Arrow IPC file format (Feather version 2), uncompressed |
| `.csv` | CSV, with a metadata file next to it |

```python
series = zx.extract(
    "era5.zarr", "t2m", lat=51.5, lon=-0.1, time=("2020-01-01", "2020-12-31")
)
zx.write(series, "series.parquet")
```

`write()` returns the paths it wrote as a tuple. It raises
`UnsupportedFormatError` for a URL, a missing suffix or an unknown one, and
`OutputExistsError` if the file exists, unless you pass `overwrite=True`. It
writes to temporary files in the same directory and then moves them into
place, so if writing fails, a file that is already there is left as it was and
the temporary files are removed.

Parquet and Arrow files keep the schema and field metadata, so the provenance
and the units survive. Parquet has no unit of seconds, so a `timestamp[s]`
column is stored in milliseconds and reads back as `timestamp[ms]`, with the
same values.

CSV can't hold metadata, so `write()` also writes `series.csv.json`: the file
name with `.json` added. It holds the schema metadata (`selections` as a JSON
object), and the Arrow type, units and long name of each column, which lets a
reader restore the types that CSV loses, such as timestamps. It is written for
every CSV, and `write()` won't replace it without `overwrite=True`.

### Reading the provenance

`zx.provenance(table_or_path)` returns a `Provenance` with the fields of the
table above, parsed: `selections` is a dict, `extracted_at` a timezone-aware
`datetime`, and `columns` holds the units and long names. For a file it reads
only the schema (for a CSV, the `.csv.json` file), not the data. It returns
`None` if there is no provenance, and raises `ZarrExtractError` if it is
malformed or comes from a newer layout.

```python
info = zx.provenance("series.parquet")
info.source, info.selections, info.extracted_at
```

## Remote stores

Local paths work as they are. URLs need `fsspec` and the backend for their
protocol, for example `s3fs` for `s3://` or `gcsfs` for `gs://`.

```python
table = zx.extract("s3://bucket/era5.zarr", "t2m", lat=51.5, lon=-0.1)
```

**Storage options.** The backend finds credentials in its usual way, for
example from environment variables or a configuration file. To pass options to
the backend, use the keyword-only `storage_options` argument of `describe()`,
`extract()` and `open()`.

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

(many-selections)=
## Many selections from one store

`extract()` and `describe()` open the store on every call. To make several
selections from the same store, open it once with `open()`:

```python
with zx.open("s3://bucket/sst.zarr", storage_options={"anon": True}) as src:
    src.describe()
    london = src.extract("analysed_sst", lat=51.5, lon=-0.1, time="2019-06")
    paris = src.extract("analysed_sst", lat=48.9, lon=2.4, time="2019-06")
```

`Source.extract()` and `Source.describe()` take the same selections and give
the same tables as the functions, including the provenance. The
`storage_options` go to `open()`, and `Source.extract()` raises `TypeError` if
you pass them there. `open()` opens the store at once, so a store that cannot
be opened raises `StoreOpenError` there.

**What is kept.** A `Source` keeps:

- the group, and the metadata of every array it has looked at;
- each coordinate that was read whole, already decoded (including CF time);
- the chunks of a long coordinate that were read to search it, and the values
  found in them, so a later selection nearby reads few or none;
- the result of `describe()` for each `max_depth`.

It does not keep the chunks of the variables you extract, so the memory used
does not grow with the amount of data you read. It is set by the size of the
store's coordinates. For MUR SST (latitude, longitude and 6,443 days) that is
under 1 MB, and for a CMIP6 `tas` store tens of kilobytes. There is no limit
on the cache. The worst case is a very long coordinate, such as a hundred
million timestamps, if every one of its chunks ends up being searched, which
takes about 1.6 GB.

**Changes to the store.** The store is assumed not to change while it is open.
If it does, a result can mix the kept metadata and coordinates with new data.
Open the store again to see the changes.

**Closing.** Using `Source` in a `with` block closes it at the end. Without
`with`, call `close()` when you are done; until then it stays usable. `close()`
drops the group and everything kept, which frees the memory. It does not close
the filesystem, which fsspec shares between users of the same store. After
`close()`, `describe()` and `extract()` raise `SourceClosedError`, and calling
`close()` again does nothing. Tables you already have are not affected.

**Things to know.**

- A repeated `describe()` returns the first result, so an
  `OmeMetadataWarning` is given only the first time.
- A `Source` has been tested from several threads only on local and in-memory
  stores, and thread safety is not promised.
- A `Source` cannot be pickled. Open the store again in the other process.
- `from zarrextract import *` does not import `open`, so that it does not
  replace Python's own `open`. Write `zx.open(...)`, or
  `from zarrextract import open as open_store`.

## Performance

- Only the chunks that overlap the selection are read.
- A selected coordinate with more than 8 chunks is searched instead of read
  whole. Each round estimates where a value lies from the values already read,
  as if they were evenly spaced, and reads the chunks around the estimate. It
  also reads the middle of what is still unknown, so uneven spacing, gaps and
  descending order cost no more rounds than bisection. Both ends of a range are
  searched together, and each round's chunks are read with one array read, so
  that they are fetched at the same time. An evenly spaced time axis takes
  about one round per selection.
  Coordinates you do not select on are read whole, because the table needs
  their values.
- `extract()` opens the store on every call. For many selections from one
  store, use `open()` (see {ref}`many-selections`).
- Dask is not supported.
