# Performance

How long ZarrExtract takes to get values out of cloud Zarr stores, compared
with xarray, on the same selections. The notebook that produced these
numbers is `notebooks/benchmarks.ipynb`, and it opens in Google Colab.

## Method

- One Google Colab session on 27 November 2026: Python 3.13, ZarrExtract
  from `main` at commit d59caca (released as 0.4.0; the notebook prints
  0.3.0 because the version number was raised afterwards), zarr 3.4.2,
  xarray 2026.9.0, Dask 2026.8.0, fsspec and s3fs 2025.12.0.
- Three contenders for one selection: `zx.extract`, xarray with Dask
  (`xr.open_zarr`, then `.sel(...).load()`), and xarray without Dask
  (`chunks=None`).
- Two contenders for repeated selections on a store that is already open:
  `zx.open` with `src.extract`, and xarray without Dask, opened once.
- Each timing includes getting the values into memory. Opening is also
  timed on its own.
- Each case ran five times. The order of the contenders rotated, after an
  untimed warm-up read from each store. The tables show the median.
- Every result was checked: where both libraries selected the same cells,
  xarray's values were identical to ZarrExtract's.

Timings over a network vary from run to run, so read the ratios rather
than the exact seconds. For example, an earlier run on the same day took
4.84 seconds for xarray without Dask in case 4, where this run took 2.37.

## One selection

Median time in seconds, from opening the store to the values in memory:

| Case | ZarrExtract | xarray without Dask | xarray with Dask | Faster by |
| --- | ---: | ---: | ---: | ---: |
| 1. MUR SST, one point | 0.98 | 6.43 | 7.10 | 6.5× |
| 2. MUR SST, one point, June 2019 | 2.12 | 6.70 | 9.43 | 3.2× |
| 3. MUR SST, box, one day (80,601 values) | 1.17 | 6.18 | 6.49 | 5.3× |
| 4. CMIP6 MRI-ESM2-0, London, 1850 to 2014 | 2.39 | 2.37 | 4.19 | 1.0× |
| 5. CMIP6 CESM2, London, 2014 (`noleap`) | 0.56 | 0.76 | 0.97 | 1.4× |
| 6. CMIP6 MRI-ESM2-0, box across 0° E, July 2014 | 0.54 | 0.73 | 0.96 | 1.3× |
| 7. IDR OME-Zarr image, one pixel, 79 time points | 3.35 | could not open | could not open | |
| 8. MUR SST, box, June 2019 (2.4 million values) | 2.70 | 8.25 | 9.99 | 3.1× |

"Faster by" compares ZarrExtract with the faster of the two xarray
contenders. Converting xarray's result to a table (`to_dataframe()`)
changed its times by less than 10%.

## Repeated selections on one open store

The store is opened once. Then the case's selection is timed, followed by
five different selections moved a little from it: a point in time by one
day each, a time range by 30 days, and in cases 4 and 5 the latitude by
one degree. Median times in seconds:

| Case | ZarrExtract open | ZarrExtract per further selection | xarray open | xarray per further selection |
| --- | ---: | ---: | ---: | ---: |
| 1. MUR SST, one point | 0.05 | 0.34 | 5.74 | 0.34 |
| 2. MUR SST, one point, a month | 0.05 | 1.38 | 5.77 | 1.01 |
| 3. MUR SST, box, one day | 0.05 | 0.56 | 5.87 | 0.57 |
| 4. CMIP6 MRI-ESM2-0, one point, 1850 to 2014 | 0.06 | 2.74 | 0.36 | 2.73 |
| 5. CMIP6 CESM2, one point, 2014 | 0.05 | 1.02 | 0.35 | 0.95 |
| 6. CMIP6 MRI-ESM2-0, box, a month | 0.05 | 0.55 | 0.34 | 0.60 |
| 7. IDR OME-Zarr image, one pixel, 79 time points | 0.14 | 2.94 | could not open | |
| 8. MUR SST, box, a month (2.4 million values) | 0.05 | 2.85 | 5.94 | 1.96 |

The first selection on an open store took 0.37 to 2.97 seconds in xarray
and 0.87 to 4.36 seconds in ZarrExtract; the notebook has them all.

## What the numbers mean

- **Long, finely chunked coordinates are where ZarrExtract gains most.**
  MUR SST stores its 6,443 days in chunks of 5, so xarray spends about
  5.8 seconds opening the store, reading every coordinate to build its
  indexes. ZarrExtract opens the store in about 0.05 seconds and finds the
  dates it needs by bisection, reading a few chunks instead of all of
  them.
- **With short coordinates, the gap is small.** The CMIP6 stores open in
  about 0.35 seconds in xarray, and ZarrExtract was as fast or up to 1.4
  times faster.
- **On an open store, the two libraries are about even.** With `zx.open()`,
  a further point or box selection took the same time as in an xarray
  dataset that is already open (cases 1, 3, 4, 5 and 6). Opening MUR SST
  costs xarray about 5.7 seconds more, so ZarrExtract stays ahead for
  several selections, about 15 in case 2 and 6 in case 8 at these rates.
- **xarray is faster for further time ranges on MUR SST and for large
  reads.** In case 2 a further month took 1.38 seconds in ZarrExtract
  against 1.01 in xarray, and in case 8 2.85 against 1.96. xarray keeps
  the whole time index in memory once it is open, while ZarrExtract
  searches for each new date by bisection and reads the chunks it hasn't
  seen. ZarrExtract also builds a table with one row per value, which for
  2.4 million values takes time that xarray's array doesn't need.
- **Not yet explained:** in case 8, the first selection on an open
  ZarrExtract store took 4.36 seconds, more than a whole `zx.extract()`
  call (2.70). This may be network variation, and will be checked.
- **Dask adds overhead to small reads.** xarray was slower with Dask than
  without it in every case here.
- **Longitude across the seam.** For London (longitude -0.1) on stores
  that run from 0 to 360, xarray's `method="nearest"` picked 358.9° or
  358.8°, about 1° away, because it doesn't wrap around. ZarrExtract
  picked 0°, 0.1° away. Both values were checked against direct reads of
  the cells each library chose.
- **Stores that can't be listed.** xarray couldn't open the IDR image,
  because the web server refuses to list folders. ZarrExtract reads it
  with a known array path.

See {ref}`many-selections` for how to use `zx.open()`.
