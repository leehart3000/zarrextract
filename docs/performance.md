# Performance

How long ZarrExtract takes to get values out of cloud Zarr stores, compared
with xarray, on the same selections. The notebook that produced these
numbers is `notebooks/benchmarks.ipynb`, and it opens in Google Colab.

## Method

- One Google Colab session on 28 November 2026: Python 3.13, ZarrExtract
  from `main` at commit 955c297 (released as 0.5.0; the notebook prints
  0.4.0 because the version number was raised afterwards), zarr 3.4.2,
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
than the exact seconds. This run was on a slower connection than the day
before: xarray took 13 seconds to open MUR SST, against 5.8.

## One selection

Median time in seconds, from opening the store to the values in memory:

| Case | ZarrExtract | xarray without Dask | xarray with Dask | Faster by |
| --- | ---: | ---: | ---: | ---: |
| 1. MUR SST, one point | 1.45 | 14.0 | 15.0 | 9.7× |
| 2. MUR SST, one point, June 2019 | 2.01 | 14.3 | 19.2 | 7.1× |
| 3. MUR SST, box, one day (80,601 values) | 1.95 | 14.3 | 14.7 | 7.3× |
| 4. CMIP6 MRI-ESM2-0, London, 1850 to 2014 | 7.54 | 8.18 | 13.9 | about the same |
| 5. CMIP6 CESM2, London, 2014 (`noleap`) | 2.11 | 2.68 | 2.42 | 1.1× |
| 6. CMIP6 MRI-ESM2-0, box across 0° E, July 2014 | 1.16 | 2.13 | 1.99 | 1.7× |
| 7. IDR OME-Zarr image, one pixel, 79 time points | 2.23 | could not open | could not open | |
| 8. MUR SST, box, June 2019 (2.4 million values) | 3.17 | 15.5 | 23.2 | 4.9× |

"Faster by" compares ZarrExtract with the faster of the two xarray
contenders. Where the two times are within 10% of each other (case 4), it
says "about the same". Converting xarray's result to a table
(`to_dataframe()`) changed most of its times by less than 10%.

## Repeated selections on one open store

The store is opened once. Then the case's selection is timed, followed by
five different selections moved a little from it: a point in time by one
day each, a time range by 30 days, and in cases 4 and 5 the latitude by
one degree. Median times in seconds:

| Case | ZarrExtract open | ZarrExtract per further selection | xarray open | xarray per further selection |
| --- | ---: | ---: | ---: | ---: |
| 1. MUR SST, one point | 0.11 | 0.64 | 13.3 | 1.18 |
| 2. MUR SST, one point, a month | 0.11 | 1.89 | 13.1 | 1.37 |
| 3. MUR SST, box, one day | 0.10 | 0.93 | 12.9 | 1.29 |
| 4. CMIP6 MRI-ESM2-0, one point, 1850 to 2014 | 0.11 | 4.61 | 0.78 | 4.95 |
| 5. CMIP6 CESM2, one point, 2014 | 0.11 | 0.81 | 0.80 | 0.75 |
| 6. CMIP6 MRI-ESM2-0, box, a month | 0.10 | 1.30 | 0.78 | 1.05 |
| 7. IDR OME-Zarr image, one pixel, 79 time points | 0.09 | 1.57 | could not open | |
| 8. MUR SST, box, a month (2.4 million values) | 0.10 | 4.72 | 13.0 | 2.67 |

The first selection on an open store took 0.71 to 5.09 seconds in xarray
and 1.28 to 6.35 seconds in ZarrExtract; the notebook has them all.

## What the numbers mean

- **This run was on a slow connection.** xarray took about 13 seconds to
  open MUR SST, where it took 5.8 in a run the day before. Both libraries
  were timed in the same session, in rotating order, so compare the
  ratios rather than the seconds.
- **Long, finely chunked coordinates are where ZarrExtract gains most.**
  MUR SST stores its 6,443 days in chunks of 5, so xarray spends most of
  its time opening the store, reading every coordinate to build its
  indexes. ZarrExtract opens the store in about 0.1 seconds, and finds the
  dates it needs by a search that guesses from the values it already
  knows: about 3 rounds of reads the first time, and 2 on an open store.
- **With short coordinates, the gap is small.** The CMIP6 stores open in
  under a second in xarray, and ZarrExtract was about as fast (case 4) or
  up to 1.7 times faster.
- **On an open store, the two libraries are close.** With `zx.open()`, a
  further point or box took about the same time as in an xarray dataset
  that is already open, or less (cases 1, 3, 4 and 5). In case 6 xarray
  was faster, 1.05 seconds against 1.30. Opening MUR SST
  costs xarray about 13 seconds here, so ZarrExtract stays ahead for many
  selections.
- **xarray is still faster for further time ranges on MUR SST and for the
  large read.** A further month took 1.89 seconds in ZarrExtract against
  1.37 in xarray (case 2), and 4.72 against 2.67 for 2.4 million values
  (case 8). The search now needs only 2 rounds of coordinate reads for
  each, so the difference lies elsewhere, most likely in how the data
  chunks are fetched. It hasn't been measured yet. Each MUR chunk holds
  5 × 1,799 × 3,600 values, so a month at one point already means
  downloading 7 chunks of about 23 MB, and case 8 about 330 MB: these
  cases mostly measure S3 bandwidth.
- **Dask adds overhead to small reads.** xarray was slower with Dask than
  without it in most cases here.
- **Longitude across the seam.** For London (longitude -0.1) on stores
  that run from 0 to 360, xarray's `method="nearest"` picked 358.9° or
  358.8°, about 1° away, because it doesn't wrap around. ZarrExtract
  picked 0°, 0.1° away. Both values were checked against direct reads of
  the cells each library chose.
- **Stores that can't be listed.** xarray couldn't open the IDR image,
  because the web server refuses to list folders. ZarrExtract reads it
  with a known array path.

See {ref}`many-selections` for how to use `zx.open()`.
