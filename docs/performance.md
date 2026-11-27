# Performance

How long ZarrExtract takes to get values out of cloud Zarr stores, compared
with xarray, on the same selections. The notebook that produced these
numbers is `notebooks/benchmarks.ipynb`, and it opens in Google Colab.

## Method

- One Google Colab session on 27 November 2026: Python 3.13, ZarrExtract
  0.3.0, zarr 3.4.2, xarray 2026.9.0, Dask 2026.8.0, fsspec and s3fs
  2025.12.0.
- Three contenders: `zx.extract`, xarray with Dask (`xr.open_zarr`, then
  `.sel(...).load()`), and xarray without Dask (`chunks=None`).
- Each timing includes opening the store and getting the values into
  memory. Opening is also timed on its own.
- Each case ran five times. The order of the contenders rotated, after an
  untimed warm-up read from each store. The tables show the median.
- Every result was checked: where both libraries selected the same cells,
  xarray's values were identical to ZarrExtract's.

Timings over a network vary from run to run, so read the ratios rather
than the exact seconds.

## Results

Median time in seconds, with the values in memory:

| Case | ZarrExtract | xarray without Dask | xarray with Dask | Faster by |
| --- | ---: | ---: | ---: | ---: |
| 1. MUR SST, one point | 1.04 | 6.23 | 7.52 | 6.0× |
| 2. MUR SST, one point, June 2019 | 1.88 | 6.68 | 9.54 | 3.6× |
| 3. MUR SST, box, one day (80,601 values) | 1.11 | 6.44 | 6.48 | 5.8× |
| 4. CMIP6 MRI-ESM2-0, London, 1850 to 2014 | 2.75 | 4.84 | 6.62 | 1.8× |
| 5. CMIP6 CESM2, London, 2014 (`noleap`) | 0.66 | 0.81 | 1.01 | 1.2× |
| 6. CMIP6 MRI-ESM2-0, box across 0° E, July 2014 | 0.57 | 0.83 | 1.05 | 1.5× |
| 7. IDR OME-Zarr image, one pixel, 79 time points | 3.50 | could not open | could not open | |
| 8. MUR SST, box, June 2019 (2.4 million values) | 2.56 | 8.41 | 9.39 | 3.3× |

"Faster by" compares ZarrExtract with the faster of the two xarray
contenders. Converting xarray's result to a table (`to_dataframe()`)
changed its times by less than 10%.

## What the numbers mean

- **Long, finely chunked coordinates are where ZarrExtract gains most.**
  MUR SST stores its 6,443 days in chunks of 5, so xarray spends about
  5.8 seconds opening the store, reading every coordinate to build its
  indexes. ZarrExtract opens the store in about 0.05 seconds and finds the
  dates it needs by bisection, reading a few chunks instead of all of
  them.
- **With short coordinates, the gap is small.** The CMIP6 stores open in
  about 0.35 seconds in xarray, and ZarrExtract is 1.2 to 1.8 times
  faster.
- **Reading the data itself is not faster.** For the larger read (case 8),
  xarray needs about 2.6 seconds after opening, the same as ZarrExtract
  in total. The difference is in finding the data, not in reading it.
- **Many small selections from one open dataset favour xarray.** Once a
  dataset is open, each further xarray selection in cases 1 to 3 took
  0.3 to 0.9 seconds. That figure was derived from the totals above (the
  median time minus the time to open the store), not measured on its own.
  The timings above are for `zx.extract()`, which opens the store on every
  call. `zx.open()` keeps the store, the metadata and the coordinates it has
  read between calls (see {ref}`many-selections`), but these timings were
  taken before it existed. The benchmark notebook now times repeated
  selections on one open store for both libraries, and the numbers will be
  added here after it has been run again.
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
