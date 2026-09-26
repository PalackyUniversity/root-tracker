# Further preprocessing optimization

Implemented on 2026-09-26, relative to commit `836bd1e`.

## Changes

- Overlap decoding of the next full-resolution photograph with processing the
  current one. Only one photograph is prefetched; CV and stochastic KMeans still
  run in their original chronological order. The GUI worker now uses the same
  series preprocessing path as the benchmark, including cancellation callbacks.
- Compute the existing full-search CCOEFF alignment score for sparse binary Canny
  edges using a parallel SciPy FFT and integral-image mean correction. Search
  bounds, image resolution, alignment crops and plant offsets are unchanged.
  Small, dense, non-binary, and ambiguous/near-tied cases use the original OpenCV
  matcher. This is mathematically the same score; floating-point implementations
  are not promised bit-identical for every possible input.
- Multiprocessing batch workers retain the previous serial inner work, avoiding
  nested CPU pools and the FFT memory increase across many processes. These
  changes primarily accelerate automatic/single-group GUI preprocessing.

The FFT adds substantial temporary memory (hundreds of MB for these large
images); prefetching holds one extra decoded photograph. SciPy was already a
required dependency. No GPU installation, downsampling, threshold changes, or
changes to user data/caches were needed.

## Exact-output verification

Fresh runs of **six groups / 23 images** (`RT_26_2-1`, `-20`, `-40`, `-60`, `-75`,
`-4`) match the current baseline exactly. Comparison covers hashes of every
stored image, processed/Canny/difference/annotated array; all plant positions and
green areas; root pixel assignments; root lengths, growth, and statistics.
Tracking was rerun after preprocessing, not compared only at an intermediate
stage. All four alternating timing runs also matched their reference outputs.

No existing processed-image cache was read. NumPy's random seed was fixed for
comparison because the existing KMeans algorithm is stochastic. All 34 tests
pass, including large registration translations, blank/non-binary fallback,
predecoded-image equivalence, ordered prefetch progress and interruption. An
independent review found no actionable issues.

## Timing observations

Preprocess totals below include computation, registration and cache saving.
Old/new versions ran alternately against fresh series. **Other benchmarks were
using several CPU cores during these measurements**, and their changing load
makes these observations unsuitable as idle-machine latency guarantees.

| Pair | Group | Before | After | Reduction |
| --- | --- | ---: | ---: | ---: |
| 1 | RT_26_2-1 | 9.39 s | 6.47 s | 31% |
| 1 | RT_26_2-4 | 5.62 s | 4.91 s | 13% |
| 2 | RT_26_2-1 | 6.10 s | 4.95 s | 19% |
| 2 | RT_26_2-4 | 3.64 s | 3.22 s | 12% |

The observed reduction was 11–31%. The ideal **one-second-per-group target is
still not met**. An earlier isolated alignment probe improved one match from
0.61 s to 0.33 s; full-stage measurements above are the more relevant evidence.

Timings exclude startup/imports, barcode detection, Qt painting, output hashing,
and OS page-cache flushing. Cache/export writes use temporary directories on the
dataset disk and are removed after each group. Raw validation-run timings were
collected under different load and should not be compared as a speedup claim.

Evidence: [reference outputs](performance-data/preprocess-v2-reference.json),
[optimized outputs](performance-data/preprocess-v2-validated.json),
[alternating timings](performance-data/preprocess-v2-timings.json).

## Reproduce

```sh
source venv/bin/activate
python -m unittest discover -s tests -q
python scripts/benchmark_pipeline.py --output /tmp/preprocess-check.json --compare docs/performance-data/preprocess-v2-reference.json
```

For baseline timings, extract `root_tracker/` from `836bd1e` to a temporary
folder and pass that folder with `--module-root`. Run old/new alternately on an
otherwise idle machine for a cleaner performance comparison.
