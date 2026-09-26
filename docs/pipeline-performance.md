# Pipeline performance

The CPU optimizations preserve the original full-resolution pipeline outputs on
six groups / 23 images from `configs/in_vitro.yaml`. The one-second-per-group
target is not yet met. The largest remaining preprocessing costs are median
filtering and full-resolution registration.

## Implemented changes

- Threshold overlap is computed inside each contour's bounding box instead of
  allocating and scanning a full-image mask for every contour. Contour hierarchy,
  hole handling, and the original 20% overlap rule are retained.
- Endpoint neighbor counting uses NumPy array operations; contour ordering,
  repeated points, and image-border behavior are retained.
- Saturated OpenCV subtraction replaces large temporary int64 image arrays.
- Disconnected root components are skeletonized within their bounding boxes.
  Dense/noisy images fall back to full-frame skeletonization.
- The 101-pixel median filter runs in up to four overlapping strips. Each strip
  includes the full 50-pixel neighborhood, preserving exact results at seams.
  Multiprocessing batch workers retain one filter call to avoid nested pools.
- Pixel sets use native Python integers and in-place updates; endpoint membership
  uses a set. Linking order and distance/angle calculations remain unchanged.
- Cache writes use uncompressed NPZ by default. Existing compressed files remain
  readable, and older versions can read newly written files. To prefer smaller
  cache files, add `cache_compressed: true` under `data:` in the YAML config.

No GPU dependency was installed and no downsampling, threshold changes, or
approximate registration were introduced.

## Measured results (2026-09-26)

The representative four-image group `RT_26_2-1` was rerun before and after the
changes without the competing Blender render seen during the initial dataset
baseline. The numbers below include cache writes and the existing tracking JPEG
exports, but exclude Qt painting and barcode detection.

| Stage | Original | Optimized |
| --- | ---: | ---: |
| Preprocessing computation | 4.59 s | 2.05 s |
| Registration | 2.00 s | 1.84 s |
| Preprocessing cache save | 6.55 s | 0.18 s |
| **Preprocess total** | 13.15 s | 4.07 s |
| Tracking including JPEG export | 5.72 s | 1.48 s |
| Tracking cache save | 10.62 s | 0.32 s |
| **Track total** | 16.34 s | 1.80 s |

Final fresh-group results across the validation subset:

| Group | Images | Preprocess + registration + save | Track + save | Outputs |
| --- | ---: | ---: | ---: | --- |
| RT_26_2-1 | 4 | 4.07 s | 1.80 s | Exact match |
| RT_26_2-20 | 4 | 3.93 s | 1.39 s | Exact match |
| RT_26_2-40 | 4 | 4.13 s | 1.68 s | Exact match |
| RT_26_2-60 | 4 | 4.07 s | 1.73 s | Exact match |
| RT_26_2-75 | 4 | 4.05 s | 1.73 s | Exact match |
| RT_26_2-4 | 3 | 2.87 s | 1.10 s | Exact match |

**The one-second target is not reached.** Representative end-to-end stage work
is approximately 3.2× faster for preprocessing and 9.1× faster for tracking.
These are single-run wall-clock observations, not latency guarantees.

Cache storage for the representative group increased from 62.5 to 281.8 MB after
preprocessing and from 110.0 to 428.8 MB after tracking. The cache file is replaced
at each stage, so these sizes are not additive. This disk-space tradeoff is
reversible with `data.cache_compressed: true`. Existing cache files are unchanged
until the corresponding group is processed again.

Raw evidence: [original output reference](performance-data/baseline-reference.json),
[uncontended representative baseline](performance-data/baseline-repeat.json), and
[final optimized run](performance-data/optimized.json). Some initial baseline
measurements were taken while a Blender render was using most CPU cores; use the
representative repeat for speedup claims, not the contended six-group baseline.

## Validation and reproduction

Run from the repository root:

```sh
source venv/bin/activate
python -m unittest discover -s tests -v
python scripts/benchmark_pipeline.py --output /tmp/root-tracker-current.json
```

`--module-root /path/to/original/checkout` runs the same benchmark against an
original source checkout. Pass `--compare /path/to/reference.json` to fail on an
output mismatch. `--groups` selects groups; `--repeats` repeats each group.

The benchmark creates fresh series and never loads processed pipeline caches.
It retains existing user masks, fixes the NumPy random seed for the original
stochastic KMeans, and times preprocessing, registration, tracking, and both cache
writes separately. Exports and cache files go to an automatically removed
temporary directory on the working-directory disk. The tracking timing includes
the pipeline's existing annotated-JPEG export.

Output comparisons cover SHA-256 hashes of every stored image array, all plant
positions and green areas, root pixel assignments, lengths, growth measurements,
and all statistics records. Hashing and metadata discovery are outside timing.
Python/import startup, barcode detection, Qt repainting, and OS page-cache flushing
are also outside timing. Thus these are fresh pipeline runs, not cold-machine I/O
benchmarks or stopwatch measurements of the GUI.

Six synthetic regression tests cover contour holes, borders, noise, empty masks,
duplicate endpoints, skeleton components, median-strip boundaries, saturated
subtraction, and compressed/uncompressed cache round trips. An independent code
review found no actionable regressions.

## GPU feasibility and next steps

This machine has an i7-9700KF (8 cores) and an RTX 5070 Ti (16 GB). The installed
OpenCV 5.0.0 reports `cv2.cuda.getCudaEnabledDeviceCount() == 0`, so the current
application has no usable OpenCV CUDA backend.

1. **GPU registration is the first GPU experiment to run.** OpenCV provides CUDA
   template matching supporting `TM_CCOEFF` for 8-bit images, which corresponds to
   the current registration score. A CUDA-enabled OpenCV/contrib build is needed.
   Compare the selected offsets, aligned pixels, and final root statistics on the
   same dataset; floating-point correlation scores need not be bit-identical.
   [OpenCV CUDA template matching](https://docs.opencv.org/4.12.0/d0/d05/group__cudaimgproc.html)
2. **Benchmark a histogram-based GPU median filter, preserving all three color
   channels and border behavior.** OpenCV's documented CUDA median filter supports
   single-channel 8-bit input, so the present BGR operation needs per-channel
   filtering. Keep subtraction and Canny on the GPU where practical to reduce
   transfers. Include GPU context initialization and transfers in the latency
   measurement; do not assume a warm kernel timing establishes a cold-stage target.
   [OpenCV CUDA median filter](https://docs.opencv.org/4.12.0/d1/d1a/namespacecv_1_1cuda.html)
3. **For tracking, prioritize native graph/contour work and independent per-image
   preparation.** Thresholding and skeletonization can be prepared independently,
   but root assignment must retain chronological order because it uses previous
   frame samples. Benchmark memory use and avoid multiplying worker pools during
   batch processing.

These are recommendations, not measured GPU speedups. A GPU implementation would
need the same output comparison before enabling it. The present changes already
remove CPU and compression overhead that a GPU alone would not address.
