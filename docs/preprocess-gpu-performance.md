# GPU, copy reduction, and coarse-to-fine alignment

This follow-up compares against the preprocessing implementation immediately
before these changes (including its CPU strip median, JPEG prefetch, and FFT
alignment). It does not compare against the much slower original application.

## Changes

- Optional CUDA implementation of the 101×101 median filter and saturated
  background subtraction. A sliding histogram produces the same integer pixels
  and replicated borders as OpenCV. Transfers are included in timings. Two
  reusable device buffers avoid repeated allocation across groups and GUI
  worker threads.
- Removed redundant full-image copies during preprocessing. Public output
  arrays still own independent, contiguous storage; source photographs remain
  unchanged. Registration allocates the final aligned image directly instead
  of padding and then copying a larger color image.
- Alignment searches a reduced edge map, refines at half resolution, then
  evaluates exact full-resolution scores in a small neighborhood. Weak,
  ambiguous, and boundary matches fall back to the existing full search.
  Smoothing before the coarse ambiguity check guards against grid-phase
  aliasing, including a regression with an exact match and a misleading decoy.

Coarse candidate selection is a heuristic: the local score is exact, but this
does not prove the global winner is identical for every possible future image.
The expanded dataset comparison below checks its effect on actual outputs.
Cache format and compression settings are unchanged by this follow-up.

## GPU setup

The optional packages were installed in this workspace's virtual environment
and tested on its NVIDIA RTX 5070 Ti. Other installations can use:

```sh
source venv/bin/activate
pip install -r requirements-gpu.txt
python scripts/run_gui.py --config configs/in_vitro.yaml
```

The optional requirement uses CuPy's CUDA 13 runtime packages; see the
[CuPy installation documentation](https://docs.cupy.dev/en/stable/install.html)
for driver requirements. The base requirements remain CPU-only. CUDA starts
lazily; missing packages, unavailable hardware, or a runtime failure use the
CPU implementation. Set `ROOT_TRACKER_GPU=0` to opt out. Multiprocessing batch
children use CPU filtering to avoid inheriting a CUDA context; the GUI's
processing threads share the accelerated backend.

Initial CUDA context creation and first-use kernel compilation add latency.
Subsequent groups reuse the backend and buffers. This is independent of the
application's processed-image cache.

## Validation method

Twenty groups, spread from `RT_26_2-1` through `RT_26_2-75`, contain 79 images.
Each run constructs fresh series and writes to a temporary cache on the dataset
disk. Existing processing caches are not read, and dataset files and user
caches are not modified. Operating-system file caches are not flushed.

The benchmark compares every image-array fingerprint, plant positions, green
areas, root-to-plant assignments, lengths, growth measurements, and exported
statistics against the pre-change snapshot. It runs tracking as well as
preprocessing to catch downstream effects of alignment changes.

Unit tests cover CUDA border/tile behavior, saturated subtraction, read-only
inputs, CPU fallback, output ownership, translations, ambiguous alignment, and
the adversarial decoy case. The GUI profiler exercises selection, processing
QThreads, cache saving, completion handling, and preview updates in an offscreen
Qt window; application startup and desktop compositor latency are excluded.

## Expanded dataset results

All **20 groups / 79 images matched exactly**, including downstream tracking
and statistics. CUDA was active for every group in the final run.

| Stage | Before, median | After, median |
| --- | ---: | ---: |
| Image preparation | 1.773 s | 1.549 s |
| Alignment | 0.939 s | 0.419 s |
| Preprocessing cache save | 0.180 s | 0.168 s |
| **Preparation + alignment + save** | **2.907 s** | **2.142 s** |

The combined median is measured per group, rather than summing the individual
stage medians. This is about **26% less elapsed time**. Alignment provides the
largest reduction. The first group in the final GPU process took 2.525 s,
including CUDA startup with an already compiled kernel cache. A separate first
run with an empty CuPy kernel cache took 2.563 s before the final alignment
safeguard. These are observations, not guaranteed startup bounds.

The one-second target is **not yet met**. These measurements exclude barcode
detection and application startup and use warm operating-system file caches,
but no previously processed group cache. Background desktop activity can change
latency substantially: a later repeated timing run overlapped a Blender render
using several CPU cores, so it is retained as validation evidence rather than
used for the before/after speedup claim.

Raw data: [before](performance-data/preprocess-gpu-reference.json),
[final expanded comparison](performance-data/preprocess-gpu-final.json),
[repeat baseline](performance-data/preprocess-gpu-timing-reference.json), and
[repeat under competing load](performance-data/preprocess-gpu-timing-final.json).

All **44 unit tests passed**, including the actual CUDA comparison on this GPU.
Four GUI selection-to-preview runs also completed successfully with CUDA active
across successive QThreads. These overlapped the competing render and took
3.758–4.527 s, so they verify integration rather than idle-machine UI latency.
Their [full spans](performance-data/preprocess-gpu-gui-profile.json) include
GPU transfers, CPU preparation, alignment, cache saving, and preview completion.

Reproduce the focused integration checks with:

```sh
source venv/bin/activate
python -m unittest discover -s tests
python scripts/benchmark_pipeline.py --groups RT_26_2-1 RT_26_2-40 --repeats 2 --output /tmp/gpu-check.json --compare docs/performance-data/preprocess-gpu-reference.json
python scripts/profile_preprocess.py --groups RT_26_2-1 RT_26_2-40 --repeats 2 --output /tmp/gpu-gui-check.json
```
