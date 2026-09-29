# CPU bottlenecks and memory pressure

Investigation started 2026-09-29, completed 2026-09-30. The implementation removes
unnecessary work while preserving the current tracking decisions. No new runtime
dependencies or GPU packages are required.

## Results

Fresh CPU processing of six local groups / 23 images, using the same configuration
and photographs before and after:

| Measurement | Before | After |
| --- | ---: | ---: |
| Tracking, summed over six groups | 28.45 s | 22.04 s |
| Tracking process CPU time | 41.41 s | 34.36 s |
| Preparation + registration + tracking | 39.49 s | 31.20 s |
| Process lifetime peak resident memory | 731 MiB | 647 MiB |
| Process lifetime peak committed memory | 1,725 MiB | 1,638 MiB |

Tracking used about **23% less elapsed time** and 17% less CPU time. Individual
groups improved by 12.5–28.0%. Main-root selection took roughly 46–60% less time.
The retained image arrays for a representative four-frame group shrank from
248 to 225 MiB without changing their pixels. The process peak includes library
imports and workspaces, not just retained arrays. Committed memory is a Windows
virtual-memory commitment and must not be described as resident RAM.

All **23 images matched exactly**: every ImageData array (including annotations,
edit/restore samples, RSML sample geometry and sparse overlays), root pixel
assignments, ordinary metadata, measurement lists and exported statistics.
The fingerprint excludes RSML document object instances; their existing import,
replacement and export tests were included in the suite.

These are sequential runs on an actively used laptop, with varying available
memory and OS file caches left warm. They include JPEG exports, but exclude GUI
startup, rendering and processed-cache writes. They establish an improvement on
this sample, not fixed latency guarantees. The earlier 20-second tracking
observations under different machine load are not the speedup baseline.

## Where the time went

Line tracing identified the two repeated cKDTree calls in main-root route-distance
construction as the largest individual computation in the selected routines.
Every geometry had its distance array built eagerly, and every graph edge ranked
its parent even when there was only one candidate. Querying both unshifted and
shifted history repeated identical work when the shift was zero.

The next substantial tracking work was junction/history support. One particular
avoidable cost was rebuilding the same exact-history set intersection for every
candidate parent. Skeletonization was smaller on these groups, so this pass
does not change its algorithm or root topology.

## Changes

- Main-root route-distance arrays are computed only when a ranking actually
  needs them. Single-parent chains skip ranking. Zero-shift queries run once.
  Route minima use a single running array, avoiding a routes-by-history
  temporary matrix. Coordinate arrays are concatenated once instead of being
  rebuilt from Python pixel lists inside the alignment loop. Point order and
  all ranking/tie rules are preserved.
- Junction exact-history overlap is checked once per plant/segment and reused
  for different candidate parents. Existence checks use `isdisjoint`, avoiding
  temporary intersection sets. Assignment order is unchanged.
- Per-plant raster/counting masks use reusable uint8 arrays instead of repeated
  float32 full-frame allocations. Label maps, root widths, skeleton workspaces
  and consumed temporal masks are released as soon as their last consumer
  finishes, before allocating the following frame's workspaces.
- Aligned Canny outputs own their cropped storage. Previously a small view kept
  the entire padded registration search image alive for each frame.
- GUI and CLI batch worker counts now consider current available RAM, image
  dimensions, group sizes and the CPU/user cap. Each worker gets a bounded
  OpenCV thread budget: at most four threads, reduced as worker count grows.
  This avoids multiplying sixteen processes by sixteen native threads.

## This machine's RAM

At the initial snapshot, Windows reported **31.63 GiB usable physical RAM** and
**3.53 GiB available**: approximately 28.10 GiB was already in use. Availability
subsequently fluctuated, falling below 1 GiB during one instrumented baseline.
This is system-wide pressure, not memory all owned by Root Tracker. Working-set
snapshots alone do not establish disk paging; Windows page-fault counts also
include soft faults.

A fresh four-image pipeline process reached roughly 0.7 GiB resident and
1.7 GiB committed before the changes. The old batch pool could start 16 such
workers regardless of the headroom. That was an avoidable memory risk.

The new scheduler uses header-only photo sizes (or already loaded dimensions),
budgets retained frames plus CV workspaces and import/geometry overhead, and
reserves 10% of physical RAM up to 2 GiB for the application/system. Full photo
dimensions deliberately overestimate cropped processing. The largest group
bounds the pool. Under the observed pressure it selects **one worker**, with
up to four OpenCV threads, rather than sixteen workers. It allows more workers
when enough memory is available. An explicit CLI `max_workers` remains an upper
bound. Windows and Linux have native memory probes; an unknown memory budget
falls back to one worker.

This is a conservative pool-start estimate, not continuous memory admission
control or an allocation guarantee for arbitrarily complex roots. It still
allows one worker when memory is tight so processing can make progress.

## Validation and reproduction

The full suite ran **429 tests**, with four failure reports across three tests
and one skip for unavailable CUDA. Every failure was reproduced against the
saved pre-change source:

- One existing recorded ownership expectation in `test_followup_routing`.
- Background metadata extraction timing in `test_image_metadata`.
- Two shapes in the bundled crop-default expectation in `test_roi_geometry`.

These behaviours were left unchanged. Seven resource-budget/spawn tests and six
registration tests passed separately after the final test additions. Existing
main-root, crossing, temporal, editing, mask, RSML and GUI tests cover the changed
pipeline. The registration test now checks that outputs own contiguous storage.

Two groups / seven images were additionally compared in real spawned workers,
both with one and with four OpenCV threads. Their outputs match the corresponding
pre-change worker baseline. Those runs confirmed lower memory use but had noisy
timings; no multi-worker throughput improvement is claimed from them.

The new profiler samples Windows RSS/private commitment every 50 ms, records
OS lifetime peaks, retained ndarray backing storage, CPU/wall stage times and
optional line attribution. Missing non-Windows memory counters are reported as
unavailable. It uses temporary JPEG output and never reads/writes dataset caches.

```powershell
python scripts/profile_pipeline.py --groups RT_25_10-1 RT_25_10-10 --output before.json --module-root PATH_TO_BASELINE
python scripts/profile_pipeline.py --groups RT_25_10-1 RT_25_10-10 --output after.json --compare before.json
python scripts/profile_pipeline.py --groups RT_25_10-1 --output trace.json --trace-lines
python scripts/profile_pipeline.py --groups RT_25_10-1 --output worker.json --batch-worker --opencv-threads 4
python -m unittest discover -s tests
```

Raw comparisons, line attribution and test summary:
[cpu-memory-2026-09-29](performance-data/cpu-memory-2026-09-29/).
The local pre-change source snapshot is retained in the ignored
`.venv/performance-baseline/root_tracker` directory.
