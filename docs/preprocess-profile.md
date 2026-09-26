# Preprocess time profile — 2026-09-26

The current code was profiled through the actual GUI selection handler and
processing QThread, including tree updates, cache saving, completion handling,
and the selected image's preview. This is an offscreen Qt window: real desktop
compositor latency and app startup are excluded. Other desktop/CAD activity was
present; these are repeatable observations under current machine load, not
exclusive-machine performance guarantees.

No algorithm changes were made for this profiling pass. Dataset images and user
caches were not modified. Each run created fresh series and wrote to a temporary
cache on the dataset disk; the temporary cache was removed afterward. OS file
caches were not flushed. Imports, data discovery, tree construction, and window
setup were outside the selection-to-preview timer.

## End-to-end results

Three groups (`RT_26_2-1`, `RT_26_2-40`, `RT_26_2-4`) were each run twice:

- Four-image groups: **2.98–3.10 s**, mean **3.031 s**.
- Three-image group: **2.15–2.26 s**, mean **2.207 s**.
- Two additional fresh runs of `RT_26_2-1` with barcode detection enabled:
  **3.10–3.17 s**, mean **3.135 s** (versus 2.985 s for the same group without
  barcode work). This comparison includes barcode cache writes and tree refresh.

The four-image group breakdown is:

| Stage | Mean elapsed time | Share of total |
| --- | ---: | ---: |
| Image preparation (including exposed JPEG load time) | 1.764 s | 58.2% |
| Alignment / registration | 0.966 s | 31.9% |
| Cache save | 0.191 s | 6.3% |
| Preview update | 0.017 s | 0.6% |
| Remaining UI / scheduling / measurement overhead | 0.093 s | 3.1% |

## Where computation goes

Nested timings below overlap the stages above and must not be added to them.

| Operation | Mean per four-image group |
| --- | ---: |
| Background removal | 0.925 s |
| Large 101-pixel median filter, within background removal | 0.745 s |
| Blue-background detection/crop | 0.277 s |
| Green-contour detection | 0.142 s |
| Plant clustering | 0.059 s |
| JPEG reads across the prefetch thread | 0.475 s |
| Full alignment score calculation | 0.910 s |
| Forward + inverse FFTs, within alignment | 0.578 s |
| Tree status refreshes, partially overlapping worker computation | 0.113 s |

Most JPEG loading overlaps image processing, so its 0.475 s is not additional
critical-path time. Similarly, summing median-filter strip workers' durations
would overstate their wall time; the 0.745 s above measures the complete parallel
filter call. The profiler records both parent and child spans and thread IDs.

Processed frames are still roughly **12–13 megapixels each** (for example,
2172 × 5640), and alignment scores use larger padded arrays. The remaining major
costs are therefore full-resolution median filtering and alignment. Those are
the highest-value next optimization targets. Cache saving and preview rendering
are relatively small, and another barcode speedup would have little effect on
this stage's total. Reaching one second requires a substantial reduction in the
filtering/alignment work, not just completion-handler tuning.

## Reproduce

```sh
source venv/bin/activate
python scripts/profile_preprocess.py --output /tmp/preprocess-profile.json
python scripts/profile_preprocess.py --groups RT_26_2-1 --repeats 2 --barcodes --output /tmp/preprocess-with-barcodes.json
```

The script uses instrumented wall-clock spans across the GUI, worker, JPEG
prefetch, and median-filter threads. No additional profiler dependency is needed.
The JSON contains call counts, inclusive durations, per-thread event timelines,
image sizes, CPU/OpenCV thread counts, and load averages. Small instrumentation
and event-loop polling overhead is included in end-to-end time.

Raw results: [preprocessing](performance-data/preprocess-gui-profile.json) and
[including barcode detection](performance-data/preprocess-gui-profile-barcodes.json).
