# Barcode performance and full-dataset verification

## Further optimization: concurrent barcode reading

The GUI now reads up to **four independent images concurrently**, both for an
individual group and for the Load step across all groups. Native JPEG decoding
and ZBar scanning overlap. The decoding algorithm, scan resolutions, fallback
sequence, barcode values, and bounding boxes are unchanged by this second change.
The smaller-resolution experiment was not adopted.

Fresh uncached runs on the same complete dataset (310 images, 78 groups):

| Measurement | Serial reader | Concurrent reader |
| --- | ---: | ---: |
| Total group wall time | 46.88 s | 18.94 s |
| Median group wall time | 0.360 s | 0.141 s |
| Slowest group | 2.240 s | 0.817 s |
| Decoded images | 277 / 310 | 277 / 310 |
| Unreadable images | 33 | 33 |
| Filename mismatches | 0 | 0 |
| Changed decoded values or rectangles | — | 0 |

This run was **2.47× faster** (60% less elapsed time).
Every group completed its barcode reads in under one second in this measurement;
this is an observation on this machine, not a latency guarantee.

Concurrency is limited to four readers (and available CPUs), with no dataset-sized
queue of work. Cancellation stops scheduling new images and waits for active
native reads before cleanup. Existing barcode cache flags are respected. Qt
updates and cache writes remain in the caller. CLI process-pool workers keep
their serial inner loop to avoid nested parallelism. Peak image-buffer memory can
increase because several photos can be decoded simultaneously.

All 30 regression tests pass, including overlapping reads, distinct results and
mismatch flags, cached results, cancellation, existing real CODE128 fixtures,
window startup, tree status, preprocessing and tracking. Independent review found
no blocking issues. Whole-dataset comparison confirms identical decoded text and
rectangles for every image, including the same 33 unreadable images.

Timings include image loading, decoding, pipeline barcode bookkeeping, and thread
pool startup/cleanup per group. They exclude imports, dataset discovery, Qt
painting and cache saving. No detection/processed-image cache was used; OS file
caches were not flushed. Per-image durations overlap during parallel execution:
use the sum of **one `group_wall_seconds` value per group**, not the sum of
`total_seconds`, for parallel elapsed time.

```sh
source venv/bin/activate
python scripts/benchmark_barcodes.py --workers 1 --output /tmp/barcode-serial.json --compare docs/performance-data/barcode-optimized.json
python scripts/benchmark_barcodes.py --workers 4 --output /tmp/barcode-parallel.json --compare /tmp/barcode-serial.json
```

Raw results: [serial](performance-data/barcode-serial.json) and
[concurrent](performance-data/barcode-parallel.json).

## First optimization: reduced JPEG decoding

Measured on 2026-09-26 using `configs/in_vitro.yaml`, covering all **310 images in
78 groups** in the current `data_in_vitro` dataset. No previous barcode detections
or processed-image caches were used. The filenames were used only for validation,
never to generate a barcode result.

| Measurement | Original | Optimized |
| --- | ---: | ---: |
| Total image loading + barcode reading | 77.12 s | 43.23 s |
| Median group loading + reading | 0.941 s | 0.348 s |
| Slowest group | 1.517 s | 1.857 s |
| Decoded images | 274 / 310 | 277 / 310 |
| Unreadable images | 36 | 33 |
| Decoded values differing from filename | 0 | 0 |
| Previously decoded values lost or changed | — | 0 |

## Implementation

`BarcodeReader.read_file()` first uses OpenCV's native half-resolution grayscale
JPEG decoding and scans the bottom half, where this dataset's labels are located.
It caps the scan width at 2000 pixels. This avoids decoding a full BGR image and
scanning the empty top half on successful fast reads. All other formats and failed
fast reads use the unchanged original multi-stage `read_fast()` implementation.
The pipeline uses this new file reader for GUI and batch barcode detection.

Bounding boxes are scaled back to original, unrotated photo coordinates, using
separate horizontal and vertical resize scales. A zero-area decoder box triggers
an attempt to obtain the original reader's result; if that fails, the decoded
fast-path text is retained. Such zero-area boxes are existing ZBar behavior when
only a very narrow barcode strip is visible. Coordinates need not be identical
between decodes at different resolutions. Images with multiple different labels
now prefer a readable bottom label on the JPEG fast path.

Unreadable images can take longer because they try the fast path before the full
fallback. The optimization improves total and typical-group time, not every
individual image. Existing cached barcode flags are unchanged. No GPU dependency
was added, and no input images or user caches were changed.

## Verification

All 274 previously readable values remain unchanged. Three additional reads match
the filename identifiers:

- `RT_26_2-13.26-04-28.JPG` → `RT_26_2-13`
- `RT_26_2-5.26-04-29.JPG` → `RT_26_2-5`
- `RT_26_2-51.26-04-27.JPG` → `RT_26_2-51`

The remaining 33 images are explicitly reported as unreadable. An inspected
failure (`RT_26_2-21.26-04-27.JPG`) has its barcode obscured by the holder; no claim
is made that every remaining failure has the same cause. The filename agreement
is a consistency check, not an independent human transcription of every label.

Fourteen regression tests pass, including real CODE128 fixture reads at top and
bottom positions, actual fast-path use, odd-sized JPEG coordinate mapping,
non-JPEG fallback, blank/missing files, degenerate boxes, and pipeline mismatch
flags. The existing preprocessing/tracking regression tests also remain green.
An independent review found no blocking issues; its vertical-scale rounding
observation was corrected and covered by the odd-sized-image test.

Timings include image loading and barcode reading, excluding process/import
startup, metadata discovery, Qt display, and cache saving. OS file caches were not
flushed, and these are wall-clock observations rather than latency guarantees.

## Reproduction

```sh
source venv/bin/activate
python -m unittest discover -s tests -v
python scripts/benchmark_barcodes.py --output /tmp/barcode-current.json --compare docs/performance-data/barcode-baseline.json
```

For a fresh baseline, extract `root_tracker/io/barcode.py` from commit `1000bdc`
and pass its path with `--reader-file`. The comparison fails if the dataset file
list differs or any previously read text is lost/changed. The summary separately
reports filename mismatches.

Raw per-image results: [baseline](performance-data/barcode-baseline.json) and
[optimized](performance-data/barcode-optimized.json).

## Still unreadable

- `RT_26_2-12.26-04-27.JPG`
- `RT_26_2-19.26-04-27.JPG`
- `RT_26_2-19.26-04-30.JPG`
- `RT_26_2-20.26-04-27.JPG`
- `RT_26_2-20.26-04-28.JPG`
- `RT_26_2-21.26-04-27.JPG`
- `RT_26_2-21.26-04-28.JPG`
- `RT_26_2-21.26-04-29.JPG`
- `RT_26_2-21.26-04-30.JPG`
- `RT_26_2-25.26-04-27.JPG`
- `RT_26_2-26.26-04-27.JPG`
- `RT_26_2-26.26-04-28.JPG`
- `RT_26_2-34.26-04-27.JPG`
- `RT_26_2-34.26-04-28.JPG`
- `RT_26_2-34.26-04-30.JPG`
- `RT_26_2-36.26-04-27.JPG`
- `RT_26_2-37.26-04-27.JPG`
- `RT_26_2-38.26-04-27.JPG`
- `RT_26_2-38.26-04-28.JPG`
- `RT_26_2-38.26-04-29.JPG`
- `RT_26_2-5.26-04-27.JPG`
- `RT_26_2-5.26-04-28.JPG`
- `RT_26_2-5.26-04-30.JPG`
- `RT_26_2-57.26-04-27.JPG`
- `RT_26_2-57.26-04-28.JPG`
- `RT_26_2-58.26-04-27.JPG`
- `RT_26_2-58.26-04-28.JPG`
- `RT_26_2-58.26-04-29.JPG`
- `RT_26_2-58.26-04-30.JPG`
- `RT_26_2-61.26-04-27.JPG`
- `RT_26_2-61.26-04-28.JPG`
- `RT_26_2-62.26-04-29.JPG`
- `RT_26_2-71.26-04-27.JPG`
