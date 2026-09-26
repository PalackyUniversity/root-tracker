# Pipeline performance implementation plan

**Goal:** Reduce fresh preprocessing and tracking latency, aiming for one second per group per GUI stage, preserving CV outputs.

**Architecture:** Profile the current full-resolution pipeline first. Replace redundant full-image work with equivalent bounded operations. Evaluate larger algorithm or hardware changes only with dataset evidence.

**Constraints:** Use the current dataset; no previous pipeline cache; preserve user files and masks. Include registration and synchronous cache saving in GUI latency. Benchmark a fixed random seed because existing KMeans is stochastic. No commits or unrelated source changes needed.

## Tasks

- [x] Profile a four-image group and inspect GPU availability.
- [x] Capture original output fingerprints and uncached stage timings on six groups (23 images, including a three-image group).
- [x] Replace per-contour full-frame threshold masks with bounding-box masks, retaining contour hierarchy and pixel overlap rules. Verify synthetic holes/borders/noise and dataset fingerprints.
- [x] Vectorize endpoint neighbor counting, preserving contour ordering and duplicate points; use saturated uint8 subtraction instead of int64 images. Verify border/duplicate endpoints and exact dataset outputs.
- [x] Reprofile, investigate median filtering, registration, skeletonization, and cache compression. Benchmark any alternatives before choosing them; reject output-changing shortcuts without quantified validation.
- [x] Run final uncached benchmarks and regression checks; document measured latency, remaining bottlenecks and GPU feasibility. Review the diff for correctness.

## Evidence

Original checkout: 1e412c870e0bfd75237a9474d556ca3e9fdc7e8d, source snapshot at /tmp/root-tracker-baseline/root_tracker.
Initial profiled RT_26_2-1: preprocess 4.52 s, registration 1.99 s, tracking 5.92 s (profiling overhead included).
Machine: i7-9700KF (8 cores), RTX 5070 Ti 16 GB; installed OpenCV reports zero CUDA devices.

Final evidence: docs/pipeline-performance.md and docs/performance-data/. All 23 images match exactly. Six regression tests pass; independent review found no actionable issues. GPU feasibility assessed, not implemented; one-second ideal remains unmet.
