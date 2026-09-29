# Adaptive GPU acceleration: feasibility investigation

Investigated 2026-09-29. Recommendation: keep the CPU implementation as the
reference, prototype a small optional PyOpenCL backend, and select acceleration
per operation and workload using measurements. Retain the existing CuPy path
for installations that already have it. GPU work can improve preprocessing;
the current tracking bottlenecks also need a separate CPU optimization effort.

No production processing code or application dependencies changed during this
investigation. OpenCL was installed in an isolated research environment. The
experimental scripts and measurements are in
[performance-data/gpu-feasibility-2026-09-29](performance-data/gpu-feasibility-2026-09-29/).

## What is already implemented

- `preprocessing/gpu_background.py` implements an exact CUDA sliding-histogram
  101×101 median with fused saturated subtraction. It lazily imports CuPy,
  reuses two buffers, serializes callers, and falls back on errors. It is enabled
  only for uint8 images of at least one million pixels in the main process.
- `ROOT_TRACKER_GPU=0` disables it. Missing CuPy/device initialization disables
  the backend for the process. A runtime error disables the entire median
  backend rather than just the failed workload.
- `requirements-gpu.txt` installs `cupy-cuda13x[ctk]`. The toolkit extra includes
  CUDA component packages for BLAS, FFT, random numbers, solvers, sparse
  operations, runtime and compilation; most are unnecessary for this kernel.
  CuPy itself is not the whole dependency footprint.
- Registration already has a CPU coarse-to-fine search, sparse exact local
  scoring, FFT fallback and an OpenCV fallback for ambiguous peaks. Porting the
  old full-resolution FFT alone would miss the usual current path.
- GUI batch preprocessing/tracking uses process pools. The current CUDA path
  deliberately declines child processes, even on Windows where workers are
  spawned. Interactive processing threads can use it.

## Evidence from this machine

The host is Windows, i7-13620H (16 logical processors), RTX 4070 Laptop GPU
(8 GiB, driver 581.57), and Intel UHD Graphics (driver 32.0.101.6790). Python
3.13.4, NumPy 2.5.3 and OpenCV 5.0.0.93 are installed. CuPy was absent.
OpenCV reports zero CUDA devices but a working OpenCL backend. PyOpenCL found
both Intel and NVIDIA GPU devices using their existing drivers.

The local dataset contains RT_25_10 groups; the earlier reports used RT_26_2
groups on an RTX 5070 Ti. Those historical measurements do not describe this
machine or establish a current speedup. The old 26% preprocessing improvement
also combined CUDA, reduced copying and changed registration; it was not an
isolated GPU speedup.

An initial cProfile run over two groups, twice each, identified main-root
geometry and junction/history routing as substantial tracking costs. Lightweight
wall timers were then used to avoid drawing latency conclusions from cProfile.
The timer records include nested operations: do not add their durations to
their enclosing stage. Tracking includes annotated JPEG exports; the focused
wall-timer runs exclude cache writes and GUI/startup time.

| Local group | Preparation | Registration | Tracking + JPEG export |
| --- | ---: | ---: | ---: |
| RT_25_10-1, four images, two runs | 3.90–5.55 s | 1.74–2.02 s | 20.02–20.44 s |
| RT_25_10-10, three images, two runs | 2.13–3.17 s | 0.34–0.97 s | 3.03–11.66 s |

For the four-image group, main-root geometry used 4.04–6.41 s and root linking
4.69–7.52 s inside tracking. Skeletonization was 0.90–0.97 s. Registration spent
essentially all matching time in the coarse/local path on these runs. Wide
latency variation means these are hotspot evidence, not performance guarantees.

One attempt to use the full cache-writing benchmark stopped on its third group
run with a temporary NPZ cache-write error (`Errno 22`). Two earlier group runs
completed. The focused timing script therefore measures computation and JPEG
export independently of cache writes. It never reads or writes dataset caches.

### Minimal OpenCL median experiment

The existing CUDA kernel was ported mechanically to OpenCL C: pointer address
spaces and thread indexing changed; histogram logic, integer arithmetic,
replicated borders and saturated subtraction were retained. This is a research
prototype, without production validation, fallback handling or concurrency
management. The application never calls it.

Both GPU devices passed 24 synthetic comparisons (12 inputs, median and
subtraction checked separately), including one-pixel dimensions and tile
boundaries. Each also matched the CPU output exactly on the first processed
image of two real groups, across four repetitions per image. Real inputs were
2444×2679×3 and 2459×2684×3, approximately 6.6 megapixels each. This validates
those cases, not every device or downstream pipeline result.

| Device/run | CPU median + subtraction, median of four | OpenCL median + subtraction, median of four |
| --- | ---: | ---: |
| NVIDIA, RT_25_10-1 | 0.143 s | 0.109 s |
| NVIDIA, RT_25_10-10 | 0.149 s | 0.103 s |
| Intel, RT_25_10-1 | 0.764 s | 0.706 s |
| Intel, RT_25_10-10 | 0.821 s | 0.680 s |

Transfers and blocking completion are included. The synthetic checks warm the
kernel before real-image measurements; the first large-image GPU call also
allocates buffers. Context/program setup took 2.51 s on NVIDIA and 7.74 s on
Intel, excluding Python imports. Driver/compiler caches were not cleared.

CPU times varied sharply between and within sessions. Compare each GPU only
against its interleaved CPU measurements; these runs are not a controlled Intel
versus NVIDIA comparison. NVIDIA's median reduction was about 24–31% for this
operation, not the entire preprocessing stage. Roughly 34–46 ms saved per
image would not repay a 2.5-second initialization cost in a single short group.
This makes cold latency, reuse and workload-based selection essential.

## Backend choices and dependency costs

Sizes below are compressed Windows x64 wheels checked on PyPI on the
investigation date, not full installed footprints or dependency totals.

| Backend | Incremental packaging | Role and limitations |
| --- | --- | --- |
| Existing OpenCV CPU | None | Always available reference and fallback. |
| OpenCV `UMat` / OpenCL | No new Python package | Trial for chains of supported operations. OpenCV 5.0's OpenCL median accepts only 3×3 or 5×5; it cannot accelerate the 101×101 median. A `UMat` result alone does not prove GPU execution. |
| PyOpenCL 2026.1.4 | ~0.72 MB wheel, plus small support packages and existing NumPy; needs a working vendor OpenCL driver | Best first lightweight custom-kernel experiment for Windows/Linux Intel, AMD and NVIDIA. Intel and NVIDIA were exercised here; AMD was not. Driver availability must be probed. |
| `wgpu` 0.32.0 | ~3.5 MB wheel with native runtime, plus dependencies including CFFI/rendercanvas | Good candidate if Metal/macOS is a first-class requirement. Uses native Vulkan/Metal/DX12, without requiring a browser. Requires a WGSL rewrite and buffer-layout work; not benchmarked here. |
| Existing CuPy 14.2 | ~36 MB wheel **before** CUDA toolkit extra | Least new kernel work for NVIDIA; useful optional fast path, especially for future FFTs. CUDA 13 is not a universal hardware/driver choice. |
| Custom CUDA OpenCV build | Build/distribution maintenance | Poor default for this dependency goal. Standard `opencv-python` wheels are CPU builds; the `cv2.cuda` namespace does not establish CUDA support. |

PyOpenCL wheels bundle an ICD loader but not the vendor GPU implementation.
An OpenCL CPU device is not evidence that GPU acceleration is available.
The base app should remain usable without any GPU packages or drivers, and
should not install packages or drivers automatically.

Sources: [PyOpenCL installation](https://documen.tician.de/pyopencl/misc.html),
[PyOpenCL files](https://pypi.org/project/pyopencl/#files),
[OpenCV 5.0 median dispatch](https://github.com/opencv/opencv/blob/5.0.0/modules/imgproc/src/median_blur.dispatch.cpp),
[OpenCV wheel packaging](https://github.com/opencv/opencv-python),
[wgpu platform support](https://wgpu-py.readthedocs.io/en/stable/start.html),
[wgpu package](https://pypi.org/project/wgpu/),
[CuPy installation](https://docs.cupy.dev/en/stable/install.html),
[CuPy package](https://pypi.org/project/cupy-cuda13x/).

## What to accelerate first

1. **Median/background removal: moderate engineering effort.** There is already
   a small kernel and an exact CPU reference. Start with a PyOpenCL equivalent.
   Then investigate keeping the 5×5 smoothing and grayscale conversion on the
   same device, returning one grayscale array instead of a three-channel
   intermediate. Those additional kernels must reproduce OpenCV rounding and
   borders. Canny is a separate equivalence problem, not automatically safe to
   swap. Avoid mixing OpenCL/CUDA contexts inside one stage and paying for
   extra transfers.
2. **Registration's current local scorer: moderate effort.** Up to 17×17
   candidate shifts gather sparse edge pixels. A reduction kernel can compute
   overlap counts for all candidates, returning just those counts; the current
   CPU mean correction and ambiguity checks can remain authoritative. Benchmark
   the whole operation, including upload. Keep coarse search and full-search
   fallbacks. Only prioritize a CuPy FFT backend if new profiles show frequent
   full FFT fallback. Never remove near-tie handling merely to stay on GPU.
3. **Tracking: first reduce CPU work.** `select_main_geometry`, junction routing
   and temporal support combine Python sets/dictionaries, graph dependencies,
   contour fragments and many small cKDTree queries. GPU conversion needs
   compact arrays, batched queries and fewer CPU/device boundaries first.
   Cache/reuse contour paths and spatial-query inputs where valid; reduce
   repeated full-image scans. Preserve chronological root ownership. GPU
   thresholding alone will not accelerate these costs.
4. **Skeletonization/connected components: higher effort, later.** Iterative
   topology-sensitive kernels and variable-size component output need more
   validation. Current component crops already reduce work. Prioritize only if
   measurements on broader datasets justify the complexity. JPEG/file/cache
   costs remain separate from these GPU targets.

## What adaptive selection should mean

Use a small internal service returning ordinary owning NumPy arrays. Keep
vendor-specific objects out of `ImageData`, cache serialization and UI code.
Select per operation rather than setting one global array backend.

1. Discover installed candidates lazily off the UI thread. Probe actual device
   access, compile a tiny kernel and check its result. Enumerate multiple GPUs;
   the first OpenCL device here was Intel, not NVIDIA.
2. Reject unsupported dtypes/shapes/kernel sizes or insufficient memory before
   dispatch. Estimate buffers, temporary workspace and headroom. Include shared
   memory pressure for integrated GPUs. Small operations should stay on CPU.
3. Compare CPU and candidate GPU on bounded representative workloads. Measure
   upload + execution + synchronized download, with cold initialization reported
   separately. Do not stall the first user action for exhaustive calibration.
   Use a background/explicit optimization check and reuse the result.
4. Store choices by operation, size bucket, device/driver, library and kernel
   version. Require a worthwhile advantage (for example 15–20% plus an absolute
   time threshold); this is a policy proposal, not a measured universal cutoff.
   Invalidate stale results and avoid switching on every noisy sample. Power
   state and sustained contention can justify a later bounded recheck.
5. Preserve CPU input until success. On allocation, build, execution or device
   failure, retry that operation on CPU and disable the affected candidate for
   the session. Treat size-specific allocation failure separately from a broken
   device; do not endlessly retry it on every frame.
6. Show **Auto / CPU** and a compact status explaining the chosen backend and
   fallback reason. Optional diagnostics can allow a device override, memory
   budget and rerun of calibration. An override still needs safe fallback.

For batch mode, use one persistent GPU owner/queue per selected device with
bounded in-flight work, while CPU workers do decoding and graph processing.
Do not simply enable GPU initialization in every process-pool child. Benchmark
IPC/shared-memory transfer costs and throughput against the existing CPU pool;
a GPU queue can otherwise become the bottleneck. Preserve ordered tracking
within each series and bound aggregate host/device memory.

## Implementation gates

- First production increment: optional OpenCL median, CPU fallback, diagnostic
  status and process-safe ownership. No blanket GPU setting for all operations.
- Next: calibration/cache and buffer reuse, followed by a batch scheduling
  experiment. Add other kernels only when their end-to-end measurements win.
- Compare full arrays, registration offsets, root assignments and final
  measurements on the same current checkout/dataset. Existing benchmark
  fingerprints and CUDA/registration tests provide starting points.
- Cover missing packages/drivers, multiple GPUs, integrated-only devices,
  tiny/noncontiguous/read-only images, borders, allocation failure, runtime
  failure, near-tied registration, concurrent GUI calls and spawned workers.
- Measure cold first group, repeated groups and batch throughput separately.
  Report transferred bytes, peak memory and end-to-end elapsed time. AMD and
  macOS need actual hardware validation before claiming support.

The uncertainty is mainly performance portability and integration, not whether
a lightweight kernel can run: the two-device experiment establishes the latter.
