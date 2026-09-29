# Research evidence, 2026-09-29

See [the investigation](../../adaptive-gpu-feasibility.md) for interpretation,
hardware, versions, limitations and recommended production work.

- `probe_opencl.py`: experimental mechanical port of the existing CUDA kernel.
  Requires PyOpenCL and a vendor GPU OpenCL driver in addition to app dependencies.
  No application code imports it. No production fallback or thread-safety contract.
- `opencl-median.json`: Intel UHD probe, including interleaved CPU comparisons.
- `opencl-nvidia.json`: NVIDIA probe, including interleaved CPU comparisons.
- `profile_current.py`: lightweight inclusive wall timers, two local groups,
  two repetitions, fresh processing, no processed-cache reads/writes. JPEG
  exports use a temporary directory. Nested durations are not additive.
- `wall-timers.json`: complete output from that focused profiler.
- `cprofile-summary.json`: filtered Python profile used to locate hotspots;
  four group runs. Do not use its instrumented times as unprofiled latency.
- `current-wall.json`: two completed runs from the earlier cache-writing
  benchmark attempt. The third run failed during a temporary cache write and
  is absent. This file is not the complete focused-profiler result.

Run from the repository root, in an environment containing the app requirements
and (for the OpenCL probe) PyOpenCL:

```powershell
python docs/performance-data/gpu-feasibility-2026-09-29/probe_opencl.py --device NVIDIA --output nvidia-probe.json
python docs/performance-data/gpu-feasibility-2026-09-29/probe_opencl.py --device Intel --output intel-probe.json
python docs/performance-data/gpu-feasibility-2026-09-29/profile_current.py
```

The scripts select `RT_25_10-1` and `RT_25_10-10` from `configs/in_vitro.yaml`;
adapt these selections for other datasets. The profiler overwrites
`wall-timers.json`. The OpenCL probe depends on the current private CUDA kernel
source, so it should be reviewed again after changing that kernel.

For this investigation only, additional packages were installed under
`.venv/gpu-feasibility-research/probe-env`, with the application environment left
unchanged. These isolated packages and raw profiling scratch files are ignored
by Git. The original scratch scripts there predate relocation; use the research
scripts in this directory for reproduction.
