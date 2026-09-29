"""Research-only port of the existing CUDA median; never used by the app."""
import argparse
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import pyopencl as cl
from root_tracker.config import Config
from root_tracker.pipeline import RootTrackingPipeline
from root_tracker.preprocessing.gpu_background import _KERNEL
from root_tracker.preprocessing import background


class OpenCLMedian:
    def __init__(self, device_name=''):
        devices = [d for p in cl.get_platforms() for d in p.get_devices()
                   if d.type & cl.device_type.GPU]
        print('OpenCL GPUs:', [(d.name, d.driver_version) for d in devices], flush=True)
        self.device = next(d for d in devices if device_name.lower() in d.name.lower())
        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.ctx)
        source = _KERNEL.replace('extern "C" __global__', '__kernel')
        source = source.replace('const unsigned char *src, unsigned char *dst',
                                '__global const unsigned char *src, __global unsigned char *dst')
        source = source.replace('blockIdx.x * blockDim.x + threadIdx.x', '(int)get_global_id(0)')
        self.program = cl.Program(self.ctx, source).build()
        self.kernel = cl.Kernel(self.program, 'median_columns')
        self.capacity = 0

    def filter(self, image, kernel=101, subtract=True):
        image = np.ascontiguousarray(image)
        if image.nbytes > self.capacity:
            self.capacity = 1 << (image.nbytes - 1).bit_length()
            self.src = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY, self.capacity)
            self.dst = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, self.capacity)
        h, w = image.shape[:2]
        channels = image.shape[2] if image.ndim == 3 else 1
        tile = 256
        jobs = ((h + tile - 1) // tile) * w * channels
        cl.enqueue_copy(self.queue, self.src, image, is_blocking=True)
        self.kernel(self.queue, (((jobs + 127) // 128) * 128,), (128,), self.src, self.dst,
                    *map(np.int32, (h, w, channels, kernel, tile, subtract)))
        out = np.empty_like(image)
        cl.enqueue_copy(self.queue, out, self.dst, is_blocking=True)
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='')
    args = parser.parse_args()
    start = time.perf_counter()
    backend = OpenCLMedian(args.device)
    startup = time.perf_counter() - start
    rng = np.random.default_rng(10)
    checks = 0
    for shape in ((1, 17), (23, 1, 3), (39, 57, 3), (519, 19, 3)):
        for kind in ('random', 'zero', 'white'):
            image = (rng.integers(0, 256, shape, dtype=np.uint8) if kind == 'random'
                     else np.full(shape, 255 if kind == 'white' else 0, np.uint8))
            expected = cv2.medianBlur(image, 101)
            np.testing.assert_array_equal(backend.filter(image, subtract=False), expected)
            np.testing.assert_array_equal(backend.filter(image), cv2.subtract(image, expected))
            checks += 2
    config = Config.from_yaml(ROOT / 'configs/in_vitro.yaml')
    pipeline = RootTrackingPipeline(config)
    groups = pipeline.loader.create_series()
    records = []
    for group in ('RT_25_10-1', 'RT_25_10-10'):
        captured = []
        def capture(image, kernel):
            captured.append(image.copy())
            return None
        with patch.object(background, 'try_subtract_median', capture):
            pipeline.preprocess_image(groups[group].images[0])
        image = captured[0]
        timings = {'cpu': [], 'opencl': []}
        for repeat in range(4):
            outputs = {}
            for mode in (('cpu', 'opencl') if repeat % 2 == 0 else ('opencl', 'cpu')):
                begin = time.perf_counter()
                outputs[mode] = (cv2.subtract(image, pipeline.background_remover._median_background(image))
                                 if mode == 'cpu' else backend.filter(image))
                timings[mode].append(time.perf_counter() - begin)
            np.testing.assert_array_equal(outputs['cpu'], outputs['opencl'])
        row = {'group': group, 'shape': list(image.shape), 'seconds': timings, 'exact': True}
        records.append(row)
        print(json.dumps(row), flush=True)
    result = {'device': backend.device.name, 'driver': backend.device.driver_version,
              'startup_seconds': startup, 'synthetic_checks': checks, 'records': records,
              'note': 'Transfers included; warm kernels after synthetic checks; CPU strip implementation.'}
    Path(args.output).write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
