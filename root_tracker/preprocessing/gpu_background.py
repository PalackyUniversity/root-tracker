"""Optional exact uint8 CUDA median subtraction, with a transparent CPU fallback.

Install requirements-gpu.txt to enable; ROOT_TRACKER_GPU=0 forces the CPU path.
CUDA is initialized lazily and never in multiprocessing batch children.
"""
import logging
from multiprocessing import current_process
import os
import threading

import numpy as np

log = logging.getLogger(__name__)
_backend = None
_unavailable = False
_initialization_lock = threading.Lock()

# Each thread walks a vertical tile, maintaining a 256-bin histogram. Adjacent
# threads read adjacent channels/pixels; replicated borders match medianBlur.
# 101x101 counts fit in uint16. Tracking the current median avoids sorting every
# window. Fused saturated subtraction also avoids a host-side BGR temporary.
_KERNEL = r"""
extern "C" __global__ void median_columns(const unsigned char *src, unsigned char *dst,
 int h, int w, int channels, int k, int tile, int subtract) {
 int job = blockIdx.x * blockDim.x + threadIdx.x;
 int xc = job % (w * channels), start = (job / (w * channels)) * tile;
 if (start >= h) return;
 int x = xc / channels, c = xc % channels, r = k/2, rank = k*k/2;
 unsigned short hist[256];
 for (int i=0; i<256; ++i) hist[i]=0;
 for (int dy=-r; dy<=r; ++dy) {
  int y=max(0,min(h-1,start+dy));
  for (int dx=-r; dx<=r; ++dx) {
   int xx=max(0,min(w-1,x+dx));
   ++hist[src[(y*w+xx)*channels+c]];
  }
 }
 int med=0,below=0;
 while (below+hist[med]<=rank) below+=hist[med++];
 for (int y=start;y<min(h,start+tile);++y) {
  int index = (y*w+x)*channels+c;
  dst[index] = subtract ? max(0, (int)src[index]-med) : med;
  if (y+1==min(h,start+tile)) break;
  int oldy=max(0,y-r),newy=min(h-1,y+r+1);
  for (int dx=-r;dx<=r;++dx) {
   int xx=max(0,min(w-1,x+dx));
   int a=src[(oldy*w+xx)*channels+c],b=src[(newy*w+xx)*channels+c];
   if(a!=b) { --hist[a]; ++hist[b]; below-=(a<med);below+=(b<med); }
  }
  while(below>rank) below-=hist[--med];
  while(below+hist[med]<=rank) below+=hist[med++];
 }
}

"""


class _CudaMedian:
    def __init__(self):
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError('No CUDA device available')
        self.cp = cp
        self.kernel = cp.RawKernel(_KERNEL, 'median_columns')
        self.lock = threading.Lock()
        self.capacity = 0
        self.source = self.destination = None

    def filter(self, image: np.ndarray, kernel: int, *, subtract: bool) -> np.ndarray:
        if (image.dtype != np.uint8 or image.ndim not in (2, 3) or not image.size
                or kernel < 1 or kernel > 255 or kernel % 2 != 1):
            raise ValueError('Expected a nonempty uint8 image and odd kernel <=255')
        image = np.ascontiguousarray(image)
        h, w = image.shape[:2]
        channels = image.shape[2] if image.ndim == 3 else 1
        with self.lock:
            # Reuse two geometrically sized buffers across images, groups and
            # GUI worker threads; do not retain one allocation per image shape.
            if image.size > self.capacity:
                self.capacity = 1 << (int(image.size)-1).bit_length()
                self.source = self.cp.empty(self.capacity, dtype=self.cp.uint8)
                self.destination = self.cp.empty_like(self.source)
            source = self.source[:image.size].reshape(image.shape)
            destination = self.destination[:image.size].reshape(image.shape)
            source.set(image)
            tile = 256
            jobs = ((h + tile - 1) // tile) * w * channels
            self.kernel(((jobs + 127) // 128,), (128,),
                        (source, destination, np.int32(h), np.int32(w),
                         np.int32(channels), np.int32(kernel), np.int32(tile), np.int32(subtract)))
            # Blocking copy completes work before buffers are reused/unlocked.
            return self.cp.asnumpy(destination)


def _get_backend():
    global _backend, _unavailable
    with _initialization_lock:
        if _unavailable:
            return None
        if _backend is None:
            try:
                _backend = _CudaMedian()
            except Exception as error:
                _unavailable = True
                log.debug('CUDA median unavailable; using CPU: %s', error)
        return _backend


def try_subtract_median(image: np.ndarray, kernel: int) -> np.ndarray | None:
    """Return exact median-subtracted pixels, or None to request CPU processing."""
    global _unavailable, _backend
    if (os.environ.get('ROOT_TRACKER_GPU', 'auto').lower() in ('0', 'off', 'false', 'cpu')
            or current_process().name != 'MainProcess'
            or image.ndim not in (2, 3) or image.dtype != np.uint8
            or image.shape[0] * image.shape[1] < 1_000_000
            or image.size >= 2**31 or kernel != 101):
        return None
    backend = _get_backend()
    if backend is None:
        return None
    try:
        return backend.filter(image, kernel, subtract=True)
    except Exception as error:
        # The original photograph is unchanged, so the CPU can safely retry.
        with _initialization_lock:
            _unavailable = True
            _backend = None
        log.warning('CUDA median failed; continuing on CPU: %s', error)
        return None
