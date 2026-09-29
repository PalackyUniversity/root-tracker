"""Conservative batch concurrency without an optional monitoring dependency."""
import logging
import os
import sys

from PIL import Image

log = logging.getLogger(__name__)
_GIB = 1024 ** 3


def system_memory() -> tuple[int, int] | None:
    """Return total/available physical bytes; unknown platforms stay serial."""
    try:
        if sys.platform == 'win32':
            import ctypes
            from ctypes import wintypes
            class MemoryStatus(ctypes.Structure):
                _fields_ = [('length', wintypes.DWORD), ('load', wintypes.DWORD)] + [
                    (name, ctypes.c_ulonglong) for name in (
                        'total', 'available', 'commit_limit', 'commit_available',
                        'virtual', 'virtual_available', 'extended')]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.total, min(status.available, status.commit_available)
        elif sys.platform.startswith('linux'):
            with open('/proc/meminfo') as stream:
                values = {key: int(value.split()[0]) * 1024
                          for key, value in (line.split(':', 1) for line in stream)}
            return values['MemTotal'], values['MemAvailable']
    except (OSError, ValueError, KeyError):
        pass
    return None


def estimate_series_memory(series) -> int:
    """Budget retained frames plus CV workspaces, using headers, never decoding.

    Full photo dimensions intentionally overestimate cropped work. This is a
    scheduling estimate, not an allocation guarantee for arbitrary root graphs.
    """
    pixels = []
    for image in series.images:
        if image.image is not None:
            h, w = image.image.shape[:2]
        else:
            try:
                with Image.open(image.path) as source:
                    w, h = source.size
            except (OSError, ValueError):
                w, h = 8000, 4000
        pixels.append(w * h)
    # Native imports/allocators, retained color and gray frames, label maps,
    # float workspaces, segmentation and temporary annotations. Include margin
    # for unmasked tracking bootstrap and Python geometry objects.
    return max(_GIB, _GIB // 2 + 12 * sum(pixels) + 80 * max(pixels, default=0))


def batch_worker_count(series, max_workers: int | None = None) -> int:
    """Cap CPU concurrency by current available RAM and the largest group."""
    if max_workers is not None and max_workers < 1:
        raise ValueError('max_workers must be positive')
    groups = list(series)
    cpu_limit = min(os.cpu_count() or 1, max_workers or (os.cpu_count() or 1),
                    max(1, len(groups)))
    memory = system_memory()
    if memory is None or not groups:
        return 1
    total, available = memory
    reserve = min(2 * _GIB, total // 10)
    per_worker = max(estimate_series_memory(group) for group in groups)
    workers = max(1, min(cpu_limit, max(0, available - reserve) // per_worker))
    log.info('Batch workers: %s (available %.2f GiB, budget %.2f GiB/worker)',
             workers, available / _GIB, per_worker / _GIB)
    return workers


def batch_thread_count(workers: int) -> int:
    """Give a memory-limited pool some native parallelism without multiplying it."""
    return min(4, max(1, (os.cpu_count() or 1) // max(1, workers)))


def initialize_batch_worker(threads: int = 1) -> None:
    """Bound each process's OpenCV pool by its share of the CPU budget."""
    import cv2
    cv2.setNumThreads(threads)
