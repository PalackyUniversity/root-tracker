"""Bounded, in-memory preview preparation; background tasks never touch Qt."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from threading import RLock


class PreviewCache:
    def __init__(self, max_bytes=256*1024*1024, max_entries=3):
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self._entries = OrderedDict()
        self._pending = {}
        self._lock = RLock()
        self._executor = None
        self._generation = 0
        self._closed = False

    def _store(self, key, result, generation):
        if result is None:
            return
        with self._lock:
            if self._closed or generation != self._generation:
                return
            self._entries[key] = result
            self._entries.move_to_end(key)
            while self._entries and (len(self._entries) > self.max_entries or
                    sum(v[0].nbytes for v in self._entries.values()) > self.max_bytes):
                self._entries.popitem(last=False)

    def _prepare(self, key, loader, generation):
        result = loader()
        self._store(key, result, generation)
        return result

    def get(self, key, loader):
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return self._entries[key]
            future = self._pending.get(key)
            generation = self._generation
        if future is not None and not future.cancelled():
            try:
                return future.result()
            except Exception:
                # A speculative failure must not prevent a foreground retry.
                pass
        return self._prepare(key, loader, generation)

    def prefetch(self, requests):
        with self._lock:
            if self._closed:
                return
            wanted = {key for key, _ in requests}
            for key, future in list(self._pending.items()):
                if future.done() or (key not in wanted and future.cancel()):
                    self._pending.pop(key, None)
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='preview')
            for key, loader in requests:
                if key in self._entries or key in self._pending or len(self._pending) >= 2:
                    continue
                self._pending[key] = self._executor.submit(self._prepare, key, loader, self._generation)

    def clear(self):
        with self._lock:
            self._generation += 1
            self._entries.clear()
            for future in self._pending.values():
                future.cancel()
            self._pending.clear()

    def close(self):
        with self._lock:
            self._closed = True
            self.clear()
            if self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)
