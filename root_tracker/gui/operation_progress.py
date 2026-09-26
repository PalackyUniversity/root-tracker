"""Weighted progress for one user request spanning several processing workers."""
import time


class OperationProgress:
    # Relative costs from the measured pipeline; fractions are estimates, not
    # equal-weight steps (barcode detection is much cheaper than preprocessing).
    WEIGHTS = {'barcode': .2, 'preprocess': 2.2, 'track': 1.7}
    LABELS = {'barcode': 'Reading barcodes', 'preprocess': 'Preparing images',
              'track': 'Tracking roots'}

    def __init__(self, operations):
        self.operations = operations
        self.started = time.monotonic()
        self.index = 0
        self.fraction = 0.0
        self.value = 0.0
        self.label = self.LABELS[operations[0]]

    def begin(self, operation):
        self.index = self.operations.index(operation)
        self.fraction = 0.0
        self.label = self.LABELS[operation]
        self.update(0, 100)

    def update(self, current, total):
        # A worker's final signal can precede cache writes / GUI completion.
        self.fraction = max(self.fraction, min(.99, max(0, current / total) if total else 0))
        weights = [self.WEIGHTS[op] for op in self.operations]
        done = sum(weights[:self.index]) + weights[self.index] * self.fraction
        self.value = max(self.value, min(.99, done / sum(weights)))

    def remaining(self):
        if self.value <= 0:
            return None
        return max(0, (time.monotonic() - self.started) * (1 - self.value) / self.value)
