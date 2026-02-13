"""
Profiling utilities for tracking processing performance.

This module provides always-on profiling for image processing operations,
with support for multiprocessing aggregation.
"""

import functools
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class TimingRecord:
    """A single timing record for a profiling operation.

    Attributes:
        operation: Name of the operation (e.g., "ImageCropper.rotate")
        duration_ms: Duration in milliseconds
        image_count: Number of images processed in this operation
    """
    operation: str
    duration_ms: float
    image_count: int = 1


class ProfileRecords:
    """Container for collecting and aggregating profiling timing data.

    This class is pickle-safe and can be transported across processes
    via to_dict()/from_dict() methods.
    """

    def __init__(self) -> None:
        """Initialize an empty profile records container."""
        self.records: list[TimingRecord] = []

    def add(self, operation: str, duration_ms: float, image_count: int = 1) -> None:
        """Add a timing record.

        Args:
            operation: Name of the operation
            duration_ms: Duration in milliseconds
            image_count: Number of images processed (default: 1)
        """
        self.records.append(TimingRecord(operation, duration_ms, image_count))

    def to_dict(self) -> dict:
        """Convert to plain dict for multiprocessing transport.

        Returns:
            Dictionary with records data suitable for pickling.
        """
        return {
            'records': [
                {
                    'operation': r.operation,
                    'duration_ms': r.duration_ms,
                    'image_count': r.image_count
                }
                for r in self.records
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ProfileRecords':
        """Recreate ProfileRecords from dict.

        Args:
            data: Dictionary created by to_dict().

        Returns:
            A new ProfileRecords instance with the restored data.
        """
        records = cls()
        for r in data.get('records', []):
            records.add(r['operation'], r['duration_ms'], r['image_count'])
        return records

    def update_from_dict(self, data: dict) -> None:
        """Merge records from another ProfileRecords (as dict).

        Args:
            data: Dictionary from to_dict() of another instance.
        """
        other = self.from_dict(data)
        self.records.extend(other.records)

    def print_report(self) -> None:
        """Print aggregate statistics grouped by operation."""
        if not self.records:
            return

        # Group by operation
        by_op: dict[str, list[TimingRecord]] = {}
        for r in self.records:
            if r.operation not in by_op:
                by_op[r.operation] = []
            by_op[r.operation].append(r)

        print("\n" + "=" * 60)
        print("PROFILING REPORT")
        print("=" * 60)

        # Sort by total time
        totals = {
            op: sum(r.duration_ms for r in recs)
            for op, recs in by_op.items()
        }
        sorted_ops = sorted(totals.items(), key=lambda x: x[1], reverse=True)

        for op, total_ms in sorted_ops:
            recs = by_op[op]
            total_images = sum(r.image_count for r in recs)
            avg_ms = total_ms / len(recs)
            avg_per_image = total_ms / total_images if total_images > 0 else 0

            print(f"\n{op}:")
            print(f"  Calls: {len(recs)}")
            print(f"  Total time: {total_ms / 1000:.2f}s")
            print(f"  Avg time per call: {avg_ms:.1f}ms")
            if total_images > 1:
                print(f"  Avg time per image: {avg_per_image:.1f}ms")
            print(f"  Total images: {total_images}")

        print("\n" + "=" * 60)


def profile_operation(operation_name: str) -> Callable:
    """Decorator to profile a method and add records to self._profile_records.

    This decorator times the execution of a method and adds the timing
    information to the ProfileRecords instance stored in self._profile_records.

    Args:
        operation_name: Name for this operation (e.g., "rotate", "threshold")

    Returns:
        A decorator function.

    Example:
        @profile_operation("rotate")
        def rotate(self, image: np.ndarray) -> np.ndarray:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            # Skip profiling if no profile_records attribute exists
            if not hasattr(self, '_profile_records') or self._profile_records is None:
                return func(self, *args, **kwargs)

            start = time.perf_counter()
            try:
                result = func(self, *args, **kwargs)
                return result
            finally:
                duration_ms = (time.perf_counter() - start) * 1000

                # Determine image count (heuristic based on first argument)
                image_count = 1
                if args and isinstance(args[0], (list, tuple)):
                    # First arg might be a list (e.g., list of images or contours)
                    image_count = len(args[0]) if args[0] else 1

                self._profile_records.add(
                    f"{self.__class__.__name__}.{operation_name}",
                    duration_ms,
                    image_count
                )
        return wrapper
    return decorator
