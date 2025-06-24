import gc
import sys
from typing import Dict, Any, List, Tuple
import traceback


def get_object_size(obj: Any, seen: set | None = None) -> int:
    """Calculate the total memory size of an object and all objects it references.

    Args:
        obj: The object to measure
        seen: Set of object ids already counted (to avoid double counting)

    Returns:
        Total size in bytes
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0

    seen.add(obj_id)

    # Handle different container types
    if isinstance(obj, dict):
        for key, value in obj.items():
            size += get_object_size(key, seen)
            size += get_object_size(value, seen)
    elif hasattr(obj, '__dict__'):
        size += get_object_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        try:
            for item in obj:
                size += get_object_size(item, seen)
        except TypeError:
            pass

    return size


def format_bytes(size_bytes: int) -> str:
    """Format bytes into human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


class MemoryTracker:
    """Track memory usage of specific objects over time."""

    def __init__(self):
        self.tracked_objects: Dict[str, List[Tuple[int, int]]] = {}  # name -> [(epoch, size_bytes)]
        self.snapshots: Dict[int, Dict[str, int]] = {}  # epoch -> {name: size_bytes}

    def track(self, name: str, obj: Any, epoch: int) -> int:
        """Track the memory usage of an object.

        Args:
            name: Name identifier for the object
            obj: The object to track
            epoch: Current epoch number

        Returns:
            Size in bytes
        """
        try:
            size = get_object_size(obj)

            if name not in self.tracked_objects:
                self.tracked_objects[name] = []
            self.tracked_objects[name].append((epoch, size))

            if epoch not in self.snapshots:
                self.snapshots[epoch] = {}
            self.snapshots[epoch][name] = size

            return size
        except Exception as e:
            print(f"Error tracking object '{name}': {e}")
            return 0

    def get_report(self, epoch: int) -> Dict[str, Any]:
        """Generate a memory report for the current epoch.

        Returns dictionary with:
        - current_sizes: Current size of each tracked object
        - growth: Growth since first measurement
        - growth_rate: Average growth per epoch
        """
        report = {
            "epoch": epoch,
            "current_sizes": {},
            "growth": {},
            "growth_rate": {},
            "formatted_sizes": {}
        }

        if epoch in self.snapshots:
            for name, size in self.snapshots[epoch].items():
                report["current_sizes"][name] = size
                report["formatted_sizes"][name] = format_bytes(size)

                # Calculate growth
                if name in self.tracked_objects and len(self.tracked_objects[name]) > 0:
                    first_epoch, first_size = self.tracked_objects[name][0]
                    growth = size - first_size
                    epochs_elapsed = epoch - first_epoch

                    report["growth"][name] = growth
                    if epochs_elapsed > 0:
                        report["growth_rate"][name] = growth / epochs_elapsed
                    else:
                        report["growth_rate"][name] = 0

        return report

    def print_report(self, epoch: int, logger=None):
        """Print a formatted memory report."""
        report = self.get_report(epoch)

        output = []
        output.append(f"\n{'='*60}")
        output.append(f"Memory Report - Epoch {epoch}")
        output.append(f"{'='*60}")

        if report["current_sizes"]:
            output.append("\nCurrent Memory Usage:")
            for name, size in sorted(report["current_sizes"].items()):
                formatted = report["formatted_sizes"][name]
                output.append(f"  {name:<30} {formatted:>15}")

            output.append("\nMemory Growth:")
            for name in sorted(report["growth"].keys()):
                growth = report["growth"][name]
                growth_rate = report["growth_rate"][name]
                output.append(f"  {name:<30} {format_bytes(growth):>15} ({format_bytes(growth_rate)}/epoch)")

        output.append(f"{'='*60}")

        message = "\n".join(output)
        if logger:
            logger.info(message)
        else:
            print(message)

    def get_potential_leaks(self, threshold_bytes_per_epoch: int = 1024 * 1024) -> List[str]:
        """Identify objects with concerning memory growth rates.

        Args:
            threshold_bytes_per_epoch: Growth rate threshold in bytes/epoch

        Returns:
            List of object names with high growth rates
        """
        leaks = []

        for name, measurements in self.tracked_objects.items():
            if len(measurements) < 2:
                continue

            first_epoch, first_size = measurements[0]
            last_epoch, last_size = measurements[-1]

            epochs_elapsed = last_epoch - first_epoch
            if epochs_elapsed > 0:
                growth_rate = (last_size - first_size) / epochs_elapsed
                if growth_rate > threshold_bytes_per_epoch:
                    leaks.append(name)

        return leaks
