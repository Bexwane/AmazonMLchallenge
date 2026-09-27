"""Memory measurement: current RSS and peak RSS (this process + finished children), in GiB."""
import os
import sys


def rss_gib():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 2 ** 30
    except ImportError:
        return float("nan")


def peak_gib():
    """Peak resident memory of this process and, separately, of its largest finished child (Linux)."""
    try:
        import resource
        scale = 2 ** 30 if sys.platform == "darwin" else 2 ** 20  # ru_maxrss: bytes on macOS, KiB on Linux
        me = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale
        kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / scale
        return {"self": round(me, 2), "children": round(kids, 2)}
    except ImportError:  # Windows
        try:
            import psutil
            return {"self": round(psutil.Process(os.getpid()).memory_info().peak_wset / 2 ** 30, 2), "children": None}
        except (ImportError, AttributeError):
            return {"self": None, "children": None}
