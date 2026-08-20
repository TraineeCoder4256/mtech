"""Device selection, shared by every entry point.

One helper so that "run it on the GPU" is a single flag everywhere rather
than something each script decides for itself.  ``--device auto`` picks CUDA
if it is there, then Apple MPS, then CPU.
"""

import torch


def pick_device(name=None):
    """Resolve a --device string to a torch device string."""
    if name and name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe(dev):
    """One line naming the hardware, for the top of a run log."""
    if dev.startswith("cuda"):
        i = torch.cuda.current_device()
        p = torch.cuda.get_device_properties(i)
        return (f"{dev}: {p.name}, {p.total_memory / 2**30:.1f} GiB, "
                f"{p.multi_processor_count} SMs, "
                f"torch {torch.__version__} (cuda {torch.version.cuda})")
    if dev.startswith("mps"):
        return f"{dev}: Apple GPU, torch {torch.__version__}"
    return (f"{dev}: {torch.get_num_threads()} threads, "
            f"torch {torch.__version__}")
