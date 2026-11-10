"""Run PIRB with its default allocator, isolated from training CUDA settings."""

from pathlib import Path
import os
import runpy
import sys


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        raise ValueError("Expected the benchmark entry point and its arguments")
    # In WSL, expandable_segments inherited from the training service can
    # fail after the Java runtime starts. Clear both aliases before PIRB's
    # first PyTorch import; keep training and evaluation in separate processes.
    keys = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")
    previous_allocator = {key: os.environ.pop(key) for key in keys if key in os.environ}
    print("[PIRB] Default CUDA allocator; training allocator settings removed", flush=True)
    target = str(Path(arguments[0]).resolve())
    previous_argv, previous_path = sys.argv, sys.path[:]
    try:
        sys.argv = [target, *arguments[1:]]
        sys.path.insert(0, str(Path(target).parent))
        runpy.run_path(target, run_name="__main__")
    finally:
        sys.argv = previous_argv
        sys.path[:] = previous_path
        for key in keys:
            os.environ.pop(key, None)
        os.environ.update(previous_allocator)


if __name__ == "__main__":
    main()
