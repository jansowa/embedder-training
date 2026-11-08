"""Release tooling: exporting trained SPLADE models to deployable formats (ONNX) and benchmarking them.

Kept apart from ``training`` and ``distill``: it reads finished checkpoints and never trains. Heavy
imports (torch, onnx, onnxruntime) stay inside the functions that need them.
"""
