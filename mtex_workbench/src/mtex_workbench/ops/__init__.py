"""The built-in operations; importing this package registers them.

MTEX is imported inside the operation functions, not here, so the interface starts and
lists the operations without paying for MTEX's import until the first computation (which
normally runs in the worker process anyway).
"""
from . import io_ops, ebsd_ops, grain_ops, texture_ops, image_ops, stats_ops  # noqa: F401

# the sample data sets of MTEX that are EBSD maps
SAMPLE_EBSD = io_ops.SAMPLE_EBSD
