"""MTEX Workbench: a non-destructive analysis environment around MTEX for Python.

Layers, each importing only those above it

  core     data kinds, parameters, the operation registry, the processing graph,
           the executors, project storage, provenance and script export
  stats    the statistical methods (pure NumPy/SciPy, no MTEX)
  ops      the operations: thin, typed wrappers over MTEX and the stats layer
  viz      figures for every data kind
  gui      the PySide6 desktop application
  cli      headless runs, batch processing and reports
"""

__version__ = "0.1.0"
