import multiprocessing
import sys

from .cli import main

# the guard matters: the worker process is started with "spawn", which imports __main__ again
if __name__ == "__main__":
  multiprocessing.freeze_support()  # a frozen (PyInstaller) Windows build starts workers through this
  sys.exit(main())
