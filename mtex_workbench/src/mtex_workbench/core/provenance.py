"""What is needed to reproduce a result: the software versions and the input file contents."""
from __future__ import annotations

import functools
import hashlib
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

PACKAGES = ("mtex", "numpy", "scipy", "matplotlib", "numba", "h5py", "PySide6", "mtex-workbench")


def sha256_file(path, chunk: int = 1 << 20) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for block in iter(lambda: f.read(chunk), b""):
      h.update(block)
  return h.hexdigest()


@functools.lru_cache(maxsize=1)
def environment() -> dict:
  env = {"python": sys.version.split()[0], "platform": platform.platform()}
  for pkg in PACKAGES:
    try:
      env[pkg] = version(pkg)
    except PackageNotFoundError:
      env[pkg] = None
  if env["mtex-workbench"] is None:  # running from a source checkout
    from .. import __version__
    env["mtex-workbench"] = __version__ + "+src"
  return env
