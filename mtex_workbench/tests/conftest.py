import os

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def voronoi_micrograph(n_seeds=40, size=160, boundary=1, seed=3):
  """a synthetic micrograph: bright Voronoi grains separated by dark boundaries, and the
  true number of cells"""
  rng = np.random.default_rng(seed)
  seeds = rng.uniform(0, size, (n_seeds, 2))
  yy, xx = np.mgrid[0:size, 0:size]
  d = (yy[..., None] - seeds[:, 0]) ** 2 + (xx[..., None] - seeds[:, 1]) ** 2
  lab = np.argmin(d, axis=-1)
  edge = np.zeros_like(lab, bool)
  for k in range(1, boundary + 1):
    edge[:-k] |= lab[:-k] != lab[k:]
    edge[:, :-k] |= lab[:, :-k] != lab[:, k:]
  img = np.where(edge, 0.15, 0.85) + rng.normal(0, 0.03, lab.shape)
  return np.clip(img, 0, 1), len(np.unique(lab))


@pytest.fixture
def micrograph_png(tmp_path):
  from PIL import Image
  img, n = voronoi_micrograph()
  path = tmp_path / "micrograph.png"
  Image.fromarray((img * 255).astype(np.uint8)).save(path)
  return path, n
