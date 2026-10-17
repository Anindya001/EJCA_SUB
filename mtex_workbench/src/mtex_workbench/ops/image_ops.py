"""Processing of plain micrographs: filtering, thresholding and geometric segmentation.

These operations never produce orientations. A micrograph segmented here yields grain
sizes and shapes, which the statistics operations treat exactly like EBSD grains."""
from __future__ import annotations

import numpy as np

from ..core.params import Bool, Choice, Float, Int
from ..core.registry import Output, Port, operation
from ..core.types import DataItem, DataKind
from .. import stats

IMAGE_IN = (Port("image", (DataKind.IMAGE,), "Image"),)
IMAGE_OUT = (Output("image", DataKind.IMAGE, "Image"),)


def _grey(item: DataItem) -> np.ndarray:
  a = np.asarray(item.value, float)
  if a.ndim == 3:
    a = a[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
  return a


@operation("image.gaussian", "Gaussian filter", "Image Processing",
           inputs=IMAGE_IN, outputs=IMAGE_OUT,
           params=(Float("sigma", "σ", 1.0, minimum=0.1, maximum=100, unit=" px"),))
def gaussian(image, sigma):
  """Convolution with an isotropic Gaussian of standard deviation σ pixels; suppresses
  noise at the cost of blurring boundaries narrower than about 2σ."""
  from scipy import ndimage
  return DataItem(DataKind.IMAGE, ndimage.gaussian_filter(_grey(image), sigma), dict(image.meta))


@operation("image.median", "Median filter", "Image Processing",
           inputs=IMAGE_IN, outputs=IMAGE_OUT,
           params=(Int("size", "Window", 3, minimum=3, maximum=51, unit=" px"),))
def median(image, size):
  """The median of a size × size window: removes salt-and-pepper noise while keeping edges."""
  from scipy import ndimage
  return DataItem(DataKind.IMAGE, ndimage.median_filter(_grey(image), size=size), dict(image.meta))


@operation("image.threshold", "Threshold", "Image Processing",
           inputs=IMAGE_IN, outputs=(Output("mask", DataKind.IMAGE, "Binary mask"),),
           params=(Choice("method", "Method", ("Otsu", "manual")),
                   Float("value", "Manual threshold", 0.5, minimum=0, maximum=1),
                   Bool("invert", "Grain interiors are dark", False,
                        help="tick when boundaries are bright and interiors dark")))
def threshold(image, method, value, invert):
  """A binary mask of the grain interiors. Otsu's method picks the threshold maximising the
  between-class variance; its effectiveness η ∈ [0, 1] (stored with the result) measures
  how well the histogram separates into two classes; below about 0.7 the segmentation
  deserves a visual check."""
  a = _grey(image)
  if method == "Otsu":
    t, eta = stats.otsu_threshold(a)
  else:
    t, eta = value, float("nan")
  mask = (a <= t) if invert else (a > t)
  return DataItem(DataKind.IMAGE, mask.astype(np.uint8),
                  {**image.meta, "binary": True, "threshold": t, "otsu_effectiveness": eta})


def region_table(labels: np.ndarray, pixel_size: float) -> dict:
  """one row per region: area, ECD and the Miles-Lantuéjoul weight

  A grain counts as whole only if it touches neither the first nor the last row or column.
  On a W × H pixel lattice a grain with a w × h bounding box can then take
  (W - w - 1)(H - h - 1) positions, so for grains placed independently of the window it is
  observed whole with probability proportional to that number: large grains are lost more
  often. Weighting each interior grain by W H / ((W - w - 1)(H - h - 1)) (Miles 1978;
  Lantuéjoul 1980) makes weighted means over the interior grains unbiased for the
  number-weighted population mean."""
  from scipy import ndimage
  H, W = labels.shape
  n = int(labels.max())
  ids = np.arange(1, n + 1)
  area_px = np.bincount(labels.ravel(), minlength=n + 1)[1:].astype(float)
  slices = ndimage.find_objects(labels)
  bh = np.array([s[0].stop - s[0].start if s else 0 for s in slices], float)
  bw = np.array([s[1].stop - s[1].start if s else 0 for s in slices], float)
  border = np.zeros(n + 1, bool)
  for edge in (labels[0], labels[-1], labels[:, 0], labels[:, -1]):
    border[edge] = True
  border = border[1:]
  free = (W - bw - 1) * (H - bh - 1)
  ml = np.full(n, np.nan)
  ok = ~border & (W - bw - 1 > 0) & (H - bh - 1 > 0)
  ml[ok] = (W * H) / free[ok]
  area = area_px * pixel_size ** 2
  return {"id": ids, "pixels": area_px.astype(int), "area": area,
          "ecd": 2 * np.sqrt(area / np.pi), "touches_border": border, "ml_weight": ml}


@operation("image.segment", "Segment grains", "Image Processing",
           inputs=(Port("mask", (DataKind.IMAGE,), "Binary mask"),),
           outputs=(Output("labels", DataKind.LABELS, "Grain labels"),
                    Output("table", DataKind.TABLE, "Grain table")),
           params=(Choice("connectivity", "Connectivity", ("4", "8"),
                          help="4: grains touching at a corner stay separate"),
                   Int("min_area", "Minimum area", 10, minimum=1, unit=" px")))
def segment(mask, connectivity, min_area):
  """Grains as the connected components of the binary mask of grain interiors. Regions
  smaller than the minimum area are discarded as noise. The table gives sizes in the
  physical unit of the image's pixel size, flags grains cut by the border and gives
  Miles–Lantuéjoul weights for an edge-corrected mean."""
  from scipy import ndimage
  m = np.asarray(mask.value) > 0
  structure = ndimage.generate_binary_structure(2, 1 if connectivity == "4" else 2)
  lab, n = ndimage.label(m, structure=structure)
  if n:
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    keep = sizes >= min_area
    keep[0] = False
    remap = np.zeros(n + 1, np.int32)
    remap[keep] = np.arange(1, keep.sum() + 1)
    lab = remap[lab]
  ps = float(mask.meta.get("pixel_size", 1.0))
  unit = mask.meta.get("unit", "px") if "pixel_size" in mask.meta else "px"
  table = region_table(lab, ps)
  return {"labels": DataItem(DataKind.LABELS, lab.astype(np.int32), dict(mask.meta)),
          "table": DataItem(DataKind.TABLE, table, {"units": {"area": f"{unit}²", "ecd": unit},
                                                    "sampling_unit": "grain"})}
