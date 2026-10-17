"""Grain reconstruction and grain-level tables."""
from __future__ import annotations

import numpy as np

from ..core.params import Float, Int
from ..core.registry import Output, Port, operation
from ..core.types import DataItem, DataKind


def _calc_grains_script(o, i, p):
  return "\n".join([
    f"{o['ebsd']} = {i['ebsd']}.copy()",
    f"{o['grains']} = calcGrains({o['ebsd']}, angle={p['angle']!r} * degree, "
    f"minPixel={p['min_pixel']}, alpha={p['alpha']!r})",
  ])


@operation("grains.calc", "Grain reconstruction", "Grain Analysis",
           inputs=(Port("ebsd", (DataKind.EBSD,), "EBSD map"),),
           outputs=(Output("grains", DataKind.GRAINS, "Grains"),
                    Output("ebsd", DataKind.EBSD, "EBSD map with grain ids")),
           params=(Float("angle", "Misorientation threshold", 10.0, minimum=0.1, maximum=62.8, unit="°",
                         help="neighbouring pixels of one phase misoriented by more are separated"),
                   Int("min_pixel", "Minimum grain size", 5, minimum=1, unit=" px",
                       help="smaller grains are dissolved into their neighbours"),
                   Float("alpha", "Unindexed closing radius α", 3.1, minimum=0, maximum=50,
                         help="unindexed seams narrower than about 2α pixel spacings are absorbed")),
           script=_calc_grains_script)
def calc_grains(ebsd, angle, min_pixel, alpha):
  """Grains as connected regions of one phase whose neighbouring pixels are misoriented by
  less than the threshold (MTEX calcGrains). The input map is not changed: a copy carrying
  the grain id of every pixel is returned as the second output, for GROD, GOS and
  grain-aware KAM. The threshold, α and the minimum size are segmentation choices, not
  measured properties; report them with every grain statistic."""
  from mtex import calcGrains, degree
  e = ebsd.value.copy()
  grains = calcGrains(e, angle=angle * degree, minPixel=min_pixel, alpha=alpha)
  meta = {"unit": ebsd.meta.get("unit", "µm"), "angle": angle, "min_pixel": min_pixel, "alpha": alpha}
  return {"grains": DataItem(DataKind.GRAINS, grains, meta),
          "ebsd": DataItem(DataKind.EBSD, e, {**ebsd.meta, "has_grain_id": True})}


@operation("grains.smooth_boundary", "Smooth grain boundaries", "Grain Analysis",
           inputs=(Port("grains", (DataKind.GRAINS,), "Grains"),),
           outputs=(Output("grains", DataKind.GRAINS, "Grains"),),
           params=(Int("iterations", "Iterations", 5, minimum=1, maximum=200),),
           script=lambda o, i, p: f"{o['grains']} = smoothBoundary({i['grains']}, {p['iterations']})")
def smooth_boundary(grains, iterations):
  """Boundaries smoothed by Laplacian iterations, removing the staircase of the scan grid.
  It changes perimeters and shape factors but not which pixel belongs to which grain."""
  from mtex import smoothBoundary
  return DataItem(DataKind.GRAINS, smoothBoundary(grains.value, iterations), dict(grains.meta))


def grain_table_values(grains) -> dict:
  """one row per indexed grain: geometry in the scan unit, GOS in degrees"""
  from mtex import degree
  g = grains["indexed"]

  def arr(name):
    v = getattr(g, name)
    return np.asarray(v() if callable(v) else v)

  phase_names = np.asarray(g.mineralList, dtype=object)
  area = arr("area").astype(float)
  return {
    "id": arr("id").astype(int),
    "phase": phase_names[arr("phaseId").astype(int)].astype(str),
    "pixels": arr("numPixel").astype(int),
    "area": area,
    "ecd": 2.0 * arr("equivalentRadius").astype(float),
    "perimeter": arr("perimeter").astype(float),
    "aspect_ratio": arr("aspectRatio").astype(float),
    "shape_factor": arr("shapeFactor").astype(float),
    "gos": arr("GOS").astype(float) / degree,
    "touches_border": arr("isBoundary").astype(bool),
  }


@operation("grains.table", "Grain table", "Grain Analysis",
           inputs=(Port("grains", (DataKind.GRAINS,), "Grains"),),
           outputs=(Output("table", DataKind.TABLE, "Grain table"),),
           script=lambda o, i, p: ("from mtex_workbench.ops.grain_ops import grain_table_values\n"
                                   f"{o['table']} = grain_table_values({i['grains']})"))
def grain_table(grains):
  """One row per indexed grain: pixel count, area, equivalent circle diameter (ECD),
  perimeter, aspect ratio, shape factor, grain orientation spread (GOS, °) and whether the
  grain is cut by the map border. Border grains are truncated, so their sizes are biased
  low; the statistics operations can exclude them."""
  unit = grains.meta.get("unit", "µm")
  return DataItem(DataKind.TABLE, grain_table_values(grains.value),
                  {"units": {"area": f"{unit}²", "ecd": unit, "perimeter": unit, "gos": "°"},
                   "sampling_unit": "grain"})
