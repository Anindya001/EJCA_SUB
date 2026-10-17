"""Pre-processing of EBSD maps and the local deformation measures."""
from __future__ import annotations

import numpy as np

from ..core.params import Bool, Choice, Float, Int, Text
from ..core.registry import Output, Port, operation
from ..core.types import DataItem, DataKind

EBSD_IN = (Port("ebsd", (DataKind.EBSD,), "EBSD map"),)
EBSD_OUT = (Output("ebsd", DataKind.EBSD, "EBSD map"),)


def _carry(item: DataItem, ebsd, **extra) -> DataItem:
  return DataItem(DataKind.EBSD, ebsd, {**item.meta, **extra})


def _phase_names(ebsd) -> list[str]:
  return [m for m in ebsd.mineralList if m != "notIndexed"]


@operation("ebsd.select_indexed", "Keep indexed pixels", "Pre-processing",
           inputs=EBSD_IN, outputs=EBSD_OUT,
           script=lambda o, i, p: f"{o['ebsd']} = {i['ebsd']}['indexed']")
def select_indexed(ebsd):
  """The measurements whose pattern was indexed; notIndexed pixels are removed."""
  return _carry(ebsd, ebsd.value["indexed"])


@operation("ebsd.select_phase", "Select phase", "Pre-processing",
           inputs=EBSD_IN, outputs=EBSD_OUT,
           params=(Text("phase", "Phase (mineral name)", "", help="empty: the first indexed phase"),),
           script=lambda o, i, p: f"{o['ebsd']} = {i['ebsd']}[{p['phase']!r}]" if p["phase"]
           else f"{o['ebsd']} = {i['ebsd']}[[m for m in {i['ebsd']}.mineralList if m != 'notIndexed'][0]]")
def select_phase(ebsd, phase):
  """The measurements of one phase."""
  names = _phase_names(ebsd.value)
  if not names:
    raise ValueError("the map has no indexed phase")
  phase = phase or names[0]
  if phase not in names:
    raise ValueError(f"no phase {phase!r}; the map has {names}")
  return _carry(ebsd, ebsd.value[phase], phase=phase)


def _threshold_script(o, i, p):
  op = ">=" if p["keep"] == "at least" else "<="
  return f"{o['ebsd']} = {i['ebsd']}[np.asarray({i['ebsd']}.prop[{p['property']!r}]) {op} {p['value']!r}]"


@operation("ebsd.threshold_property", "Threshold a property", "Pre-processing",
           inputs=EBSD_IN, outputs=EBSD_OUT,
           params=(Text("property", "Property", "ci", help="ci, iq, fit, mad, bc, ... as the file names them"),
                   Choice("keep", "Keep pixels with", ("at least", "at most")),
                   Float("value", "Value", 0.1)),
           script=_threshold_script)
def threshold_property(ebsd, property, keep, value):
  """Keep the pixels whose measured property (confidence index, image quality, MAD, band
  contrast, ...) passes a threshold. Removing low-confidence pixels before grain
  reconstruction stops mis-indexed points from forming spurious one-pixel grains."""
  e = ebsd.value
  if property not in e.prop:
    raise ValueError(f"no property {property!r}; the map has {sorted(e.prop)}")
  x = np.asarray(e.prop[property], float)
  mask = x >= value if keep == "at least" else x <= value
  return _carry(ebsd, e[mask])


@operation("ebsd.crop", "Crop rectangle", "Pre-processing",
           inputs=EBSD_IN, outputs=EBSD_OUT,
           params=(Float("x", "x min", 0.0), Float("y", "y min", 0.0),
                   Float("width", "Width", 10.0, minimum=0), Float("height", "Height", 10.0, minimum=0)),
           script=lambda o, i, p: (f"{o['ebsd']} = {i['ebsd']}[{i['ebsd']}.inpolygon("
                                   f"np.array([{p['x']!r}, {p['y']!r}, {p['width']!r}, {p['height']!r}]))]"))
def crop(ebsd, x, y, width, height):
  """The pixels inside an axis-aligned rectangle, in the scan unit of the map."""
  e = ebsd.value
  sub = e[e.inpolygon(np.array([x, y, width, height]))]
  if sub.size == 0:
    raise ValueError("the rectangle contains no pixels")
  return _carry(ebsd, sub, roi=[x, y, width, height])


FILTERS = ("median", "mean", "spline", "half-quadratic", "infimal convolution", "Kuwahara")


def _filter(name, neighbours, alpha):
  import mtex
  a = None if alpha == 0 else alpha
  return {
    "median": lambda: mtex.medianFilter(numNeighbours=neighbours),
    "mean": lambda: mtex.meanFilter(numNeighbours=neighbours),
    "spline": lambda: mtex.splineFilter(alpha=a),
    "half-quadratic": lambda: mtex.halfQuadraticFilter(alpha=a),
    "infimal convolution": lambda: mtex.infimalConvolutionFilter(alpha=a),
    "Kuwahara": lambda: mtex.KuwaharaFilter(numNeighbours=neighbours),
  }[name]()


def _filter_code(p):
  a = "None" if p["alpha"] == 0 else repr(p["alpha"])
  return {
    "median": f"medianFilter(numNeighbours={p['neighbours']})",
    "mean": f"meanFilter(numNeighbours={p['neighbours']})",
    "spline": f"splineFilter(alpha={a})",
    "half-quadratic": f"halfQuadraticFilter(alpha={a})",
    "infimal convolution": f"infimalConvolutionFilter(alpha={a})",
    "Kuwahara": f"KuwaharaFilter(numNeighbours={p['neighbours']})",
  }[p["filter"]]


@operation("ebsd.denoise", "Denoise orientations", "Pre-processing",
           inputs=EBSD_IN, outputs=EBSD_OUT,
           params=(Choice("filter", "Filter", FILTERS, default="half-quadratic"),
                   Int("neighbours", "Neighbourhood order", 1, minimum=1, maximum=10,
                       help="median, mean and Kuwahara filters only"),
                   Float("alpha", "Regularisation α", 0.0, minimum=0,
                         help="spline, half-quadratic and infimal convolution filters; 0 chooses it by generalised cross-validation"),
                   Bool("fill", "Fill unindexed pixels", False)),
           script=lambda o, i, p: f"{o['ebsd']} = {i['ebsd']}.smooth({_filter_code(p)}, fill={p['fill']})")
def denoise(ebsd, filter, neighbours, alpha, fill):
  """Orientation denoising on the rotation group by MTEX's EBSD filters. The variational
  filters (half-quadratic, infimal convolution) preserve sub-grain boundaries while
  removing measurement noise; the median filter removes isolated wild pixels. If the map
  carries grain ids (the map output of grain reconstruction), smoothing stays inside grains."""
  e = ebsd.value.smooth(_filter(filter, neighbours, alpha), fill=fill)
  return _carry(ebsd, e, denoised=filter)


# -- local deformation -----------------------------------------------------------------------

SCALAR_OUT = (Output("map", DataKind.SCALAR_MAP, "Scalar map"),)


def _kam_threshold(p):
  return "np.inf" if p["threshold"] == 0 else f"{p['threshold']!r} * degree"


@operation("ebsd.kam", "Kernel average misorientation", "Local Deformation",
           inputs=EBSD_IN, outputs=SCALAR_OUT,
           params=(Int("order", "Neighbour order", 1, minimum=1, maximum=10),
                   Float("threshold", "Threshold", 2.5, minimum=0, maximum=180, unit="°",
                         help="neighbour pairs above it are left out of the average; 0 = no threshold")),
           script=lambda o, i, p: (f"{o['map']} = {{'ebsd': {i['ebsd']}, 'values': np.asarray({i['ebsd']}.KAM("
                                   f"order={p['order']}, threshold={_kam_threshold(p)})) / degree}}"))
def kam(ebsd, order, threshold):
  """The mean disorientation angle between a pixel and its neighbours up to the given
  order, in degrees. KAM grows with lattice curvature but also with the step size and the
  angular noise, so maps are comparable only at equal step size and preprocessing. Pairs
  across grain boundaries are excluded when the map carries grain ids."""
  from mtex import degree
  e = ebsd.value
  thr = np.inf if threshold == 0 else threshold * degree
  values = np.asarray(e.KAM(order=order, threshold=thr), float) / degree
  return DataItem(DataKind.SCALAR_MAP, {"ebsd": e, "values": values},
                  {"label": "KAM", "unit": "°", "colormap": "LaboTeX"})


@operation("ebsd.grod", "Grain reference orientation deviation", "Local Deformation",
           inputs=(Port("ebsd", (DataKind.EBSD,), "EBSD map with grain ids"),
                   Port("grains", (DataKind.GRAINS,), "Grains")),
           outputs=SCALAR_OUT,
           script=lambda o, i, p: (f"{o['map']} = {{'ebsd': {i['ebsd']}, 'values': "
                                   f"np.asarray(angle({i['ebsd']}.calcGROD({i['grains']}))) / degree}}"))
def grod(ebsd, grains):
  """The disorientation angle between every pixel and the mean orientation of its grain
  (GROD, MTEX's mis2mean), in degrees. Its grain average is the grain orientation spread."""
  from mtex import angle, degree
  e = ebsd.value
  if "grainId" not in e.prop:
    raise ValueError("this map carries no grain ids: connect the 'EBSD map' output of Grain reconstruction")
  values = np.asarray(angle(e.calcGROD(grains.value)), float) / degree
  return DataItem(DataKind.SCALAR_MAP, {"ebsd": e, "values": values},
                  {"label": "GROD", "unit": "°", "colormap": "LaboTeX"})
