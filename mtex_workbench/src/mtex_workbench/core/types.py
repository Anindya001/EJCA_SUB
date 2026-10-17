"""The kinds of data the workbench moves between operations.

Every value flowing along an edge of the processing graph is a DataItem: a kind, the value
itself and a small JSON-serialisable metadata dictionary (units, labels, provenance hints).
The kind is what the registry uses to decide which operations apply to a dataset; the value
is never inspected for that purpose, so the filtering is cheap and predictable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class DataKind(str, Enum):
  EBSD = "ebsd"              # an MTEX EBSD map: orientations, phases, properties
  GRAINS = "grains"          # an MTEX grain2d object with its boundaries
  SCALAR_MAP = "scalar_map"  # one value per pixel of an EBSD map: {"ebsd": EBSD, "values": ndarray}
  ODF = "odf"                # an MTEX SO3Fun (orientation distribution function)
  TABLE = "table"            # a dict of equally long 1-D numpy columns
  IMAGE = "image"            # a 2-D (grey) or 3-D (RGB) ndarray of a micrograph
  LABELS = "labels"          # a 2-D integer ndarray, 0 = background, k > 0 = region k
  STATS = "stats"            # a dict of statistical results, rendered by meta["plot"]

  @property
  def label(self) -> str:
    return _LABELS[self]

  @property
  def crystallographic(self) -> bool:
    """whether the kind carries orientations; an image never does"""
    return self in (DataKind.EBSD, DataKind.GRAINS, DataKind.SCALAR_MAP, DataKind.ODF)


_LABELS = {
  DataKind.EBSD: "EBSD map",
  DataKind.GRAINS: "Grains",
  DataKind.SCALAR_MAP: "Scalar map",
  DataKind.ODF: "Orientation distribution",
  DataKind.TABLE: "Table",
  DataKind.IMAGE: "Image",
  DataKind.LABELS: "Label image",
  DataKind.STATS: "Statistics",
}


@dataclass
class DataItem:
  kind: DataKind
  value: Any
  meta: dict = field(default_factory=dict)

  def summary(self) -> str:
    """one line that describes the value without rendering it"""
    v = self.value
    try:
      if self.kind == DataKind.EBSD:
        phases = [m for m in v.mineralList if m != "notIndexed"]
        return f"{v.size} pixels, phases: {', '.join(phases) or 'none'}"
      if self.kind == DataKind.GRAINS:
        return f"{len(v)} grains, {len(v.boundary)} boundary segments"
      if self.kind == DataKind.SCALAR_MAP:
        x = np.asarray(v["values"], float)
        fin = x[np.isfinite(x)]
        unit = self.meta.get("unit", "")
        if fin.size == 0:
          return "no finite values"
        return f"median {np.median(fin):.3g} {unit}, {fin.size} finite of {x.size}"
      if self.kind == DataKind.TABLE:
        n = len(next(iter(v.values()))) if v else 0
        return f"{n} rows × {len(v)} columns"
      if self.kind in (DataKind.IMAGE, DataKind.LABELS):
        a = np.asarray(v)
        extra = f", {int(a.max())} regions" if self.kind == DataKind.LABELS else ""
        return f"{'×'.join(map(str, a.shape))} {a.dtype}{extra}"
      if self.kind == DataKind.ODF:
        return "orientation distribution function"
      if self.kind == DataKind.STATS:
        return self.meta.get("headline", f"{len(v)} results")
    except Exception as exc:  # a summary must never break the interface
      return f"<{exc.__class__.__name__}>"
    return ""


def table_rows(table: dict) -> int:
  return len(next(iter(table.values()))) if table else 0


def check_table(table: dict) -> dict:
  """a table with numpy columns of one common length, or a ValueError"""
  out = {str(k): np.asarray(v) for k, v in table.items()}
  lengths = {len(c) for c in out.values()}
  if len(lengths) > 1:
    raise ValueError(f"table columns differ in length: {sorted(lengths)}")
  for k, c in out.items():
    if c.ndim != 1:
      raise ValueError(f"table column {k!r} is not one-dimensional")
  return out
