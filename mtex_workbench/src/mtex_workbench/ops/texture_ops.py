"""Orientation distribution functions and texture strength."""
from __future__ import annotations

import numpy as np

from ..core.params import Float, Text
from ..core.registry import Output, Port, operation
from ..core.types import DataItem, DataKind


def _phase(ebsd, phase):
  names = [m for m in ebsd.mineralList if m != "notIndexed"]
  if not names:
    raise ValueError("the map has no indexed phase")
  phase = phase or names[0]
  if phase not in names:
    raise ValueError(f"no phase {phase!r}; the map has {names}")
  return phase


def _odf_script(o, i, p):
  sel = repr(p["phase"]) if p["phase"] else f"[m for m in {i['ebsd']}.mineralList if m != 'notIndexed'][0]"
  hw = f"{p['halfwidth']!r} * degree" if p["halfwidth"] > 0 else None
  args = f"{i['ebsd']}[{sel}].orientations" + (f", halfwidth={hw}" if hw else "")
  return f"{o['odf']} = calcDensity({args})"


@operation("texture.odf", "ODF from orientations", "Texture & ODF",
           inputs=(Port("ebsd", (DataKind.EBSD,), "EBSD map"),),
           outputs=(Output("odf", DataKind.ODF, "ODF"),),
           params=(Text("phase", "Phase", "", help="empty: the first indexed phase"),
                   Float("halfwidth", "Kernel halfwidth", 5.0, minimum=0, maximum=45, unit="°",
                         help="0 selects it by unbiased cross-validation")),
           script=_odf_script)
def odf(ebsd, phase, halfwidth):
  """The orientation distribution function of one phase by kernel density estimation on
  SO(3) with the de la Vallée Poussin kernel (MTEX calcDensity). Every pixel counts once,
  so large grains weigh more: this is the volume-weighted texture. The halfwidth trades
  bias for variance; with few grains, sharp ODF peaks reflect individual grains, not a
  texture of the material."""
  from mtex import calcDensity, degree
  phase = _phase(ebsd.value, phase)
  ori = ebsd.value[phase].orientations
  kw = {"halfwidth": halfwidth * degree} if halfwidth > 0 else {}
  f = calcDensity(ori, **kw)
  return DataItem(DataKind.ODF, f, {"phase": phase, "halfwidth": halfwidth, "n_orientations": int(ebsd.value[phase].size)})


def texture_index_values(f) -> dict:
  from mtex import entropy, textureIndex
  fmax, ori = f.max()
  euler = np.degrees(np.asarray(ori.Euler(), float)).ravel()
  return {"quantity": np.array(["texture index", "entropy", "max (mrd)", "max at φ1", "max at Φ", "max at φ2"]),
          "value": np.array([float(textureIndex(f)), float(entropy(f)), float(fmax), *euler[:3]])}


@operation("texture.indices", "Texture strength", "Texture & ODF",
           inputs=(Port("odf", (DataKind.ODF,), "ODF"),),
           outputs=(Output("table", DataKind.TABLE, "Texture indices"),),
           script=lambda o, i, p: ("from mtex_workbench.ops.texture_ops import texture_index_values\n"
                                   f"{o['table']} = texture_index_values({i['odf']})"))
def texture_indices(odf):
  """The texture index ∫f² (1 for a random texture), the entropy −∫f log f (0 for random,
  more negative for sharper textures), and the ODF maximum in multiples of a random
  distribution with its Bunge Euler angles (°)."""
  return DataItem(DataKind.TABLE, texture_index_values(odf.value), {"sampling_unit": "ODF"})
