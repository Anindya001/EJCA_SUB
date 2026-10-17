"""Statistical analyses of grains, boundaries and maps.

The sampling unit is stated with every result. Grains (and grain pairs, for boundaries)
are treated as the independent units; pixels never are, because neighbouring pixels of an
EBSD map are strongly correlated and a pixel-level interval would be falsely narrow."""
from __future__ import annotations

import numpy as np

from ..core.params import Bool, Choice, Float, Int, Text
from ..core.registry import Output, Port, operation
from ..core.types import DataItem, DataKind
from .. import stats

STATS_OUT = (Output("stats", DataKind.STATS, "Statistics"),)
CONF = Float("confidence", "Confidence level", 0.95, minimum=0.5, maximum=0.999)
NBOOT = Int("n_boot", "Bootstrap resamples", 1999, minimum=99, maximum=100000)
SEED = Int("seed", "Random seed", 0, minimum=0)


def _size_table(item: DataItem) -> tuple[dict, str]:
  if item.kind == DataKind.GRAINS:
    from .grain_ops import grain_table_values
    return grain_table_values(item.value), item.meta.get("unit", "µm")
  table = item.value
  unit = item.meta.get("units", {}).get("ecd", "")
  return table, unit


@operation("stats.grain_size", "Grain size distribution", "Statistics",
           inputs=(Port("grains", (DataKind.GRAINS, DataKind.TABLE), "Grains or grain table"),),
           outputs=STATS_OUT,
           params=(Text("phase", "Phase", "", help="empty: all phases"),
                   Bool("exclude_border", "Exclude grains cut by the border", True),
                   Choice("weighting", "Mean weighted by", ("number", "area", "Miles-Lantuéjoul"),
                          help="number: the mean grain; area: the grain a random point falls in; "
                               "Miles-Lantuéjoul: number-weighted, corrected for border loss (micrographs)"),
                   CONF, NBOOT, SEED))
def grain_size(grains, phase, exclude_border, weighting, confidence, n_boot, seed):
  """The distribution of equivalent circle diameters (ECD), one observation per grain.

  Reports: a lognormal maximum likelihood fit with exact intervals for the median and the
  log-scale spread and Cox's interval for the mean; Kolmogorov–Smirnov and
  Anderson–Darling tests of the lognormal model with p-values calibrated by the parametric
  bootstrap (the parameters are estimated, so tabulated p-values would be too large); a
  BCa bootstrap interval for the chosen mean that assumes no distribution; and, for maps in
  micrometres, the ASTM E112 planimetric grain size number from the mean grain area.

  Excluding border grains removes truncated grains but loses large grains preferentially;
  the Miles–Lantuéjoul weighting corrects that loss (available for segmented micrographs)."""
  table, unit = _size_table(grains)
  if "ecd" not in table:
    raise ValueError("the table has no 'ecd' column")
  ecd = np.asarray(table["ecd"], float)
  keep = np.isfinite(ecd) & (ecd > 0)
  if phase:
    if "phase" not in table:
      raise ValueError("the table has no phase column")
    keep &= np.asarray(table["phase"]) == phase
  n_border = 0
  if "touches_border" in table:
    border = np.asarray(table["touches_border"], bool)
    n_border = int(np.sum(keep & border))
    if exclude_border or weighting == "Miles-Lantuéjoul":
      keep &= ~border
  d = ecd[keep]
  if d.size < 3:
    raise ValueError(f"only {d.size} grains remain after the selection; at least 3 are needed")
  fit = stats.lognormal_fit(d, confidence)
  gof = stats.lognormal_gof(d, n_boot=n_boot, seed=seed)
  if weighting == "number":
    boot = stats.bootstrap_ci(d, "mean", n_boot, confidence, seed)
  elif weighting == "area":
    area = np.asarray(table["area"], float)[keep] if "area" in table else np.pi * d ** 2 / 4
    boot = stats.bootstrap_ci(d, "weighted_mean", n_boot, confidence, seed, weights=area)
  else:
    if "ml_weight" not in table:
      raise ValueError("Miles-Lantuéjoul weights need a table from 'Segment grains'")
    boot = stats.bootstrap_ci(d, "weighted_mean", n_boot, confidence, seed,
                              weights=np.asarray(table["ml_weight"], float)[keep])
  result = {"ecd": d, "unit": unit, "n": int(d.size), "n_border": n_border,
            "excluded_border": bool(exclude_border or weighting == "Miles-Lantuéjoul"),
            "weighting": weighting, "fit": fit.to_dict(), "gof": gof, "mean_ci": boot,
            "sampling_unit": "grain"}
  if unit in ("um", "µm") and "area" in table:
    area = np.asarray(table["area"], float)[keep]
    result["mean_area"] = float(area.mean())
    result["astm_G"] = stats.astm_grain_size_number(float(area.mean()))
  lo, hi = boot["ci"]
  head = (f"{weighting}-weighted mean ECD {boot['estimate']:.3g} {unit} "
          f"[{lo:.3g}, {hi:.3g}], n = {d.size}")
  return DataItem(DataKind.STATS, result, {"plot": "grain_size", "headline": head})


def group_median(keys: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """(unique keys, median value, count) of values grouped by integer keys"""
  order = np.lexsort((values, keys))
  k, v = keys[order], values[order]
  uniq, start, count = np.unique(k, return_index=True, return_counts=True)
  lo = start + (count - 1) // 2
  hi = start + count // 2
  return uniq, 0.5 * (v[lo] + v[hi]), count


@operation("stats.boundary_misorientation", "Boundary misorientation distribution", "Statistics",
           inputs=(Port("grains", (DataKind.GRAINS,), "Grains"),),
           outputs=STATS_OUT,
           params=(Text("phase", "Phase", "", help="empty: the first indexed phase"),
                   Float("segmentation_angle", "Segmentation threshold", 0.0, minimum=0, maximum=62.8, unit="°",
                         help="the angle the grains were reconstructed with; 0 takes it from the grains"),
                   Float("low_angle", "Low-angle limit", 15.0, minimum=0.5, maximum=60, unit="°"),
                   CONF))
def boundary_misorientation(grains, phase, segmentation_angle, low_angle, confidence):
  """The disorientation angles of the boundaries between grains of one phase, compared with
  the Mackenzie distribution of uncorrelated random orientations.

  The sampling unit is the grain pair: all boundary segments between two grains share
  nearly the same misorientation, so treating segments as independent would inflate n by
  the boundary length. Each pair contributes the median angle of its segments. The KS test
  against the fully specified Mackenzie CDF then has an exact p-value. A small p-value
  says the boundary population differs from the random reference, which texture alone can
  cause; the reference for a textured material is the uncorrelated MDF of its ODF. The
  low-angle fraction gets a Wilson interval over grain pairs and, descriptively, a length
  fraction.

  Grains reconstructed with a threshold θ cannot be separated by boundaries below θ, so
  the reference is the Mackenzie distribution conditioned on ω ≥ θ,
  F_θ(ω) = (F(ω) − F(θ)) / (1 − F(θ)), and only pairs with median angle ≥ θ enter the
  test; comparing with the unconditioned distribution would reject by construction."""
  from mtex import angle, calcAngleDistribution, degree
  g = grains.value
  names = [m for m in g.mineralList if m != "notIndexed"]
  if not names:
    raise ValueError("no indexed phase")
  phase = phase or names[0]
  if phase not in names:
    raise ValueError(f"no phase {phase!r}; the grains have {names}")
  B = g.boundary[phase, phase]
  if len(B) == 0:
    raise ValueError(f"there are no {phase}–{phase} boundaries")
  seg_angle = np.asarray(angle(B.misorientation), float) / degree
  pairs = np.sort(np.asarray(B.grainId, np.int64), axis=1)
  seg_len = np.asarray(B.segLength, float)
  keys = pairs[:, 0] * (int(pairs.max()) + 1) + pairs[:, 1]
  _, pair_angle, n_seg = group_median(keys, seg_angle)
  cs = g[phase].CS
  dens, omega = calcAngleDistribution(cs, cs)
  om = np.asarray(omega, float) / degree
  cdf = stats.tabulated_cdf(om, np.asarray(dens, float))
  theta = segmentation_angle or float(grains.meta.get("angle", 0.0))
  f_theta = float(cdf(theta))
  if f_theta >= 1:
    raise ValueError("the segmentation threshold leaves no admissible angle")
  cond = lambda t: np.clip((cdf(t) - f_theta) / (1 - f_theta), 0, 1)
  tested = pair_angle[pair_angle >= theta]
  if tested.size < 3:
    raise ValueError(f"only {tested.size} grain pairs lie above the {theta:g}° segmentation threshold")
  ks = stats.ks_against(tested, cond)
  k = int(np.sum(pair_angle < low_angle))
  wil = stats.wilson_interval(k, pair_angle.size, confidence)
  length_fraction = float(seg_len[seg_angle < low_angle].sum() / seg_len.sum())
  result = {"pair_angle": pair_angle, "segments_per_pair": n_seg, "omega": om,
            "mackenzie_density": np.asarray(dens, float), "omax": float(om[-1]),
            "ks": ks, "tested_angle": tested, "segmentation_angle": theta, "reference_mass_below": f_theta,
            "low_angle": low_angle, "low_angle_pairs": k, "n_pairs": int(pair_angle.size),
            "low_angle_fraction": k / pair_angle.size, "low_angle_ci": wil,
            "low_angle_length_fraction": length_fraction, "phase": phase,
            "confidence": confidence, "sampling_unit": "grain pair"}
  head = (f"{pair_angle.size} {phase} grain pairs, KS D = {ks['D']:.3f} (p = {ks['p']:.2g}) vs random ≥ {theta:g}°; "
          f"low-angle {100 * k / pair_angle.size:.1f}%")
  return DataItem(DataKind.STATS, result, {"plot": "misorientation", "headline": head})


@operation("stats.map_summary", "Map statistics", "Statistics",
           inputs=(Port("map", (DataKind.SCALAR_MAP,), "Scalar map"),),
           outputs=STATS_OUT,
           params=(CONF, NBOOT, SEED))
def map_summary(map, confidence, n_boot, seed):
  """Descriptive statistics of a per-pixel quantity (KAM, GROD, ...) and, when the map
  carries grain ids, inference over grains: each grain contributes the mean of its
  pixels, and the BCa bootstrap interval for the mean over grains treats grains as the
  independent units. No interval is given at the pixel level, where neighbouring values
  are correlated."""
  v = np.asarray(map.value["values"], float)
  e = map.value["ebsd"]
  result = {"label": map.meta.get("label", "value"), "unit": map.meta.get("unit", ""),
            "pixels": stats.robust_summary(v), "values": v[np.isfinite(v)].ravel()}
  head = f"pixel median {result['pixels'].get('median', float('nan')):.3g} {result['unit']}"
  if "grainId" in e.prop:
    ids, means, counts = stats.group_means(v, np.asarray(e.prop["grainId"]))
    result["grains"] = {"id": ids, "mean": means, "pixels": counts}
    if ids.size >= 3:
      result["grain_mean_ci"] = stats.bootstrap_ci(means, "mean", n_boot, confidence, seed)
      lo, hi = result["grain_mean_ci"]["ci"]
      head += f"; mean over {ids.size} grains {means.mean():.3g} [{lo:.3g}, {hi:.3g}]"
  result["sampling_unit"] = "grain" if "grains" in result else "pixel (descriptive only)"
  return DataItem(DataKind.STATS, result, {"plot": "map_summary", "headline": head})


@operation("stats.column", "Column statistics", "Statistics",
           inputs=(Port("table", (DataKind.TABLE,), "Table"),),
           outputs=STATS_OUT,
           params=(Text("column", "Column", "gos"), Bool("exclude_border", "Exclude border rows", True),
                   CONF, NBOOT, SEED))
def column(table, column, exclude_border, confidence, n_boot, seed):
  """Robust summary and BCa bootstrap intervals for the mean and the median of one column of
  a table, each row (grain) an independent observation."""
  t = table.value
  if column not in t:
    raise ValueError(f"no column {column!r}; the table has {list(t)}")
  x = np.asarray(t[column], float)
  if exclude_border and "touches_border" in t:
    x = x[~np.asarray(t["touches_border"], bool)]
  x = x[np.isfinite(x)]
  unit = table.meta.get("units", {}).get(column, "")
  result = {"label": column, "unit": unit, "values": x, "summary": stats.robust_summary(x),
            "mean_ci": stats.bootstrap_ci(x, "mean", n_boot, confidence, seed),
            "median_ci": stats.bootstrap_ci(x, "median", n_boot, confidence, seed),
            "sampling_unit": "row"}
  m = result["mean_ci"]
  head = f"mean {column} {m['estimate']:.3g} [{m['ci'][0]:.3g}, {m['ci'][1]:.3g}] {unit}, n = {x.size}"
  return DataItem(DataKind.STATS, result, {"plot": "column", "headline": head})
