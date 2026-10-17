"""A figure for every data kind.

MTEX draws through pyplot into figures it creates itself, so `render` lets it draw, takes
the figure it drew into and detaches that figure from pyplot: the caller (a Qt canvas, a
PDF page) owns it afterwards and pyplot keeps no reference. The statistics figures are
plain matplotlib and show, next to every estimate, the band that says how far to trust it.
"""
from __future__ import annotations

import contextlib

import numpy as np
from scipy import stats as sps

from ..core.types import DataItem, DataKind

# the views a kind offers; the first is the default
VIEWS = {
  DataKind.EBSD: ("IPF map", "Phase map"),
  DataKind.GRAINS: ("Mean orientation", "Boundaries"),
  DataKind.SCALAR_MAP: ("Map", "Histogram"),
  DataKind.ODF: ("Pole figures",),
  DataKind.TABLE: ("Histograms", "Table"),
  DataKind.IMAGE: ("Image", "Histogram"),
  DataKind.LABELS: ("Labels",),
  DataKind.STATS: ("Summary",),
}

INK = "#1f2a36"
ACCENT = "#c0392b"
FILL = "#4c78a8"


@contextlib.contextmanager
def _captured():
  """run pyplot drawing code; yields a holder whose .fig is the drawn figure, detached"""
  import matplotlib.pyplot as plt
  before = set(plt.get_fignums())
  holder = type("Holder", (), {"fig": None})()
  try:
    yield holder
  finally:
    new = [n for n in plt.get_fignums() if n not in before]
    if new:
      holder.fig = plt.figure(new[-1]) if holder.fig is None else holder.fig
    for n in new:
      plt.close(n)


def render(item: DataItem, view: str | None = None, overlay: DataItem | None = None):
  """the figure of a data item; overlay may be grains whose boundaries are drawn on top"""
  view = view or VIEWS[item.kind][0]
  fn = {
    DataKind.EBSD: _ebsd, DataKind.GRAINS: _grains, DataKind.SCALAR_MAP: _scalar_map,
    DataKind.ODF: _odf, DataKind.TABLE: _table, DataKind.IMAGE: _image,
    DataKind.LABELS: _labels, DataKind.STATS: _stats,
  }[item.kind]
  return fn(item, view, overlay)


def _new(figsize=(7.5, 6)):
  from matplotlib.figure import Figure
  return Figure(figsize=figsize, layout="constrained")


def _boundaries(overlay):
  if overlay is not None and overlay.kind == DataKind.GRAINS:
    from mtex import hold, plot
    hold(True)
    plot(overlay.value.boundary, lineWidth=1)
    hold(False)


def _indexed_phases(ebsd):
  return [m for m in ebsd.mineralList if m != "notIndexed"]


def _ebsd(item, view, overlay):
  from mtex import ipfColorKey, plot
  e = item.value
  phases = _indexed_phases(e)
  with _captured() as cap:
    if view == "IPF map" and len(phases) == 1:
      sub = e[phases[0]]
      plot(sub, ipfColorKey(sub.CS).orientation2color(sub.orientations), micronbar=True)
    else:
      plot(e)
    _boundaries(overlay)
  _title(cap.fig, f"{'IPF-Z' if view == 'IPF map' and len(phases) == 1 else 'Phases'} · {item.summary()}")
  return cap.fig


def _grains(item, view, overlay):
  from mtex import ipfColorKey, plot
  g = item.value
  phases = _indexed_phases(g)
  with _captured() as cap:
    if view == "Mean orientation" and len(phases) == 1:
      gi = g[phases[0]]
      plot(gi, ipfColorKey(gi.CS).orientation2color(gi.meanOrientation))
      from mtex import hold
      hold(True)
      plot(g.boundary, lineWidth=0.8)
      hold(False)
    elif view == "Mean orientation":
      plot(g)
    else:
      plot(g.boundary, lineWidth=1)
  _title(cap.fig, item.summary())
  return cap.fig


def _robust_range(v):
  fin = v[np.isfinite(v)]
  if fin.size == 0:
    return 0.0, 1.0
  lo, hi = np.quantile(fin, [0.01, 0.99])
  return float(lo), float(hi if hi > lo else lo + 1)


def _scalar_map(item, view, overlay):
  v = np.asarray(item.value["values"], float)
  label, unit = item.meta.get("label", "value"), item.meta.get("unit", "")
  if view == "Histogram":
    fig = _new((7, 4.5))
    ax = fig.add_subplot()
    _hist(ax, v[np.isfinite(v)], f"{label} ({unit})")
    _title(fig, f"{label}: pixel distribution (descriptive; pixels are not independent)")
    return fig
  from mtex import mtexColorMap, mtexColorbar, plot, setColorRange
  with _captured() as cap:
    plot(item.value["ebsd"], v, micronbar=True)
    setColorRange(list(_robust_range(v)))
    with contextlib.suppress(Exception):
      mtexColorMap(item.meta.get("colormap", "LaboTeX"))
    mtexColorbar(title=f"{label} ({unit})")
    _boundaries(overlay)
  _title(cap.fig, f"{label} · colour range: 1st–99th percentile")
  return cap.fig


def pole_figure_planes(cs):
  from mtex import Miller
  lat = str(cs.lattice).lower()
  if "cubic" in lat:
    idx = [[1, 0, 0], [1, 1, 0], [1, 1, 1]]
  elif "hexagonal" in lat or "trigonal" in lat:
    idx = [[0, 0, 0, 1], [1, 0, -1, 0], [1, 1, -2, 0]]
  else:
    idx = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
  return Miller(np.array(idx), cs)


def _odf(item, view, overlay):
  from mtex import mtexColorbar, plotPF
  f = item.value
  with _captured() as cap:
    plotPF(f, pole_figure_planes(f.CS))
    mtexColorbar(title="mrd")
  _title(cap.fig, f"Pole figures · {item.meta.get('phase', '')} · halfwidth {item.meta.get('halfwidth', '?')}°")
  return cap.fig


def _title(fig, text):
  if fig is not None:
    fig.suptitle(text, fontsize=9, color=INK)


def _hist(ax, x, xlabel, bins="fd", density=False, color=FILL):
  x = np.asarray(x, float)
  x = x[np.isfinite(x)]
  if x.size == 0:
    ax.text(0.5, 0.5, "no finite values", ha="center", transform=ax.transAxes)
    return
  edges = np.histogram_bin_edges(x, bins=bins)
  if edges.size > 81:
    edges = np.histogram_bin_edges(x, bins=80)
  ax.hist(x, bins=edges, density=density, color=color, alpha=0.75, edgecolor="white", linewidth=0.4)
  ax.set_xlabel(xlabel)
  ax.set_ylabel("density" if density else "count")
  ax.grid(alpha=0.25)


def _table(item, view, overlay):
  t = item.value
  numeric = [k for k, c in t.items() if np.asarray(c).dtype.kind in "fiu" and k not in ("id",)]
  if view == "Table" or not numeric or len(next(iter(t.values()))) < 3:
    fig = _new((7.5, 0.6 + 0.3 * min(25, len(next(iter(t.values()))))))
    ax = fig.add_subplot()
    ax.axis("off")
    cols = list(t)
    n = min(25, len(t[cols[0]]))
    cells = [[_fmt(t[c][i]) for c in cols] for i in range(n)]
    tab = ax.table(cellText=cells, colLabels=cols, loc="upper center", cellLoc="right")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    _title(fig, item.summary() + ("" if n == len(t[cols[0]]) else f" (first {n} rows)"))
    return fig
  numeric = numeric[:6]
  rows = int(np.ceil(len(numeric) / 2))
  fig = _new((8, 2.8 * rows))
  units = item.meta.get("units", {})
  for i, k in enumerate(numeric):
    ax = fig.add_subplot(rows, 2 if len(numeric) > 1 else 1, i + 1)
    _hist(ax, t[k], f"{k} ({units[k]})" if k in units else k)
  _title(fig, item.summary())
  return fig


def _fmt(v):
  if isinstance(v, (float, np.floating)):
    return f"{v:.4g}"
  return str(v)


def _scale_bar(ax, shape, pixel_size, unit):
  H, W = shape[:2]
  target = 0.2 * W * pixel_size
  nice = 10 ** np.floor(np.log10(target))
  for m in (5, 2, 1):
    if m * nice <= target:
      length = m * nice
      break
  px = length / pixel_size
  x0, y0 = 0.05 * W, 0.93 * H
  ax.plot([x0, x0 + px], [y0, y0], color="white", lw=4, solid_capstyle="butt")
  ax.plot([x0, x0 + px], [y0, y0], color="black", lw=2, solid_capstyle="butt")
  ax.text(x0 + px / 2, y0 - 0.02 * H, f"{length:g} {unit}", ha="center", va="bottom", fontsize=8,
          color="black", bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))


def _image(item, view, overlay):
  a = np.asarray(item.value)
  fig = _new((7, 6) if view == "Image" else (7, 4))
  ax = fig.add_subplot()
  if view == "Histogram":
    _hist(ax, a.ravel(), "intensity", bins=256)
    if "threshold" in item.meta:
      ax.axvline(item.meta["threshold"], color=ACCENT)
    _title(fig, "Intensity histogram")
    return fig
  ax.imshow(a, cmap="gray" if a.ndim == 2 else None, interpolation="nearest")
  ax.set_axis_off()
  if "pixel_size" in item.meta:
    _scale_bar(ax, a.shape, item.meta["pixel_size"], item.meta.get("unit", "µm"))
  extra = ""
  if "threshold" in item.meta:
    eta = item.meta.get("otsu_effectiveness")
    extra = f" · threshold {item.meta['threshold']:.3f}" + (f", η = {eta:.2f}" if eta == eta else "")
  _title(fig, item.summary() + extra)
  return fig


def _labels(item, view, overlay):
  from matplotlib.colors import ListedColormap
  lab = np.asarray(item.value)
  n = int(lab.max())
  rng = np.random.default_rng(1)
  colours = np.vstack([[0, 0, 0], 0.25 + 0.75 * rng.random((max(n, 1), 3))])
  fig = _new()
  ax = fig.add_subplot()
  ax.imshow(lab, cmap=ListedColormap(colours), vmin=0, vmax=max(n, 1), interpolation="nearest")
  ax.set_axis_off()
  if "pixel_size" in item.meta:
    _scale_bar(ax, lab.shape, item.meta["pixel_size"], item.meta.get("unit", "µm"))
  _title(fig, item.summary())
  return fig


# -- statistics figures ----------------------------------------------------------------------

def _stats(item, view, overlay):
  kind = item.meta.get("plot")
  fn = {"grain_size": _grain_size_fig, "misorientation": _misorientation_fig,
        "map_summary": _map_summary_fig, "column": _column_fig}.get(kind)
  if fn is None:
    fig = _new((7, 4))
    ax = fig.add_subplot()
    ax.axis("off")
    ax.text(0.02, 0.98, item.meta.get("headline", ""), va="top", family="monospace", fontsize=9)
    return fig
  return fn(item.value, item.meta)


def _text_panel(ax, lines):
  ax.axis("off")
  ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8, color=INK,
          transform=ax.transAxes)


def _grain_size_fig(r, meta):
  d = np.asarray(r["ecd"], float)
  f, gof, boot, unit = r["fit"], r["gof"], r["mean_ci"], r["unit"]
  n = d.size
  fig = _new((9, 7.5))
  gs = fig.add_gridspec(2, 2)
  # (a) histogram with the fitted lognormal density
  ax = fig.add_subplot(gs[0, 0])
  edges = np.histogram_bin_edges(np.log(d), bins="auto")
  if edges.size > 61:
    edges = np.histogram_bin_edges(np.log(d), bins=60)
  ax.hist(d, bins=np.exp(edges), density=True, color=FILL, alpha=0.7, edgecolor="white", lw=0.4)
  x = np.geomspace(d.min() * 0.7, d.max() * 1.3, 400)
  ax.plot(x, sps.lognorm.pdf(x, f["s"], scale=np.exp(f["mu"])), color=INK, lw=1.5, label="lognormal fit")
  lo, hi = boot["ci"]
  ax.axvspan(lo, hi, color=ACCENT, alpha=0.18, lw=0)
  ax.axvline(boot["estimate"], color=ACCENT, lw=1.2, label=f"{r['weighting']} mean, {int(100 * boot['confidence'])}% BCa")
  ax.set_xscale("log")
  ax.set_xlabel(f"equivalent circle diameter ({unit})")
  ax.set_ylabel("density")
  ax.legend(fontsize=7, frameon=False)
  ax.grid(alpha=0.25, which="both")
  ax.set_title("(a) distribution", fontsize=9, loc="left")
  # (b) normal Q-Q plot of log d with the exact pointwise band of the order statistics
  ax = fig.add_subplot(gs[0, 1])
  y = np.sort(np.log(d))
  i = np.arange(1, n + 1)
  q = sps.norm.ppf((i - 0.375) / (n + 0.25))  # Blom plotting positions
  band = sps.norm.ppf(sps.beta.ppf([[0.025], [0.975]], i, n - i + 1))
  ax.fill_between(q, f["mu"] + f["s"] * band[0], f["mu"] + f["s"] * band[1], color=FILL, alpha=0.18, lw=0,
                  label="95% pointwise band")
  ax.plot(q, f["mu"] + f["s"] * q, color=INK, lw=1)
  ax.plot(q, y, ".", ms=3, color=ACCENT)
  ax.set_xlabel("standard normal quantile")
  ax.set_ylabel(f"log ECD (log {unit})")
  ax.legend(fontsize=7, frameon=False, loc="upper left")
  ax.grid(alpha=0.25)
  ax.set_title("(b) lognormal Q–Q", fontsize=9, loc="left")
  # (c) empirical CDF with the DKW band and the fitted CDF
  ax = fig.add_subplot(gs[1, 0])
  ds = np.sort(d)
  ecdf = i / n
  eps = np.sqrt(np.log(2 / 0.05) / (2 * n))
  ax.fill_between(ds, np.clip(ecdf - eps, 0, 1), np.clip(ecdf + eps, 0, 1), step="post", color=FILL, alpha=0.18,
                  lw=0, label="95% DKW band")
  ax.step(ds, ecdf, where="post", color=ACCENT, lw=1.2, label="empirical")
  ax.plot(x, sps.lognorm.cdf(x, f["s"], scale=np.exp(f["mu"])), color=INK, lw=1, label="lognormal fit")
  ax.set_xscale("log")
  ax.set_xlabel(f"equivalent circle diameter ({unit})")
  ax.set_ylabel("cumulative fraction")
  ax.legend(fontsize=7, frameon=False, loc="lower right")
  ax.grid(alpha=0.25, which="both")
  ax.set_title("(c) cumulative distribution", fontsize=9, loc="left")
  # (d) the numbers
  ax = fig.add_subplot(gs[1, 1])
  c = int(100 * f["confidence"])
  lines = [
    f"grains n = {n}  (sampling unit: grain)",
    f"border grains {'excluded' if r['excluded_border'] else 'included'}: {r['n_border']}",
    "",
    "lognormal maximum likelihood",
    f"  median   {f['median']:.4g}  [{f['median_ci'][0]:.4g}, {f['median_ci'][1]:.4g}]",
    f"  mean     {f['mean']:.4g}  [{f['mean_ci'][0]:.4g}, {f['mean_ci'][1]:.4g}]  Cox",
    f"  σ(log)   {f['s']:.4f}  [{f['sigma_ci'][0]:.4f}, {f['sigma_ci'][1]:.4f}]",
    f"  area-weighted mean  {f['area_weighted_mean']:.4g}",
    "",
    f"{r['weighting']}-weighted mean, BCa bootstrap",
    f"  {boot['estimate']:.4g}  [{boot['ci'][0]:.4g}, {boot['ci'][1]:.4g}]  {c}%",
    f"  se {boot['se']:.3g}, B = {boot['n_boot']}",
    "",
    "goodness of fit (parametric bootstrap)",
    f"  KS  D  = {gof['ks']:.4f}   p = {gof['ks_p']:.3f}",
    f"  AD  A² = {gof['ad']:.3f}   p = {gof['ad_p']:.3f}",
  ]
  if "astm_G" in r:
    lines += ["", f"ASTM E112 G ≈ {r['astm_G']:.2f}  (planimetric, from mean area)"]
  _text_panel(ax, lines)
  fig.suptitle(f"Grain size · {meta.get('headline', '')}", fontsize=9, color=INK)
  return fig


def _misorientation_fig(r, meta):
  a = np.asarray(r["pair_angle"], float)
  om, dens = np.asarray(r["omega"]), np.asarray(r["mackenzie_density"])
  theta = r["segmentation_angle"]
  cum = np.concatenate([[0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(om))])
  cum /= cum[-1]
  f_theta = np.interp(theta, om, cum)
  # the Mackenzie density conditioned on omega >= theta
  cond_dens = np.where(om >= theta, dens / (np.trapezoid(dens, om) * (1 - f_theta)), 0.0)
  cond_cum = np.clip((cum - f_theta) / (1 - f_theta), 0, 1)
  fig = _new((9, 4.2))
  gs = fig.add_gridspec(1, 3)
  ax = fig.add_subplot(gs[0, 0])
  edges = np.linspace(0, r["omax"], 26)
  ax.hist(a, bins=edges, density=True, color=FILL, alpha=0.75, edgecolor="white", lw=0.4, label="grain pairs")
  ax.plot(om, cond_dens, color=INK, lw=1.5, label=f"random, ω ≥ {theta:g}°")
  ax.axvline(r["low_angle"], color=ACCENT, lw=1, ls="--")
  if theta > 0:
    ax.axvspan(0, theta, color="0.85", lw=0, zorder=0)
  ax.set_xlabel("disorientation angle (°)")
  ax.set_ylabel("density (1/°)")
  ax.legend(fontsize=7, frameon=False, loc="upper left")
  ax.grid(alpha=0.25)
  ax.set_title("(a) distribution", fontsize=9, loc="left")
  ax = fig.add_subplot(gs[0, 1])
  s = np.sort(np.asarray(r.get("tested_angle", a), float))
  n = s.size
  ecdf = np.arange(1, n + 1) / n
  cum = cond_cum
  ax.step(s, ecdf, where="post", color=ACCENT, lw=1.2, label="grain pairs")
  ax.plot(om, cum, color=INK, lw=1.2, label="random")
  ref = np.interp(s, om, cum)
  j = int(np.argmax(np.maximum(np.abs(ecdf - ref), np.abs(ecdf - 1 / n - ref))))
  ax.annotate("", xy=(s[j], ecdf[j]), xytext=(s[j], ref[j]), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8))
  ax.set_xlabel("disorientation angle (°)")
  ax.set_ylabel("cumulative fraction")
  ax.legend(fontsize=7, frameon=False, loc="upper left")
  ax.grid(alpha=0.25)
  ax.set_title("(b) KS distance", fontsize=9, loc="left")
  lo, hi = r["low_angle_ci"]
  c = int(100 * r["confidence"])
  _text_panel(fig.add_subplot(gs[0, 2]), [
    f"phase {r['phase']}",
    f"grain pairs n = {r['n_pairs']}",
    "(sampling unit: grain pair)",
    "",
    f"KS vs random, ω ≥ {theta:g}°",
    f"  pairs tested {r['ks']['n']}",
    f"  D = {r['ks']['D']:.4f}",
    f"  p = {r['ks']['p']:.3g}  (exact)",
    "",
    f"low-angle (< {r['low_angle']:g}°)",
    f"  pairs  {100 * r['low_angle_fraction']:.1f}%",
    f"         [{100 * lo:.1f}, {100 * hi:.1f}]% {c}% Wilson",
    f"  length {100 * r['low_angle_length_fraction']:.1f}%",
    "",
    "texture alone shifts the",
    "distribution from random",
  ])
  fig.suptitle("Boundary misorientation", fontsize=9, color=INK)
  return fig


def _map_summary_fig(r, meta):
  label, unit = r["label"], r["unit"]
  has_grains = "grains" in r
  fig = _new((9, 4.2))
  gs = fig.add_gridspec(1, 3 if has_grains else 2)
  _hist(fig.add_subplot(gs[0, 0]), r["values"], f"{label} per pixel ({unit})")
  fig.axes[-1].set_title("(a) pixels, descriptive", fontsize=9, loc="left")
  p = r["pixels"]
  lines = [f"pixels: {p['n_finite']} finite of {p['n_total']}"]
  if p["n_finite"]:
    lines += [f"  median {p['median']:.4g}  IQR {p['iqr']:.4g}", f"  mean   {p['mean']:.4g}  MAD {p['mad']:.4g}",
              f"  5–95%  [{p['q05']:.4g}, {p['q95']:.4g}]"]
  if has_grains:
    ax = fig.add_subplot(gs[0, 1])
    _hist(ax, r["grains"]["mean"], f"grain mean {label} ({unit})", color="#72b7b2")
    ax.set_title("(b) grain means", fontsize=9, loc="left")
    if "grain_mean_ci" in r:
      b = r["grain_mean_ci"]
      ax.axvspan(*b["ci"], color=ACCENT, alpha=0.2, lw=0)
      ax.axvline(b["estimate"], color=ACCENT, lw=1.2)
      lines += ["", f"grains n = {b['n']} (sampling unit)", f"  mean {b['estimate']:.4g}",
                f"  [{b['ci'][0]:.4g}, {b['ci'][1]:.4g}] {int(100 * b['confidence'])}% BCa"]
  _text_panel(fig.add_subplot(gs[0, -1]), lines)
  fig.suptitle(f"{label} statistics", fontsize=9, color=INK)
  return fig


def _column_fig(r, meta):
  x = np.asarray(r["values"], float)
  fig = _new((9, 4))
  gs = fig.add_gridspec(1, 3)
  ax = fig.add_subplot(gs[0, 0])
  _hist(ax, x, f"{r['label']} ({r['unit']})" if r["unit"] else r["label"])
  m = r["mean_ci"]
  ax.axvspan(*m["ci"], color=ACCENT, alpha=0.2, lw=0)
  ax.axvline(m["estimate"], color=ACCENT, lw=1.2)
  ax = fig.add_subplot(gs[0, 1])
  s = np.sort(x)
  ax.step(s, np.arange(1, s.size + 1) / s.size, where="post", color=ACCENT)
  ax.set_xlabel(r["label"])
  ax.set_ylabel("cumulative fraction")
  ax.grid(alpha=0.25)
  su, md = r["summary"], r["median_ci"]
  c = int(100 * m["confidence"])
  _text_panel(fig.add_subplot(gs[0, 2]), [
    f"n = {su['n_finite']} (sampling unit: row)", "",
    f"mean    {m['estimate']:.4g}", f"        [{m['ci'][0]:.4g}, {m['ci'][1]:.4g}] {c}% BCa",
    f"median  {md['estimate']:.4g}", f"        [{md['ci'][0]:.4g}, {md['ci'][1]:.4g}] {c}% BCa", "",
    f"sd {su['sd']:.4g}  IQR {su['iqr']:.4g}", f"5–95%  [{su['q05']:.4g}, {su['q95']:.4g}]",
  ])
  fig.suptitle(meta.get("headline", ""), fontsize=9, color=INK)
  return fig
