"""The MTEX operations on the ferrite map of the MTEX documentation."""
import numpy as np
import pytest

from mtex_workbench.core import DataItem, DataKind, Pipeline, Ref, Status, load_builtin_operations
from mtex_workbench.core.script import export_script
from mtex_workbench.core.storage import load_project, save_project

pytestmark = pytest.mark.mtex_data


@pytest.fixture(scope="module")
def analysed(ferrite_module):
  load_builtin_operations()
  p = Pipeline()
  src = p.add("io.sample_data", {"name": "ferrite"})
  g = p.add("grains.calc", {"angle": 10, "min_pixel": 8}, {"ebsd": Ref(src.id, "ebsd")})
  kam = p.add("ebsd.kam", {"order": 1, "threshold": 2.5}, {"ebsd": Ref(g.id, "ebsd")})
  grod = p.add("ebsd.grod", {}, {"ebsd": Ref(g.id, "ebsd"), "grains": Ref(g.id, "grains")})
  tab = p.add("grains.table", {}, {"grains": Ref(g.id, "grains")})
  size = p.add("stats.grain_size", {"n_boot": 499}, {"grains": Ref(g.id, "grains")})
  mis = p.add("stats.boundary_misorientation", {}, {"grains": Ref(g.id, "grains")})
  summ = p.add("stats.map_summary", {"n_boot": 499}, {"map": Ref(kam.id, "map")})
  odf = p.add("texture.odf", {"halfwidth": 10}, {"ebsd": Ref(src.id, "ebsd")})
  ti = p.add("texture.indices", {}, {"odf": Ref(odf.id, "odf")})
  p.run_all()
  return p, dict(src=src.id, g=g.id, kam=kam.id, grod=grod.id, tab=tab.id, size=size.id, mis=mis.id,
                 summ=summ.id, odf=odf.id, ti=ti.id)


@pytest.fixture(scope="module")
def ferrite_module():
  mtex = pytest.importorskip("mtex")
  try:
    return mtex.mtexdata("ferrite")
  except Exception as exc:
    pytest.skip(f"MTEX sample data unavailable: {exc}")


def out(p, nid, name):
  return p.nodes[nid].outputs[name]


def test_everything_ran(analysed):
  p, ids = analysed
  assert all(n.status == Status.DONE for n in p.nodes.values())


def test_grain_reconstruction_is_non_destructive(analysed):
  p, ids = analysed
  assert "grainId" not in out(p, ids["src"], "ebsd").value.prop
  e = out(p, ids["g"], "ebsd").value
  assert "grainId" in e.prop
  grains = out(p, ids["g"], "grains").value
  assert 150 < len(grains) < 300


def test_grain_table_is_consistent(analysed):
  p, ids = analysed
  t = out(p, ids["tab"], "table").value
  n = len(t["id"])
  assert all(len(c) == n for c in t.values())
  # the equivalent circle diameter is that of the area
  np.testing.assert_allclose(t["ecd"], 2 * np.sqrt(t["area"] / np.pi), rtol=1e-9)
  assert np.all(t["gos"] >= 0) and np.all(t["aspect_ratio"] >= 1 - 1e-9)
  assert set(t["phase"]) == {"Ferrite"}


def test_local_misorientation_maps(analysed):
  p, ids = analysed
  kam = out(p, ids["kam"], "map").value["values"]
  fin = kam[np.isfinite(kam)]
  assert np.all(fin >= 0) and np.all(fin <= 2.5 + 1e-9)  # the 2.5° threshold bounds every pair
  assert 0.3 < np.median(fin) < 1.0
  grod = out(p, ids["grod"], "map").value["values"]
  assert np.nanmin(grod) >= 0 and np.nanmax(grod) < 62.8


def test_grod_needs_grain_ids(analysed):
  p, ids = analysed
  op = p.registry["ebsd.grod"]
  with pytest.raises(ValueError, match="grain ids"):
    op.run({"ebsd": out(p, ids["src"], "ebsd"), "grains": out(p, ids["g"], "grains")}, {})


def test_statistics_are_sane(analysed):
  p, ids = analysed
  s = out(p, ids["size"], "stats").value
  lo, hi = s["mean_ci"]["ci"]
  assert lo < s["mean_ci"]["estimate"] < hi
  assert s["fit"]["median_ci"][0] < s["fit"]["median"] < s["fit"]["median_ci"][1]
  assert s["n"] + s["n_border"] <= 206 and s["excluded_border"]
  assert 0 < s["gof"]["ks_p"] <= 1
  m = out(p, ids["mis"], "stats").value
  assert m["segmentation_angle"] == 10
  assert np.all(m["tested_angle"] >= 10)  # the conditioned test only sees admissible angles
  assert 0 <= m["ks"]["D"] <= 1 and 0 <= m["low_angle_ci"][0] <= m["low_angle_fraction"] <= m["low_angle_ci"][1]
  # the Mackenzie reference for cubic symmetry ends at 62.8°
  assert m["omax"] == pytest.approx(62.8, abs=0.1)
  sm = out(p, ids["summ"], "stats").value
  assert sm["sampling_unit"] == "grain" and sm["grain_mean_ci"]["n"] > 100


def test_texture_indices(analysed):
  p, ids = analysed
  t = out(p, ids["ti"], "table").value
  values = dict(zip(t["quantity"], t["value"]))
  assert values["texture index"] > 1  # a measured texture is never weaker than random
  assert values["entropy"] < 0
  assert values["max (mrd)"] > 1


def test_mackenzie_reference_mean():
  """the mean of the random cubic disorientation angle is 40.7° (Mackenzie 1958)"""
  from mtex import calcAngleDistribution, crystalFrame, degree
  cs = crystalFrame("m-3m")
  dens, om = calcAngleDistribution(cs, cs)
  om = np.asarray(om) / degree
  dens = np.asarray(dens)
  assert np.trapezoid(om * dens, om) / np.trapezoid(dens, om) == pytest.approx(40.7, abs=0.2)


def test_exported_script_reproduces(analysed):
  p, ids = analysed
  ns = {}
  exec(export_script(p, [ids["kam"], ids["tab"], ids["grod"], ids["mis"], ids["size"]]), ns)
  # operations run through run_op see the metadata their inputs had: the segmentation angle
  mis = ns[f"{ids['mis']}_stats"]
  assert mis["segmentation_angle"] == 10
  assert mis["ks"] == out(p, ids["mis"], "stats").value["ks"]
  size = ns[f"{ids['size']}_stats"]
  assert size["mean_ci"] == out(p, ids["size"], "stats").value["mean_ci"] and size["unit"] == "µm"
  np.testing.assert_array_equal(ns[f"{ids['kam']}_map"]["values"], out(p, ids["kam"], "map").value["values"])
  np.testing.assert_array_equal(ns[f"{ids['grod']}_map"]["values"], out(p, ids["grod"], "map").value["values"])
  for k, v in out(p, ids["tab"], "table").value.items():
    np.testing.assert_array_equal(ns[f"{ids['tab']}_table"][k], v)


def test_project_round_trip_with_mtex_objects(analysed, tmp_path):
  p, ids = analysed
  path = save_project(p, tmp_path / "ferrite")
  q, warnings = load_project(path)
  assert warnings == []
  assert all(n.status == Status.DONE for n in q.nodes.values())
  assert len(out(q, ids["g"], "grains").value) == len(out(p, ids["g"], "grains").value)
  np.testing.assert_array_equal(out(q, ids["kam"], "map").value["values"], out(p, ids["kam"], "map").value["values"])
  # a restored map is a working MTEX object
  kam_again = np.asarray(out(q, ids["g"], "ebsd").value.KAM())
  assert kam_again.shape == out(p, ids["kam"], "map").value["values"].shape


def test_project_from_another_mtex_version_recomputes(analysed, tmp_path, monkeypatch):
  import mtex_workbench.core.storage as storage
  p, ids = analysed
  path = save_project(p, tmp_path / "old")
  env = dict(storage.environment())
  env["mtex"] = "0.0.1"
  monkeypatch.setattr(storage, "environment", lambda: env)
  q, warnings = load_project(path)
  assert warnings  # MTEX objects are not unpickled across versions
  assert q.nodes[ids["src"]].status == Status.STALE
  # plain arrays and tables do not depend on MTEX's version, but sit downstream of stale nodes
  assert q.nodes[ids["tab"]].status == Status.STALE


def test_worker_process_matches_inline(analysed):
  from mtex_workbench.core.executor import ProcessExecutor
  p, ids = analysed
  ex = ProcessExecutor()
  try:
    r = ex.execute("ebsd.kam", {"ebsd": out(p, ids["g"], "ebsd")}, {"order": 1, "threshold": 2.5})
  finally:
    ex.shutdown()
  np.testing.assert_array_equal(r["map"].value["values"], out(p, ids["kam"], "map").value["values"])


def test_preprocessing(analysed):
  p, ids = analysed
  reg = p.registry
  e = out(p, ids["src"], "ebsd")
  thr = reg["ebsd.threshold_property"].run({"ebsd": e}, {"property": "ci", "keep": "at least", "value": 0.1})["ebsd"]
  assert thr.value.size < e.value.size
  with pytest.raises(ValueError, match="no property"):
    reg["ebsd.threshold_property"].run({"ebsd": e}, {"property": "nope", "keep": "at least", "value": 0.1})
  cr = reg["ebsd.crop"].run({"ebsd": e}, {"x": 10, "y": 10, "width": 20, "height": 15})["ebsd"]
  assert 0 < cr.value.size < e.value.size and cr.meta["roi"] == [10, 10, 20, 15]
  dn = reg["ebsd.denoise"].run({"ebsd": e}, {"filter": "median", "neighbours": 1, "alpha": 0, "fill": False})["ebsd"]
  assert dn.value.size == e.value.size
  sel = reg["ebsd.select_phase"].run({"ebsd": e}, {"phase": ""})["ebsd"]
  assert sel.meta["phase"] == "Ferrite"


def test_every_kind_renders(analysed):
  from mtex_workbench.viz import VIEWS, render
  p, ids = analysed
  grains = out(p, ids["g"], "grains")
  seen = set()
  for n in p.nodes.values():
    for item in n.outputs.values():
      for view in VIEWS[item.kind]:
        fig = render(item, view, overlay=grains if item.kind == DataKind.SCALAR_MAP else None)
        assert fig is not None and fig.axes
        seen.add(item.kind)
  assert seen >= {DataKind.EBSD, DataKind.GRAINS, DataKind.SCALAR_MAP, DataKind.ODF, DataKind.TABLE, DataKind.STATS}


def test_render_leaves_no_pyplot_figures(analysed):
  import matplotlib.pyplot as plt
  from mtex_workbench.viz import render
  p, ids = analysed
  before = len(plt.get_fignums())
  render(out(p, ids["odf"], "odf"))
  render(out(p, ids["src"], "ebsd"))
  assert len(plt.get_fignums()) == before


def test_summary_strings(analysed):
  p, ids = analysed
  for n in p.nodes.values():
    for item in n.outputs.values():
      assert isinstance(item.summary(), str) and not item.summary().startswith("<")
  assert DataItem(DataKind.TABLE, {}).summary() == "0 rows × 0 columns"
