"""The engine on a toy registry: parameters, the graph, invalidation, storage and scripts."""
import json
import time

import numpy as np
import pytest

from mtex_workbench.core import (
  Bool, Choice, DataItem, DataKind, Float, Int, ParamError, Pipeline, PipelineError, Port, Ref, Status, Text,
  validate_params,
)
from mtex_workbench.core.registry import Output, Registry, operation
from mtex_workbench.core.script import export_script
from mtex_workbench.core.storage import load_project, save_project


# -- parameters ------------------------------------------------------------------------------

def test_params_strict():
  specs = [Float("a", "A", 1.0, minimum=0, maximum=2), Int("n", "N", 3, minimum=1), Bool("b", "B", True),
           Choice("c", "C", ("x", "y")), Text("t", "T", "")]
  assert validate_params(specs, {}) == {"a": 1.0, "n": 3, "b": True, "c": "x", "t": ""}
  assert validate_params(specs, {"a": "1.5", "n": 2.0})["a"] == 1.5
  for bad in ({"a": 3}, {"a": float("nan")}, {"n": 0}, {"n": 2.5}, {"b": "yes"}, {"c": "z"}, {"t": 5},
              {"typo": 1}, {"a": True}):
    with pytest.raises(ParamError):
      validate_params(specs, bad)


def test_param_errors_are_collected():
  specs = [Float("a", "A", 1.0, minimum=0), Int("n", "N", 1, minimum=1)]
  with pytest.raises(ParamError, match="A.*N"):
    validate_params(specs, {"a": -1, "n": 0})


# -- a toy registry --------------------------------------------------------------------------

@pytest.fixture
def reg():
  r = Registry()
  calls = []

  @operation("t.src", "Source", "Import & Data", outputs=(Output("x", DataKind.IMAGE),),
             params=(Int("n", "n", 4, minimum=1),), registry=r,
             script=lambda o, i, p: f"{o['x']} = np.arange({p['n']}, dtype=float)")
  def src(n):
    calls.append("src")
    return DataItem(DataKind.IMAGE, np.arange(n, dtype=float), {"unit": "px"})

  @operation("t.scale", "Scale", "Image Processing", inputs=(Port("x", (DataKind.IMAGE,)),),
             outputs=(Output("y", DataKind.IMAGE),), params=(Float("k", "k", 2.0),), registry=r,
             script=lambda o, i, p: f"{o['y']} = {i['x']} * {p['k']!r}")
  def scale(x, k):
    calls.append("scale")
    return x.value * k

  @operation("t.add", "Add", "Image Processing",
             inputs=(Port("a", (DataKind.IMAGE,)), Port("b", (DataKind.IMAGE,))),
             outputs=(Output("s", DataKind.IMAGE), Output("t", DataKind.TABLE)), registry=r,
             script=lambda o, i, p: f"{o['s']} = {i['a']} + {i['b']}\n{o['t']} = {{'s': {o['s']}}}")
  def add(a, b):
    calls.append("add")
    s = a.value + b.value
    return {"s": s, "t": {"s": s}}

  @operation("t.stats", "Stats", "Statistics", inputs=(Port("t", (DataKind.TABLE,)),),
             outputs=(Output("r", DataKind.STATS),), registry=r)
  def st(t):
    return DataItem(DataKind.STATS, {"mean": float(np.mean(t.value["s"])), "arr": t.value["s"]},
                    {"plot": "none"})

  @operation("t.fail", "Fail", "Statistics", inputs=(Port("x", (DataKind.IMAGE,)),),
             outputs=(Output("y", DataKind.IMAGE),), registry=r)
  def fail(x):
    raise ValueError("boom")

  r.calls = calls
  return r


def build(reg):
  p = Pipeline(reg)
  a = p.add("t.src", {"n": 5})
  b = p.add("t.scale", {"k": 3}, {"x": Ref(a.id, "x")})
  c = p.add("t.add", {}, {"a": Ref(a.id, "x"), "b": Ref(b.id, "y")})
  d = p.add("t.stats", {}, {"t": Ref(c.id, "t")})
  return p, a, b, c, d


def test_registry_filters_by_kind(reg):
  assert {o.id for o in reg.applicable(None)} == {"t.src"}
  assert {o.id for o in reg.applicable(DataKind.IMAGE)} == {"t.scale", "t.add", "t.fail"}
  assert {o.id for o in reg.applicable(DataKind.TABLE)} == {"t.stats"}
  assert reg.applicable(DataKind.EBSD) == []


def test_registry_rejects_duplicates_and_bad_category(reg):
  with pytest.raises(ValueError):
    operation("t.src", "x", "Statistics", registry=reg)(lambda: None)
  with pytest.raises(ValueError):
    operation("t.new", "x", "Nonsense", registry=reg)(lambda: None)


def test_run_and_lazy_reuse(reg):
  p, a, b, c, d = build(reg)
  p.run(d.id)
  assert [p.nodes[n].status for n in p.nodes] == [Status.DONE] * 4
  np.testing.assert_allclose(p.nodes[c.id].outputs["s"].value, np.arange(5) * 4.0)
  assert p.nodes[d.id].outputs["r"].value["mean"] == pytest.approx(8.0)
  n_calls = len(reg.calls)
  p.run(d.id)  # nothing stale: nothing recomputed
  assert len(reg.calls) == n_calls


def test_invalidation_is_downstream_only(reg):
  p, a, b, c, d = build(reg)
  p.run(d.id)
  assert p.set_params(b.id, {"k": 3}) == []  # no change, nothing invalidated
  inv = p.set_params(b.id, {"k": 1})
  assert inv == [b.id, c.id, d.id]
  assert p.nodes[a.id].status == Status.DONE
  assert all(p.nodes[n].status == Status.STALE and not p.nodes[n].outputs for n in inv)
  reg.calls.clear()
  p.run(d.id)
  assert reg.calls == ["scale", "add"]  # the source is reused
  np.testing.assert_allclose(p.nodes[c.id].outputs["s"].value, np.arange(5) * 2.0)


def test_inputs_are_not_modified(reg):
  p, a, b, c, d = build(reg)
  p.run(d.id)
  np.testing.assert_array_equal(p.nodes[a.id].outputs["x"].value, np.arange(5.0))


def test_type_checks_and_cycles(reg):
  p, a, b, c, d = build(reg)
  with pytest.raises(PipelineError):
    p.add("t.stats", {}, {"t": Ref(a.id, "x")})  # an image is not a table
  with pytest.raises(PipelineError):
    p.add("t.scale", {}, {})  # missing input
  with pytest.raises(PipelineError, match="cycle"):
    p.set_input(b.id, "x", Ref(c.id, "s"))
  with pytest.raises(ParamError):
    p.add("t.src", {"n": 0})


def test_topology(reg):
  p, a, b, c, d = build(reg)
  assert p.descendants(a.id) == [b.id, c.id, d.id]
  assert p.upstream(d.id) == [a.id, b.id, c.id, d.id]
  assert p.children(a.id) == [b.id, c.id]
  removed = p.remove(b.id)
  assert removed == [b.id, c.id, d.id] and list(p.nodes) == [a.id]


def test_errors_are_recorded(reg):
  p = Pipeline(reg)
  a = p.add("t.src")
  f = p.add("t.fail", {}, {"x": Ref(a.id, "x")})
  with pytest.raises(ValueError, match="boom"):
    p.run(f.id)
  assert p.nodes[f.id].status == Status.ERROR and "boom" in p.nodes[f.id].error
  assert p.nodes[a.id].status == Status.DONE
  assert [h["event"] for h in p.history].count("error") == 1


def test_fingerprint_tracks_parameters(reg):
  p, a, b, c, d = build(reg)
  f0 = p.fingerprint(d.id)
  p.set_params(b.id, {"k": 7})
  assert p.fingerprint(d.id) != f0
  p.set_params(b.id, {"k": 3})
  assert p.fingerprint(d.id) == f0


def test_spec_round_trip(reg):
  p, *_ = build(reg)
  spec = json.loads(json.dumps(p.to_spec()))
  q = Pipeline.from_spec(spec, reg)
  assert q.to_spec() == p.to_spec()
  e = q.add("t.src")
  assert e.id == "n5"  # ids continue after the loaded ones


def test_project_round_trip(reg, tmp_path, monkeypatch):
  import mtex_workbench.core.storage as storage
  monkeypatch.setattr(storage, "REGISTRY", reg)
  monkeypatch.setattr("mtex_workbench.core.registry.load_builtin_operations", lambda: reg)
  p, a, b, c, d = build(reg)
  p.run(d.id)
  p.set_params(b.id, {"k": 5})
  p.run(c.id)  # d stays stale
  path = save_project(p, tmp_path / "proj")
  assert path.name == "proj.mtexwb"
  save_project(p, path)  # overwriting keeps a valid project
  q, warnings = load_project(path, reg)
  assert warnings == []
  assert [n.status for n in q.nodes.values()] == [n.status for n in p.nodes.values()]
  np.testing.assert_array_equal(q.nodes[c.id].outputs["s"].value, p.nodes[c.id].outputs["s"].value)
  np.testing.assert_array_equal(q.nodes[c.id].outputs["t"].value["s"], p.nodes[c.id].outputs["t"].value["s"])
  assert q.nodes[a.id].outputs["x"].meta == {"unit": "px"}
  assert q.nodes[b.id].provenance["fingerprint"] == p.nodes[b.id].provenance["fingerprint"]
  assert len(q.history) == len(p.history)
  q.run(d.id)
  r = q.nodes[d.id].outputs["r"].value
  assert isinstance(r["arr"], np.ndarray) and r["mean"] == pytest.approx(float(np.mean(np.arange(5) * 6.0)))


def test_script_export_reproduces(reg):
  p, a, b, c, d = build(reg)
  p.run(c.id)
  code = export_script(p, [c.id])
  assert "n4" not in code.split('"""', 2)[2]  # only what the target needs
  ns = {}
  exec(code, ns)
  np.testing.assert_array_equal(ns["n3_s"], p.nodes[c.id].outputs["s"].value)


def test_source_file_hash_detects_changes(tmp_path):
  from mtex_workbench.core.registry import load_builtin_operations
  from PIL import Image
  load_builtin_operations()
  f = tmp_path / "a.png"
  Image.fromarray(np.full((8, 8), 100, np.uint8)).save(f)
  p = Pipeline()
  n = p.add("io.load_image", {"path": str(f), "as_gray": True, "pixel_size": 1.0})
  p.run(n.id)
  assert len(n.provenance["file_sha256"]) == 64
  assert p.stale_sources() == []
  Image.fromarray(np.full((8, 8), 101, np.uint8)).save(f)
  assert p.stale_sources() == [n.id]


def test_process_executor_matches_inline_and_cancels(tmp_path):
  from PIL import Image
  from mtex_workbench.core.executor import Cancelled, InlineExecutor, ProcessExecutor
  from mtex_workbench.core.registry import load_builtin_operations
  load_builtin_operations()
  f = tmp_path / "a.png"
  Image.fromarray(np.random.default_rng(0).integers(0, 255, (600, 600), dtype=np.uint8)).save(f)
  img = InlineExecutor().execute("io.load_image", {}, {"path": str(f), "as_gray": True, "pixel_size": 1.0})["image"]
  ex = ProcessExecutor()
  try:
    remote = ex.execute("image.gaussian", {"image": img}, {"sigma": 2.0})["image"]
    local = InlineExecutor().execute("image.gaussian", {"image": img}, {"sigma": 2.0})["image"]
    np.testing.assert_array_equal(remote.value, local.value)
    # a slow operation cancelled from another thread
    import threading
    t = threading.Timer(0.5, ex.cancel)
    t.start()
    t0 = time.perf_counter()
    with pytest.raises(Cancelled):
      ex.execute("image.median", {"image": img}, {"size": 51})
    assert time.perf_counter() - t0 < 10
    # the executor recovers with a fresh worker
    again = ex.execute("image.gaussian", {"image": img}, {"sigma": 2.0})["image"]
    np.testing.assert_array_equal(again.value, local.value)
  finally:
    ex.shutdown()


def test_worker_errors_carry_the_remote_traceback():
  from mtex_workbench.core.executor import ProcessExecutor, WorkerError
  ex = ProcessExecutor()
  try:
    with pytest.raises(WorkerError, match="FileNotFoundError") as info:
      ex.execute("io.load_image", {}, {"path": "/nonexistent.png", "as_gray": True, "pixel_size": 1.0})
    assert "Traceback" in info.value.remote_traceback
  finally:
    ex.shutdown()
