"""The micrograph path end to end, the report export and the command line."""
import csv
import json

import numpy as np
import pytest

from mtex_workbench.cli import flatten_stats, main
from mtex_workbench.core import Pipeline, Ref, Status, load_builtin_operations
from mtex_workbench.report import export_results


def image_pipeline(path, pixel_size=0.5):
  load_builtin_operations()
  p = Pipeline()
  img = p.add("io.load_image", {"path": str(path), "as_gray": True, "pixel_size": pixel_size})
  sm = p.add("image.gaussian", {"sigma": 0.7}, {"image": Ref(img.id, "image")})
  th = p.add("image.threshold", {"method": "Otsu"}, {"image": Ref(sm.id, "image")})
  seg = p.add("image.segment", {"connectivity": "4", "min_area": 5}, {"mask": Ref(th.id, "mask")})
  gs = p.add("stats.grain_size", {"weighting": "Miles-Lantuéjoul", "n_boot": 299}, {"grains": Ref(seg.id, "table")})
  col = p.add("stats.column", {"column": "area", "n_boot": 299}, {"table": Ref(seg.id, "table")})
  return p, (img, sm, th, seg, gs, col)


def test_micrograph_segmentation(micrograph_png):
  path, n_true = micrograph_png
  p, (img, sm, th, seg, gs, col) = image_pipeline(path)
  p.run_all()
  assert all(n.status == Status.DONE for n in p.nodes.values())
  mask = p.nodes[th.id].outputs["mask"]
  assert mask.meta["otsu_effectiveness"] > 0.8  # boundaries and interiors separate cleanly
  labels = p.nodes[seg.id].outputs["labels"].value
  assert abs(int(labels.max()) - n_true) <= 2
  table = p.nodes[seg.id].outputs["table"].value
  assert table["area"].sum() == pytest.approx(table["pixels"].sum() * 0.25)
  stats = p.nodes[gs.id].outputs["stats"].value
  assert stats["weighting"] == "Miles-Lantuéjoul" and stats["unit"] == "µm"
  assert stats["n"] == int(np.sum(~table["touches_border"]))


def test_grain_size_refuses_too_few_grains(micrograph_png):
  path, _ = micrograph_png
  p, (img, sm, th, seg, gs, col) = image_pipeline(path)
  p.set_params(seg.id, {"min_area": 10 ** 6})
  with pytest.raises(ValueError, match="at least 3"):
    p.run(gs.id)


def test_export_results(micrograph_png, tmp_path):
  path, _ = micrograph_png
  p, nodes = image_pipeline(path)
  p.run_all()
  written = export_results(p, tmp_path / "out", name="test")
  out = tmp_path / "out"
  assert (out / "report.pdf").stat().st_size > 10_000
  assert len(written["figures"]) == sum(len(n.outputs) for n in p.nodes.values())
  rows = list(csv.reader(open(out / "tables" / f"{nodes[3].id}_table.csv")))
  assert rows[0][:3] == ["id", "pixels", "area"] and len(rows) - 1 == len(p.nodes[nodes[3].id].outputs["table"].value["id"])
  prov = json.loads((out / "provenance.json").read_text())
  assert prov["nodes"]["n1"]["file_sha256"] and prov["environment"]["mtex"]
  spec = json.loads((out / "pipeline.json").read_text())
  assert Pipeline.from_spec(spec).to_spec() == p.to_spec()
  # the exported script runs and gives the same segmentation
  ns = {}
  exec((out / "analysis.py").read_text(), ns)
  np.testing.assert_array_equal(ns[f"{nodes[3].id}_labels"], p.nodes[nodes[3].id].outputs["labels"].value)
  # the pixel size reaches the script's segmentation, so the areas are physical
  for k, v in p.nodes[nodes[3].id].outputs["table"].value.items():
    np.testing.assert_array_equal(ns[f"{nodes[3].id}_table"][k], v)
  assert ns[f"{nodes[4].id}_stats"]["mean_ci"] == p.nodes[nodes[4].id].outputs["stats"].value["mean_ci"]


def test_flatten_stats():
  flat = flatten_stats({"a": 1.0, "b": {"c": 2, "ci": (0.5, 1.5)}, "arr": np.arange(3), "flag": True}, "n1")
  assert flat == {"n1.a": 1.0, "n1.b.c": 2.0, "n1.b.ci.lo": 0.5, "n1.b.ci.hi": 1.5}


def test_cli_run_and_batch(micrograph_png, tmp_path, capsys):
  from PIL import Image
  from conftest import voronoi_micrograph
  path, _ = micrograph_png
  p, _ = image_pipeline(path)
  template = tmp_path / "template.json"
  template.write_text(json.dumps(p.to_spec()))
  assert main(["run", str(template), "-o", str(tmp_path / "run"), "--set", "n5.n_boot=199",
               "--save", str(tmp_path / "proj")]) == 0
  assert (tmp_path / "run" / "report.pdf").exists() and (tmp_path / "proj.mtexwb" / "project.sqlite").exists()
  files = []
  for k, seeds in enumerate((30, 60)):
    img, _ = voronoi_micrograph(n_seeds=seeds, seed=k)
    f = tmp_path / f"specimen{k}.png"
    Image.fromarray((img * 255).astype(np.uint8)).save(f)
    files.append(str(f))
  assert main(["batch", str(template), "-i", *files, "-o", str(tmp_path / "batch")]) == 0
  rows = list(csv.DictReader(open(tmp_path / "batch" / "summary.csv")))
  assert [r["specimen"] for r in rows] == ["specimen0", "specimen1"]
  # more seeds, smaller grains: the batch summary carries the comparison
  key = "n5.stats.mean_ci.estimate"
  assert float(rows[1][key]) < float(rows[0][key])
  assert all(len(r["sha256"]) == 64 for r in rows)
  main(["script", str(template)])
  assert "run_op('image.segment'" in capsys.readouterr().out
  main(["ops"])
  assert "grains.calc" in capsys.readouterr().out
