"""The command line: the GUI, headless runs, batch processing and script export.

  mtex-workbench                              the desktop application
  mtex-workbench gui [project.mtexwb]
  mtex-workbench ops                          the operations and their parameters
  mtex-workbench run template.json -o out/    run a pipeline, export report and files
  mtex-workbench batch template.json -i a.ang b.ang -o out/
                                              the same pipeline for every specimen, plus
                                              summary.csv with one row per specimen
  mtex-workbench script template.json         print the reproducing Python script
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def _load_pipeline(path):
  from .core.pipeline import Pipeline
  from .core.registry import load_builtin_operations
  from .core.storage import SUFFIX, load_project
  load_builtin_operations()
  path = Path(path)
  if path.suffix == SUFFIX or (path.is_dir() and (path / "project.sqlite").exists()):
    p, warnings = load_project(path)
    for w in warnings:
      print("warning:", w, file=sys.stderr)
    return p
  return Pipeline.from_spec(json.loads(path.read_text()))


def _apply_sets(p, sets):
  for s in sets or ():
    target, value = s.split("=", 1)
    nid, name = target.split(".", 1)
    try:
      value = json.loads(value)
    except json.JSONDecodeError:
      pass
    p.set_params(nid, {name: value})


def flatten_stats(item_value: dict, prefix: str) -> dict:
  """the scalar results of a statistics output, one column each"""
  out = {}

  def walk(d, key, depth):
    if isinstance(d, dict) and depth < 3:
      for k, v in d.items():
        walk(v, f"{key}.{k}", depth + 1)
    elif isinstance(d, (int, float, np.integer, np.floating)) and not isinstance(d, bool):
      out[key] = float(d)
    elif isinstance(d, (tuple, list)) and len(d) == 2 and all(isinstance(x, (int, float)) for x in d):
      out[f"{key}.lo"], out[f"{key}.hi"] = float(d[0]), float(d[1])

  walk(item_value, prefix, 0)
  return out


def cmd_ops(args):
  from .core.registry import load_builtin_operations
  reg = load_builtin_operations()
  for cat, ops in reg.by_category().items():
    if not ops:
      continue
    print(f"\n{cat}")
    for op in ops:
      ins = ", ".join(f"{p.name}:{'|'.join(k.value for k in p.kinds)}" for p in op.inputs) or "—"
      outs = ", ".join(f"{o.name}:{o.kind.value}" for o in op.outputs)
      print(f"  {op.id:<32} {op.label}   ({ins}) → ({outs})")
      for prm in op.params:
        rng = ""
        if prm.minimum is not None or prm.maximum is not None:
          rng = f" [{prm.minimum}, {prm.maximum}]"
        if prm.choices:
          rng = f" {list(prm.choices)}"
        print(f"      {prm.name} = {prm.default!r}{prm.unit}{rng}")


def cmd_run(args):
  from .core.executor import InlineExecutor
  from .core.storage import save_project
  from .report import export_results
  p = _load_pipeline(args.pipeline)
  _apply_sets(p, args.set)
  p.run_all(InlineExecutor())
  written = export_results(p, args.out, name=args.name or Path(args.pipeline).stem)
  if args.save:
    save_project(p, args.save)
  print(f"report: {written['report']}")
  return p


def cmd_batch(args):
  from .core.executor import InlineExecutor
  from .core.pipeline import Pipeline
  from .core.registry import load_builtin_operations
  from .core.types import DataKind
  from .report import export_results
  load_builtin_operations()
  spec = json.loads(Path(args.pipeline).read_text())
  template = Pipeline.from_spec(spec)
  sources = [n.id for n in template.nodes.values()
             if template.registry[n.op_id].hashes_file and template.registry[n.op_id].is_source]
  source = args.source or (sources[0] if len(sources) == 1 else None)
  if source is None:
    raise SystemExit(f"the template has file sources {sources}; choose one with --source")
  op = template.registry[template.nodes[source].op_id]
  rows, failures = [], []
  for path in args.inputs:
    p = Pipeline.from_spec(spec)
    _apply_sets(p, args.set)
    p.set_params(source, {op.hashes_file: str(Path(path).resolve())})
    specimen = Path(path).stem
    try:
      p.run_all(InlineExecutor())
    except Exception as exc:  # one bad specimen must not stop the batch
      failures.append((path, str(exc)))
      print(f"FAILED {path}: {exc}", file=sys.stderr)
      continue
    export_results(p, Path(args.out) / specimen, name=specimen)
    row = {"specimen": specimen, "file": str(path), "sha256": p.nodes[source].provenance.get("file_sha256")}
    for n in p.nodes.values():
      for name, item in n.outputs.items():
        if item.kind == DataKind.STATS:
          row.update(flatten_stats(item.value, f"{n.id}.{name}"))
    rows.append(row)
    print(f"done {specimen}")
  if rows:
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(Path(args.out) / "summary.csv", "w", newline="") as f:
      w = csv.DictWriter(f, fieldnames=cols)
      w.writeheader()
      w.writerows(rows)
  if failures:
    raise SystemExit(f"{len(failures)} of {len(args.inputs)} specimens failed")
  return rows


def cmd_script(args):
  from .core.script import export_script
  print(export_script(_load_pipeline(args.pipeline)), end="")


def cmd_gui(args):
  from .gui.app import main as gui_main
  return gui_main(getattr(args, "project", None))


def main(argv=None):
  ap = argparse.ArgumentParser(prog="mtex-workbench", description="MTEX Workbench")
  sub = ap.add_subparsers(dest="cmd")
  g = sub.add_parser("gui", help="the desktop application")
  g.add_argument("project", nargs="?")
  sub.add_parser("ops", help="list the operations")
  r = sub.add_parser("run", help="run a pipeline template or project and export the results")
  r.add_argument("pipeline")
  r.add_argument("-o", "--out", required=True)
  r.add_argument("--set", action="append", metavar="NODE.PARAM=VALUE")
  r.add_argument("--save", metavar="PROJECT", help="also save the computed project")
  r.add_argument("--name")
  b = sub.add_parser("batch", help="run one template over many input files")
  b.add_argument("pipeline")
  b.add_argument("-i", "--inputs", nargs="+", required=True)
  b.add_argument("-o", "--out", required=True)
  b.add_argument("--source", help="the node whose file is replaced (default: the only file source)")
  b.add_argument("--set", action="append", metavar="NODE.PARAM=VALUE")
  s = sub.add_parser("script", help="print the Python script of a template or project")
  s.add_argument("pipeline")
  args = ap.parse_args(argv)
  fn = {"ops": cmd_ops, "run": cmd_run, "batch": cmd_batch, "script": cmd_script, "gui": cmd_gui,
        None: cmd_gui}[args.cmd]
  fn(args)
  return 0


if __name__ == "__main__":
  sys.exit(main())
