"""Export of an analysis: a PDF report and the machine-readable files behind it.

  report.pdf        a title page with the provenance, then one page per result
  figures/*.png     every figure
  tables/*.csv      every table
  stats/*.json      every statistical result, arrays included
  pipeline.json     the graph as a reusable template (parameters, no results)
  provenance.json   software versions, input file hashes, run times, the history
  analysis.py       the script that reproduces the results
"""
from __future__ import annotations

import csv
import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .core.pipeline import Pipeline, Status
from .core.provenance import environment
from .core.script import export_script
from .core.storage import dumps
from .core.types import DataKind


def write_table_csv(table: dict, path: Path) -> None:
  cols = list(table)
  with open(path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(cols)
    for row in zip(*(np.asarray(table[c]).tolist() for c in cols)):
      w.writerow(row)


def provenance(pipeline: Pipeline) -> dict:
  return {
    "exported": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "environment": environment(),
    "nodes": {n.id: {"op": n.op_id, "label": n.label, "params": n.params, "status": n.status.value,
                     "inputs": {k: r.as_list() for k, r in n.inputs.items()}, **n.provenance}
              for n in pipeline.nodes.values()},
    "history": pipeline.history,
  }


def _caption(pipeline, node, out_name, item) -> str:
  params = ", ".join(f"{k} = {v}" for k, v in node.params.items()) or "no parameters"
  src = ", ".join(f"{k} ← {r.node}.{r.output}" for k, r in node.inputs.items())
  return f"{node.id} · {node.label} → {out_name}   [{params}]" + (f"   inputs: {src}" if src else "")


def _title_page(pipeline: Pipeline, name: str):
  from matplotlib.figure import Figure
  fig = Figure(figsize=(8.27, 11.69))
  env = environment()
  lines = [name, "", f"MTEX Workbench report · {datetime.now().strftime('%Y-%m-%d %H:%M')}", "",
           "Software"]
  lines += [f"  {k:<16}{v}" for k, v in env.items() if v]
  lines += ["", "Processing steps"]
  for n in pipeline.nodes.values():
    params = ", ".join(f"{k}={v}" for k, v in n.params.items())
    lines += textwrap.wrap(f"{n.id:>4}  {n.label} [{n.status.value}]  {params}", 92,
                           subsequent_indent=" " * 6)
    if "file_sha256" in n.provenance:
      lines.append(f"      sha256 {n.provenance['file_sha256']}")
  fig.text(0.06, 0.96, "\n".join(lines[:110]), va="top", ha="left", family="monospace", fontsize=7.5)
  return fig


def export_results(pipeline: Pipeline, outdir, name: str = "MTEX Workbench analysis",
                   overlay_grains: bool = True) -> dict:
  """write everything computed so far; returns the paths written"""
  from matplotlib.backends.backend_pdf import PdfPages
  from .viz import render
  out = Path(outdir)
  for sub in ("figures", "tables", "stats"):
    (out / sub).mkdir(parents=True, exist_ok=True)
  written = {"figures": [], "tables": [], "stats": []}
  (out / "pipeline.json").write_text(json.dumps(pipeline.to_spec(), indent=2))
  (out / "provenance.json").write_text(dumps(provenance(pipeline)))
  (out / "analysis.py").write_text(export_script(pipeline))
  pdf_path = out / "report.pdf"
  with PdfPages(pdf_path) as pdf:
    pdf.savefig(_title_page(pipeline, name))
    for node in pipeline.nodes.values():
      if node.status != Status.DONE:
        continue
      grains = next((o for o in node.outputs.values() if o.kind == DataKind.GRAINS), None)
      if grains is None:
        for r in node.inputs.values():
          up = pipeline.nodes[r.node].outputs
          grains = next((o for o in up.values() if o.kind == DataKind.GRAINS), None) or grains
      for out_name, item in node.outputs.items():
        stem = f"{node.id}_{out_name}"
        if item.kind == DataKind.TABLE:
          p = out / "tables" / f"{stem}.csv"
          write_table_csv(item.value, p)
          written["tables"].append(str(p))
        if item.kind == DataKind.STATS:
          p = out / "stats" / f"{stem}.json"
          p.write_text(dumps({"meta": item.meta, "result": item.value}))
          written["stats"].append(str(p))
        overlay = grains if overlay_grains and item.kind == DataKind.SCALAR_MAP else None
        fig = render(item, overlay=overlay)
        if fig is None:
          continue
        fig.text(0.01, 0.003, textwrap.fill(_caption(pipeline, node, out_name, item), 160), fontsize=6,
                 va="bottom", color="#555")
        p = out / "figures" / f"{stem}.png"
        fig.savefig(p, dpi=150)
        written["figures"].append(str(p))
        pdf.savefig(fig)
  written["report"] = str(pdf_path)
  written["script"] = str(out / "analysis.py")
  return written
