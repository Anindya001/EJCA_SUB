"""Export a processing graph as a standalone Python script.

Operations that are single MTEX calls write themselves as MTEX code, so the script reads
like one written by hand. Operations implemented in the workbench (the statistics, the
image segmentation) are called through `run_op`, which executes exactly the code the
interface ran. In the script every variable holds the plain value (an EBSD map, a dict
of columns, an array), named <node id>_<output>.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from .pipeline import Pipeline
from .provenance import environment
from .types import DataItem


def var(node_id: str, output: str) -> str:
  return f"{node_id}_{output}"


def run_op(op_id: str, inputs: dict, params: dict, meta: dict | None = None) -> dict:
  """run a registered operation on plain values; the script-side twin of a graph node

  meta gives the metadata the inputs carried when the analysis ran (a pixel size, the
  threshold grains were reconstructed with), which plain values cannot carry"""
  from .registry import load_builtin_operations
  op = load_builtin_operations()[op_id]
  meta = meta or {}
  items = {}
  for port in op.inputs:
    v = inputs[port.name]
    items[port.name] = v if isinstance(v, DataItem) else DataItem(_guess_kind(port, v), v, dict(meta.get(port.name, {})))
  return {k: item.value for k, item in op.run(items, params).items()}


def _plain(o):
  import numpy as np
  if isinstance(o, np.generic):
    return o.item()
  if isinstance(o, np.ndarray):
    return o.tolist()
  return str(o)


def _guess_kind(port, value):
  if len(port.kinds) == 1:
    return port.kinds[0]
  from .types import DataKind
  if isinstance(value, dict):
    return DataKind.SCALAR_MAP if "values" in value and "ebsd" in value and DataKind.SCALAR_MAP in port.kinds \
      else DataKind.TABLE
  name = type(value).__name__
  for k, cls in ((DataKind.GRAINS, "grain2d"), (DataKind.EBSD, "EBSD")):
    if name == cls and k in port.kinds:
      return k
  return port.kinds[0]


def generic_script(op_id: str, out: dict, inp: dict, params: dict, meta: dict | None = None) -> str:
  args = "{" + ", ".join(f"{k!r}: {v}" for k, v in inp.items()) + "}"
  meta = {k: m for k, m in (meta or {}).items() if m}
  extra = f", meta=json.loads({json.dumps(meta, default=_plain)!r})" if meta else ""
  lines = [f"_r = run_op({op_id!r}, {args}, {params!r}{extra})"]
  lines += [f"{v} = _r[{k!r}]" for k, v in out.items()]
  return "\n".join(lines)


def export_script(pipeline: Pipeline, targets: list[str] | None = None) -> str:
  ids = list(pipeline.nodes)
  if targets:
    keep = set()
    for t in targets:
      keep.update(pipeline.upstream(t))
    ids = [i for i in ids if i in keep]
  env = environment()
  imports = {"import numpy as np", "from mtex import *"}
  body, uses_run_op = [], False
  for nid in ids:
    node = pipeline.nodes[nid]
    op = pipeline.registry[node.op_id]
    out = {o.name: var(nid, o.name) for o in op.outputs}
    inp = {k: var(r.node, r.output) for k, r in node.inputs.items()}
    body.append(f"# {nid} · {node.label}  [{op.id}]")
    if "file_sha256" in node.provenance:
      body.append(f"# input file sha256: {node.provenance['file_sha256']}")
    if op.script is not None:
      body.append(op.script(out, inp, dict(node.params)))
      imports.update(op.imports)
    else:
      meta = {}
      for k, r in node.inputs.items():
        up = pipeline.nodes[r.node].outputs.get(r.output)
        meta[k] = up.meta if up is not None else {}
      body.append(generic_script(op.id, out, inp, dict(node.params), meta))
      uses_run_op = True
    body.append("")
  if uses_run_op:
    imports.update({"import json", "from mtex_workbench.core.script import run_op"})
  stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
  head = [
    '"""Analysis exported from MTEX Workbench',
    "",
    f"exported      {stamp}",
    f"workbench     {env['mtex-workbench']}",
    f"mtex          {env['mtex']}",
    f"python        {env['python']}",
    f"numpy/scipy   {env['numpy']} / {env['scipy']}",
    '"""',
  ]
  ordered = sorted(i for i in imports if i.startswith("import ")) + \
            sorted(i for i in imports if i.startswith("from "))
  return "\n".join(head + ordered + ["", ""] + body).rstrip() + "\n"
