"""Project files: SQLite for the graph and its history, HDF5 for the numbers.

A project is a folder `name.mtexwb` holding

  project.sqlite   the nodes, their parameters, inputs, status and provenance, the history
  results.h5       /nodes/<node id>/<output name>, one group per computed output

Arrays, tables and scalar maps are stored as plain HDF5 datasets that any HDF5 tool reads.
Statistics are stored as JSON. MTEX objects (maps, grains, ODFs) have no portable HDF5
form, so they are stored pickled together with the MTEX and Python versions that wrote
them; a project opened with a different MTEX version recomputes them from the recorded
parameters instead of trusting a pickle across versions.
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import sqlite3
from pathlib import Path

import numpy as np

from .pipeline import Pipeline, Ref, Status
from .provenance import environment
from .registry import REGISTRY
from .types import DataItem, DataKind

FORMAT_VERSION = 1
SUFFIX = ".mtexwb"

_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE nodes (id TEXT PRIMARY KEY, seq INTEGER, op_id TEXT, label TEXT, params TEXT,
                    inputs TEXT, status TEXT, error TEXT, provenance TEXT, created TEXT);
CREATE TABLE history (seq INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT, event TEXT, node TEXT,
                      detail TEXT);
"""


# -- JSON that keeps numpy arrays ------------------------------------------------------------

def _encode(o):
  if isinstance(o, np.ndarray):
    return {"__ndarray__": o.tolist(), "dtype": str(o.dtype)}
  if isinstance(o, np.generic):
    return o.item()
  raise TypeError(f"cannot store {type(o).__name__} as JSON")


def _decode(d):
  if "__ndarray__" in d:
    return np.asarray(d["__ndarray__"], dtype=d["dtype"])
  return d


def dumps(o) -> str:
  return json.dumps(o, default=_encode, allow_nan=True)


def loads(s: str):
  return json.loads(s, object_hook=_decode)


# -- HDF5 encoding of one output -------------------------------------------------------------

def _write_item(group, item: DataItem) -> None:
  import h5py
  group.attrs["kind"] = item.kind.value
  group.attrs["meta"] = dumps(item.meta)
  k, v = item.kind, item.value
  if k in (DataKind.IMAGE, DataKind.LABELS):
    group.attrs["encoding"] = "array"
    group.create_dataset("data", data=np.asarray(v), compression="gzip")
  elif k == DataKind.TABLE:
    group.attrs["encoding"] = "table"
    group.attrs["columns"] = dumps(list(v))
    cols = group.create_group("columns")
    for i, (name, col) in enumerate(v.items()):
      col = np.asarray(col)
      if col.dtype.kind in "OUS":
        col = col.astype(object).astype(str).astype(h5py.string_dtype())
      cols.create_dataset(f"c{i}", data=col)
  elif k == DataKind.SCALAR_MAP:
    group.attrs["encoding"] = "scalar_map"
    group.create_dataset("values", data=np.asarray(v["values"], float), compression="gzip")
    _write_pickle(group, "ebsd", v["ebsd"])
    rest = {kk: vv for kk, vv in v.items() if kk not in ("values", "ebsd")}
    group.attrs["extra"] = dumps(rest)
  elif k == DataKind.STATS:
    group.attrs["encoding"] = "json"
    group.attrs["data"] = dumps(v)
  else:
    group.attrs["encoding"] = "pickle"
    _write_pickle(group, "object", v)


def _write_pickle(group, name, obj) -> None:
  blob = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
  ds = group.create_dataset(name, data=np.frombuffer(blob, dtype=np.uint8), compression="gzip")
  env = environment()
  ds.attrs["mtex"] = env["mtex"] or ""
  ds.attrs["python"] = env["python"]


class VersionMismatch(RuntimeError):
  pass


def _read_pickle(ds):
  env = environment()
  if ds.attrs.get("mtex", "") != (env["mtex"] or "") or \
     ds.attrs.get("python", "").split(".")[:2] != env["python"].split(".")[:2]:
    raise VersionMismatch(f"written by mtex {ds.attrs.get('mtex')} / Python {ds.attrs.get('python')}")
  return pickle.loads(ds[()].tobytes())


def _read_item(group) -> DataItem:
  kind = DataKind(group.attrs["kind"])
  meta = loads(group.attrs["meta"])
  enc = group.attrs["encoding"]
  if enc == "array":
    value = group["data"][()]
  elif enc == "table":
    names = loads(group.attrs["columns"])
    value = {}
    for i, name in enumerate(names):
      col = group["columns"][f"c{i}"][()]
      if col.dtype.kind in "OS":
        col = np.asarray([c.decode() if isinstance(c, bytes) else c for c in col], dtype=str)
      value[name] = col
  elif enc == "scalar_map":
    value = {"values": group["values"][()], "ebsd": _read_pickle(group["ebsd"]), **loads(group.attrs["extra"])}
  elif enc == "json":
    value = loads(group.attrs["data"])
  else:
    value = _read_pickle(group["object"])
  return DataItem(kind, value, meta)


# -- the project -----------------------------------------------------------------------------

def project_path(path) -> Path:
  p = Path(path)
  return p if p.suffix == SUFFIX else p.with_name(p.name + SUFFIX)


def save_project(pipeline: Pipeline, path, name: str | None = None) -> Path:
  """write the project atomically: a crash mid-save leaves the previous version intact"""
  import h5py
  final = project_path(path)
  tmp = final.with_name(final.name + ".saving")
  if tmp.exists():
    shutil.rmtree(tmp)
  tmp.mkdir(parents=True)
  db = sqlite3.connect(tmp / "project.sqlite")
  try:
    db.executescript(_SCHEMA)
    env = environment()
    meta = {"format_version": FORMAT_VERSION, "name": name or final.stem,
            "workbench": env["mtex-workbench"], "mtex": env["mtex"]}
    db.executemany("INSERT INTO meta VALUES (?, ?)", [(k, json.dumps(v)) for k, v in meta.items()])
    for seq, n in enumerate(pipeline.nodes.values()):
      db.execute("INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?)",
                 (n.id, seq, n.op_id, n.label, dumps(n.params),
                  json.dumps({k: r.as_list() for k, r in n.inputs.items()}),
                  n.status.value if n.status != Status.RUNNING else Status.STALE.value,
                  n.error, dumps(n.provenance), n.created))
    db.executemany("INSERT INTO history (time, event, node, detail) VALUES (?,?,?,?)",
                   [(h["time"], h["event"], h["node"],
                     dumps({k: v for k, v in h.items() if k not in ("time", "event", "node")}))
                    for h in pipeline.history])
    db.commit()
  finally:
    db.close()
  with h5py.File(tmp / "results.h5", "w") as h5:
    h5.attrs["format_version"] = FORMAT_VERSION
    root = h5.create_group("nodes")
    for n in pipeline.nodes.values():
      if n.status != Status.DONE:
        continue
      g = root.create_group(n.id)
      for out_name, item in n.outputs.items():
        _write_item(g.create_group(out_name), item)
  if final.exists():
    old = final.with_name(final.name + ".old")
    if old.exists():
      shutil.rmtree(old)
    os.replace(final, old)
    os.replace(tmp, final)
    shutil.rmtree(old)
  else:
    os.replace(tmp, final)
  return final


def load_project(path, registry=REGISTRY) -> tuple[Pipeline, list[str]]:
  """the pipeline with every result that could be restored, and warnings for the rest"""
  import h5py
  from .registry import load_builtin_operations
  load_builtin_operations()
  root = project_path(path)
  db = sqlite3.connect(root / "project.sqlite")
  try:
    meta = {k: json.loads(v) for k, v in db.execute("SELECT key, value FROM meta")}
    if meta.get("format_version") != FORMAT_VERSION:
      raise RuntimeError(f"unsupported project format {meta.get('format_version')}")
    rows = db.execute("SELECT id, op_id, label, params, inputs, status, error, provenance, created "
                      "FROM nodes ORDER BY seq").fetchall()
    history = db.execute("SELECT time, event, node, detail FROM history ORDER BY seq").fetchall()
  finally:
    db.close()
  p = Pipeline(registry)
  warnings = []
  for nid, op_id, label, params, inputs, status, error, prov, created in rows:
    node = p.add(op_id, loads(params), {k: Ref(*v) for k, v in json.loads(inputs).items()}, label, node_id=nid)
    node.status, node.error, node.provenance, node.created = Status(status), error, loads(prov), created
  p.history = [{"time": t, "event": e, "node": n, **loads(d)} for t, e, n, d in history]
  with h5py.File(root / "results.h5", "r") as h5:
    for node in p.nodes.values():
      if node.status != Status.DONE:
        continue
      try:
        g = h5["nodes"][node.id]
        node.outputs = {name: _read_item(g[name]) for name in g}
      except (KeyError, VersionMismatch) as exc:
        node.status, node.outputs = Status.STALE, {}
        warnings.append(f"{node.id} ({node.label}): result not restored ({exc}); it will be recomputed")
  # a node is only DONE if everything it depends on is
  for node in p.nodes.values():
    if node.status == Status.DONE and any(p.nodes[r.node].status != Status.DONE for r in node.inputs.values()):
      node.status, node.outputs = Status.STALE, {}
  return p, warnings
