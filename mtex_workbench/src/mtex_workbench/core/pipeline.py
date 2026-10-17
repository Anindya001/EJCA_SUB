"""The processing graph.

Every analysis step is a node: an operation, its validated parameters and references to the
outputs of upstream nodes. Nodes never modify their inputs; each run stores new outputs on
the node itself. Changing a node's parameters or inputs invalidates it and everything
downstream of it, so a result on screen is always consistent with the parameters shown
next to it. The graph is acyclic by construction: an input may only reference a node that
is not downstream of the node being connected.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Iterable

from .registry import REGISTRY, Registry
from .types import DataItem, DataKind

SPEC_VERSION = 1


class Status(str, Enum):
  STALE = "stale"      # parameters known, outputs not (yet) computed
  RUNNING = "running"
  DONE = "done"
  ERROR = "error"


@dataclass(frozen=True)
class Ref:
  node: str
  output: str

  def as_list(self) -> list:
    return [self.node, self.output]


@dataclass
class Node:
  id: str
  op_id: str
  params: dict
  inputs: dict[str, Ref]
  label: str
  status: Status = Status.STALE
  outputs: dict[str, DataItem] = field(default_factory=dict)
  error: str = ""
  provenance: dict = field(default_factory=dict)
  created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


class PipelineError(RuntimeError):
  pass


class Pipeline:
  def __init__(self, registry: Registry = REGISTRY):
    self.registry = registry
    self.nodes: dict[str, Node] = {}
    self._next = 1
    self._listeners: list[Callable[[str, str], None]] = []
    self.history: list[dict] = []  # an append-only log of what was done, for the report

  # -- events -------------------------------------------------------------------------------

  def subscribe(self, fn: Callable[[str, str], None]) -> None:
    """fn(event, node_id) on add, remove, invalidate, start, done, error"""
    self._listeners.append(fn)

  def _emit(self, event: str, node_id: str, **detail) -> None:
    self.history.append({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         "event": event, "node": node_id, **detail})
    for fn in list(self._listeners):
      fn(event, node_id)

  # -- construction -------------------------------------------------------------------------

  def add(self, op_id: str, params: dict | None = None, inputs: dict | None = None,
          label: str | None = None, node_id: str | None = None) -> Node:
    op = self.registry[op_id]
    params = op.validate(params or {})
    refs = {k: (v if isinstance(v, Ref) else Ref(*v)) for k, v in (inputs or {}).items()}
    ports = {p.name for p in op.inputs}
    if set(refs) != ports:
      raise PipelineError(f"{op.label}: inputs {sorted(ports)} expected, got {sorted(refs)}")
    for port in op.inputs:
      self._check_ref(port, refs[port.name])
    if node_id is None:
      node_id = f"n{self._next}"
    if node_id in self.nodes:
      raise PipelineError(f"node {node_id} exists")
    self._bump(node_id)
    node = Node(node_id, op_id, params, refs, label or op.label)
    self.nodes[node_id] = node
    self._emit("add", node_id, op=op_id, params=params)
    return node

  def _bump(self, node_id: str) -> None:
    if node_id.startswith("n") and node_id[1:].isdigit():
      self._next = max(self._next, int(node_id[1:]) + 1)

  def _check_ref(self, port, ref: Ref) -> None:
    if ref.node not in self.nodes:
      raise PipelineError(f"input {port.name!r} references unknown node {ref.node}")
    src = self.registry[self.nodes[ref.node].op_id]
    kind = src.output(ref.output).kind
    if not port.accepts(kind):
      raise PipelineError(f"input {port.name!r} takes {[k.value for k in port.kinds]}, "
                          f"{ref.node}.{ref.output} is {kind.value}")

  def set_params(self, node_id: str, params: dict) -> list[str]:
    """new parameters for a node; returns the nodes invalidated by the change"""
    node = self.nodes[node_id]
    new = self.registry[node.op_id].validate({**node.params, **params})
    if new == node.params:
      return []
    node.params = new
    self._emit("params", node_id, params=new)
    return self.invalidate(node_id)

  def set_input(self, node_id: str, port_name: str, ref: Ref) -> list[str]:
    node = self.nodes[node_id]
    op = self.registry[node.op_id]
    port = next((p for p in op.inputs if p.name == port_name), None)
    if port is None:
      raise PipelineError(f"{op.label} has no input {port_name!r}")
    if ref.node == node_id or ref.node in self.descendants(node_id):
      raise PipelineError("that connection would create a cycle")
    self._check_ref(port, ref)
    node.inputs[port_name] = ref
    return self.invalidate(node_id)

  def remove(self, node_id: str) -> list[str]:
    """remove a node and everything that depends on it"""
    doomed = [node_id] + self.descendants(node_id)
    for nid in reversed(doomed):
      del self.nodes[nid]
      self._emit("remove", nid)
    return doomed

  # -- topology -----------------------------------------------------------------------------

  def children(self, node_id: str) -> list[str]:
    return [n.id for n in self.nodes.values() if any(r.node == node_id for r in n.inputs.values())]

  def descendants(self, node_id: str) -> list[str]:
    """every node downstream, in topological order"""
    seen, stack = set(), [node_id]
    while stack:
      for c in self.children(stack.pop()):
        if c not in seen:
          seen.add(c)
          stack.append(c)
    # insertion order of self.nodes is a topological order, since inputs precede users
    return [nid for nid in self.nodes if nid in seen]

  def upstream(self, node_id: str) -> list[str]:
    """the node and everything it needs, in topological order"""
    seen, stack = {node_id}, [node_id]
    while stack:
      for r in self.nodes[stack.pop()].inputs.values():
        if r.node not in seen:
          seen.add(r.node)
          stack.append(r.node)
    return [nid for nid in self.nodes if nid in seen]

  def roots(self) -> list[str]:
    return [n.id for n in self.nodes.values() if not n.inputs]

  # -- state --------------------------------------------------------------------------------

  def invalidate(self, node_id: str) -> list[str]:
    ids = [node_id] + self.descendants(node_id)
    for nid in ids:
      n = self.nodes[nid]
      n.status, n.outputs, n.error = Status.STALE, {}, ""
      self._emit("invalidate", nid)
    return ids

  def output(self, ref: Ref) -> DataItem:
    node = self.nodes[ref.node]
    if node.status != Status.DONE:
      raise PipelineError(f"{node.label} ({node.id}) has not been run")
    return node.outputs[ref.output]

  def available_outputs(self, kinds: Iterable[DataKind] | None = None, exclude_downstream_of: str | None = None):
    """(Ref, label, kind) for every declared output, optionally of the given kinds"""
    kinds = set(kinds) if kinds is not None else None
    banned = set()
    if exclude_downstream_of is not None:
      banned = {exclude_downstream_of, *self.descendants(exclude_downstream_of)}
    out = []
    for n in self.nodes.values():
      if n.id in banned:
        continue
      for o in self.registry[n.op_id].outputs:
        if kinds is None or o.kind in kinds:
          out.append((Ref(n.id, o.name), f"{n.id} · {n.label} → {o.label or o.name}", o.kind))
    return out

  def fingerprint(self, node_id: str) -> str:
    """a hash of everything that determines a node's outputs: the operation and its version,
    the parameters, the fingerprints of the inputs and the content of any source file"""
    node = self.nodes[node_id]
    op = self.registry[node.op_id]
    payload = {
      "op": node.op_id, "version": op.version, "params": node.params,
      "inputs": {k: [self.fingerprint(r.node), r.output] for k, r in sorted(node.inputs.items())},
      "file": node.provenance.get("file_sha256"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

  # -- execution ----------------------------------------------------------------------------

  def run(self, node_id: str, executor=None, force: bool = False) -> Node:
    """compute the node and every stale node upstream of it"""
    from .executor import InlineExecutor
    executor = executor or InlineExecutor(self.registry)
    for nid in self.upstream(node_id):
      node = self.nodes[nid]
      if node.status == Status.DONE and not (force and nid == node_id):
        continue
      self._run_one(node, executor)
    return self.nodes[node_id]

  def run_all(self, executor=None) -> None:
    for nid in list(self.nodes):
      if self.nodes[nid].status != Status.DONE:
        self.run(nid, executor)

  def _run_one(self, node: Node, executor) -> None:
    from .provenance import environment, sha256_file
    op = self.registry[node.op_id]
    inputs = {k: self.output(r) for k, r in node.inputs.items()}
    node.status, node.error = Status.RUNNING, ""
    self._emit("start", node.id)
    t0, started = time.perf_counter(), datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
      prov = {}
      if op.hashes_file:
        prov["file_sha256"] = sha256_file(node.params[op.hashes_file])
      outputs = executor.execute(node.op_id, inputs, node.params)
    except Exception as exc:
      node.status, node.error = Status.ERROR, f"{exc.__class__.__name__}: {exc}"
      node.provenance = {"started": started, "duration_s": time.perf_counter() - t0}
      self._emit("error", node.id, error=node.error)
      raise
    node.outputs = outputs
    node.status = Status.DONE
    node.provenance = {**prov, "started": started, "duration_s": round(time.perf_counter() - t0, 4),
                       "op_version": op.version, "environment": environment()}
    node.provenance["fingerprint"] = self.fingerprint(node.id)
    self._emit("done", node.id, duration_s=node.provenance["duration_s"])

  def stale_sources(self) -> list[str]:
    """source nodes whose file changed on disk since they were run"""
    from .provenance import sha256_file
    changed = []
    for n in self.nodes.values():
      op = self.registry[n.op_id]
      if op.hashes_file and n.status == Status.DONE:
        try:
          now = sha256_file(n.params[op.hashes_file])
        except OSError:
          now = None
        if now != n.provenance.get("file_sha256"):
          changed.append(n.id)
    return changed

  # -- serialisation ------------------------------------------------------------------------

  def to_spec(self) -> dict:
    """the graph without results: what a template, a batch run or a project file needs"""
    return {"spec_version": SPEC_VERSION, "nodes": [
      {"id": n.id, "op": n.op_id, "label": n.label, "params": n.params,
       "inputs": {k: r.as_list() for k, r in n.inputs.items()}}
      for n in self.nodes.values()]}

  @classmethod
  def from_spec(cls, spec: dict, registry: Registry = REGISTRY) -> "Pipeline":
    if spec.get("spec_version") != SPEC_VERSION:
      raise PipelineError(f"unsupported pipeline spec version {spec.get('spec_version')}")
    p = cls(registry)
    for d in spec["nodes"]:
      p.add(d["op"], d.get("params", {}), {k: Ref(*v) for k, v in d.get("inputs", {}).items()},
            d.get("label"), node_id=d["id"])
    p.history.clear()
    return p
