"""The operation registry.

An operation declares what it accepts, what it produces, which parameters it takes and how
to write itself as a line of a reproducible Python script. The interface is generated from
these declarations: the library lists only the operations whose primary input accepts the
kind of the selected dataset, and the inspector builds its controls from the parameters.
Adding an operation therefore never needs a new dialog.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from .params import Param, validate_params
from .types import DataItem, DataKind

# the library's menu, in display order; the last four are reached through the function
# browser and the console until they get dedicated operations
CATEGORIES = (
  "Import & Data",
  "Pre-processing",
  "Image Processing",
  "Grain Analysis",
  "Local Deformation",
  "Texture & ODF",
  "Statistics",
  "Crystallographic Analysis",
  "Material Properties",
  "3D Microstructure",
  "Advanced Mathematics",
)


@dataclass(frozen=True)
class Port:
  name: str
  kinds: tuple[DataKind, ...]
  label: str = ""

  def accepts(self, kind: DataKind) -> bool:
    return kind in self.kinds


@dataclass(frozen=True)
class Output:
  name: str
  kind: DataKind
  label: str = ""


ScriptFn = Callable[[dict, dict, dict], str]


@dataclass(frozen=True)
class Operation:
  id: str
  label: str
  category: str
  func: Callable
  inputs: tuple[Port, ...] = ()
  outputs: tuple[Output, ...] = ()
  params: tuple[Param, ...] = ()
  doc: str = ""
  script: ScriptFn | None = None
  imports: tuple[str, ...] = ()
  version: int = 1
  hashes_file: str | None = None  # the path parameter whose file content is hashed

  @property
  def is_source(self) -> bool:
    return not self.inputs

  def accepts(self, kind: DataKind) -> bool:
    """whether the primary input takes this kind"""
    return bool(self.inputs) and self.inputs[0].accepts(kind)

  def output(self, name: str) -> Output:
    for o in self.outputs:
      if o.name == name:
        return o
    raise KeyError(f"{self.id} has no output {name!r}")

  def validate(self, params: dict) -> dict:
    return validate_params(self.params, params)

  def run(self, inputs: dict[str, DataItem], params: dict) -> dict[str, DataItem]:
    """the outputs, each checked against the declared kind"""
    missing = [p.name for p in self.inputs if p.name not in inputs]
    if missing:
      raise ValueError(f"{self.label}: missing inputs {missing}")
    for p in self.inputs:
      if not p.accepts(inputs[p.name].kind):
        raise TypeError(f"{self.label}: input {p.name!r} takes {[k.value for k in p.kinds]}, "
                        f"got {inputs[p.name].kind.value}")
    params = self.validate(params)
    result = self.func(**inputs, **params)
    if not isinstance(result, dict):
      if len(self.outputs) != 1:
        raise TypeError(f"{self.id} must return a dict of its {len(self.outputs)} outputs")
      result = {self.outputs[0].name: result}
    out = {}
    for o in self.outputs:
      if o.name not in result:
        raise ValueError(f"{self.id} did not produce output {o.name!r}")
      v = result[o.name]
      item = v if isinstance(v, DataItem) else DataItem(o.kind, v)
      if item.kind != o.kind:
        raise TypeError(f"{self.id}: output {o.name!r} declared {o.kind.value}, produced {item.kind.value}")
      out[o.name] = item
    return out


class Registry:
  def __init__(self):
    self._ops: dict[str, Operation] = {}

  def add(self, op: Operation) -> Operation:
    if op.id in self._ops:
      raise ValueError(f"operation {op.id!r} registered twice")
    if op.category not in CATEGORIES:
      raise ValueError(f"{op.id}: unknown category {op.category!r}")
    names = [p.name for p in op.params]
    if len(set(names)) != len(names):
      raise ValueError(f"{op.id}: duplicate parameter names")
    self._ops[op.id] = op
    return op

  def __getitem__(self, op_id: str) -> Operation:
    try:
      return self._ops[op_id]
    except KeyError:
      raise KeyError(f"unknown operation {op_id!r}") from None

  def __contains__(self, op_id: str) -> bool:
    return op_id in self._ops

  def __iter__(self):
    return iter(self._ops.values())

  def __len__(self):
    return len(self._ops)

  def by_category(self) -> dict[str, list[Operation]]:
    out = {c: [] for c in CATEGORIES}
    for op in self._ops.values():
      out[op.category].append(op)
    return out

  def applicable(self, kind: DataKind | None) -> list[Operation]:
    """the operations a dataset of this kind can feed; the sources when nothing is selected"""
    if kind is None:
      return [op for op in self._ops.values() if op.is_source]
    return [op for op in self._ops.values() if op.accepts(kind)]

  def search(self, text: str) -> list[Operation]:
    t = text.lower().strip()
    return [op for op in self._ops.values()
            if t in op.label.lower() or t in op.id.lower() or t in op.doc.lower()]


REGISTRY = Registry()


def operation(id: str, label: str, category: str, *, inputs: Sequence[Port] = (),
              outputs: Sequence[Output] = (), params: Iterable[Param] = (),
              script: ScriptFn | None = None, imports: Sequence[str] = (), version: int = 1,
              hashes_file: str | None = None, registry: Registry = REGISTRY):
  """register the decorated function as an operation; the docstring is its help text"""
  def deco(func):
    registry.add(Operation(id=id, label=label, category=category, func=func,
                           inputs=tuple(inputs), outputs=tuple(outputs), params=tuple(params),
                           doc=(func.__doc__ or "").strip(), script=script,
                           imports=tuple(imports), version=version, hashes_file=hashes_file))
    return func
  return deco


def load_builtin_operations() -> Registry:
  """import the operation modules, which register themselves; idempotent"""
  from .. import ops  # noqa: F401
  return REGISTRY
