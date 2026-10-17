"""Typed parameter specifications.

An operation declares its parameters once; the inspector panel builds its widgets from the
declaration and the engine validates every value against it before a run. Validation is
strict: an out-of-range or mistyped value is an error, never silently clipped, because a
clipped value would make a recorded analysis differ from the one the user believes ran.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence


class ParamError(ValueError):
  pass


@dataclass(frozen=True)
class Param:
  name: str
  label: str
  type: str  # float | int | bool | choice | str | text (multi-line) | path
  default: Any
  minimum: float | None = None
  maximum: float | None = None
  step: float | None = None
  unit: str = ""
  choices: Sequence[str] = field(default_factory=tuple)
  help: str = ""
  file_filter: str = ""  # for path parameters, a Qt file dialog filter

  def validate(self, value: Any) -> Any:
    """the value coerced to the declared type, or ParamError"""
    t = self.type
    if t == "float":
      if isinstance(value, bool):
        raise ParamError(f"{self.label}: expected a number, got {value!r}")
      try:
        v = float(value)
      except (TypeError, ValueError):
        raise ParamError(f"{self.label}: expected a number, got {value!r}") from None
      if not math.isfinite(v):
        raise ParamError(f"{self.label}: must be finite")
      self._range(v)
      return v
    if t == "int":
      if isinstance(value, bool):
        raise ParamError(f"{self.label}: expected an integer, got {value!r}")
      if isinstance(value, float) and not value.is_integer():
        raise ParamError(f"{self.label}: expected an integer, got {value!r}")
      try:
        v = int(value)
      except (TypeError, ValueError):
        raise ParamError(f"{self.label}: expected an integer, got {value!r}") from None
      self._range(v)
      return v
    if t == "bool":
      if isinstance(value, bool):
        return value
      if value in (0, 1):
        return bool(value)
      raise ParamError(f"{self.label}: expected true or false, got {value!r}")
    if t == "choice":
      if value not in self.choices:
        raise ParamError(f"{self.label}: {value!r} is not one of {list(self.choices)}")
      return value
    if t in ("str", "text", "path"):
      if not isinstance(value, str):
        raise ParamError(f"{self.label}: expected text, got {value!r}")
      if t == "path" and not value:
        raise ParamError(f"{self.label}: a file is required")
      return value
    raise ParamError(f"{self.name}: unknown parameter type {t!r}")

  def _range(self, v: float) -> None:
    if self.minimum is not None and v < self.minimum:
      raise ParamError(f"{self.label}: {v} is below the minimum {self.minimum}{self.unit}")
    if self.maximum is not None and v > self.maximum:
      raise ParamError(f"{self.label}: {v} is above the maximum {self.maximum}{self.unit}")


def validate_params(specs: Sequence[Param], values: dict) -> dict:
  """all declared parameters, defaults filled in, every value validated

  Unknown names are an error: a typo in a batch template must not pass silently."""
  names = {p.name for p in specs}
  unknown = set(values) - names
  if unknown:
    raise ParamError(f"unknown parameters: {sorted(unknown)}")
  out, errors = {}, []
  for p in specs:
    try:
      out[p.name] = p.validate(values.get(p.name, p.default))
    except ParamError as e:
      errors.append(str(e))
  if errors:
    raise ParamError("; ".join(errors))
  return out


# shorthands, so operation declarations stay one line per parameter
def Float(name, label, default, minimum=None, maximum=None, step=None, unit="", help=""):
  return Param(name, label, "float", default, minimum, maximum, step, unit, (), help)


def Int(name, label, default, minimum=None, maximum=None, unit="", help=""):
  return Param(name, label, "int", default, minimum, maximum, 1, unit, (), help)


def Bool(name, label, default, help=""):
  return Param(name, label, "bool", default, help=help)


def Choice(name, label, choices, default=None, help=""):
  choices = tuple(choices)
  return Param(name, label, "choice", choices[0] if default is None else default, choices=choices, help=help)


def Text(name, label, default="", help=""):
  return Param(name, label, "str", default, help=help)


def Code(name, label, default="", help=""):
  return Param(name, label, "text", default, help=help)


def Path(name, label, file_filter="All files (*)", help=""):
  return Param(name, label, "path", "", help=help, file_filter=file_filter)
