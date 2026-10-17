"""Where operations run.

InlineExecutor runs in the calling thread: tests, scripts and batch runs. ProcessExecutor
keeps one warm worker process (MTEX imported once) and ships inputs and outputs across a
pipe; a long reconstruction can then be cancelled by terminating the worker, which a thread
cannot offer, and a crash inside a numerical library cannot take the interface down.
"""
from __future__ import annotations

import multiprocessing as mp
import threading
import traceback

from .registry import REGISTRY, Registry

STARTUP_TIMEOUT = 180  # seconds; the first start compiles MTEX's numba kernels


class Cancelled(RuntimeError):
  pass


class WorkerError(RuntimeError):
  def __init__(self, type_name: str, message: str, tb: str):
    super().__init__(f"{type_name}: {message}")
    self.type_name, self.remote_traceback = type_name, tb


class InlineExecutor:
  def __init__(self, registry: Registry = REGISTRY):
    self.registry = registry

  def execute(self, op_id, inputs, params):
    if op_id not in self.registry:
      from .registry import load_builtin_operations
      load_builtin_operations()
    return self.registry[op_id].run(inputs, params)

  def cancel(self):  # nothing to cancel in the calling thread
    pass

  def shutdown(self):
    pass


def _worker(conn):
  import matplotlib
  matplotlib.use("Agg")
  from .registry import load_builtin_operations
  registry = load_builtin_operations()
  conn.send(("ready", None))
  while True:
    try:
      msg = conn.recv()
    except EOFError:
      return
    if msg is None:
      return
    op_id, inputs, params = msg
    try:
      conn.send(("ok", registry[op_id].run(inputs, params)))
    except BaseException as exc:  # report everything; the parent decides
      conn.send(("err", (exc.__class__.__name__, str(exc), traceback.format_exc())))


class ProcessExecutor:
  def __init__(self):
    self._ctx = mp.get_context("spawn")
    self._proc = None
    self._conn = None
    self._lock = threading.Lock()
    self._cancelled = False

  @property
  def started(self) -> bool:
    return self._proc is not None and self._proc.is_alive()

  def _ensure(self):
    """the live worker and its pipe, started if needed"""
    proc, conn = self._proc, self._conn
    if proc is not None and proc.is_alive():
      return proc, conn
    parent, child = self._ctx.Pipe()
    proc = self._ctx.Process(target=_worker, args=(child,), daemon=True, name="mtex-workbench-worker")
    proc.start()
    child.close()
    if not parent.poll(STARTUP_TIMEOUT):
      proc.terminate()
      raise RuntimeError(f"the worker process did not start within {STARTUP_TIMEOUT} s; if this is a script, "
                         "guard its entry point with `if __name__ == '__main__':`")
    tag, _ = parent.recv()
    if tag != "ready":
      raise RuntimeError(f"unexpected message from the worker: {tag!r}")
    self._proc, self._conn = proc, parent
    return proc, parent

  def execute(self, op_id, inputs, params):
    with self._lock:
      self._cancelled = False
      try:
        proc, conn = self._ensure()
        conn.send((op_id, inputs, params))
        while not conn.poll(0.05):
          if self._cancelled or not proc.is_alive():
            break
        if self._cancelled:
          raise Cancelled("the run was cancelled")
        tag, payload = conn.recv()
      except (EOFError, OSError):
        if self._cancelled:
          raise Cancelled("the run was cancelled") from None
        self._proc = None
        raise RuntimeError("the worker process died") from None
    if tag == "err":
      raise WorkerError(*payload)
    return payload

  def cancel(self):
    """stop the running operation; the next run starts a fresh worker"""
    proc = self._proc
    self._proc = None  # detached first, so that a run started right after never picks it up
    self._cancelled = True
    if proc is not None and proc.is_alive():
      proc.terminate()
      proc.join(5)

  def shutdown(self):
    proc, conn = self._proc, self._conn
    self._proc = None
    if proc is not None and proc.is_alive():
      try:
        conn.send(None)
        proc.join(2)
      except OSError:
        pass
      if proc.is_alive():
        proc.terminate()
