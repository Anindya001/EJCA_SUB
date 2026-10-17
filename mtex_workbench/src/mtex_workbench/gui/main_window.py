"""The main window: datasets on the left, the viewer in the centre, the inspector on the
right, the history, log and console at the bottom."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
  QApplication, QDockWidget, QFileDialog, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QMainWindow,
  QMenu, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QTableWidget, QTableWidgetItem,
  QTabWidget, QVBoxLayout, QWidget,
)

from .. import __version__
from ..core.executor import Cancelled, InlineExecutor, ProcessExecutor
from ..core.pipeline import Pipeline, PipelineError, Status
from ..core.registry import load_builtin_operations
from ..core.script import export_script
from ..core.storage import SUFFIX, load_project, save_project
from ..core.types import DataKind
from ..ops.io_ops import SAMPLE_EBSD
from .tools import BatchDialog, Console, FunctionBrowser
from .widgets import DatasetExplorer, Library, ParamForm, Viewer

ABOUT = f"""<h3>MTEX Workbench {__version__}</h3>
<p>A non-destructive analysis environment for EBSD maps, grains, textures and micrographs,
built on <b>MTEX for Python</b> (R. Hielscher et al., <i>mtex-toolbox.github.io</i>).</p>
<p>When publishing results, cite MTEX: F. Bachmann, R. Hielscher, H. Schaeben,
<i>Texture analysis with MTEX – free and open source software toolbox</i>,
Solid State Phenomena 160 (2010) 63–68.</p>
<p>MTEX Workbench is free software under the GNU General Public License version 2, the
licence of MTEX, and comes with ABSOLUTELY NO WARRANTY.</p>"""


def doc_html(doc: str) -> str:
  """a docstring as paragraphs that reflow with the panel"""
  paras = [" ".join(p.split()) for p in doc.split("\n\n") if p.strip()]
  return "".join(f"<p>{html.escape(p)}</p>" for p in paras)


class Bridge(QObject):
  """carries pipeline events from the worker thread to the interface thread"""
  event = Signal(str, str)


class RunThread(QThread):
  """computes the given nodes (and what they need) in turn; stops at the first failure"""
  succeeded = Signal(str)
  failed = Signal(str, str)

  def __init__(self, pipeline: Pipeline, node_ids: list[str], executor):
    super().__init__()
    self.pipeline, self.node_ids, self.executor = pipeline, node_ids, executor

  def run(self):
    for nid in self.node_ids:
      try:
        self.pipeline.run(nid, self.executor)
      except Cancelled:
        self.failed.emit(nid, "cancelled")
        return
      except Exception as exc:
        failed = next((n for n in self.pipeline.upstream(nid) if self.pipeline.nodes[n].status == Status.ERROR), nid)
        self.failed.emit(failed, f"{exc.__class__.__name__}: {exc}")
        return
    self.succeeded.emit(self.node_ids[-1])


class Inspector(QWidget):
  """the selected operation (a new step) or node (an existing one) and its parameters"""
  run_requested = Signal()

  def __init__(self, parent=None):
    super().__init__(parent)
    lay = QVBoxLayout(self)
    self.header = QLabel("No operation selected")
    self.header.setStyleSheet("font-size: 14px; font-weight: 600;")
    self.header.setWordWrap(True)
    self.sub = QLabel("")
    self.sub.setStyleSheet("color: #5b6b7a;")
    self.doc = QLabel("")
    self.doc.setWordWrap(True)
    self.doc.setTextFormat(Qt.RichText)
    self.doc.setTextInteractionFlags(Qt.TextSelectableByMouse)
    self.doc.setStyleSheet("color: #33414e;")
    self.form_area = QScrollArea()
    self.form_area.setWidgetResizable(True)
    self.status = QLabel("")
    self.status.setWordWrap(True)
    row = QHBoxLayout()
    self.run = QPushButton("Run")
    self.run.setDefault(True)
    self.run.clicked.connect(self.run_requested)
    self.reset = QPushButton("Defaults")
    self.reset.clicked.connect(self._defaults)
    row.addWidget(self.reset)
    row.addStretch(1)
    row.addWidget(self.run)
    lay.addWidget(self.header)
    lay.addWidget(self.sub)
    lay.addWidget(self.form_area, 1)
    lay.addWidget(self.status)
    lay.addLayout(row)
    doc_box = QScrollArea()
    doc_box.setWidgetResizable(True)
    doc_box.setWidget(self.doc)
    doc_box.setMaximumHeight(220)
    lay.addWidget(doc_box)
    self.form = None
    self.mode = None  # ("new", op_id) | ("edit", node_id)
    self.pipeline = None
    self.clear()

  def clear(self):
    self.mode, self.form = None, None
    self.header.setText("No operation selected")
    self.sub.setText("Choose an operation in the Processing Library, or select a dataset.")
    self.doc.setText("")
    self.form_area.setWidget(QWidget())
    self.status.setText("")
    self.run.setEnabled(False)
    self.reset.setEnabled(False)

  def show_new(self, pipeline, op, inputs=None):
    self.pipeline, self.mode = pipeline, ("new", op.id)
    self.header.setText(op.label)
    self.sub.setText(f"{op.category} · new step · {op.id}")
    self.run.setText("Add && run")
    self._form(ParamForm(op, pipeline, inputs=inputs), op)

  def show_node(self, pipeline, node_id):
    node = pipeline.nodes[node_id]
    op = pipeline.registry[node.op_id]
    self.pipeline, self.mode = pipeline, ("edit", node_id)
    self.header.setText(f"{node.id} · {node.label}")
    extra = ""
    if node.status == Status.DONE:
      extra = f" · ran {node.provenance.get('started', '')} in {node.provenance.get('duration_s', 0):.2f} s"
    self.sub.setText(f"{op.category} · {node.status.value}{extra}")
    self.run.setText("Apply && run")
    self._form(ParamForm(op, pipeline, node.params, node.inputs, node_id=node_id), op)
    if node.status == Status.ERROR:
      self.status.setText(f"<span style='color:#c0392b'>{node.error}</span>")

  def _form(self, form, op):
    self.form = form
    form.changed.connect(self._validate)
    self.form_area.setWidget(form)
    self.doc.setText(doc_html(op.doc))
    self.reset.setEnabled(bool(op.params))
    self._validate()

  def _defaults(self):
    if self.form is None:
      return
    op = self.form.op
    inputs = self.form.inputs()
    nid = self.mode[1] if self.mode and self.mode[0] == "edit" else None
    self._form(ParamForm(op, self.pipeline, {}, inputs, node_id=nid), op)

  def _validate(self):
    ok, msg = self.form.validate()
    colour = "#27ae60" if ok else "#c0392b"
    self.status.setText(f"<span style='color:{colour}'>{'✓' if ok else '✗'} {msg}</span>")
    self.run.setEnabled(ok)

  def set_busy(self, busy: bool):
    if self.form is not None:
      ok, _ = self.form.validate()
      self.run.setEnabled(ok and not busy)


class MainWindow(QMainWindow):
  def __init__(self, project: str | None = None):
    super().__init__()
    self.registry = load_builtin_operations()
    self.settings = QSettings("MTEX Workbench", "MTEX Workbench")
    self.setWindowTitle("MTEX Workbench")
    self.resize(1500, 920)
    self.bridge = Bridge()
    self.bridge.event.connect(self._on_event)
    self.pipeline = None
    self.project_path = None
    self.dirty = False
    self.thread = None
    self.process_executor = ProcessExecutor()
    self.inline_executor = InlineExecutor(self.registry)
    self.current = (None, None)
    self.pinned = None
    self.history_rows = 0

    # the docks
    self.explorer = DatasetExplorer(Pipeline(self.registry))
    self.explorer.selected.connect(self._on_select)
    self.explorer.setContextMenuPolicy(Qt.CustomContextMenu)
    self.explorer.customContextMenuRequested.connect(self._explorer_menu)
    self.library = Library(self.registry)
    self.library.chosen.connect(self._on_choose)
    self.inspector = Inspector()
    self.inspector.run_requested.connect(self._on_run_requested)
    self.viewer = Viewer()
    self.viewer.view_changed.connect(self._refresh_viewer)
    self.viewer.pin.toggled.connect(self._on_pin)
    self.setCentralWidget(self.viewer)

    self._dock("Dataset Explorer", self.explorer, Qt.LeftDockWidgetArea)
    self._dock("Processing Library", self.library, Qt.LeftDockWidgetArea)
    self._dock("Inspector", self.inspector, Qt.RightDockWidgetArea)
    self.history = QTableWidget(0, 4)
    self.history.setHorizontalHeaderLabels(["Time", "Event", "Node", "Detail"])
    self.history.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
    self.history.verticalHeader().setVisible(False)
    self.log = QPlainTextEdit()
    self.log.setReadOnly(True)
    self.console = Console(self._console_namespace())
    bottom = QTabWidget()
    bottom.addTab(self.history, "Processing History")
    bottom.addTab(self.log, "Log")
    bottom.addTab(self.console, "Python Console")
    self.bottom = bottom
    self._dock("History", bottom, Qt.BottomDockWidgetArea)

    self._menus()
    self.statusBar().showMessage("Ready")
    self._set_pipeline(Pipeline(self.registry))
    geo = self.settings.value("geometry")
    if geo is not None:
      self.restoreGeometry(geo)
    if project:
      QTimer.singleShot(0, lambda: self.open_project(project))

  # -- layout -------------------------------------------------------------------------------

  def _dock(self, title, widget, area):
    d = QDockWidget(title, self)
    d.setObjectName(title)
    d.setWidget(widget)
    self.addDockWidget(area, d)
    return d

  def _action(self, menu, text, slot, shortcut=None, checkable=False):
    a = QAction(text, self)
    if shortcut:
      a.setShortcut(QKeySequence(shortcut))
    a.setCheckable(checkable)
    a.triggered.connect(slot)
    menu.addAction(a)
    return a

  def _menus(self):
    mb = self.menuBar()
    m = mb.addMenu("&Project")
    self._action(m, "New project", self.new_project, "Ctrl+N")
    self._action(m, "Open project…", lambda: self.open_project(), "Ctrl+O")
    self._action(m, "Save", self.save_project, "Ctrl+S")
    self._action(m, "Save as…", lambda: self.save_project(ask=True), "Ctrl+Shift+S")
    m.addSeparator()
    self._action(m, "Save pipeline template…", self.save_template)
    self._action(m, "Load pipeline template…", self.load_template)
    m.addSeparator()
    self._action(m, "Quit", self.close, "Ctrl+Q")
    m = mb.addMenu("&Import")
    self._action(m, "EBSD file…", self.import_ebsd, "Ctrl+I")
    self._action(m, "MTEX import wizard…", self.import_wizard)
    self._action(m, "MTEX sample data…", self.import_sample)
    self._action(m, "Micrograph (image)…", self.import_image)
    m = mb.addMenu("&Run")
    self._action(m, "Run selected", self.run_selected, "F5")
    self._action(m, "Run all stale", self.run_all, "Shift+F5")
    self.cancel_action = self._action(m, "Cancel", self.cancel, "Esc")
    self.cancel_action.setEnabled(False)
    m.addSeparator()
    self.use_process = self._action(m, "Compute in worker process", lambda: None, checkable=True)
    self.use_process.setChecked(self.settings.value("use_process", True, type=bool))
    self.use_process.setToolTip("a separate process can be cancelled and cannot crash the interface")
    m = mb.addMenu("&Export")
    self._action(m, "Report and results…", self.export_report, "Ctrl+E")
    self._action(m, "Python script…", self.export_script)
    self._action(m, "Current figure…", self.export_figure)
    m = mb.addMenu("&Tools")
    self._action(m, "MTEX function browser…", self.function_browser, "Ctrl+F")
    self._action(m, "Batch processing…", self.batch)
    self._action(m, "Python console", lambda: (self.bottom.setCurrentWidget(self.console), self.console.input.setFocus()),
                 "Ctrl+`")
    m = mb.addMenu("&Help")
    self._action(m, "About MTEX Workbench", lambda: QMessageBox.about(self, "About", ABOUT))

  # -- the pipeline -------------------------------------------------------------------------

  def _set_pipeline(self, p: Pipeline):
    self.pipeline = p
    p.subscribe(lambda ev, nid: self.bridge.event.emit(ev, nid))
    self.explorer.pipeline = p
    self.console.ns["pipeline"] = p
    self.current = (None, None)
    self.pinned = None
    self.viewer.pin.setChecked(False)
    self.explorer.rebuild()
    self.inspector.clear()
    self.library.update_kind(None)
    self.history.setRowCount(0)
    self.history_rows = 0
    for h in p.history:
      self._history_row(h)
    self._refresh_viewer()
    self._title()

  def _title(self):
    name = Path(self.project_path).name if self.project_path else "untitled"
    self.setWindowTitle(f"MTEX Workbench — {name}{' •' if self.dirty else ''}")

  def _on_event(self, event, node_id):
    if event in ("add", "remove", "params", "done", "error"):
      self.dirty = True
      self._title()
    self.explorer.rebuild(keep=self.current)
    while self.history_rows < len(self.pipeline.history):
      self._history_row(self.pipeline.history[self.history_rows])
    if event == "start":
      self.statusBar().showMessage(f"Running {node_id} · {self.pipeline.nodes[node_id].label}…")
    elif event == "done":
      n = self.pipeline.nodes[node_id]
      self._log(f"{node_id} {n.label}: done in {n.provenance.get('duration_s', 0):.2f} s")
    elif event == "error":
      self._log(f"{node_id}: {self.pipeline.nodes[node_id].error}")
    if node_id == self.current[0] and event in ("done", "error", "invalidate"):
      self._refresh_viewer()

  def _history_row(self, h):
    r = self.history_rows
    self.history.insertRow(r)
    detail = {k: v for k, v in h.items() if k not in ("time", "event", "node")}
    text = ", ".join(f"{k}={v}" for k, v in detail.items()) if detail else ""
    for j, v in enumerate([h["time"][11:19], h["event"], h["node"], text[:300]]):
      self.history.setItem(r, j, QTableWidgetItem(v))
    self.history_rows += 1
    self.history.scrollToBottom()

  def _log(self, text):
    self.log.appendPlainText(text)

  # -- selection ----------------------------------------------------------------------------

  def _on_select(self, node_id, output):
    if node_id is None:
      self.current = (None, None)
      self.library.update_kind(None)
      self._refresh_viewer()
      return
    op = self.registry[self.pipeline.nodes[node_id].op_id]
    if output is None and op.outputs:
      output = op.outputs[0].name
    self.current = (node_id, output)
    kind = op.output(output).kind if output else None
    self.library.update_kind(kind)
    self.inspector.show_node(self.pipeline, node_id)
    self.inspector.set_busy(self.thread is not None)
    self._refresh_viewer()

  def _on_choose(self, op_id):
    op = self.registry[op_id]
    inputs = {}
    nid, out = self.current
    if op.inputs and nid is not None:
      cur_kind = self.registry[self.pipeline.nodes[nid].op_id].output(out).kind
      for port in op.inputs:
        if port.accepts(cur_kind):
          inputs[port.name] = (nid, out)
          continue
        # a second input (grains for GROD, ...) from the same node or upstream of it
        for up in reversed(self.pipeline.upstream(nid)):
          o = next((o for o in self.registry[self.pipeline.nodes[up].op_id].outputs if port.accepts(o.kind)), None)
          if o is not None:
            inputs[port.name] = (up, o.name)
            break
    from ..core.pipeline import Ref
    self.inspector.show_new(self.pipeline, op, {k: Ref(*v) for k, v in inputs.items()})
    self.inspector.set_busy(self.thread is not None)

  def _on_run_requested(self):
    form, mode = self.inspector.form, self.inspector.mode
    if form is None or mode is None or self.thread is not None:
      return
    try:
      if mode[0] == "new":
        node = self.pipeline.add(mode[1], form.values(), form.inputs())
        nid = node.id
        out = self.registry[node.op_id].outputs[0].name
        self.current = (nid, out)
      else:
        nid = mode[1]
        for port, ref in form.inputs().items():
          if self.pipeline.nodes[nid].inputs[port] != ref:
            self.pipeline.set_input(nid, port, ref)
        self.pipeline.set_params(nid, form.values())
    except (PipelineError, ValueError) as exc:
      QMessageBox.warning(self, "Cannot run", str(exc))
      return
    self.explorer.rebuild(keep=self.current)
    self.start_run(nid)

  def _explorer_menu(self, pos):
    nid, _ = self.explorer.current()
    if nid is None:
      return
    m = QMenu(self)
    m.addAction("Run", lambda: self.start_run(nid))
    m.addAction("Recompute (force)", lambda: self.start_run(nid, force=True))
    m.addAction("Export script up to here…", lambda: self.export_script(nid))
    m.addSeparator()
    n_dep = len(self.pipeline.descendants(nid))
    m.addAction(f"Remove{f' with {n_dep} dependent steps' if n_dep else ''}", lambda: self.remove_node(nid))
    m.exec(self.explorer.viewport().mapToGlobal(pos))

  def remove_node(self, nid):
    if self.thread is not None:
      return
    deps = self.pipeline.descendants(nid)
    if deps and QMessageBox.question(self, "Remove", f"Also remove the dependent steps {', '.join(deps)}?") \
        != QMessageBox.Yes:
      return
    self.pipeline.remove(nid)
    self.current = (None, None)
    self.inspector.clear()
    self.explorer.rebuild()
    self._refresh_viewer()

  # -- running ------------------------------------------------------------------------------

  @property
  def executor(self):
    return self.process_executor if self.use_process.isChecked() else self.inline_executor

  def start_run(self, nid, force=False):
    """compute a node, or a list of nodes, in the background"""
    if self.thread is not None:
      self.statusBar().showMessage("A computation is running; wait or cancel it")
      return
    nids = list(nid) if isinstance(nid, (list, tuple)) else [nid]
    if force:
      for n in nids:
        self.pipeline.invalidate(n)
    self.thread = RunThread(self.pipeline, nids, self.executor)
    self.thread.succeeded.connect(self._run_done)
    self.thread.failed.connect(self._run_failed)
    self.thread.finished.connect(self._thread_finished)
    self.cancel_action.setEnabled(True)
    self.inspector.set_busy(True)
    if self.use_process.isChecked() and not self.process_executor.started:
      self.statusBar().showMessage("Starting the worker process (MTEX is imported once)…")
    QApplication.setOverrideCursor(Qt.BusyCursor)
    self.thread.start()

  def _run_done(self, nid):
    self.statusBar().showMessage(f"{nid} done", 5000)

  def _run_failed(self, nid, msg):
    self.statusBar().showMessage(f"{nid}: {msg}", 10000)
    self._log(f"{nid} failed: {msg}")
    if msg != "cancelled":
      QMessageBox.warning(self, "Operation failed", f"{self.pipeline.nodes[nid].label}\n\n{msg}")

  def _thread_finished(self):
    QApplication.restoreOverrideCursor()
    self.thread = None
    self.cancel_action.setEnabled(False)
    self.inspector.set_busy(False)
    nid, _ = self.current
    if nid in self.pipeline.nodes:
      self.inspector.show_node(self.pipeline, nid)
    self.explorer.rebuild(keep=self.current)
    self._refresh_viewer()

  def run_selected(self):
    nid, _ = self.current
    if nid:
      self.start_run(nid)

  def run_all(self):
    stale = [nid for nid, n in self.pipeline.nodes.items() if n.status != Status.DONE]
    if not stale:
      return
    # every stale node is upstream of some stale leaf: running the leaves runs them all
    self.start_run([nid for nid in stale if not any(c in stale for c in self.pipeline.children(nid))])

  def cancel(self):
    if self.thread is not None:
      self.executor.cancel()
      self.statusBar().showMessage("Cancelling…")

  # -- the viewer ---------------------------------------------------------------------------

  def _grains_for(self, nid):
    for up in reversed(self.pipeline.upstream(nid)):
      n = self.pipeline.nodes[up]
      for item in n.outputs.values():
        if item.kind == DataKind.GRAINS:
          return item
    return None

  def _render(self, nid, out, view, overlay):
    from ..viz import render
    item = self.pipeline.nodes[nid].outputs[out]
    grains = self._grains_for(nid) if overlay and item.kind in (DataKind.SCALAR_MAP, DataKind.EBSD) else None
    return render(item, view, grains)

  def _refresh_viewer(self):
    from ..viz import VIEWS
    v = self.viewer
    nid, out = self.current
    if nid is None or nid not in self.pipeline.nodes:
      v.title.setText("")
      v.main.message("Select a dataset, or import one from the Import menu")
      v.data.show_item(None)
      return
    node = self.pipeline.nodes[nid]
    v.title.setText(f"{nid} · {node.label} → {out}")
    item = node.outputs.get(out)
    if item is None:
      msg = {Status.STALE: "Not computed yet: press Run (F5)", Status.RUNNING: "Computing…",
             Status.ERROR: f"Failed: {node.error}"}.get(node.status, "")
      v.main.message(msg)
      v.data.show_item(None)
      return
    views = VIEWS[item.kind]
    current_view = v.view.currentText()
    v.set_views(views, current_view)
    v.overlay.setEnabled(item.kind in (DataKind.SCALAR_MAP, DataKind.EBSD) and self._grains_for(nid) is not None)
    QApplication.setOverrideCursor(Qt.WaitCursor)
    try:
      fig = self._render(nid, out, v.view.currentText(), v.overlay.isChecked() and v.overlay.isEnabled())
      v.main.set_figure(fig)
      if self.pinned is not None and self.pinned[:2] != (nid, out):
        pn, po, pview, pov = self.pinned
        if pn in self.pipeline.nodes and po in self.pipeline.nodes[pn].outputs:
          pfig = self._render(pn, po, pview, pov)
          v.pinned.set_figure(pfig)
          v.show_pinned(True)
          from .widgets import link_axes
          kinds = {self.pipeline.nodes[pn].outputs[po].kind, item.kind}
          if kinds <= {DataKind.EBSD, DataKind.SCALAR_MAP, DataKind.GRAINS, DataKind.IMAGE, DataKind.LABELS}:
            link_axes(fig, pfig)
      else:
        v.show_pinned(self.pinned is not None)
    except Exception as exc:
      v.main.message(f"Cannot draw this result: {exc.__class__.__name__}: {exc}")
      self._log(f"render {nid}.{out}: {exc}")
    finally:
      QApplication.restoreOverrideCursor()
    v.data.show_item(item)

  def _on_pin(self, on):
    nid, out = self.current
    if on and nid is not None and out in self.pipeline.nodes[nid].outputs:
      self.pinned = (nid, out, self.viewer.view.currentText(), self.viewer.overlay.isChecked())
      fig = self._render(*self.pinned)
      self.viewer.pinned.set_figure(fig)
      self.viewer.show_pinned(True)
      self.statusBar().showMessage("Pinned: select another dataset to compare side by side, with linked zoom", 6000)
    else:
      self.pinned = None
      self.viewer.show_pinned(False)
      self.viewer.pin.setChecked(False)

  # -- import -------------------------------------------------------------------------------

  def _last_dir(self):
    return self.settings.value("last_dir", str(Path.home()))

  def _add_and_run(self, op_id, params):
    node = self.pipeline.add(op_id, params)
    self.current = (node.id, self.registry[op_id].outputs[0].name)
    self.explorer.rebuild(keep=self.current)
    self.inspector.show_node(self.pipeline, node.id)
    self.start_run(node.id)

  def import_ebsd(self):
    from ..ops.io_ops import EBSD_FILTER
    f, _ = QFileDialog.getOpenFileName(self, "Import EBSD file", self._last_dir(), EBSD_FILTER)
    if f:
      self.settings.setValue("last_dir", os.path.dirname(f))
      self._add_and_run("io.load_ebsd", {"path": f})

  def import_sample(self):
    name, ok = QInputDialog.getItem(self, "MTEX sample data", "Data set (downloaded on first use):",
                                    list(SAMPLE_EBSD), 0, False)
    if ok:
      self._add_and_run("io.sample_data", {"name": name})

  def import_image(self):
    from ..ops.io_ops import IMAGE_FILTER
    f, _ = QFileDialog.getOpenFileName(self, "Import micrograph", self._last_dir(), IMAGE_FILTER)
    if not f:
      return
    size, ok = QInputDialog.getDouble(self, "Pixel size", "Width of one pixel (µm):", 1.0, 1e-6, 1e6, 6)
    if ok:
      self.settings.setValue("last_dir", os.path.dirname(f))
      self._add_and_run("io.load_image", {"path": f, "as_gray": True, "pixel_size": size})

  def import_wizard(self):
    """MTEX's own import wizard; its recipe becomes a reproducible import step"""
    try:
      from mtex.wizard.callbacks import scriptOf
      from mtex.wizard.window import runWizard
    except ImportError as exc:
      QMessageBox.warning(self, "Import wizard", f"The MTEX import wizard is not available: {exc}")
      return
    w = runWizard(self._last_dir(), exec=False)
    st = w.st
    self._wizards = getattr(self, "_wizards", []) + [w]

    def closed(*_):
      if st.ebsd is None or not st.selected:
        return
      recipe = scriptOf(st)
      self._log(f"import wizard recipe for {st.selected}:\n{recipe}")
      self._add_and_run("io.import_recipe", {"path": st.selected, "recipe": recipe, "variable": st.variable})

    w.destroyed.connect(closed)

  # -- projects -----------------------------------------------------------------------------

  def _confirm_discard(self) -> bool:
    if not self.dirty or not self.pipeline.nodes:
      return True
    r = QMessageBox.question(self, "Unsaved changes", "Save the current project first?",
                             QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
    if r == QMessageBox.Save:
      return self.save_project()
    return r == QMessageBox.Discard

  def new_project(self):
    if self.thread is None and self._confirm_discard():
      self.project_path = None
      self.dirty = False
      self._set_pipeline(Pipeline(self.registry))

  def open_project(self, path=None):
    if self.thread is not None or not self._confirm_discard():
      return
    if path is None:
      path = QFileDialog.getExistingDirectory(self, "Open project (a .mtexwb folder)", self._last_dir())
      if not path:
        return
    try:
      p, warnings = load_project(path)
    except Exception as exc:
      QMessageBox.warning(self, "Open project", f"{exc.__class__.__name__}: {exc}")
      return
    self.project_path = str(path)
    self.dirty = False
    self._set_pipeline(p)
    for w in warnings:
      self._log(w)
    changed = p.stale_sources()
    if changed:
      QMessageBox.warning(self, "Input files changed",
                          "These input files changed on disk since they were analysed:\n" + "\n".join(
                            f"{nid}: {p.nodes[nid].label}" for nid in changed) +
                          "\n\nTheir results and everything downstream are marked stale.")
      for nid in changed:
        p.invalidate(nid)
    self.explorer.rebuild()

  def save_project(self, ask=False) -> bool:
    path = self.project_path
    if ask or path is None:
      path, _ = QFileDialog.getSaveFileName(self, "Save project", self._last_dir(), f"MTEX Workbench project (*{SUFFIX})")
      if not path:
        return False
    if self.thread is not None:
      QMessageBox.information(self, "Save", "Wait for the running computation to finish.")
      return False
    QApplication.setOverrideCursor(Qt.WaitCursor)
    try:
      final = save_project(self.pipeline, path)
    finally:
      QApplication.restoreOverrideCursor()
    self.project_path = str(final)
    self.dirty = False
    self._title()
    self.statusBar().showMessage(f"Saved {final}", 5000)
    return True

  def save_template(self):
    path, _ = QFileDialog.getSaveFileName(self, "Save pipeline template", self._last_dir(), "Pipeline (*.json)")
    if path:
      Path(path).write_text(json.dumps(self.pipeline.to_spec(), indent=2))

  def load_template(self):
    if self.thread is not None or not self._confirm_discard():
      return
    path, _ = QFileDialog.getOpenFileName(self, "Load pipeline template", self._last_dir(), "Pipeline (*.json)")
    if path:
      try:
        p = Pipeline.from_spec(json.loads(Path(path).read_text()), self.registry)
      except Exception as exc:
        QMessageBox.warning(self, "Template", str(exc))
        return
      self.project_path = None
      self._set_pipeline(p)

  # -- export -------------------------------------------------------------------------------

  def export_report(self):
    from ..report import export_results
    d = QFileDialog.getExistingDirectory(self, "Export report into folder", self._last_dir())
    if not d:
      return
    QApplication.setOverrideCursor(Qt.WaitCursor)
    try:
      written = export_results(self.pipeline, d, name=Path(self.project_path).stem if self.project_path else "analysis")
    except Exception as exc:
      QMessageBox.warning(self, "Export", f"{exc.__class__.__name__}: {exc}")
      return
    finally:
      QApplication.restoreOverrideCursor()
    QMessageBox.information(self, "Export", f"Report: {written['report']}\n{len(written['figures'])} figures, "
                                            f"{len(written['tables'])} tables, script {written['script']}")

  def export_script(self, target=None):
    path, _ = QFileDialog.getSaveFileName(self, "Export Python script", self._last_dir(), "Python (*.py)")
    if path:
      Path(path).write_text(export_script(self.pipeline, [target] if target else None))
      self.statusBar().showMessage(f"Wrote {path}", 5000)

  def export_figure(self):
    fig = self.viewer.main.figure
    if fig is None:
      return
    path, _ = QFileDialog.getSaveFileName(self, "Export figure", self._last_dir(), "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
    if path:
      fig.savefig(path, dpi=300)

  # -- tools --------------------------------------------------------------------------------

  def function_browser(self):
    QApplication.setOverrideCursor(Qt.WaitCursor)
    try:
      fb = FunctionBrowser(self)
    finally:
      QApplication.restoreOverrideCursor()
    fb.insert.connect(lambda text: (self.bottom.setCurrentWidget(self.console), self.console.insert(text)))
    fb.show()

  def batch(self):
    BatchDialog(self.pipeline, self).exec()

  def _console_namespace(self):
    def item(node, output=None):
      n = self.pipeline.nodes[node]
      return n.outputs[output or next(iter(n.outputs))]

    return {"wb": self, "np": np, "item": item, "out": lambda node, output=None: item(node, output).value,
            "__name__": "__console__"}

  def closeEvent(self, event):
    if not self._confirm_discard():
      event.ignore()
      return
    self.cancel()
    if self.thread is not None:
      self.thread.wait(5000)
    self.process_executor.shutdown()
    self.settings.setValue("geometry", self.saveGeometry())
    self.settings.setValue("use_process", self.use_process.isChecked())
    super().closeEvent(event)
