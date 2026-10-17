"""The building blocks of the main window."""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
  QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QHeaderView, QLabel,
  QLineEdit, QPlainTextEdit, QPushButton, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
  QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from ..core.params import Param, ParamError
from ..core.pipeline import Pipeline, Ref, Status
from ..core.registry import Operation, Registry
from ..core.types import DataItem, DataKind

STATUS_COLOURS = {Status.STALE: "#9aa5b1", Status.RUNNING: "#2f80ed", Status.DONE: "#27ae60", Status.ERROR: "#c0392b"}


def dot(colour: str, size: int = 10) -> QIcon:
  pm = QPixmap(size, size)
  pm.fill(Qt.transparent)
  p = QPainter(pm)
  p.setRenderHint(QPainter.Antialiasing)
  p.setBrush(QBrush(QColor(colour)))
  p.setPen(Qt.NoPen)
  p.drawEllipse(1, 1, size - 2, size - 2)
  p.end()
  return QIcon(pm)


# -- parameters ------------------------------------------------------------------------------

class ParamForm(QWidget):
  """the controls of an operation's inputs and parameters, built from its declaration"""
  changed = Signal()

  def __init__(self, op: Operation, pipeline: Pipeline, params: dict | None = None,
               inputs: dict | None = None, node_id: str | None = None, parent=None):
    super().__init__(parent)
    self.op, self.pipeline = op, pipeline
    self.widgets: dict[str, QWidget] = {}
    self.ports: dict[str, QComboBox] = {}
    form = QFormLayout(self)
    form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    params = params or {}
    for port in op.inputs:
      box = QComboBox()
      box.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
      box.setMinimumContentsLength(16)
      for ref, label, kind in pipeline.available_outputs(port.kinds, exclude_downstream_of=node_id):
        box.addItem(label, ref)
      want = (inputs or {}).get(port.name)
      if want is not None:
        i = next((k for k in range(box.count()) if box.itemData(k) == want), -1)
        if i >= 0:
          box.setCurrentIndex(i)
      elif box.count():
        box.setCurrentIndex(box.count() - 1)
      box.currentIndexChanged.connect(self.changed)
      box.setToolTip(f"accepts {', '.join(k.label for k in port.kinds)}")
      form.addRow(f"{port.label or port.name}:", box)
      self.ports[port.name] = box
    for p in op.params:
      w = self._widget(p, params.get(p.name, p.default))
      self.widgets[p.name] = w
      label = QLabel(p.label + ":")
      if p.help:
        label.setToolTip(p.help)
        w.setToolTip(p.help)
      form.addRow(label, w)

  def _widget(self, p: Param, value):
    if p.type == "float":
      w = QDoubleSpinBox()
      w.setDecimals(4 if (p.maximum or 1e9) <= 1 else 3)
      w.setRange(p.minimum if p.minimum is not None else -1e12, p.maximum if p.maximum is not None else 1e12)
      w.setSingleStep(p.step or (0.05 if (p.maximum or 1e9) <= 1 else 0.5))
      w.setSuffix(p.unit)
      w.setValue(float(value))
      w.valueChanged.connect(self.changed)
    elif p.type == "int":
      w = QSpinBox()
      w.setRange(int(p.minimum) if p.minimum is not None else -2 ** 31 + 1,
                 int(p.maximum) if p.maximum is not None else 2 ** 31 - 1)
      w.setSuffix(p.unit)
      w.setValue(int(value))
      w.valueChanged.connect(self.changed)
    elif p.type == "bool":
      w = QCheckBox()
      w.setChecked(bool(value))
      w.toggled.connect(self.changed)
    elif p.type == "choice":
      w = QComboBox()
      w.addItems(list(p.choices))
      w.setCurrentText(str(value))
      w.currentIndexChanged.connect(self.changed)
    elif p.type == "text":
      w = QPlainTextEdit(str(value))
      w.setFont(QFont("monospace"))
      w.setMinimumHeight(120)
      w.textChanged.connect(self.changed)
    elif p.type == "path":
      w = QWidget()
      lay = QHBoxLayout(w)
      lay.setContentsMargins(0, 0, 0, 0)
      edit = QLineEdit(str(value))
      btn = QPushButton("Browse…")
      lay.addWidget(edit, 1)
      lay.addWidget(btn)
      w.edit = edit

      def browse():
        f, _ = QFileDialog.getOpenFileName(self, p.label, edit.text(), p.file_filter)
        if f:
          edit.setText(f)
      btn.clicked.connect(browse)
      edit.textChanged.connect(self.changed)
    else:
      w = QLineEdit(str(value))
      w.textChanged.connect(self.changed)
    return w

  def values(self) -> dict:
    out = {}
    for p in self.op.params:
      w = self.widgets[p.name]
      if p.type in ("float", "int"):
        out[p.name] = w.value()
      elif p.type == "bool":
        out[p.name] = w.isChecked()
      elif p.type == "choice":
        out[p.name] = w.currentText()
      elif p.type == "text":
        out[p.name] = w.toPlainText()
      elif p.type == "path":
        out[p.name] = w.edit.text()
      else:
        out[p.name] = w.text()
    return out

  def inputs(self) -> dict[str, Ref]:
    return {name: box.currentData() for name, box in self.ports.items()}

  def validate(self) -> tuple[bool, str]:
    missing = [n for n, box in self.ports.items() if box.currentData() is None]
    if missing:
      return False, f"no compatible dataset for {', '.join(missing)}"
    try:
      self.op.validate(self.values())
    except ParamError as e:
      return False, str(e)
    return True, "Parameters valid"


# -- figures ---------------------------------------------------------------------------------

class FigureView(QWidget):
  """a matplotlib figure with the zoom and pan toolbar"""

  def __init__(self, parent=None):
    super().__init__(parent)
    self.lay = QVBoxLayout(self)
    self.lay.setContentsMargins(0, 0, 0, 0)
    self.canvas = None
    self.toolbar = None
    self.placeholder = QLabel("Select a dataset, or import one from the File menu")
    self.placeholder.setAlignment(Qt.AlignCenter)
    self.placeholder.setStyleSheet("color: #7a8794;")
    self.lay.addWidget(self.placeholder)

  def set_figure(self, fig):
    for w in (self.canvas, self.toolbar):
      if w is not None:
        self.lay.removeWidget(w)
        w.setParent(None)
        w.deleteLater()
    self.canvas = self.toolbar = None
    self.placeholder.setVisible(fig is None)
    if fig is None:
      return
    self.canvas = FigureCanvasQTAgg(fig)
    self.toolbar = NavigationToolbar2QT(self.canvas, self)
    self.lay.addWidget(self.toolbar)
    self.lay.addWidget(self.canvas, 1)
    self.canvas.draw_idle()

  def message(self, text: str):
    self.set_figure(None)
    self.placeholder.setText(text)

  @property
  def figure(self):
    return self.canvas.figure if self.canvas is not None else None


def link_axes(fig_a, fig_b):
  """linked zoom and pan between the first axes of two figures of the same map"""
  if fig_a is None or fig_b is None or not fig_a.axes or not fig_b.axes:
    return
  a, b = fig_a.axes[0], fig_b.axes[0]
  busy = [False]

  def sync(src, dst):
    def cb(_ax):
      if busy[0]:
        return
      busy[0] = True
      try:
        dst.set_xlim(src.get_xlim())
        dst.set_ylim(src.get_ylim())
        dst.figure.canvas.draw_idle()
      finally:
        busy[0] = False
    return cb

  for src, dst in ((a, b), (b, a)):
    src.callbacks.connect("xlim_changed", sync(src, dst))
    src.callbacks.connect("ylim_changed", sync(src, dst))


class DataView(QWidget):
  """the numbers behind a result: a table, or the statistics as text"""

  def __init__(self, parent=None):
    super().__init__(parent)
    lay = QVBoxLayout(self)
    lay.setContentsMargins(0, 0, 0, 0)
    self.table = QTableWidget()
    self.text = QPlainTextEdit()
    self.text.setReadOnly(True)
    self.text.setFont(QFont("monospace"))
    lay.addWidget(self.table)
    lay.addWidget(self.text)
    self.show_item(None)

  def show_item(self, item: DataItem | None, max_rows: int = 5000):
    self.table.setVisible(False)
    self.text.setVisible(True)
    if item is None:
      self.text.setPlainText("")
      return
    if item.kind == DataKind.TABLE:
      t = item.value
      cols = list(t)
      n = min(max_rows, len(t[cols[0]]) if cols else 0)
      self.table.clear()
      self.table.setColumnCount(len(cols))
      self.table.setRowCount(n)
      units = item.meta.get("units", {})
      self.table.setHorizontalHeaderLabels([f"{c} ({units[c]})" if c in units else c for c in cols])
      for j, c in enumerate(cols):
        col = np.asarray(t[c])
        for i in range(n):
          v = col[i]
          cell = QTableWidgetItem(f"{v:.5g}" if isinstance(v, (float, np.floating)) else str(v))
          cell.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
          self.table.setItem(i, j, cell)
      self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
      self.table.setVisible(True)
      self.text.setVisible(False)
      return
    lines = [f"{item.kind.label}: {item.summary()}", ""]
    if item.meta:
      lines += ["metadata"] + [f"  {k}: {v}" for k, v in item.meta.items()] + [""]
    if item.kind == DataKind.STATS:
      lines += ["results"] + list(_describe(item.value, 1))
    self.text.setPlainText("\n".join(lines))


def _describe(d, depth):
  pad = "  " * depth
  for k, v in d.items():
    if isinstance(v, dict):
      yield f"{pad}{k}:"
      yield from _describe(v, depth + 1)
    elif isinstance(v, np.ndarray):
      yield f"{pad}{k}: array{v.shape}"
    elif isinstance(v, float):
      yield f"{pad}{k}: {v:.6g}"
    elif isinstance(v, (tuple, list)) and all(isinstance(x, float) for x in v):
      yield f"{pad}{k}: [{', '.join(f'{x:.6g}' for x in v)}]"
    else:
      yield f"{pad}{k}: {v}"


# -- the explorer and the library ------------------------------------------------------------

class DatasetExplorer(QTreeWidget):
  """the processing graph as a tree: a node under its primary input, its outputs under it"""
  selected = Signal(object, object)  # node id, output name

  def __init__(self, pipeline: Pipeline, parent=None):
    super().__init__(parent)
    self.pipeline = pipeline
    self.setHeaderLabels(["Dataset", "Summary"])
    self.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
    self.itemSelectionChanged.connect(self._emit)

  def rebuild(self, keep: tuple | None = None):
    self.blockSignals(True)
    self.clear()
    items = {}
    target = None
    for n in self.pipeline.nodes.values():
      op = self.pipeline.registry[n.op_id]
      parent = None
      if op.inputs:
        first = n.inputs[op.inputs[0].name]
        parent = items.get(first.node)
      it = QTreeWidgetItem([f"{n.id} · {n.label}", n.error if n.status == Status.ERROR else n.status.value])
      it.setIcon(0, dot(STATUS_COLOURS[n.status]))
      it.setData(0, Qt.UserRole, (n.id, None))
      tip = ", ".join(f"{k}={v}" for k, v in n.params.items() if not isinstance(v, str) or len(v) < 60)
      it.setToolTip(0, f"{op.label} [{op.id}]\n{tip}")
      (parent.addChild(it) if parent else self.addTopLevelItem(it))
      for o in op.outputs:
        item = n.outputs.get(o.name)
        child = QTreeWidgetItem([o.label or o.name, item.summary() if item else ""])
        child.setData(0, Qt.UserRole, (n.id, o.name))
        child.setForeground(0, QBrush(QColor("#222" if item else "#9aa5b1")))
        it.addChild(child)
        if keep == (n.id, o.name):
          target = child
      if keep == (n.id, None):
        target = it
      it.setExpanded(True)
      items[n.id] = it
    self.blockSignals(False)
    if target is not None:
      self.setCurrentItem(target)

  def current(self) -> tuple:
    it = self.currentItem()
    return it.data(0, Qt.UserRole) if it else (None, None)

  def _emit(self):
    nid, out = self.current()
    self.selected.emit(nid, out)


class Library(QWidget):
  """the operations by category; the ones the selected dataset cannot feed are disabled"""
  chosen = Signal(str)

  def __init__(self, registry: Registry, parent=None):
    super().__init__(parent)
    self.registry = registry
    lay = QVBoxLayout(self)
    lay.setContentsMargins(0, 0, 0, 0)
    self.search = QLineEdit()
    self.search.setPlaceholderText("Search operations…")
    self.search.textChanged.connect(lambda _: self.update_kind(self.kind))
    self.tree = QTreeWidget()
    self.tree.setHeaderHidden(True)
    self.tree.itemActivated.connect(self._activate)
    self.tree.itemClicked.connect(self._activate)
    lay.addWidget(self.search)
    lay.addWidget(self.tree)
    self.kind = None
    self.update_kind(None)

  def update_kind(self, kind: DataKind | None):
    self.kind = kind
    text = self.search.text().lower().strip()
    applicable = {op.id for op in self.registry.applicable(kind)}
    sources = {op.id for op in self.registry.applicable(None)}
    self.tree.clear()
    bold = QFont()
    bold.setBold(True)
    for cat, ops in self.registry.by_category().items():
      head = QTreeWidgetItem([cat])
      head.setFont(0, bold)
      head.setFlags(Qt.ItemIsEnabled)
      shown = 0
      for op in ops:
        if text and text not in op.label.lower() and text not in op.id.lower():
          continue
        it = QTreeWidgetItem([op.label])
        it.setData(0, Qt.UserRole, op.id)
        ok = op.id in applicable or op.id in sources
        if not ok:
          it.setFlags(Qt.NoItemFlags)
          needs = " or ".join(k.label for k in op.inputs[0].kinds) if op.inputs else ""
          it.setToolTip(0, f"needs {needs}")
        else:
          it.setToolTip(0, op.doc.split("\n")[0])
        head.addChild(it)
        shown += 1
      if not ops and not text:
        it = QTreeWidgetItem(["via Function Browser / Python console"])
        it.setFlags(Qt.NoItemFlags)
        f = QFont()
        f.setItalic(True)
        it.setFont(0, f)
        head.addChild(it)
        shown += 1
      if shown:
        self.tree.addTopLevelItem(head)
        head.setExpanded(cat in ("Import & Data",) or bool(ops and any(o.id in applicable for o in ops)) or bool(text))

  def _activate(self, item, _col=0):
    op_id = item.data(0, Qt.UserRole)
    if op_id and item.flags() & Qt.ItemIsEnabled:
      self.chosen.emit(op_id)


class Viewer(QWidget):
  """the central viewer: the figure (optionally beside a pinned one) and the numbers"""
  view_changed = Signal()

  def __init__(self, parent=None):
    super().__init__(parent)
    lay = QVBoxLayout(self)
    lay.setContentsMargins(2, 2, 2, 2)
    bar = QHBoxLayout()
    self.title = QLabel("")
    self.title.setStyleSheet("font-weight: 600;")
    self.view = QComboBox()
    self.view.currentIndexChanged.connect(lambda _: self.view_changed.emit())
    self.overlay = QCheckBox("Grain boundaries")
    self.overlay.toggled.connect(lambda _: self.view_changed.emit())
    self.pin = QPushButton("Pin for comparison")
    self.pin.setCheckable(True)
    bar.addWidget(self.title, 1)
    bar.addWidget(QLabel("View:"))
    bar.addWidget(self.view)
    bar.addWidget(self.overlay)
    bar.addWidget(self.pin)
    lay.addLayout(bar)
    self.tabs = QTabWidget()
    split = QSplitter(Qt.Horizontal)
    self.pinned = FigureView()
    self.main = FigureView()
    split.addWidget(self.pinned)
    split.addWidget(self.main)
    self.pinned.setVisible(False)
    self.split = split
    self.data = DataView()
    self.tabs.addTab(split, "Figure")
    self.tabs.addTab(self.data, "Data")
    lay.addWidget(self.tabs, 1)

  def show_pinned(self, on: bool):
    was = self.pinned.isVisible()
    self.pinned.setVisible(on)
    if on and not was:
      half = max(1, self.split.width() // 2)
      self.split.setSizes([half, half])

  def set_views(self, views, current=None):
    self.view.blockSignals(True)
    self.view.clear()
    self.view.addItems(list(views))
    if current in views:
      self.view.setCurrentText(current)
    self.view.blockSignals(False)
