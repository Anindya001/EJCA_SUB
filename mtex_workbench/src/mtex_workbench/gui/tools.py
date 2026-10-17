"""Expert tools: the Python console, the MTEX function browser and batch processing."""
from __future__ import annotations

import contextlib
import inspect
import io
import json
import sys
import traceback
from pathlib import Path

from PySide6.QtCore import QProcess, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
  QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget, QPlainTextEdit,
  QPushButton, QSplitter, QTextBrowser, QVBoxLayout, QWidget,
)


class Console(QWidget):
  """a Python console sharing a namespace with the workbench

  Names: pipeline, out(node, output) the value of a result, item(node, output) the DataItem,
  wb the main window, np, and after `from mtex import *` all of MTEX."""

  def __init__(self, namespace: dict, parent=None):
    super().__init__(parent)
    self.ns = namespace
    lay = QVBoxLayout(self)
    lay.setContentsMargins(0, 0, 0, 0)
    self.output = QPlainTextEdit()
    self.output.setReadOnly(True)
    self.output.setFont(QFont("monospace"))
    self.input = QLineEdit()
    self.input.setFont(QFont("monospace"))
    self.input.setPlaceholderText(">>> python   (try: out('n1', 'ebsd'),  from mtex import *)")
    self.input.returnPressed.connect(self.execute)
    lay.addWidget(self.output, 1)
    lay.addWidget(self.input)
    self.history: list[str] = []
    self.output.appendPlainText(self.__doc__.split("\n\n", 1)[1].strip().replace("\n  ", "\n"))

  def execute(self):
    src = self.input.text()
    if not src.strip():
      return
    self.history.append(src)
    self.input.clear()
    self.output.appendPlainText(f">>> {src}")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
      try:
        try:
          code = compile(src, "<console>", "eval")
        except SyntaxError:
          exec(compile(src, "<console>", "exec"), self.ns)
        else:
          r = eval(code, self.ns)
          if r is not None:
            print(repr(r))
      except Exception:
        traceback.print_exc()
    text = buf.getvalue().rstrip()
    if text:
      self.output.appendPlainText(text)

  def insert(self, text: str):
    self.input.setText(text)
    self.input.setFocus()


class FunctionBrowser(QDialog):
  """every public name of MTEX with its signature and documentation"""
  insert = Signal(str)

  def __init__(self, parent=None):
    super().__init__(parent)
    self.setWindowTitle("MTEX function browser")
    self.resize(1000, 640)
    import mtex
    self.mtex = mtex
    self.names = sorted(mtex.__all__, key=str.lower)
    lay = QVBoxLayout(self)
    self.filter = QLineEdit()
    self.filter.setPlaceholderText(f"Filter {len(self.names)} functions and classes…")
    self.filter.textChanged.connect(self._fill)
    split = QSplitter(Qt.Horizontal)
    self.list = QListWidget()
    self.list.currentTextChanged.connect(self._show)
    self.doc = QTextBrowser()
    self.doc.setFont(QFont("monospace"))
    split.addWidget(self.list)
    split.addWidget(self.doc)
    split.setSizes([280, 720])
    lay.addWidget(self.filter)
    lay.addWidget(split, 1)
    row = QHBoxLayout()
    btn = QPushButton("Insert into console")
    btn.clicked.connect(self._insert)
    row.addStretch(1)
    row.addWidget(btn)
    lay.addLayout(row)
    self._fill("")

  def _fill(self, text):
    t = text.lower()
    self.list.clear()
    self.list.addItems([n for n in self.names if t in n.lower()])

  def _show(self, name):
    if not name:
      return
    obj = getattr(self.mtex, name, None)
    try:
      sig = str(inspect.signature(obj))
    except (TypeError, ValueError):
      sig = ""
    kind = "class" if inspect.isclass(obj) else ("function" if callable(obj) else type(obj).__name__)
    mod = getattr(obj, "__module__", "")
    self.doc.setPlainText(f"{name}{sig}\n\n{kind} in {mod}\n\n{inspect.getdoc(obj) or '(no documentation)'}")

  def _insert(self):
    it = self.list.currentItem()
    if it:
      self.insert.emit(f"{it.text()}(")


class BatchDialog(QDialog):
  """the current pipeline applied to many files, one report each and one summary table"""

  def __init__(self, pipeline, parent=None):
    super().__init__(parent)
    self.setWindowTitle("Batch processing")
    self.resize(720, 480)
    self.pipeline = pipeline
    lay = QVBoxLayout(self)
    sources = [n for n in pipeline.nodes.values() if pipeline.registry[n.op_id].hashes_file]
    if not sources:
      lay.addWidget(QLabel("The pipeline has no file source to vary: build it on a loaded file first."))
      self.ok = False
    else:
      self.ok = True
    self.source = sources[0].id if sources else None
    lay.addWidget(QLabel(f"Template: the current pipeline ({len(pipeline.nodes)} steps); "
                         f"the file of {self.source} is replaced by each input."))
    self.files = QListWidget()
    lay.addWidget(self.files, 1)
    row = QHBoxLayout()
    add = QPushButton("Add files…")
    add.clicked.connect(self._add)
    self.out = QLineEdit(str(Path.home() / "mtex-workbench-batch"))
    browse = QPushButton("Output folder…")
    browse.clicked.connect(lambda: self.out.setText(QFileDialog.getExistingDirectory(self, "Output folder") or self.out.text()))
    row.addWidget(add)
    row.addWidget(self.out, 1)
    row.addWidget(browse)
    lay.addLayout(row)
    self.log = QPlainTextEdit()
    self.log.setReadOnly(True)
    self.log.setMaximumHeight(140)
    lay.addWidget(self.log)
    self.buttons = QDialogButtonBox(QDialogButtonBox.Close)
    self.run_btn = self.buttons.addButton("Run batch", QDialogButtonBox.AcceptRole)
    self.run_btn.setEnabled(self.ok)
    self.run_btn.clicked.disconnect()
    self.run_btn.clicked.connect(self._run)
    self.buttons.rejected.connect(self.reject)
    lay.addWidget(self.buttons)
    self.proc = None

  def _add(self):
    files, _ = QFileDialog.getOpenFileNames(self, "Specimens")
    self.files.addItems(files)

  def _run(self):
    inputs = [self.files.item(i).text() for i in range(self.files.count())]
    if not inputs:
      return
    out = Path(self.out.text())
    out.mkdir(parents=True, exist_ok=True)
    spec = out / "template.json"
    spec.write_text(json.dumps(self.pipeline.to_spec(), indent=2))
    self.run_btn.setEnabled(False)
    self.log.appendPlainText(f"running {len(inputs)} specimens in a separate process…")
    # a separate process: the batch's figures never share pyplot with the interface
    self.proc = QProcess(self)
    self.proc.setProcessChannelMode(QProcess.MergedChannels)
    self.proc.readyReadStandardOutput.connect(
      lambda: self.log.appendPlainText(bytes(self.proc.readAllStandardOutput()).decode(errors="replace").rstrip()))
    self.proc.finished.connect(self._done)
    self.proc.start(sys.executable, ["-m", "mtex_workbench", "batch", str(spec), "-i", *inputs,
                                     "-o", str(out), "--source", self.source])

  def _done(self, code, _status):
    self.log.appendPlainText("batch finished" if code == 0 else f"batch ended with exit code {code}")
    self.log.appendPlainText(f"summary: {Path(self.out.text()) / 'summary.csv'}")
    self.run_btn.setEnabled(True)
