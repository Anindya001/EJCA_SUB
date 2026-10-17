"""The desktop application, offscreen."""
import time

import pytest

pytestmark = pytest.mark.gui
pytest.importorskip("PySide6")


@pytest.fixture
def window(qapp, monkeypatch, tmp_path):
  from PySide6.QtCore import QSettings
  QSettings.setDefaultFormat(QSettings.IniFormat)
  QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path))
  from mtex_workbench.gui.main_window import MainWindow
  w = MainWindow()
  w.use_process.setChecked(False)  # in-thread: fast and deterministic for a test
  w.show()
  yield w
  w.dirty = False
  w.close()


@pytest.fixture(scope="module")
def qapp():
  from PySide6.QtWidgets import QApplication
  return QApplication.instance() or QApplication([])


def wait(w, app, timeout=120):
  t0 = time.time()
  while w.thread is not None and time.time() - t0 < timeout:
    app.processEvents()
    time.sleep(0.01)
  for _ in range(20):
    app.processEvents()


def test_library_filters_by_selection(window, qapp, micrograph_png):
  path, _ = micrograph_png
  lib = window.library

  def enabled():
    out = set()
    for i in range(lib.tree.topLevelItemCount()):
      head = lib.tree.topLevelItem(i)
      for j in range(head.childCount()):
        it = head.child(j)
        if it.data(0, 0x0100) and it.flags():
          out.add(it.data(0, 0x0100))
    return out

  assert "io.load_image" in enabled() and "grains.calc" not in enabled()
  window._add_and_run("io.load_image", {"path": str(path), "as_gray": True, "pixel_size": 1.0})
  wait(window, qapp)
  window._on_select("n1", "image")
  e = enabled()
  assert "image.gaussian" in e and "grains.calc" not in e and "ebsd.kam" not in e


def test_inspector_validates_and_runs(window, qapp, micrograph_png):
  path, _ = micrograph_png
  window._add_and_run("io.load_image", {"path": str(path), "as_gray": True, "pixel_size": 1.0})
  wait(window, qapp)
  window._on_select("n1", "image")
  window._on_choose("image.threshold")
  insp = window.inspector
  assert insp.mode == ("new", "image.threshold") and insp.run.isEnabled()
  assert insp.form.inputs()["image"].node == "n1"
  window._on_run_requested()
  wait(window, qapp)
  assert window.pipeline.nodes["n2"].status.value == "done"
  # editing a parameter of n1 invalidates n2; the viewer then says so
  window._on_select("n1", "image")
  window.inspector.form.widgets["pixel_size"].setValue(2.0)
  window.pipeline.set_params("n1", window.inspector.form.values())
  qapp.processEvents()
  assert window.pipeline.nodes["n2"].status.value == "stale"
  window._on_select("n2", "mask")
  assert "Not computed" in window.viewer.main.placeholder.text()


def test_project_save_and_open(window, qapp, micrograph_png, tmp_path):
  path, _ = micrograph_png
  window._add_and_run("io.load_image", {"path": str(path), "as_gray": True, "pixel_size": 1.0})
  wait(window, qapp)
  window.project_path = str(tmp_path / "p")
  assert window.save_project()
  window.open_project(window.project_path)
  assert list(window.pipeline.nodes) == ["n1"] and window.pipeline.nodes["n1"].status.value == "done"


def test_console(window, qapp):
  window.console.input.setText("1 + 1")
  window.console.execute()
  assert window.console.output.toPlainText().rstrip().endswith("2")
  window.console.input.setText("raise ValueError('x')")
  window.console.execute()
  assert "ValueError" in window.console.output.toPlainText()
