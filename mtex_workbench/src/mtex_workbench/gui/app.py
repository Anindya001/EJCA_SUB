"""Start the desktop application."""
from __future__ import annotations

import sys


def main(project: str | None = None) -> int:
  import matplotlib
  matplotlib.use("Agg")  # pyplot draws off-screen; the viewer wraps the figures in Qt canvases
  from PySide6.QtWidgets import QApplication
  from .main_window import MainWindow
  app = QApplication.instance() or QApplication(sys.argv[:1])
  app.setApplicationName("MTEX Workbench")
  app.setOrganizationName("MTEX Workbench")
  w = MainWindow(project)
  w.show()
  return app.exec()


if __name__ == "__main__":
  sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
