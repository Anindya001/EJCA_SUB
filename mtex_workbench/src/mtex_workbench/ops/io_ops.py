"""Import: EBSD files, the MTEX sample data and plain micrographs."""
from __future__ import annotations

import os

import numpy as np

from ..core.params import Bool, Choice, Code, Float, Path, Text
from ..core.registry import Output, operation
from ..core.types import DataItem, DataKind

SAMPLE_EBSD = ("ferrite", "forsterite", "copper", "twins", "titanium", "martensite",
               "alphaBetaTitanium", "small", "sharp")

EBSD_FILTER = ("EBSD files (*.ang *.ctf *.osc *.crc *.cpr *.h5 *.h5oina *.oh5 *.hdf5 *.edaxh5 *.txt *.csv);;"
               "All files (*)")
IMAGE_FILTER = "Images (*.png *.tif *.tiff *.jpg *.jpeg *.bmp);;All files (*)"


def _ebsd_meta(ebsd, source: str) -> dict:
  unit = str(getattr(ebsd, "scanUnit", "um"))
  return {"source": source, "unit": "µm" if unit == "um" else unit,
          "phases": [m for m in ebsd.mineralList if m != "notIndexed"]}


@operation("io.sample_data", "MTEX sample data", "Import & Data",
           outputs=(Output("ebsd", DataKind.EBSD, "EBSD map"),),
           params=(Choice("name", "Data set", SAMPLE_EBSD,
                          help="downloaded from mtex-toolbox/mtex-data on first use"),),
           script=lambda o, i, p: f"{o['ebsd']} = mtexdata({p['name']!r})")
def sample_data(name):
  """One of the EBSD maps of the MTEX documentation, read with the recipe of its page."""
  from mtex import EBSD, mtexdata
  ebsd = mtexdata(name)
  if not isinstance(ebsd, EBSD):
    raise TypeError(f"the sample {name!r} is a {type(ebsd).__name__}, not an EBSD map")
  return DataItem(DataKind.EBSD, ebsd, _ebsd_meta(ebsd, f"mtexdata:{name}"))


@operation("io.load_ebsd", "Load EBSD file", "Import & Data",
           outputs=(Output("ebsd", DataKind.EBSD, "EBSD map"),),
           params=(Path("path", "File", EBSD_FILTER),),
           hashes_file="path",
           script=lambda o, i, p: f"{o['ebsd']} = loadEBSD({p['path']!r})")
def load_ebsd(path):
  """An EBSD map read by MTEX's loaders, the format chosen by the extension (TSL .ang,
  Oxford .ctf/.crc/.cpr/.h5oina, EDAX .osc/.oh5, ...). For files whose phases or conventions
  need choosing, use File ▸ Import Wizard, which is MTEX's own import wizard."""
  from mtex import loadEBSD
  if not os.path.isfile(path):
    raise FileNotFoundError(path)
  ebsd = loadEBSD(path)
  return DataItem(DataKind.EBSD, ebsd, _ebsd_meta(ebsd, os.path.abspath(path)))


def _recipe_for(recipe: str, path: str) -> str:
  """the wizard's recipe reading `path`: its `fname = ...` line rebound, so that a batch run
  can apply one recipe to many files"""
  import re
  if not re.search(r"^fname = .*$", recipe, flags=re.M):
    raise ValueError("the recipe has no 'fname = ...' line; it was not written by the import wizard")
  return re.sub(r"^fname = .*$", lambda m: f"fname = {path!r}", recipe, count=1, flags=re.M)


def _recipe_script(o, i, p):
  body = _recipe_for(p["recipe"], p["path"])
  lines = ["def _import():"] + ["  " + ln if ln.strip() else "" for ln in body.splitlines()]
  lines += [f"  return {p['variable']}", f"{o['ebsd']} = _import()"]
  return "\n".join(lines)


@operation("io.import_recipe", "Import with wizard recipe", "Import & Data",
           outputs=(Output("ebsd", DataKind.EBSD, "EBSD map"),),
           params=(Path("path", "File", EBSD_FILTER),
                   Code("recipe", "Recipe (Python)", "",
                        help="the script MTEX's import wizard wrote: phases, conventions, corrections"),
                   Text("variable", "Map variable", "ebsd")),
           hashes_file="path", script=_recipe_script)
def import_recipe(path, recipe, variable):
  """An EBSD map imported by the recipe that MTEX's import wizard wrote for it, with the
  phases, reference frame conventions and vendor corrections chosen in the wizard. The
  recipe is Python code and is executed as such: only run recipes you or the wizard wrote."""
  from mtex import EBSD
  if not os.path.isfile(path):
    raise FileNotFoundError(path)
  ns: dict = {}
  exec(compile(_recipe_for(recipe, path), "<import recipe>", "exec"), ns)
  ebsd = ns.get(variable)
  if not isinstance(ebsd, EBSD):
    raise TypeError(f"the recipe did not define an EBSD map named {variable!r}")
  return DataItem(DataKind.EBSD, ebsd, _ebsd_meta(ebsd, os.path.abspath(path)))


def read_image(path, as_gray=True) -> np.ndarray:
  """the image as float in [0, 1]; grey by the Rec. 709 luminance of linear RGB weights"""
  from PIL import Image
  with Image.open(path) as im:
    a = np.asarray(im)
  if a.dtype == bool:
    a = a.astype(float)
  elif np.issubdtype(a.dtype, np.integer):
    a = a.astype(float) / np.iinfo(a.dtype).max
  else:
    a = a.astype(float)
    if a.max() > 1:
      a = (a - a.min()) / (a.max() - a.min())
  if a.ndim == 3 and a.shape[2] == 4:
    a = a[..., :3]
  if as_gray and a.ndim == 3:
    a = a @ np.array([0.2126, 0.7152, 0.0722])
  return a


@operation("io.load_image", "Load micrograph", "Import & Data",
           outputs=(Output("image", DataKind.IMAGE, "Image"),),
           params=(Path("path", "File", IMAGE_FILTER),
                   Bool("as_gray", "Convert to grey", True),
                   Float("pixel_size", "Pixel size", 1.0, minimum=1e-9, unit=" µm",
                         help="the physical width of one pixel, from the scale bar or the microscope")),
           hashes_file="path")
def load_image(path, as_gray, pixel_size):
  """A light or electron micrograph (PNG, TIFF, JPEG). An image carries intensities only:
  it supports filtering and geometric segmentation, but no orientations, ODFs or GND
  densities; those need indexed EBSD data."""
  if not os.path.isfile(path):
    raise FileNotFoundError(path)
  img = read_image(path, as_gray)
  return DataItem(DataKind.IMAGE, img, {"pixel_size": pixel_size, "unit": "µm",
                                        "source": os.path.abspath(path)})
