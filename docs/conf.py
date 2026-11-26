"""Sphinx configuration for the zarrextract documentation."""

from importlib.metadata import version as _version

project = "zarrextract"
author = "Lee Hart"
copyright = "Lee Hart"
release = _version("zarrextract")
version = release

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]

html_theme = "furo"
html_title = f"zarrextract {release}"

exclude_patterns = ["_build"]

source_suffix = {".md": "myst-nb", ".ipynb": "myst-nb"}

myst_heading_anchors = 3

# Show the outputs saved in the notebooks. Nothing is run.
nb_execution_mode = "off"

autodoc_typehints = "description"
autodoc_member_order = "bysource"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pyarrow": ("https://arrow.apache.org/docs", None),
    "zarr": ("https://zarr.readthedocs.io/en/stable", None),
}
