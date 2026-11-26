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
html_title = f"ZarrExtract {release}"

html_static_path = ["_static"]
html_favicon = "_static/brand/favicon.svg"

html_theme_options = {
    "light_logo": "brand/zarrextract-lockup.svg",
    "dark_logo": "brand/zarrextract-lockup-dark.svg",
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#0B5D7A",
        "color-brand-content": "#0B5D7A",
    },
    "dark_css_variables": {
        "color-brand-primary": "#5FC1D9",
        "color-brand-content": "#5FC1D9",
    },
}

exclude_patterns = ["_build"]

source_suffix = {".md": "myst-nb", ".ipynb": "myst-nb"}

myst_heading_anchors = 3

# Show the outputs saved in the notebooks. Nothing is run.
nb_execution_mode = "off"

# Show the "Attributes:" sections of the classes as a list.
napoleon_use_ivar = True

autodoc_typehints = "description"
autodoc_member_order = "bysource"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pyarrow": ("https://arrow.apache.org/docs", None),
    "zarr": ("https://zarr.readthedocs.io/en/stable", None),
}
