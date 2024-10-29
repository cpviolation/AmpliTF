###############################################################################
# (c) Copyright 2023 CERN for the benefit of the LHCb Collaboration           #
#                                                                             #
# This software is distributed under the terms of the GNU General Public      #
# Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import datetime
#sys.path.append(os.path.abspath("./_ext"))

# -- Project information -----------------------------------------------------

project = "AmpliTF"
year = datetime.date.today().strftime("%Y")
copyright = f"2020-{year}, Anton Poluetkov (CNRS)"
author = "Anton Poluetkov (CNRS)"

# -- General configuration ---------------------------------------------------

master_doc = "index"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.graphviz",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    #"sphinx.ext.imgmath",
    "autoapi.extension",
    #"graphviz_linked",
]

# AutoAPI options
autoapi_dirs = ['../amplitf' ]
autoapi_add_toctree_entry = True
autoapi_generate_api_docs = True
autoapi_keep_files = True

# Assume unmarked references (in backticks) refer to Python objects
default_role = "py:obj"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_*",
    "Thumbs.db",
    ".DS_Store",
    #"make_functor_docs.py",
    #"selection/thor_functors_reference.generated.rst",
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ["_static"]

# Global file metadata
html_context = {
    "display_github": True,
    "github_host": "github.com",
    "github_user": "apoluekt",
    "github_repo": "AmpliTF",
    "github_version": "master/docs/",
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False

# A list of regular expressions that match URIs that should not be
# checked when doing a linkcheck build.
linkcheck_ignore = [
]
