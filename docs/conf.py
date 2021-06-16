import inspect
import os
import sys
import glob
from os.path import join

import vitessce

import sphinx_rtd_theme
import nbclean

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'vitessce'
copyright = '2020, Gehlenborg Lab'
author = 'Gehlenborg Lab'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.linkcode',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme',
    'nbsphinx'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    'notebooks/README.md',
    'notebooks/environment.yml',
    'notebooks/example-config-*.json',
    'notebooks/data/**'
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = [
    'stylesheet.css',
]

autoclass_content = 'both'

def linkcode_resolve(domain, info):
    def find_source():
        # try to find the file and line number, based on code from numpy:
        # https://github.com/numpy/numpy/blob/master/doc/source/conf.py#L286
        obj = sys.modules[info['module']]
        for part in info['fullname'].split('.'):
            obj = getattr(obj, part)
        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start=os.path.dirname(vitessce.__file__))
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != 'py' or not info['module']:
        return None
    try:
        filename = 'vitessce/%s#L%d-L%d' % find_source()
    except Exception as e:
        print(str(e))
        filename = info['module'].replace('.', '/') + '.py'
    return "https://github.com/vitessce/vitessce-jupyter/blob/master/%s" % filename


# -- Options for intersphinx -------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'anndata': ('https://anndata.readthedocs.io/en/latest', None),
    'loompy': ('http://linnarssonlab.org/loompy/', None),
}

# -- Options for nbsphinx -------------------------------------------------

nbsphinx_execute = 'never'

# -- Strip notebook output -------------------------------------------------


for filename in glob.glob(join('notebooks', '*.ipynb'), recursive=True):
    ntbk = nbclean.NotebookCleaner(filename)
    ntbk.clear('stderr')
    ntbk.clear('output')
    ntbk.remove_cells(empty=True)
    ntbk.save(filename)