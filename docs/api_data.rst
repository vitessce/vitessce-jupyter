Data config APIs
#####################

Dataset wrapper classes
provide functionality for adding in-memory or local data objects
to datasets when rendering Vitessce as a Jupyter widget.

We provide default wrapper class implementations for data formats
used by popular single-cell and imaging packages.

To write your own custom wrapper class, create a subclass
of the ``AbstractWrapper`` class, implementing the
getter functions for the data types that can be derived from your object.

vitessce.wrappers
*****************

.. automodule:: vitessce.wrappers
 :members:

vitessce.export
*****************

.. automodule:: vitessce.export
 :members:
