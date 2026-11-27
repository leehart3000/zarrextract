# API reference

```{eval-rst}
.. currentmodule:: zarrextract

Functions
---------

.. autofunction:: describe

.. autofunction:: extract

.. autofunction:: open

.. autofunction:: write

.. autofunction:: provenance

Open stores
-----------

.. autoclass:: Source
   :members: store, closed, describe, extract, close
   :special-members: __enter__, __exit__

Provenance
----------

.. autoclass:: Provenance
   :members:

Store description
-----------------

.. autoclass:: StoreDescription
   :members:
   :special-members: __iter__, __len__, __contains__, __getitem__

.. autoclass:: ArrayInfo
   :members:

OME-Zarr groups
---------------

.. autoclass:: OmeImage
   :members:

.. autoclass:: OmeMultiscale
   :members:

.. autoclass:: OmeLevel
   :members:

.. autoclass:: OmeAxis
   :members:

.. autoclass:: OmeLabels
   :members:

.. autoclass:: OmePlate
   :members:

.. autoclass:: OmeWell
   :members:

.. autoclass:: OmeContainer
   :members:

Exceptions
----------

.. autoexception:: ZarrExtractError

.. autoexception:: StoreOpenError

.. autoexception:: SourceClosedError

.. autoexception:: VariableNotFoundError

.. autoexception:: MissingDimensionNamesError

.. autoexception:: SelectionError

.. autoexception:: UnknownDimensionError

.. autoexception:: EmptySelectionError

.. autoexception:: AntimeridianError

.. autoexception:: TimeDecodingError

.. autoexception:: UnsupportedCalendarError

.. autoexception:: OmeMetadataError

.. autoexception:: WriteError

.. autoexception:: UnsupportedFormatError

.. autoexception:: OutputExistsError

Warnings
--------

.. autoexception:: OmeMetadataWarning
```
