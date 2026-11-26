# API reference

```{eval-rst}
.. currentmodule:: zarrextract

Functions
---------

.. autofunction:: describe

.. autofunction:: extract

Store description
-----------------

.. autoclass:: StoreDescription
   :members:
   :special-members: __iter__, __len__, __contains__, __getitem__

.. autoclass:: ArrayInfo
   :members:

Exceptions
----------

.. autoexception:: ZarrExtractError

.. autoexception:: StoreOpenError

.. autoexception:: VariableNotFoundError

.. autoexception:: MissingDimensionNamesError

.. autoexception:: SelectionError

.. autoexception:: UnknownDimensionError

.. autoexception:: EmptySelectionError

.. autoexception:: AntimeridianError

.. autoexception:: TimeDecodingError

.. autoexception:: UnsupportedCalendarError
```
