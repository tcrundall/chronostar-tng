Installation
============

Download the source code

.. code::

   git clone https://github.com/tcrundall/chronostar-tng.git

Change into the directory and create a new conda environment, and activate it.

.. code::

   cd chronostar-tng
   conda env create -n chron -f environment.yml    # set up py39 environment
   conda activate chron

Alternatively you can use pip, installing all the libraries in :code:`requirements.txt`.


You can install chronostar from source so that it is importable from everywhere, and your command line has access to the command line tools. This also installs the libraries that conda can't find but pip can.

.. code::

   python -m pip install .

Now that all depedencies are installed, you can run all the tests. Unit tests should take less than 10 seconds. Integration tests take up to an hour.

.. code::

    pytest tests/unit
    pytest tests/integration

You should now be able to import :code:`chronostar`:

.. code::

    >>> import chronostar
    >>> chronostar.__file__
    /path/to/env/lib/python3.9/site-packages/chronostar/__init__.py
    >>> from chronostar.component.spacetimecomponent import SpaceTimeComponent

.. note::

   For developers of chronostar, make sure this isn't clashing with any local version of chronostar hiding in your :code:`PYTHONPATH`.

You should also have access to the command line tools:

.. code::

    $ fit-component -h
    $ fit-mixture -h
    $ fit-chronostar -h
