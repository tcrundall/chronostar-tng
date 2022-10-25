Installation
============

Download
--------
Download the source code

.. code::

   git clone https://github.com/tcrundall/chronostar-tng.git

Virtual Environment
-------------------
Chronostar-TNG utilises features from :code:`python >= 3.9`, so we recommend making
a virtual environment.

Conda
^^^^^
Change into the directory and create a new conda environment, and activate it.
Conda will automatically set up a python 3.9 environment.

.. code::

   cd chronostar-tng
   conda env create -n chron -f environment.yml    # set up py39 environment
   conda activate chron

Chronostar developers should use the :code:`dev-environment.yml`, as this includes extra packages that assist in development of the code and documentation.

Pip
^^^
Alternatively you can use pip, but you need to ensure you already have python 3.9 installed
somewhere.

.. code::

   python -m pip install virtualenv
   python -m virtualenv -p /path/to/your/3.9/bin/python chron
   source chron/bin/activate

Once you've set up your virtual environment install all the libraries in :code:`requirements.txt` (or :code:`dev-requirements.txt`):

.. code::

   cd chronostar-tng
   pip install -r requirements.txt

Installing Chronostar
---------------------
You can install chronostar from source so that it is importable from everywhere, and your command line has access to the command line tools. This also installs the libraries that conda can't find but pip can.

.. code::

   python -m pip install .

Developers should install with the editable flag, enabling changes you make to the source code to be immediately reflected in the build libraries and command line tools:

.. code::
   
   python -m pip install -e .

Testing
-------
Now that all depedencies are installed, you can run all the tests. Unit tests should take less than a minute. Integration tests take over an hour.

.. code::

    pytest tests/unit
    pytest tests/integration

.. note::

   These tests run automatically on github on every push to main.

Confirming access
-----------------
You should now be able to import :code:`chronostar`:

.. code::

    >>> import chronostar
    >>> chronostar.__file__
    /path/to/env/lib/python3.9/site-packages/chronostar/__init__.py
    >>> from chronostar.component.spherespacetimecomponent import SphereSpaceTimeComponent

.. note::

   For developers of chronostar, make sure this isn't clashing with any local version of chronostar hiding in your :code:`PYTHONPATH`.

You should also have access to the command line tools:

.. code::

    $ fit-component -h
    $ fit-mixture -h
    $ fit-chronostar -h
