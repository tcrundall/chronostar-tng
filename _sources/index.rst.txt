.. chronostar-dev documentation master file, created by
   sphinx-quickstart on Sat Sep 17 16:22:57 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Chronostar-TNG's documentation!
==========================================
**Chronostar** is a Python library for discovering and characterising
unbound stellar associations (aka "moving groups") and their members using 
features such as kinematics, age markers, chemical composition. The code 
can be found on `github <https://github.com/tcrundall/chronostar-tng/>`_.

These docs will sketch out the structure of the next generation of Chronostar
code.


It turns out fitting Gaussian Mixture Models is quite complex, so a lot of
the design will be shamelessly *inspired* by 
`scikit-learn <https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html>`_.

..
   Check out the :doc:`usage` section for further information, including how to :ref:`install <installation>` the project.

.. note::

   This project is under active development

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :glob:

   introduction
   Getting Started <gettingstarted>
   General Overview <generaloverview>
   Worked Examples <examples/workedexamples>
   API <api/modules>
   FAQ <faq>


..
   .. toctree::
      :maxdepth: 2
      :caption: Lost docs:
      :glob:

      uncertaintyfinder


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
