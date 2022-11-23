.. chronostar-dev documentation master file, created by
   sphinx-quickstart on Sat Sep 17 16:22:57 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Chronostar-TNG's documentation!
==========================================
**Chronostar** is a Python library for discovering and characterising
unbound stellar associations (aka
"`moving groups <https://en.wikipedia.org/wiki/Stellar_kinematics#Moving_groups>`_")
and their members using
features such as kinematics, age markers, chemical composition. The code 
can be found on `github <https://github.com/tcrundall/chronostar-tng/>`_.

This documentation details the usage of the next generation of Chronostar
code.

It turns out fitting Gaussian Mixture Models is quite complex, so a lot of
the API design (`Buitinck et al. 2013 <https://arxiv.org/abs/1309.0238>`_) will be shamelessly *inspired* by
`scikit-learn <https://scikit-learn.org/stable/about.html>`_.

Publications
------------
`Chronostar: a novel Bayesion method for kinematic age determination. I. Derivation and application to the Beta Pictoris Moving Group -
Crundall et al. (2019)
<https://arxiv.org/abs/1902.07732>`_

`Chronostar. II. Kinematic age and substruction of the Scorpius-Cenaurus OB2 association -
Žerjal et al. (submitted)
<https://arxiv.org/abs/2111.09897>`_.


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
   Configuration Settings <userdocs/settings>
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
