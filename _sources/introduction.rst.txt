Introduction to Chronostar
==========================

Chronostar is a wonderful tool for identifying and characterising
unbound stellar associations and is particular adept at determining
kinematic ages.

The next generation of Chronostar prioritises simplicity, flexibility and extensibility. The default behaviour addresses the simplest scenario but the means of extending the capabilties for more complex scenarios is straight forward; by default Chronostar-TNG (will) provide abstract base classes which clearly dictate required interfaces as well as many different example reifications for the most basic scenarios. How one actually extends these example implementations to address complex scenarios is left to the user. ;)

The framework of Chronostar TNG consists of 5 classes. A :doc:`Driver <guides/driverguide>`, a :doc:`Initial Conditions Pool <guides/icpoolguide>`, an :doc:`Introducer <guides/introducerguide>`, a :doc:`Mixture <guides/mixtureguide>` and a :doc:`Component <guides/componentguide>`.

.. image:: images/simple_snapshot.svg
  :width: 800
  :alt: A graphical representation of how the classes connect.


The goal of Chronostar is to be a flexible framework for fitting Gaussian
Mixture Models to astronomical data. This will be achieved by utilising
"injected dependencies". The :doc:`Driver <guides/driverguide>` is composed of a collection
of 4 classes (:doc:`ICPool <guides/icpoolguide>`, :doc:`Introducer <guides/introducerguide>`, :doc:`Mixture <guides/mixtureguide>` and :doc:`Component <guides/componentguide>`) for which
Chronostar provides many implementations. Anyone wishing to modify 
aspects of Chronostar (e.g. input data, fitting method, models of 
components) simply needs to provide the :doc:`driver <guides/driverguide>` with their own
class that matches the required interface, whether that be by writing
the class from scratch, or by using inheritance to extend pre-existing
classes.
