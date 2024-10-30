AmpliTF
=======

Library of primitives for amplitude analyses in high-energy physics using TensorFlow v2.

Introduction
------------

The package includes the primitives to operate with relativistic kinematics (operations with Lorentz vectors, elements of helicity formalism), 
and descriptions of the dynamical functions (such as Breit-Wigner line shapes) using Google TensorFlow library. 

This package is a fork of `TensorFlowAnalysis <https://gitlab.cern.ch/poluekt/TensorFlowAnalysis>`_ that includes only the "stable" basic functionality. 
The rest of TensorFlowAnalysis functions is moved to a separate package called `TFA2 <https://github.com/apoluekt/TFA2>`_. 

The package is compatible with TensorFlow v2. 

Prerequisites
-------------

   * TensorFlow v2.1
   * NumPy
   * SymPy

Installation
------------

After checking out the package from git, run

.. code-block:: bash

   $ python setup.py build
   $ python setup.py install


Links
-----

   * `TensorFlowAnalysis <https://gitlab.cern.ch/poluekt/TensorFlowAnalysis>`_
   * `ZFit <https://github.com/zfit/zfit>`_
   * `ComPWA <https://github.com/ComPWA>`_


.. .. toctree::
..    :caption: User Guide
..    :maxdepth: 3

..    guide/guide_index

.. .. toctree::
..    :caption: Summary Plots
..    :maxdepth: 6

..    summary_plots/summary_plots_index

.. toctree::
   :caption: API Reference
   :maxdepth: 2

   autoapi/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`