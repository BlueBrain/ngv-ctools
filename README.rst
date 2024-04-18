Overview
========

ngv-ctools provide c++ algorithms for ngv_ exposed via pybind11_ python bindings.

Algorithms
==========

* second_order_solutions
* grow_waves_on_triangulated_surface

Installation
============

If one is on a Linux platform, one should be able to use the compiled Python wheels.
This is the recommended way.

.. code-block:: bash

  pip install ngv-ctools

Tests
=====

.. code-block:: bash

  pip install tox
  tox

Acknowledgements
================

The development of this software was supported by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology.

For license and authors, see LICENSE.txt and AUTHORS.txt respectively.

Copyright (c) 2022-2024 Blue Brain Project/EPFL

.. _ngv: https://github.com/BlueBrain/ArchNGV
.. _pybind11: https://pybind11.readthedocs.io
