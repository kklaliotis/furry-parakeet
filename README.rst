furry-parakeet: C kernels and wrappers for image combination
############################################################

This repository contains the linear algebra and interpolation kernels used by `PyIMCOM <https://github.com/Roman-HLIS-Cosmology-PIT/pyimcom>`_.

Installation
============

You should be able to pip install by going to the ``furry-parakeet`` directory and running::

    pip install .

This uses the Numpy/C API. If you want to adjust the default settings for C compilers, you can edit ``setup.py`` (for example: openmp is enabled by default).

Helpful information
===================

There are `interface instructions <docs/interface_instructions.rst>`_ if you are interested in how ``furry-parakeet`` can interface with other tools in the Roman ecosystem.

Links
=====

Some useful links include:

* `Hirata et al. (2024) <https://arxiv.org/abs/2303.08749>`_ describes the D5512 interpolator. The tables in Appendix A are available in machine-readable form `here <docs/Tables_A2A3.txt>`_.

* `Cao et al. (2025) <https://arxiv.org/abs/2410.05442>`_ describes the various linear algebra kernels.
