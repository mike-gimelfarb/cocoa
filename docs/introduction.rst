Introduction
============

COCOA is a suite of algorithms written in C++ for the optimization of continuous 
black-box functions (mostly without using derivative information). 
Main advantages:

- a single unified interface for all algorithms
- a variety of classical algorithms with state-of-the-art improvements (e.g. automatic parameter adaptation)
- convenient wrappers for Python with a user-friendly API

Installation
============

To use this library in a Python project, you will need:

- C++ compiler (e.g., MS Build Tools)
- git
- pybind11

Then install directly from source:

.. code-block:: shell

    pip install git+https://github.com/mike-gimelfarb/cocoa