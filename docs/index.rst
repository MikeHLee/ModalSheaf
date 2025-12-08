ModalSheaf Documentation
========================

**Practical Sheaf Theory for Multimodal Machine Learning**

ModalSheaf is a Python library that brings the power of sheaf theory to ML practitioners
without requiring a PhD in algebraic topology. It provides intuitive tools for:

- Moving data between modalities (text, images, code, embeddings)
- Tracking information loss in transformations
- Detecting inconsistencies in multimodal data
- Building consensus from multiple sources

.. note::
   
   This documentation is designed to teach you the underlying mathematics
   gently, using examples from machine learning. You don't need to understand
   the theory to use the library, but understanding it will make you a better
   ML practitioner.

Quick Start
-----------

.. code-block:: python

   from modalsheaf import ModalityGraph, diagnose_gluing_problem
   
   # Create a graph of modalities
   graph = ModalityGraph("my_project")
   graph.add_modality("text")
   graph.add_modality("embedding", shape=(768,))
   graph.add_transformation("text", "embedding", forward=my_encoder)
   
   # Transform data
   embedding = graph.transform("text", "embedding", "Hello world")

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/installation
   getting_started/quickstart
   getting_started/concepts

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/01_modalities
   tutorials/02_transformations
   tutorials/03_gluing
   tutorials/04_diagnostics
   tutorials/05_real_world

.. toctree::
   :maxdepth: 2
   :caption: Mathematical Background

   theory/00_why_topology
   theory/01_spaces_and_continuity
   theory/02_sheaves_intuition
   theory/03_restriction_extension
   theory/04_gluing_and_cohomology
   theory/05_applications_to_ml

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/graph
   api/modalities
   api/gluing
   api/diagnostics

.. toctree::
   :maxdepth: 1
   :caption: Resources

   resources/bibliography
   resources/further_reading
   resources/glossary

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
