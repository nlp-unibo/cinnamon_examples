.. Cinnamon Examples documentation master file, created by
   sphinx-quickstart on Wed Aug 23 15:13:50 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Quickstart
=============================================

This project contains a complete example of a machine-learning pipeline.

In particular, we'll see how

- Define a ``DataLoader``
- Define multiple data ``Processor``
- Define a SVM classifier ``Model``
- Train and evaluate our classifier with a ``Routine``

We consider the IMDB data to perform a binary classification for sentiment analysis.

===============================================
Usage
===============================================

Want to use the ``Component`` and ``Configuration`` of this example project?

You can quickly integrate them into your project as follows

1. Download the repository
2. Pick any runnable script that starts with ``setup_registry``
3. Specify the repository base path in ``module_directories`` of ``setup_registry``

.. code-block:: python

    setup_registry(directory=Path(__file__).parent.parent.resolve(),
                   module_directories=['/path/to/this/repository'],
                   registrations_to_file=True)


Done! Cinnamon will automatically import all registered ``Configuration``.

You can code your custom ``Configuration`` with children pointing to the ``RegistrationKey`` of this project!

===============================================
Contact
===============================================

Don't hesitate to contact:
- `Federico Ruggeri <federico.ruggeri6@unibo.it>`_

for questions/doubts/issues!


.. toctree::
   :maxdepth: 4
   :hidden:
   :caption: Contents:
   :titlesonly:

   Data Loader <data_loader.rst>
   Processor  <processor.rst>
   Model <model.rst>
   Routine <routine.rst>
   Catalog <catalog.rst>