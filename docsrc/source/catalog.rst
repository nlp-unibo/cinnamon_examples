.. _catalog:

Available ``Configuration``
*************************************

Currently, ``cinnamon-examples`` provides the following registered ``Configuration``.

-------------
Data Loader
-------------

- ``name='data_loader', tags={'imdb'}, namespace='examples'``: the default ``ExampleLoader``.

------------
Processor
------------

- ``name='processor', tags={'tf-idf', 'default'}, namespace='examples'``: the default ``TfIdfProcessor``.
- ``name='processor', tags={'label', 'default'}, namespace='examples'``: the default ``LabelProcessor``.
- ``name='processor', tags={'ml', 'default'}, namespace='examples'``: the default ``MLProcessor``.
- ``name='processor', tags={'tf-idf', 'label', 'ml'}, namespace='examples'``: the default ``ProcessorPipeline``.

-------------
Model
-------------

- ``name='model', tags={'svm', 'default'}, namespace='examples'``: the default ``ExampleModel``.

------------
Metrics
------------

- ``name='metrics', tags={'binary_f1'}, namespace='examples'``: the binary f1-score ``LambdaMetric``.
- ``name='metrics', tags={'macro_f1'}, namespace='examples'``: the macro f1-score ``LambdaMetric``.
- ``name='metrics', tags={'accuracy'}, namespace='examples'``: the accuracy ``LambdaMetric``.
- ``name='Metrics', tags={'binary_f1', 'macro_f1', 'accuracy'}, namespace='examples'``: the ``Pipeline`` that computes binary and macro f1-score, and accuracy metrics.


-----------
Routine
-----------

- ``name='routine', tags={'train_and_test'}, namespace='examples'``: the ``TrainAndTestRoutine`` that evaluate ``ExampleModel`` on the IMDB dataset.
