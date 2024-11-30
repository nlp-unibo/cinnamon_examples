.. _model:

Defining a SVM classifier
*************************************

We are ready to define our SVM classifier.

We define the ``ExampleModel`` component to wrap  a SVC from sklearn.

Then, we define its associated ``ExampleModelConfig`` and perform registrations.

Lastly, we define the runnable script to run our ``ExampleModel``.

------------------
``ExampleModel``
------------------

.. code-block:: python

    class ExampleModel(Model):

        def save_model(
                self,
                filepath: Union[AnyStr, Path],
        ):
            filepath = Path(filepath) if type(filepath) != Path else filepath
            save_pickle(filepath=filepath.joinpath('model.pkl'),
                        data=self.model)

        def load_model(
                self,
                filepath: Union[AnyStr, Path],
        ) -> Any:
            filepath = Path(filepath) if type(filepath) != Path else filepath
            return load_pickle(filepath=filepath.joinpath('model.pkl'))

        def build(
                self,
                processor: Processor,
                callbacks: Optional[Callback] = None
        ):
            self.model = SVC(C=self.C,
                             kernel=self.kernel,
                             class_weight=self.class_weight)

        def predict(
                self,
                data: FieldDict,
                callbacks: Optional[Callback] = None,
                metrics: Optional[Metric] = None,
                model_processor: Optional[Processor] = None,
                suffixes: Optional[Dict] = None
        ) -> FieldDict:
            predictions = self.model.predict(X=data.X)

            return_field = FieldDict()
            return_field.add(name='predictions',
                             value=predictions)
            if suffixes is not None:
                return_field.add(name='suffixes',
                                 value=suffixes)

            if 'y' in data and metrics is not None:
                metrics_result = metrics.run(y_pred=predictions,
                                             y_true=data.y,
                                             as_dict=True)
                return_field.add(name='metrics',
                                 value=metrics_result)

            return return_field

        def fit(
                self,
                train_data: FieldDict,
                val_data: Optional[FieldDict] = None,
                metrics: Optional[Metric] = None,
                callbacks: Optional[Callback] = None,
                model_processor: Optional[Processor] = None
        ) -> FieldDict:
            self.model.fit(X=train_data.X, y=train_data.y)

            return_field = FieldDict()

            if val_data is not None:
                val_info = self.evaluate(data=val_data,
                                         callbacks=callbacks,
                                         metrics=metrics)
                return_field.add(name='val_info',
                                 value=val_info)
            return return_field

        def evaluate(
                self,
                data: FieldDict,
                callbacks: Optional[Callback] = None,
                metrics: Optional[Metric] = None,
                model_processor: Optional[Processor] = None,
                suffixes: Optional[Dict] = None
        ) -> FieldDict:
            return self.predict(data=data,
                                callbacks=callbacks,
                                metrics=metrics,
                                model_processor=model_processor,
                                suffixes=suffixes)


Note how ``fit()`` and ``predict()`` functions simply wrap the ``model.fit()`` and ``model.predict()`` functions of the SVC.

To keep our code clean, we have defined a ``MLProcessor`` such that input data ``FieldDict`` contains two fields: ``X`` and ``y``.


-----------------------
``ExampleModelConfig``
-----------------------

The ``ExampleModel`` uses ``ExampleModelConfig`` as default configuration template.

.. code-block:: python

    class ExampleModelConfig(Configuration):

        @classmethod
        def get_default(
                cls
        ):
            config = super().get_default()

            config.add(name='C',
                       value=1.0,
                       type_hint=float,
                       description='C parameter of SVC')

            config.add(name='kernel',
                       type_hint=str,
                       value='linear',
                       description='The kernel of the SVC')

            config.add(name='class_weight',
                       type_hint=Optional[str],
                       value='balanced',
                       description='The weighting technique for addressing class imbalance.'
                                   'Each sample in the training set receives a weight based on'
                                   ' its class distribution')

            return config

Next, we register ``ExampleModelConfig`` and bind it to ``ExampleModel``.

.. code-block:: python

    @register
    def register_models():
        Registry.add_and_bind(config_class=ExampleModelConfig,
                              component_class=ExampleModel,
                              name='model',
                              tags={'svm'},
                              is_default=True,
                              namespace='examples')


----------------------------
Running ``ExampleModel``
----------------------------

We can now write a script to test ``ExampleModel``.

.. code-block:: python

    from pathlib import Path
    from cinnamon_core.utility import logging_utility
    from cinnamon_generic.api.commands import setup_registry
    from cinnamon_generic.components.data_loader import DataLoader
    from cinnamon_generic.components.metrics import Metric
    from cinnamon_generic.components.model import Model
    from cinnamon_generic.components.processor import Processor, ProcessorPipeline

    if __name__ == '__main__':
        """
        In this demo script, we manually define a simple code pipeline:
        - Data loading
        - Data pre-processing
        - Model definition
        - Model training
        """

        setup_registry(directory=Path(__file__).parent.parent.resolve(),
                       registrations_to_file=True)

        # DataLoader (dl)
        dl = DataLoader.build_component(name='data_loader',
                                        tags={'default', 'imdb'},
                                        namespace='examples')
        data = dl.run()

        # ProcessorPipeline (pp)
        pp = Processor.build_component(name='processor',
                                        tags={'tf-idf', 'label', 'ml'},
                                        namespace='examples')
        pp.run(data=data.train, is_training_data=True)
        pp.run(data=data.val)
        pp.run(data=data.test)

        # Model
        model = Model.build_component(name='model',
                                      tags={'default', 'svm'},
                                      namespace='examples')

        # Training
        model.build(processor=pp)
        fit_info = model.fit(train_data=data.train,
                             val_data=data.val,
                             metrics=None,
                             callbacks=None)
        logging_utility.logger.info(f'Fit info: {fit_info}')

        predict_info = model.predict(data=data.test,
                                     metrics=metrics)
        logging_utility.logger.info(f'Predict info: {predict_info}')

----------------
Metrics
----------------

We can do a bit better by defining some metrics to evaluate our model.

We can quickly define some basic metrics like binary and macro f1-score, and accuracy as follows.

.. code-block:: python

    @register
    def register_metrics_configurations():
        Registry.add_and_bind(config_class=LambdaMetricConfig,
                              component_class=LambdaMetric,
                              config_constructor=LambdaMetricConfig.get_delta_class_copy,
                              config_kwargs={
                                  'params': {
                                      'name': 'binary_f1',
                                      'method': f1_score,
                                      'method_args': {'average': 'binary', 'pos_label': 1}
                                  }
                              },
                              name='metrics',
                              tags={'binary_f1'},
                              namespace='examples')
        Registry.add_and_bind(config_class=LambdaMetricConfig,
                              component_class=LambdaMetric,
                              config_constructor=LambdaMetricConfig.get_delta_class_copy,
                              config_kwargs={
                                  'params': {
                                      'name': 'macro_f1',
                                      'method': f1_score,
                                      'method_args': {'average': 'macro'}
                                  }
                              },
                              name='metrics',
                              tags={'macro_f1'},
                              namespace='examples')
        Registry.add_and_bind(config_class=LambdaMetricConfig,
                              component_class=LambdaMetric,
                              config_constructor=LambdaMetricConfig.get_delta_class_copy,
                              config_kwargs={
                                  'params': {
                                      'name': 'accuracy',
                                      'method': accuracy_score,
                                  }
                              },
                              name='metrics',
                              tags={'accuracy'},
                              namespace='examples')


As for ``Processor``, we can wrap all these metrics into a single component via ``Pipeline``.

.. code-block:: python

    Registry.add_and_bind(config_class=PipelineConfig,
                          config_constructor=PipelineConfig.from_keys,
                          config_kwargs={
                              'keys': [
                                  RegistrationKey(name='metrics', tags={'binary_f1'}, namespace='examples'),
                                  RegistrationKey(name='metrics', tags={'macro_f1'}, namespace='examples'),
                                  RegistrationKey(name='metrics', tags={'accuracy'}, namespace='examples')
                              ],
                              'names': [
                                  'binary_f1',
                                  'macro_f1',
                                  'accuracy'
                              ]
                          },
                          component_class=MetricPipeline,
                          name='metrics',
                          tags={'binary_f1', 'macro_f1', 'accuracy'},
                          namespace='examples')

Now, our test script becomes

.. code-block:: python

    from pathlib import Path
    from cinnamon_core.utility import logging_utility
    from cinnamon_generic.api.commands import setup_registry
    from cinnamon_generic.components.data_loader import DataLoader
    from cinnamon_generic.components.metrics import Metric
    from cinnamon_generic.components.model import Model
    from cinnamon_generic.components.processor import Processor, ProcessorPipeline

    if __name__ == '__main__':
        setup_registry(directory=Path(__file__).parent.parent.resolve(),
                       registrations_to_file=True)

        # DataLoader (dl)
        dl = DataLoader.build_component(name='data_loader',
                                        tags={'default', 'imdb'},
                                        namespace='examples')
        data = dl.run()

        # ProcessorPipeline (pp)
        pp = Processor.build_component(name='processor',
                                        tags={'tf-idf', 'label', 'ml'},
                                        namespace='examples')
        pp.run(data=data.train, is_training_data=True)
        pp.run(data=data.val)
        pp.run(data=data.test)

        # Metrics
        metrics = Metric.build_component(name='metrics',
                                     tags={'accuracy', 'binary_f1', 'macro_f1'},
                                     namespace='examples')

        # Model
        model = Model.build_component(name='model',
                                      tags={'default', 'svm'},
                                      namespace='examples')

        # Training
        model.build(processor=pp)
        fit_info = model.fit(train_data=data.train,
                             val_data=data.val,
                             metrics=metrics,
                             callbacks=None)
        logging_utility.logger.info(f'Fit info: {fit_info}')

        predict_info = model.predict(data=data.test,
                                     metrics=metrics)
        logging_utility.logger.info(f'Predict info: {predict_info}')

The same reasoning can also be applied to ``Callback``.

You can try on your own by experimenting with additional ``Metric``, ``Callback`` and ``Processor`` components!

----------------
Next!
----------------

That's it! We have define our SVM classifier as a custom ``Model`` component, along with some custom ``Metric`` to evaluate it.

Next, we define a proper evaluation criteria by wrapping our data, processing, and model pipeline into a ``Routine``.