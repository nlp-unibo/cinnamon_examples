.. _processor:

Parsing data with ``Processor``
*************************************

We still need to parse loaded data in order to train and evaluate our SVM classifier.

We can define several plug-and-play ``Processor`` to

- Process input data
- Process classification labels
- Process data for the classifier

--------------------
Input data
--------------------

To process input data, we rely on tf-idf processing since we are dealing with a SVM classifier.

We define a ``TfIdfProcessor`` as follows

.. code-block:: python

    class TfIdfProcessor(Processor):

        def __init__(
                self,
                **kwargs
        ):
            super().__init__(**kwargs)
            self.vectorizer = TfidfVectorizer(ngram_range=self.ngram_range)

        def prepare_save_data(
                self
        ) -> Dict:
            data = super().prepare_save_data()

            data['vectorizer'] = self.vectorizer
            return data

        def process(
                self,
                data: FieldDict,
                is_training_data: bool = False,
        ):
            if is_training_data:
                text_data = data.search_by_tag(tags='text')
                text_data = list(itertools.chain.from_iterable([field for field in text_data.values()]))
                self.vectorizer.fit(text_data)

            text_fields = data.search_by_tag(tags='text')
            for index, field in text_fields.items():
                data[index] = self.vectorizer.transform(field)
            return data


The ``TfIdfProcessor`` has an internal ``TfidfVectorizer`` from sklearn. The vectorizer is used in ``process()`` to convert textual input data into numerical format.

We define a corresponding ``TfIdfProcessorConfig`` with minimal view (for simplicity) of the vectorizer.

.. code-block:: python

    class TfIdfProcessorConfig(Configuration):

        @classmethod
        def get_default(
                cls
        ):
            config = super().get_default()

            config.add(name='ngram_range',
                       value=(1, 1),
                       type_hint=Any,
                       description='Vectorizer ngram_range hyper-parameter')

            return config


Lastly, we register the ``TfIdfProcessorConfig`` and bind it to ``TfIdfProcessor``

.. code-block:: python

    @register
    def register_processors():
        Registry.add_and_bind(config_class=TfIdfProcessorConfig,
                              component_class=TfIdfProcessor,
                              name='processor',
                              tags={'tf-idf'},
                              is_default=True,
                              namespace='examples')


-----------------------
Classification Labels
-----------------------

To process classification labels, we rely on one-hot encoding via ``LabelEncoder`` from sklearn.

We define a ``LabelProcessor`` as follows

.. code-block:: python

    class LabelProcessor(Processor):

        def __init__(
                self,
                **kwargs
        ):
            super().__init__(**kwargs)
            self.encoders = dict()

        def prepare_save_data(
                self
        ) -> Dict:
            data = super().prepare_save_data()

            data['encoders'] = self.encoders
            return data

        def process(
                self,
                data: FieldDict,
                is_training_data: bool = False
        ):
            label_fields = data.search_by_tag(tags='label')
            for field_name, field in label_fields.items():
                field_encoder = self.encoders.get(field_name, LabelEncoder())
                data[field_name] = field_encoder.fit_transform(field) \
                    if is_training_data else field_encoder.transform(field)
                if field_name not in self.encoders and is_training_data:
                    self.encoders[field_name] = field_encoder

            return data

The ``LabelProcessor`` doesn't require any specific configuration since it has no hyper-parameters.

Thus, we can bind it to ``Configuration``.

.. code-block:: python

    Registry.add_and_bind(config_class=Configuration,
                          component_class=LabelProcessor,
                          name='processor',
                          tags={'label'},
                          is_default=True,
                          namespace='examples')


-----------------------
Classifier input
-----------------------

We have processed both input data and classification labels.

We need just to provide a proper data view for our model, such that it is independent of the underlying processing components.

We do by defining a ``MLProcessor`` component

.. code-block:: python

    class MLProcessor(Processor):

        def process(
                self,
                data: FieldDict,
                is_training_data: bool = False
        ) -> FieldDict:
            return_dict = FieldDict()

            text_data = list(data.search_by_tag(tags='text').values())[0]
            return_dict.add(name='X',
                            value=text_data)

            label_data = data.search_by_tag(tags='label')
            if len(label_data):
                label_data = list(label_data.values())[0]
                return_dict.add(name='y',
                                value=label_data)

            return return_dict

Now input data is wrapped as ``X``, while classification labels are wrapped as ``y``. Much easier to remember, isn't it?

.. note::
    If you don't need such level of abstraction, you just need to remove the ``MLProcessor`` from your component pipelines. No code contamination :)

Lastly, we register ``MLProcessor`` and use a ``Configuration`` to bind it since no hyper-parameters are defined.

.. code-block:: python

    Registry.add_and_bind(config_class=Configuration,
                          component_class=MLProcessor,
                          name='processor',
                          tags={'ml'},
                          is_default=True,
                          namespace='examples')


------------------
Making a Pipeline
------------------

We have first split the pre-processing logic into several ``Processor``, each with a specific purpose.

We now want to provide a single interface to execute all the ``Processor`` as a single one.

We can do so via ``Pipeline`` component by defining a ``ProcessorPipeline``.

.. code-block:: python

    class ProcessorPipeline(OrderedPipeline, Processor):

        def run(
                self,
                data: Optional[FieldDict] = None,
                is_training_data: bool = False
        ) -> FieldDict:
            components = self.get_pipeline()
            for component in components:
                data = component.run(data=data, is_training_data=is_training_data)
            return data



We only need to write the corresponding ``OrderedPipelineConfig`` to wrap ``TfIdfProcessor``, ``LabelProcessor``, and ``MLProcessor`` altogether.

.. code-block:: python

    Registry.add_and_bind(config_class=OrderedPipelineConfig,
                          config_constructor=OrderedPipelineConfig.from_keys,
                          config_kwargs={
                              'keys': [
                                  RegistrationKey(name='processor', tags={'default', 'tf-idf'}, namespace='examples'),
                                  RegistrationKey(name='processor', tags={'default', 'label'}, namespace='examples'),
                                  RegistrationKey(name='processor', tags={'default', 'ml'}, namespace='examples')
                              ],
                              'names': [
                                  'text_processor',
                                  'label_processor',
                                  'ml_processor'
                              ]
                          },
                          component_class=ProcessorPipeline,
                          name='processor',
                          tags={'tf-idf', 'label', 'ml'},
                          namespace='examples')


----------------------------
Running the processors
----------------------------

We can now write a script to test our processors in combination with the previously defined ``ExampleLoader``.

.. code-block:: python

    from pathlib import Path
    from cinnamon_core.utility import logging_utility
    from cinnamon_generic.api.commands import setup_registry
    from cinnamon_generic.components.data_loader import DataLoader
    from cinnamon_generic.components.processor import Processor

    if __name__ == '__main__':
        """
        In this demo script, we retrieve and build our IMDB data loader and input processor components.
        We first load the IMDB data via the related data loader and subsequently process the data via the processors.
        You may notice that processors modify input data in-place.
        """

        setup_registry(directory=Path(__file__).parent.parent.resolve(),
                       registrations_to_file=True)

        # DataLoader (dl)
        dl = DataLoader.build_component(name='data_loader',
                                        tags={'default', 'imdb'},
                                        namespace='examples')
        data = dl.run()

        # TfIdfProcessor (tip)
        tip = Processor.build_component(name='processor',
                                        tags={'default', 'tf-idf'},
                                        namespace='examples')
        tip.run(data=data.train, is_training_data=True)
        tip.run(data=data.val)
        tip.run(data=data.test)

        # LabelProcessor (lp)
        lp = Processor.build_component(name='processor',
                                       tags={'default', 'label'},
                                       namespace='examples')
        lp.run(data=data.train, is_training_data=True)
        lp.run(data=data.val)
        lp.run(data=data.test)

        # MLProcessor (mp)
        mp = Processor.build_component(name='processor',
                                       tags={'default', 'ml'},
                                       namespace='examples')
        mp.run(data=data.train)
        mp.run(data=data.val)
        mp.run(data=data.test)

        logging_utility.logger.info(f'Train: {data.train}')
        logging_utility.logger.info(f'Val: {data.val}')
        logging_utility.logger.info(f'Test: {data.test}')

Alternatively, we can use our ``ProcessorPipeline`` to quickly execute all processors in a sequential ordered fashion.

.. code-block:: python

    from pathlib import Path
    from cinnamon_core.utility import logging_utility
    from cinnamon_generic.api.commands import setup_registry
    from cinnamon_generic.components.data_loader import DataLoader
    from cinnamon_generic.components.processor import Processor

    if __name__ == '__main__':
        """
        In this demo script, we retrieve and build our IMDB data loader and input processor components.
        We first load the IMDB data via the related data loader and subsequently process the data via the processors.
        You may notice that processors modify input data in-place.
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

        logging_utility.logger.info(f'Train: {data.train}')
        logging_utility.logger.info(f'Val: {data.val}')
        logging_utility.logger.info(f'Test: {data.test}')

----------------
Next!
----------------

That's it! We have defined several processors to parse input data so that it can be digested by our SVM classifier.

Next, we define the SVM classifier as a custom ``Model`` component.