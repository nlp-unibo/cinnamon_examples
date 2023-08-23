.. _routine:

Defining a custom ``Routine``
*************************************

Routine components are already defined in ``cinnamon-generic``.

We can rely on them and write a custom ``RoutineConfig`` to run our machine-learning pipeline in **one shot**!

-------------------------
``ExampleRoutineConfig``
-------------------------

We first define our custom ``ExampleRoutineConfig`` that extends ``RoutineConfig``.

.. code-block:: python

    class ExampleRoutineConfig(RoutineConfig):

        @classmethod
        def get_default(
                cls
        ):
            config = super().get_default()

            config.data_loader = RegistrationKey(name='data_loader',
                                                 tags={'imdb', 'default'},
                                                 namespace='examples')
            config.pre_processor = RegistrationKey(name='processor',
                                                   tags={'tf-idf', 'label', 'ml'},
                                                   namespace='examples')
            config.data_splitter = RegistrationKey(name='data_splitter',
                                                   tags={'sklearn', 'tt'},
                                                   namespace='generic')
            config.model = RegistrationKey(name='model',
                                           tags={'svm', 'default'},
                                           namespace='examples')
            config.metrics = RegistrationKey(name='metrics',
                                             tags={'binary_f1', 'macro_f1', 'accuracy'},
                                             namespace='examples')
            config.helper = RegistrationKey(name='helper',
                                            tags={'default'},
                                            namespace='generic')
            config.routine_processor = RegistrationKey(name='routine_processor',
                                                       tags={'average'},
                                                       namespace='generic')

            config.seeds = [15000, 42]

            return config

Note how we are mixing ``RegistrationKey`` from this example project and ``cinnamon-generic``.

Lastly, we register ``ExampleRoutineConfig`` and bind it to ``TrainAndTestRoutine`` component to define a custom train, validation and test evaluation criteria.

.. code-block:: python

    @register
    def register_routines():
        Registry.add_and_bind(config_class=ExampleRoutineConfig,
                              component_class=TrainAndTestRoutine,
                              name='routine',
                              tags={'train_and_test'},
                              namespace='examples')


--------------------------------
Running the custom ``Routine``
--------------------------------

We can now write a script to test our custom routine.

.. code-block:: python

    from pathlib import Path
    from cinnamon_core.core.registry import Registry
    from cinnamon_core.utility import logging_utility
    from cinnamon_generic.api.commands import setup_registry, routine_train

    if __name__ == '__main__':
        """
        This demo script is the cinnamon command-compliant version of ``demo_routine.py``.
        We can use the ``routine_train`` command to run the training phase of a routine.
        We make use of command configurations (see ``configurations/commands.py``) to quickly load our command configuration.
        """

        setup_registry(directory=Path(__file__).parent.parent.resolve(),
                       registrations_to_file=True)

        cmd_config = Registry.build_configuration(name='command',
                                                  tags={'imdb', 'routine', 'train'},
                                                  namespace='examples')
        result = routine_train(name=cmd_config.name,
                               tags=cmd_config.tags,
                               namespace=cmd_config.namespace,
                               serialize=True,
                               run_name=cmd_config.run_name)
        logging_utility.logger.info(result)

The above code does the following:

- Loads data with ``ExampleLoader``.
- Parses data with ``ProcessorPipeline``.
- Defines a ``ExampleModel``.
- Trains and evaluates ``ExampleModel``.
- Repeats the above step for each specified seed in ``ExampleRoutineConfig``.

------------------
Congratulations!
------------------

That's it! We have successfully defined a **customizable**, **plug-and-play**, and **re-usable** machine-learning pipeline.

Feel free to play to download this repository and play with ``Component`` and ``Configuration``.

Cheers!