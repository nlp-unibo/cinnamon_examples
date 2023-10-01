from cinnamon_core.core.registry import register, Registry
from cinnamon_generic.components.command_runner import ComponentRunner
from cinnamon_generic.configurations.command_runner import ComponentRunnerConfig, RoutineInferenceRunnerConfig


@register
def register_command_configurations():
    # Data loader
    Registry.add_and_bind(config_class=ComponentRunnerConfig,
                          config_constructor=ComponentRunnerConfig.get_delta_class_copy,
                          config_kwargs={
                              'params': {
                                  'name': 'data_loader',
                                  'namespace': 'examples',
                                  'tags': {'default', 'imdb'},
                                  'run_name': 'imdb_loader'
                              }
                          },
                          component_class=ComponentRunner,
                          name='command',
                          namespace='examples',
                          tags={'imdb', 'data_loader'})

    # Routine
    Registry.add_and_bind(config_class=ComponentRunnerConfig,
                          config_constructor=ComponentRunnerConfig.get_delta_class_copy,
                          config_kwargs={
                              'params': {
                                  'name': 'routine',
                                  'namespace': 'examples',
                                  'tags': {'train_and_test'},
                                  'run_name': 'imdb_svm'
                              }
                          },
                          component_class=ComponentRunner,
                          name='command',
                          namespace='examples',
                          tags={'imdb', 'routine', 'train'})

    Registry.add_and_bind(config_class=RoutineInferenceRunnerConfig,
                          config_constructor=RoutineInferenceRunnerConfig.get_delta_class_copy,
                          config_kwargs={
                              'params': {
                                  'namespace': 'examples',
                                  'tags': {'train_and_test'},
                                  'run_name': 'imdb_svm'
                              }
                          },
                          component_class=ComponentRunner,
                          name='command',
                          namespace='examples',
                          tags={'imdb', 'routine', 'inference'})
