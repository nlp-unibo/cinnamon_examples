from cinnamon_core.core.registry import register, Registry
from cinnamon_generic.configurations.commands import ComponentRunConfig


@register
def register_command_configurations():
    # Data loader
    Registry.add_configuration(config_class=ComponentRunConfig,
                               config_constructor=ComponentRunConfig.get_delta_class_copy,
                               config_kwargs={
                                   'params': {
                                       'name': 'data_loader',
                                       'namespace': 'examples',
                                       'tags': {'default', 'imdb'},
                                       'run_name': 'imdb_loader'
                                   }
                               },
                               name='command',
                               namespace='examples',
                               tags={'imdb', 'data_loader'})

    # Routine
    Registry.add_configuration(config_class=ComponentRunConfig,
                               config_constructor=ComponentRunConfig.get_delta_class_copy,
                               config_kwargs={
                                   'params': {
                                       'name': 'routine',
                                       'namespace': 'examples',
                                       'tags': {'train_and_test'},
                                       'run_name': 'imdb_svm'
                                   }
                               },
                               name='command',
                               namespace='examples',
                               tags={'imdb', 'routine'})
