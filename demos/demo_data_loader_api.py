from pathlib import Path

from cinnamon_core.utility import logging_utility
from cinnamon_generic.api.commands import setup_registry, run_component
from cinnamon_core.core.registry import Registry

if __name__ == '__main__':
    """
    This demo script is the cinnamon command-compliant version of ``demo_data_loader.py``.
    We can use the ``run_component`` command to run a generic component (our data loader in this case).
    We make use of command configurations (see ``configurations/commands.py``) to quickly load our command configuration.
    """

    setup_registry(directory=Path(__file__).parent.parent.resolve(),
                   registrations_to_file=True)

    cmd_config = Registry.build_configuration(name='command',
                                              tags={'imdb', 'data_loader'},
                                              namespace='examples')
    result, _ = run_component(name=cmd_config.name,
                              tags=cmd_config.tags,
                              namespace=cmd_config.namespace,
                              run_name=cmd_config.run_name,
                              serialize=False)
    logging_utility.logger.info(result)
