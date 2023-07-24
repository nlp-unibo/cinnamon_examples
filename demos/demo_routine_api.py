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
                                              tags={'imdb', 'routine'},
                                              namespace='examples')
    result = routine_train(name=cmd_config.name,
                           tags=cmd_config.tags,
                           namespace=cmd_config.namespace,
                           serialize=True,
                           run_name=cmd_config.run_name)
    logging_utility.logger.info(result)
