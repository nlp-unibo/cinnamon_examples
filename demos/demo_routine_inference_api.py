from pathlib import Path

from cinnamon_core.core.registry import Registry
from cinnamon_core.utility import logging_utility
from cinnamon_generic.api.commands import setup_registry, routine_inference

if __name__ == '__main__':
    """
    We can use the ``routine_inference`` command to run the inference phase of a pre-trained routine.
    We make use of command configurations (see ``configurations/commands.py``) to quickly load our command configuration.
    """

    setup_registry(directory=Path(__file__).parent.parent.resolve(),
                   registrations_to_file=True)

    runner = Registry.build_component(name='command',
                                      tags={'imdb', 'routine', 'inference'},
                                      namespace='examples')

    cmd_config = runner.run()
    result = routine_inference(routine_path=cmd_config.routine_path,
                               namespace=cmd_config.namespace,
                               serialize=False,
                               run_name=cmd_config.run_name)
    logging_utility.logger.info(result)
