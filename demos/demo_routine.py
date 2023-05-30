from pathlib import Path

from cinnamon_core.utility import logging_utility
from cinnamon_generic.api.commands import setup_registry
from cinnamon_generic.components.routine import Routine

if __name__ == '__main__':
    directory = Path(__file__).parent.parent.resolve()

    setup_registry(directory=directory,
                   registrations_to_file=True)

    logging_utility.logger.info(f'Directory: {directory}')

    # Routine
    routine = Routine.build_component(name='routine',
                                      tags={'train_and_test'},
                                      namespace='examples')
    routine_info = routine.run(is_training=True)
    logging_utility.logger.info(f'Routine info: {routine_info}')
