from pathlib import Path

from cinnamon_core.utility import logging_utility
from cinnamon_generic.api.commands import setup_registry, routine_train

if __name__ == '__main__':
    directory = Path(__file__).parent.parent.resolve()

    setup_registry(directory=directory,
                   registrations_to_file=True)

    logging_utility.logger.info(f'Directory: {directory}')

    routine_train(name='routine',
                  tags={'train_and_test'},
                  namespace='examples',
                  serialize=True,
                  run_name='test')
