from pathlib import Path

from cinnamon_core.utility import logging_utility
from cinnamon_generic.api.commands import setup_registry, run_component

if __name__ == '__main__':
    directory = Path(__file__).parent.parent.resolve()

    setup_registry(directory=directory,
                   registrations_to_file=True)

    logging_utility.logger.info(f'Directory: {directory}')
    run_component(name='data_loader',
                  tags={'default', 'imdb'},
                  namespace='examples',
                  run_name='imdb_loader_test',
                  serialize=True)
