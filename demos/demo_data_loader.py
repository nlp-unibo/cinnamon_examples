from pathlib import Path

from cinnamon_core.utility import logging_utility
from cinnamon_generic.api.commands import setup_registry
from cinnamon_generic.components.data_loader import DataLoader

if __name__ == '__main__':
    directory = Path(__file__).parent.parent.resolve()

    setup_registry(directory=directory,
                   registrations_to_file=True)

    logging_utility.logger.info(f'Directory: {directory}')

    loader = DataLoader.build_component(name='data_loader',
                                        tags={'default', 'imdb'},
                                        namespace='examples')
    data = loader.run()
    logging_utility.logger.info(data)
