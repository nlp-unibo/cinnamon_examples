from pathlib import Path

from cinnamon_core.utility import logging_utility
from cinnamon_generic.api.commands import setup_registry
from cinnamon_generic.components.data_loader import DataLoader

if __name__ == '__main__':
    """
    In this demo script, we retrieve and build our IMDB data loader.
    Once built, we run the data loader to load the IMDB dataset and print it for visualization purposes.
    """

    setup_registry(directory=Path(__file__).parent.parent.resolve(),
                   registrations_to_file=True)

    loader = DataLoader.build_component(name='data_loader',
                                        tags={'default', 'imdb'},
                                        namespace='examples')
    data = loader.run()
    logging_utility.logger.info(data)
