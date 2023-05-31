from pathlib import Path

import pytest

from cinnamon_core.utility import logging_utility
from cinnamon_generic.api.commands import setup_registry
from cinnamon_generic.components.data_loader import DataLoader
from cinnamon_generic.components.file_manager import FileManager
from cinnamon_generic.components.processor import Processor


@pytest.fixture
def get_data():
    directory = Path(__file__).parent.parent.resolve()

    file_manager_regr_key = setup_registry(directory=directory,
                                           registrations_to_file=True)

    logging_utility.logger.info(f'Directory: {directory}')

    loader = DataLoader.build_component(name='data_loader',
                                        tags={'default', 'imdb'},
                                        namespace='examples')
    data = loader.run()

    file_manager = FileManager.retrieve_built_component_from_key(config_registration_key=file_manager_regr_key)

    return data, file_manager.runs_directory


def test_processor_serialization(
        get_data
):
    data, serialization_path = get_data

    tip = Processor.build_component(name='processor',
                                    tags={'default', 'tf-idf'},
                                    namespace='examples')
    tip.run(data=data.train,
            is_training_data=True)
