from pathlib import Path
from typing import cast

import pytest
from cinnamon_core.core.registry import RegistrationKey, Registry
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

    loader_registration_key = RegistrationKey(name='data_loader',
                                              tags={'default', 'imdb'},
                                              namespace='examples')
    loader = Registry.build_component_from_key(config_registration_key=loader_registration_key)
    loader = cast(DataLoader, loader)
    data = loader.run()

    file_manager = Registry.retrieve_built_component_from_key(config_registration_key=file_manager_regr_key)
    file_manager = cast(FileManager, file_manager)
    serialization_path = file_manager.run(filepath=file_manager.serialization_directory)

    return data, serialization_path


def test_processor_serialization(
        get_data
):
    data, serialization_path = get_data

    tip_regr_key = RegistrationKey(name='processor',
                                   tags={'default', 'tf-idf'},
                                   namespace='examples')
    tip = Registry.build_component_from_key(config_registration_key=tip_regr_key)
    tip = cast(Processor, tip)
    tip.run(data=data.train,
            is_training_data=True)
