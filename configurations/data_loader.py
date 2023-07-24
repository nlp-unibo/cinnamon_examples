from pathlib import Path
from typing import AnyStr, Union

from cinnamon_core.core.registry import Registry, RegistrationKey, register
from cinnamon_generic.configurations.data_loader import DataLoaderConfig
from components.data_loader import ExampleLoader


class ExampleLoaderConfig(DataLoaderConfig):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        config.has_val_split = False
        config.name = 'example_dataset'

        config.add(name='file_manager_key',
                   value=RegistrationKey(
                       name='file_manager',
                       tags={'default'},
                       namespace='generic'
                   ),
                   type_hint=RegistrationKey,
                   description="registration info of built FileManager component."
                               " Used for filesystem interfacing")
        config.add(name='data_url',
                   value='http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
                   type_hint=Union[AnyStr, Path],
                   description='URL to dataset archive file')
        config.add(name='download_directory',
                   value='imdb',
                   type_hint=str,
                   description='Folder the archive file is downloaded',
                   is_required=True)
        config.add(name='download_filename',
                   value='imdb.tar.gz',
                   type_hint=str,
                   description='Name of the archive file',
                   is_required=True)
        config.add(name='samples_amount',
                   value=500,
                   type_hint=int,
                   description='Number of samples per split to consider at maximum')

        return config


@register
def register_data_loaders():
    Registry.add_and_bind(config_class=ExampleLoaderConfig,
                          component_class=ExampleLoader,
                          name='data_loader',
                          tags={'imdb'},
                          is_default=True,
                          namespace='examples')
