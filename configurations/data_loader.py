from pathlib import Path
from typing import AnyStr, Union

from cinnamon.configuration import Configuration
from cinnamon.registry import register_method
from components.data_loader import IMDBLoader


class IMDBLoaderConfig(Configuration):

    @classmethod
    @register_method(name='data_loader',
                     tags={'imdb'},
                     namespace='examples',
                     component_class=IMDBLoader)
    def default(
            cls
    ):
        config = super().default()

        config.add(name='download_directory',
                   value=Path(__file__).resolve().parent.parent.joinpath('datasets'),
                   type_hint=Path,
                   description='Folder the archive file is downloaded')
        config.add(name='download_filename',
                   value='imdb.tar.gz',
                   type_hint=str,
                   description='Name of the archive file')
        config.add(name='dataset_name',
                   value='dataset.csv',
                   type_hint=str,
                   description='.csv filename')
        config.add(name='download_url',
                   value='http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
                   type_hint=Union[AnyStr, Path],
                   description='URL to dataset archive file')
        config.add(name='samples_amount',
                   value=500,
                   type_hint=int,
                   description='Number of samples per split to consider at maximum')

        return config
