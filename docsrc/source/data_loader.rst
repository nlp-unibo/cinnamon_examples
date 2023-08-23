.. _data_loader:

Loading data with ``DataLoader``
*************************************

We consider the **IMDB** dataset for this example.

We first define our custom ``ExampleLoader`` by extending the ``DataLoader`` of ``cinnamon-generic``.

Then, we define its associated ``ExampleLoaderConfig`` and perform registrations.

Lastly, we define the runnable script to run our ``ExampleLoader`` and check loaded data.

------------------
``ExampleLoader``
------------------

.. code-block:: python

    import tarfile
    from typing import Tuple, Optional, Iterable
    from urllib import request
    import pandas as pd
    from cinnamon_core.core.data import FieldDict
    from cinnamon_core.utility import logging_utility
    from cinnamon_generic.components.data_loader import DataLoader
    from cinnamon_generic.components.file_manager import FileManager


    class ExampleLoader(DataLoader):

        def __init__(
                self,
                **kwargs
        ):
            super().__init__(**kwargs)

            # Update directory paths
            file_manager = FileManager.retrieve_built_component_from_key(self.file_manager_key)

            self.download_directory = file_manager.dataset_directory.joinpath(self.download_directory)
            self.download_path = self.download_directory.joinpath(self.download_filename)
            self.extraction_path = self.download_path.parents[0]
            self.dataframe_path = self.extraction_path.joinpath('dataset.csv')

        def download(
                self
        ):
            if not self.download_directory.exists():
                self.download_directory.mkdir(parents=True)

            # Download
            if not self.download_path.exists():
                request.urlretrieve(self.data_url, self.download_path)

            # Extract
            with tarfile.open(self.download_path) as loaded_tar:
                loaded_tar.extractall(self.extraction_path)

            # Clean
            self.download_path.unlink()

        def read_df_from_files(
                self
        ) -> pd.DataFrame:
            dataframe_rows = []
            for split in ['train', 'test']:
                for sentiment in ['pos', 'neg']:
                    folder = self.extraction_path.joinpath('aclImdb', split, sentiment)
                    for filepath in folder.glob('**/*'):
                        if not filepath.is_file():
                            continue

                        filename = filepath.name
                        with filepath.open(mode='r', encoding='utf-8') as text_file:
                            text = text_file.read()
                            score = filename.split("_")[1].split(".")[0]
                            file_id = filename.split("_")[0]

                            # create single dataframe row
                            dataframe_row = {
                                "file_id": file_id,
                                "score": score,
                                "sentiment": sentiment,
                                "split": split,
                                "text": text
                            }
                            dataframe_rows.append(dataframe_row)

            df = pd.DataFrame(dataframe_rows)
            df = df[["file_id",
                     "score",
                     "sentiment",
                     "split",
                     "text"]]

            # Save dataframe for quick retrieval
            df.to_csv(path_or_buf=self.dataframe_path, index=None)

            return df

        def load_data(
                self
        ) -> pd.DataFrame:
            if not self.dataframe_path.is_file():
                logging_utility.logger.info('First time loading dataset...Downloading...')
                self.download()
                df = self.read_df_from_files()
            else:
                if self.dataframe_path.is_file():
                    logging_utility.logger.info('Loaded pre-loaded dataset...')
                    df = pd.read_csv(self.dataframe_path)
                else:
                    logging_utility.logger.info("Couldn't find pre-loaded dataset...Building dataset from files...")
                    df = self.read_df_from_files()
                    df.to_csv(self.dataframe_path, index=False)

            return df

        def get_splits(
                self,
        ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
            df = self.load_data()
            train = df[df.split == 'train'].sample(frac=1).reset_index(drop=True)[:self.samples_amount]
            val = None
            test = df[df.split == 'test'].sample(frac=1).reset_index(drop=True)[:self.samples_amount]

            return train, val, test

        def parse(
                self,
                data: Optional[pd.DataFrame] = None,
        ) -> Optional[FieldDict]:
            if data is None:
                return data

            return_field = FieldDict()
            return_field.add(name='text',
                             value=data['text'].values,
                             type_hint=Iterable[str],
                             tags={'text'},
                             description='Input text to classify')
            return_field.add(name='sentiment',
                             value=data['sentiment'].values,
                             type_hint=Iterable[str],
                             tags={'label'},
                             description='Sentiment associated to text')
            return return_field


The ``ExampleLoader`` does the following:

- ``download``: checks if the dataset has to be downloaded from the web. If yes, the loader downloads it and extracts the archive file.
- ``read_df_from_files``: an internal utility function that reads extracted files to build a ``pandas.DataFrame`` view of the IMDB dataset.
- ``load_data``: the API to invoke to obtain the ``pandas.DataFrame`` of the dataset.
- ``get_splits``: retrieves the train, validation and test data splits, if available.
- ``parse``: parse each data split to provide a ``FieldDict`` view of it.

.. note::
    We provide tags to each input field like ``text`` and ``label`` to quickly search for data independently of their specified name.


-------------------------
``ExampleLoaderConfig``
-------------------------

The ``ExampleLoader`` uses ``ExampleLoaderConfig`` as default configuration template.

.. code-block:: python

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


Next, we register ``ExampleLoaderConfig`` and bind it to ``ExampleLoader``.

.. code-block:: python

    @register
    def register_data_loaders():
        Registry.add_and_bind(config_class=ExampleLoaderConfig,
                              component_class=ExampleLoader,
                              name='data_loader',
                              tags={'imdb'},
                              is_default=True,
                              namespace='examples')


----------------------------
Running ``ExampleLoader``
----------------------------

We can now write a script to test ``ExampleLoader``.

.. code-block:: python

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

We use ``logging_utility.logger`` to print the returned data (a ``FieldDict``).
Check ``ExampleLoader.parse()`` and ``DataLoader.run()`` (from ``cinnamon-generic``) to know more about the structure of the returned ``FieldDict``.

We can further simplify the execution by relying on command runners.

.. code-block:: python

    from pathlib import Path
    from cinnamon_core.utility import logging_utility
    from cinnamon_generic.api.commands import setup_registry, run_component
    from cinnamon_core.core.registry import Registry

    if __name__ == '__main__':
        """
        This demo script is the cinnamon command-compliant version of ``demo_data_loader.py``.
        We can use the ``run_component`` command to run a generic component (our data loader in this case).
        We make use of command configurations (see ``configurations/commands.py``) to quickly load our command configuration.
        """

        setup_registry(directory=Path(__file__).parent.parent.resolve(),
                       registrations_to_file=True)

        runner = Registry.build_configuration(name='command',
                                                  tags={'imdb', 'data_loader'},
                                                  namespace='examples')
        cmd_config = runner.run()
        result, _ = run_component(name=cmd_config.name,
                                  tags=cmd_config.tags,
                                  namespace=cmd_config.namespace,
                                  run_name=cmd_config.run_name,
                                  serialize=False)
        logging_utility.logger.info(result)

----------------
Next!
----------------

That's it! We have defined our data loader component to load and parse the IMDB dataset.

Next, we define data ``Processor`` to further parse our input data for our classifier.