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

        self.download_path = file_manager.run(filepath=self.download_directory).joinpath(self.download_filename)
        self.extraction_path = self.download_path.parents[0]
        self.dataframe_path = self.extraction_path.joinpath('dataset.csv')

    def download(
            self
    ):
        request.urlretrieve(self.data_url, self.download_path)

        logging_utility.logger.info('Download complete...Extracting files...')
        with tarfile.open(self.download_path) as loaded_tar:
            loaded_tar.extractall(self.extraction_path)
        logging_utility.logger.info('Extraction complete...')

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
        if not self.download_path.is_file():
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
        return_field.add_short(name='text',
                               value=data['text'].values,
                               type_hint=Iterable[str],
                               tags={'text'},
                               description='Input text to classify')
        return_field.add_short(name='sentiment',
                               value=data['sentiment'].values,
                               type_hint=Iterable[str],
                               tags={'label'},
                               description='Sentiment associated to text')
        return return_field
