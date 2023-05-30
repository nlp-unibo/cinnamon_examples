import itertools

from cinnamon_core.core.data import FieldDict
from cinnamon_generic.components.processor import Processor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


class TfIdfProcessor(Processor):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.vectorizer = TfidfVectorizer(ngram_range=self.ngram_range)

    def process(
            self,
            data: FieldDict,
            is_training_data: bool = False,
    ):
        if is_training_data:
            text_data = data.search_by_tag(tags='text')
            text_data = list(itertools.chain.from_iterable([field for field in text_data.values()]))
            self.vectorizer.fit(text_data)

        text_fields = data.search_by_tag(tags='text')
        for index, field in text_fields.items():
            data[index] = self.vectorizer.transform(field)
        return data


class LabelProcessor(Processor):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.encoders = dict()

    def process(
            self,
            data: FieldDict,
            is_training_data: bool = False
    ):
        if is_training_data and data is not None:
            label_fields = data.search_by_tag(tags='label')
            for field_name, field in label_fields.items():
                field_encoder = LabelEncoder() if field_name not in self.encoders else self.encoders[field_name]
                field_encoder.fit(field)
                self.encoders[field_name] = field_encoder

        label_fields = data.search_by_tag(tags='label')
        for field_name, field in label_fields.items():
            field_encoder = self.encoders[field_name]
            data[field_name] = field_encoder.transform(field)
        return data
