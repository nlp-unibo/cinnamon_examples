import itertools
from typing import Optional, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from cinnamon_core.core.data import FieldDict
from cinnamon_generic.components.pipeline import OrderedPipeline
from cinnamon_generic.components.processor import Processor


class TfIdfProcessor(Processor):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.vectorizer = TfidfVectorizer(ngram_range=self.ngram_range)

    def prepare_save_data(
            self
    ) -> Dict:
        data = super().prepare_save_data()

        data['vectorizer'] = self.vectorizer
        return data

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

    def prepare_save_data(
            self
    ) -> Dict:
        data = super().prepare_save_data()

        data['encoders'] = self.encoders
        return data

    def process(
            self,
            data: FieldDict,
            is_training_data: bool = False
    ):
        label_fields = data.search_by_tag(tags='label')
        for field_name, field in label_fields.items():
            field_encoder = self.encoders.get(field_name, LabelEncoder())
            data[field_name] = field_encoder.fit_transform(field) \
                if is_training_data else field_encoder.transform(field)
            if field_name not in self.encoders and is_training_data:
                self.encoders[field_name] = field_encoder

        return data


class MLProcessor(Processor):

    def process(
            self,
            data: FieldDict,
            is_training_data: bool = False
    ) -> FieldDict:
        return_dict = FieldDict()

        text_data = list(data.search_by_tag(tags='text').values())[0]
        return_dict.add(name='X',
                        value=text_data)

        label_data = data.search_by_tag(tags='label')
        if len(label_data):
            label_data = list(label_data.values())[0]
            return_dict.add(name='y',
                            value=label_data)

        return return_dict


class ProcessorPipeline(OrderedPipeline, Processor):

    def run(
            self,
            data: Optional[FieldDict] = None,
            is_training_data: bool = False
    ) -> FieldDict:
        components = self.get_pipeline()
        for component in components:
            data = component.run(data=data, is_training_data=is_training_data)
        return data
