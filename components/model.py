from pathlib import Path
from typing import AnyStr, Any, Dict, Optional, Iterable, Union

from cinnamon_core.core.data import FieldDict
from cinnamon_core.utility.pickle_utility import save_pickle, load_pickle
from cinnamon_generic.components.callback import CallbackPipeline
from cinnamon_generic.components.metrics import MetricPipeline
from cinnamon_generic.components.model import Model
from sklearn.svm import SVC


class ExampleModel(Model):

    def save_model(
            self,
            filepath: Union[AnyStr, Path],
    ):
        filepath = Path(filepath) if type(filepath) != Path else filepath
        save_pickle(filepath=filepath.joinpath('model.pkl'),
                    data=self.model)

    def load_model(
            self,
            filepath: Union[AnyStr, Path],
    ) -> Any:
        filepath = Path(filepath) if type(filepath) != Path else filepath
        return load_pickle(filepath=filepath.joinpath('model.pkl'))

    def build_model(
            self,
            processor_state: FieldDict,
            callbacks: Optional[CallbackPipeline] = None
    ):
        self.model = SVC(C=self.C,
                         kernel=self.kernel,
                         class_weight=self.class_weight)

    def predict(
            self,
            data: FieldDict,
            callbacks: Optional[CallbackPipeline] = None,
            metrics: Optional[MetricPipeline] = None
    ) -> FieldDict:
        model_data = self.get_model_data(data=data, with_labels=True)
        predictions = self.model.predict(X=model_data['X'])

        return_field = FieldDict()
        return_field.add_short(name='predictions',
                               value=predictions)

        if 'y' in model_data and metrics is not None:
            metrics_result = metrics.run(y_pred=predictions,
                                         y_true=model_data['y'])
            return_field.add_short(name='metrics',
                                   value=metrics_result)

        return return_field

    def get_model_data(
            self,
            data: FieldDict,
            with_labels: bool = False
    ) -> Dict[str, Iterable]:
        return_data = dict()

        text_data = list(data.search_by_tag(tags='text').values())[0]
        return_data['X'] = text_data

        if with_labels:
            label_data = data.search_by_tag(tags='label')
            if len(label_data):
                label_data = list(label_data.values())[0]
                return_data['y'] = label_data

        return return_data

    def fit(
            self,
            train_data: FieldDict,
            val_data: Optional[FieldDict] = None,
            metrics: Optional[MetricPipeline] = None,
            callbacks: Optional[CallbackPipeline] = None
    ) -> FieldDict:
        self.model.fit(**self.get_model_data(data=train_data, with_labels=True))

        return_field = FieldDict()

        if val_data is not None:
            val_info = self.evaluate_and_predict(data=val_data,
                                                 callbacks=callbacks,
                                                 metrics=metrics)
            return_field.add_short(name='val_info',
                                   value=val_info)
        return return_field
