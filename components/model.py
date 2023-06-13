from pathlib import Path
from typing import AnyStr, Any, Optional, Union

from sklearn.svm import SVC

from cinnamon_core.core.data import FieldDict
from cinnamon_core.utility.pickle_utility import save_pickle, load_pickle
from cinnamon_generic.components.callback import Callback
from cinnamon_generic.components.metrics import Metric
from cinnamon_generic.components.model import Model
from cinnamon_generic.components.processor import Processor


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

    def build(
            self,
            processor: Processor,
            callbacks: Optional[Callback] = None
    ):
        self.model = SVC(C=self.C,
                         kernel=self.kernel,
                         class_weight=self.class_weight)

    def predict(
            self,
            data: FieldDict,
            callbacks: Optional[Callback] = None,
            metrics: Optional[Metric] = None,
            model_processor: Optional[Processor] = None
    ) -> FieldDict:
        predictions = self.model.predict(X=data.X)

        return_field = FieldDict()
        return_field.add_short(name='predictions',
                               value=predictions)

        if 'y' in data and metrics is not None:
            metrics_result = metrics.run(y_pred=predictions,
                                         y_true=data.y,
                                         as_dict=True)
            return_field.add_short(name='metrics',
                                   value=metrics_result)

        return return_field

    def fit(
            self,
            train_data: FieldDict,
            val_data: Optional[FieldDict] = None,
            metrics: Optional[Metric] = None,
            callbacks: Optional[Callback] = None,
            model_processor: Optional[Processor] = None
    ) -> FieldDict:
        self.model.fit(X=train_data.X, y=train_data.y)

        return_field = FieldDict()

        if val_data is not None:
            val_info = self.evaluate(data=val_data,
                                     callbacks=callbacks,
                                     metrics=metrics)
            return_field.add_short(name='val_info',
                                   value=val_info)
        return return_field

    def evaluate(
            self,
            data: FieldDict,
            callbacks: Optional[Callback] = None,
            metrics: Optional[Metric] = None,
            model_processor: Optional[Processor] = None
    ) -> FieldDict:
        return self.predict(data=data,
                            callbacks=callbacks,
                            metrics=metrics)
