from typing import Optional

from cinnamon.configuration import Configuration
from cinnamon.registry import register_method
from components.model import SVCModel


class SVCModelConfig(Configuration):

    @classmethod
    @register_method(name='model',
                     tags={'svc'},
                     namespace='examples',
                     component_class=SVCModel)
    def default(
            cls
    ):
        config = super().default()

        config.add(name='C',
                   value=1.0,
                   type_hint=float,
                   description='C parameter of SVC')
        config.add(name='kernel',
                   type_hint=str,
                   value='linear',
                   description='The kernel of the SVC')
        config.add(name='class_weight',
                   type_hint=Optional[str],
                   value='balanced',
                   description='The weighting technique for addressing class imbalance.'
                               'Each sample in the training set receives a weight based on'
                               ' its class distribution')

        return config
