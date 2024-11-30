from typing import Optional

from cinnamon_core.configuration import Configuration
from cinnamon_core.registry import Registry, register
from components.model import SVCModel


class SVCModelConfig(Configuration):

    @classmethod
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


@register
def register_models():
    Registry.register_configuration(config_class=SVCModelConfig,
                                    component_class=SVCModel,
                                    name='model',
                                    tags={'svc'},
                                    namespace='examples')
