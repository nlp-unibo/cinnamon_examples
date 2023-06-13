from typing import Optional

from cinnamon_core.core.configuration import Configuration
from cinnamon_core.core.registry import Registry, register
from components.model import ExampleModel


class ExampleModelConfig(Configuration):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        config.add_short(name='C',
                         value=1.0,
                         type_hint=float,
                         description='C parameter of SVC')
        config.add_short(name='kernel',
                         type_hint=str,
                         value='linear',
                         description='The kernel of the SVC')
        config.add_short(name='class_weight',
                         type_hint=Optional[str],
                         value='balanced',
                         description='The weighting technique for addressing class imbalance.'
                                     'Each sample in the training set receives a weight based on'
                                     ' its class distribution')

        return config


@register
def register_models():
    Registry.add_and_bind(config_class=ExampleModelConfig,
                          component_class=ExampleModel,
                          name='model',
                          tags={'svm'},
                          is_default=True,
                          namespace='examples')
