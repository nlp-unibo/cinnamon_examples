from typing import Any

from cinnamon.configuration import Configuration
from cinnamon.registry import Registry, register_method, register
from components.processor import TfIdfProcessor, LabelProcessor


class TfIdfProcessorConfig(Configuration):

    @classmethod
    @register_method(component_class=TfIdfProcessor,
                     name='processor',
                     tags={'tf-idf'},
                     namespace='examples')
    def get_default(
            cls
    ):
        config = super().default()

        config.add(name='ngram_range',
                   value=(1, 1),
                   type_hint=Any,
                   description='Vectorizer ngram_range hyper-parameter')

        return config


@register
def register_processors():
    Registry.register_configuration(config_class=Configuration,
                                    component_class=LabelProcessor,
                                    name='processor',
                                    tags={'label'},
                                    namespace='examples')
