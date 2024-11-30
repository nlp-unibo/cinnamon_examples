from typing import Any

from cinnamon_core.configuration import Configuration
from cinnamon_core.registry import Registry, register
from components.processor import TfIdfProcessor, LabelProcessor


class TfIdfProcessorConfig(Configuration):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        config.add(name='ngram_range',
                   value=(1, 1),
                   type_hint=Any,
                   description='Vectorizer ngram_range hyper-parameter')

        return config


@register
def register_processors():
    Registry.register_configuration(config_class=TfIdfProcessorConfig,
                                    component_class=TfIdfProcessor,
                                    name='processor',
                                    tags={'tf-idf'},
                                    namespace='examples')
    Registry.register_configuration(config_class=Configuration,
                                    component_class=LabelProcessor,
                                    name='processor',
                                    tags={'label'},
                                    namespace='examples')
