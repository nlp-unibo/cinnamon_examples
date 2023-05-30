from typing import Any

from cinnamon_core.core.configuration import Configuration
from cinnamon_core.core.registry import Registry, register
from cinnamon_generic.components.processor import ProcessorPipeline
from cinnamon_generic.configurations.processor import ProcessorPipelineConfig

from components.processor import TfIdfProcessor, LabelProcessor


class TfIdfProcessorConfig(Configuration):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        config.add_short(name='ngram_range',
                         value=(1, 1),
                         type_hint=Any,
                         description='Vectorizer ngram_range hyper-parameter')

        return config


@register
def register_processors():
    tfidf_regr_key = Registry.register_and_bind(configuration_class=TfIdfProcessorConfig,
                                                component_class=TfIdfProcessor,
                                                name='processor',
                                                tags={'tf-idf'},
                                                is_default=True,
                                                namespace='examples')

    label_regr_key = Registry.register_and_bind(configuration_class=Configuration,
                                                component_class=LabelProcessor,
                                                name='processor',
                                                tags={'label'},
                                                is_default=True,
                                                namespace='examples')

    Registry.register_and_bind(configuration_class=ProcessorPipelineConfig,
                               configuration_constructor=ProcessorPipelineConfig.get_delta_class_copy,
                               configuration_kwargs={
                                   'params_dict': {
                                       'processors': [
                                           tfidf_regr_key,
                                           label_regr_key
                                       ]
                                   }
                               },
                               component_class=ProcessorPipeline,
                               name='processor',
                               tags={'tf-idf', 'label'},
                               namespace='examples')
