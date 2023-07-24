from typing import Any

from cinnamon_core.core.configuration import Configuration
from cinnamon_core.core.registry import Registry, register, RegistrationKey
from cinnamon_generic.configurations.pipeline import OrderedPipelineConfig
from components.processor import TfIdfProcessor, LabelProcessor, ProcessorPipeline, MLProcessor


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
    Registry.add_and_bind(config_class=TfIdfProcessorConfig,
                          component_class=TfIdfProcessor,
                          name='processor',
                          tags={'tf-idf'},
                          is_default=True,
                          namespace='examples')

    Registry.add_and_bind(config_class=Configuration,
                          component_class=LabelProcessor,
                          name='processor',
                          tags={'label'},
                          is_default=True,
                          namespace='examples')
    Registry.add_and_bind(config_class=Configuration,
                          component_class=MLProcessor,
                          name='processor',
                          tags={'ml'},
                          is_default=True,
                          namespace='examples')

    Registry.add_and_bind(config_class=OrderedPipelineConfig,
                          config_constructor=OrderedPipelineConfig.from_keys,
                          config_kwargs={
                              'keys': [
                                  RegistrationKey(name='processor', tags={'default', 'tf-idf'}, namespace='examples'),
                                  RegistrationKey(name='processor', tags={'default', 'label'}, namespace='examples'),
                                  RegistrationKey(name='processor', tags={'default', 'ml'}, namespace='examples')
                              ],
                              'names': [
                                  'text_processor',
                                  'label_processor',
                                  'ml_processor'
                              ]
                          },
                          component_class=ProcessorPipeline,
                          name='processor',
                          tags={'tf-idf', 'label', 'ml'},
                          namespace='examples')
