from typing import Any, Type

from cinnamon_core.core.configuration import Configuration, C
from cinnamon_core.core.registry import Registry, register
from cinnamon_generic.components.processor import Processor
from cinnamon_generic.configurations.pipeline import OrderedPipelineConfig
from components.processor import TfIdfProcessor, LabelProcessor, ProcessorPipeline, MLProcessor


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


class ProcessorPipelineConfig(OrderedPipelineConfig):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.add_pipeline_component(name='text_processor',
                                      is_required=True,
                                      build_type_hint=Processor,
                                      description='Processor for text processing (e.g., tree parsing)')
        config.add_pipeline_component(name='label_processor',
                                      is_required=True,
                                      build_type_hint=Processor,
                                      description='Processor for label processing')
        config.add_pipeline_component(name='model_processor',
                                      is_required=True,
                                      build_type_hint=Processor,
                                      description='Processor that format data to be digested by a ML model')

        return config


@register
def register_processors():
    tfidf_key = Registry.add_and_bind(config_class=TfIdfProcessorConfig,
                                      component_class=TfIdfProcessor,
                                      name='processor',
                                      tags={'tf-idf'},
                                      is_default=True,
                                      namespace='examples')

    label_key = Registry.add_and_bind(config_class=Configuration,
                                      component_class=LabelProcessor,
                                      name='processor',
                                      tags={'label'},
                                      is_default=True,
                                      namespace='examples')
    ml_key = Registry.add_and_bind(config_class=Configuration,
                                   component_class=MLProcessor,
                                   name='processor',
                                   tags={'ml'},
                                   is_default=True,
                                   namespace='examples')

    Registry.add_and_bind(config_class=ProcessorPipelineConfig,
                          config_constructor=ProcessorPipelineConfig.get_delta_class_copy,
                          config_kwargs={
                              'params': {
                                  'text_processor': tfidf_key,
                                  'label_processor': label_key,
                                  'model_processor': ml_key
                              }
                          },
                          component_class=ProcessorPipeline,
                          name='processor',
                          tags={'tf-idf', 'label', 'ml'},
                          namespace='examples')
