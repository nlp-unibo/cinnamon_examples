from sklearn.metrics import f1_score, accuracy_score

from cinnamon_core.core.registry import Registry, register, RegistrationKey
from cinnamon_generic.components.metrics import LambdaMetric, MetricPipeline
from cinnamon_generic.configurations.metrics import LambdaMetricConfig
from cinnamon_generic.configurations.pipeline import PipelineConfig


@register
def register_metrics_configurations():
    Registry.add_and_bind(config_class=LambdaMetricConfig,
                          component_class=LambdaMetric,
                          config_constructor=LambdaMetricConfig.get_delta_class_copy,
                          config_kwargs={
                              'params': {
                                  'name': 'binary_f1',
                                  'method': f1_score,
                                  'method_args': {'average': 'binary', 'pos_label': 1}
                              }
                          },
                          name='metrics',
                          tags={'binary_f1'},
                          namespace='examples')
    Registry.add_and_bind(config_class=LambdaMetricConfig,
                          component_class=LambdaMetric,
                          config_constructor=LambdaMetricConfig.get_delta_class_copy,
                          config_kwargs={
                              'params': {
                                  'name': 'macro_f1',
                                  'method': f1_score,
                                  'method_args': {'average': 'macro'}
                              }
                          },
                          name='metrics',
                          tags={'macro_f1'},
                          namespace='examples')
    Registry.add_and_bind(config_class=LambdaMetricConfig,
                          component_class=LambdaMetric,
                          config_constructor=LambdaMetricConfig.get_delta_class_copy,
                          config_kwargs={
                              'params': {
                                  'name': 'accuracy',
                                  'method': accuracy_score,
                              }
                          },
                          name='metrics',
                          tags={'accuracy'},
                          namespace='examples')

    Registry.add_and_bind(config_class=PipelineConfig,
                          config_constructor=PipelineConfig.from_keys,
                          config_kwargs={
                              'keys': [
                                  RegistrationKey(name='metrics', tags={'binary_f1'}, namespace='examples'),
                                  RegistrationKey(name='metrics', tags={'macro_f1'}, namespace='examples'),
                                  RegistrationKey(name='metrics', tags={'accuracy'}, namespace='examples')
                              ],
                              'names': [
                                  'binary_f1',
                                  'macro_f1',
                                  'accuracy'
                              ]
                          },
                          component_class=MetricPipeline,
                          name='metrics',
                          tags={'binary_f1', 'macro_f1', 'accuracy'},
                          namespace='examples')
