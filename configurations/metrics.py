from sklearn.metrics import f1_score, accuracy_score

from cinnamon_core.core.configuration import add_variant, supports_variants
from cinnamon_core.core.registry import Registry, register
from cinnamon_generic.components.metrics import LambdaMetric, MetricPipeline
from cinnamon_generic.configurations.metrics import LambdaMetricConfig
from cinnamon_generic.configurations.pipeline import PipelineConfig


@supports_variants
class ExampleLambdaMetricConfig(LambdaMetricConfig):

    @classmethod
    @add_variant('binary_F1')
    def get_sklearn_binary_f1(
            cls
    ):
        config = cls.get_default()
        config.name = 'binary_f1'
        config.method = f1_score
        config.method_args = {'average': 'binary', 'pos_label': 1}
        return config

    @classmethod
    @add_variant('macro_F1')
    def get_sklearn_macro_f1(
            cls
    ):
        config = cls.get_default()
        config.name = 'macro_f1'
        config.method = f1_score
        config.method_args = {'average': 'macro'}
        return config

    @classmethod
    @add_variant('accuracy')
    def get_sklearn_accuracy(
            cls
    ):
        config = cls.get_default()
        config.name = 'accuracy'
        config.method = accuracy_score
        return config


@register
def register_metrics_configurations():
    metric_keys = Registry.register_and_bind_variants(config_class=ExampleLambdaMetricConfig,
                                                 component_class=LambdaMetric,
                                                 name='metrics',
                                                 namespace='examples')

    Registry.add_and_bind(config_class=PipelineConfig,
                          config_constructor=PipelineConfig.from_keys,
                          config_kwargs={
                              'keys': metric_keys
                          },
                          component_class=MetricPipeline,
                          name='metrics',
                          tags={'binary_f1', 'macro_f1', 'accuracy'},
                          namespace='examples')
