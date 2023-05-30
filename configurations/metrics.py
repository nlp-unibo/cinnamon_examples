from cinnamon_core.core.configuration import add_variant, supports_variants
from cinnamon_core.core.registry import Registry, register
from cinnamon_generic.components.metrics import MetricPipeline, LambdaMetric
from cinnamon_generic.configurations.metrics import MetricPipelineConfig, LambdaMetricConfig
from sklearn.metrics import f1_score, accuracy_score


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
    variant_regr_keys = Registry.register_and_bind_configuration_variants(configuration_class=ExampleLambdaMetricConfig,
                                                                          component_class=LambdaMetric,
                                                                          name='metrics',
                                                                          namespace='examples')

    Registry.register_and_bind(configuration_class=MetricPipelineConfig,
                               configuration_constructor=MetricPipelineConfig.get_delta_class_copy,
                               configuration_kwargs={
                                   'params_dict': {
                                       'metrics': variant_regr_keys
                                   }
                               },
                               component_class=MetricPipeline,
                               name='metrics',
                               tags={'binary_f1', 'macro_f1', 'accuracy'},
                               namespace='examples')
