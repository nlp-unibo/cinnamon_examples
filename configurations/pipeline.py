from cinnamon_core.configuration import Configuration
from cinnamon_core.registry import Registry, register, RegistrationKey
from components.pipeline import SVCPipeline


class SVCPipelineConfig(Configuration):

    @classmethod
    def default(
            cls
    ):
        config = super().default()

        config.add(name='data_loader',
                   value=RegistrationKey(name='data_loader',
                                         tags={'imdb'},
                                         namespace='examples'))

        config.add(name='text_processor',
                   value=RegistrationKey(name='processor',
                                         tags={'tf-idf'},
                                         namespace='examples'))
        config.add(name='label_processor',
                   value=RegistrationKey(name='processor',
                                         tags={'label'},
                                         namespace='examples'))

        config.add(name='model',
                   value=RegistrationKey(name='model',
                                         tags={'svc'},
                                         namespace='examples'))

        return config


@register
def register_routines():
    Registry.register_configuration(config_class=SVCPipelineConfig,
                                    component_class=SVCPipeline,
                                    name='pipeline',
                                    tags={'svc'},
                                    namespace='examples')
