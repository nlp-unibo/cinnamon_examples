from cinnamon.configuration import Configuration
from cinnamon.registry import register_method, RegistrationKey
from components.benchmark import SVCBenchmark


class SVCPipelineConfig(Configuration):

    @classmethod
    @register_method(name='benchmark',
                     tags={'svc'},
                     namespace='examples',
                     component_class=SVCBenchmark)
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
