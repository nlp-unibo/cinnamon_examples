from cinnamon_core.core.registry import Registry, RegistrationKey, register
from cinnamon_generic.components.routine import TrainAndTestRoutine
from cinnamon_generic.configurations.routine import TrainAndTestRoutineConfig


class ExampleRoutineConfig(TrainAndTestRoutineConfig):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        config.data_loader = RegistrationKey(name='data_loader',
                                             tags={'imdb', 'default'},
                                             namespace='examples')
        config.pre_processor = RegistrationKey(name='processor',
                                               tags={'tf-idf', 'label'},
                                               namespace='examples')
        config.model = RegistrationKey(name='model',
                                       tags={'svm', 'default'},
                                       namespace='examples')
        config.metrics = RegistrationKey(name='metrics',
                                         tags={'binary_f1', 'macro_f1', 'accuracy'},
                                         namespace='examples')
        config.helper = RegistrationKey(name='helper',
                                        tags={'default'},
                                        namespace='generic')
        config.routine_processor = RegistrationKey(name='routine_processor',
                                                   tags={'average'},
                                                   namespace='generic')

        config.seeds = [15000, 42]
        config.has_val_split = False
        config.has_test_split = True

        return config


@register
def register_routines():
    Registry.register_and_bind(configuration_class=ExampleRoutineConfig,
                               component_class=TrainAndTestRoutine,
                               name='routine',
                               tags={'train_and_test'},
                               namespace='examples')
