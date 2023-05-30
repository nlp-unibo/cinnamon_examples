from pathlib import Path
from typing import cast

from cinnamon_core.utility import logging_utility
from cinnamon_generic.api.commands import setup_registry
from cinnamon_generic.components.data_loader import DataLoader
from cinnamon_generic.components.metrics import Metric
from cinnamon_generic.components.model import Model
from cinnamon_generic.components.processor import Processor

if __name__ == '__main__':
    directory = Path(__file__).parent.parent.resolve()

    setup_registry(directory=directory,
                   registrations_to_file=True)

    # DataLoader (dl)
    dl = DataLoader.build_component(name='data_loader',
                                    tags={'default', 'imdb'},
                                    namespace='examples')
    data = dl.run()

    # TfIdfProcessor (tip)
    tip = Processor.build_component(name='processor',
                                    tags={'default', 'tf-idf'},
                                    namespace='examples')
    tip.run(data=data.train, is_training_data=True)
    tip.run(data=data.val)
    tip.run(data=data.test)

    # LabelProcessor (lp)
    lp = Processor.build_component(name='processor',
                                   tags={'default', 'label'},
                                   namespace='examples')
    lp = cast(Processor, lp)
    lp.run(data=data.train, is_training_data=True)
    lp.run(data=data.val)
    lp.run(data=data.test)
    logging_utility.logger.info(f'Train: {data.train}')
    logging_utility.logger.info(f'Val: {data.val}')
    logging_utility.logger.info(f'Test: {data.test}')

    # Metrics
    metrics = Metric.build_component(name='metrics',
                                     tags={'accuracy', 'binary_f1', 'macro_f1'},
                                     namespace='examples')

    # Model
    model = Model.build_component(name='model',
                                  tags={'default', 'svm'},
                                  namespace='examples')

    # Training
    model.build_model(processor_state=lp.state)
    fit_info = model.fit(train_data=data.train,
                         val_data=data.val,
                         metrics=metrics,
                         callbacks=None)
    logging_utility.logger.info(f'Fit info: {fit_info}')

    predict_info = model.predict(data=data.test,
                                 metrics=metrics)
    logging_utility.logger.info(f'Predict info: {predict_info}')
