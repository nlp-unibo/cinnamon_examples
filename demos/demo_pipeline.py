from pathlib import Path

from cinnamon.registry import Registry
from components.pipeline import SVCPipeline

if __name__ == '__main__':
    """
    In this demo script, we retrieve and build our SVC pipeline.
    The pipeline covers data loading, data processing, and model evaluation.
    """
    directory = Path(__file__).parent.parent.resolve()
    Registry.setup(directory=directory)

    pipeline = SVCPipeline.build_component(name='pipeline',
                                           tags={'svc'},
                                           namespace='examples')
    pipeline.run()