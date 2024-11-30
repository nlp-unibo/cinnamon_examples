from pathlib import Path

from cinnamon_core.registry import Registry
from components.pipeline import SVCPipeline

if __name__ == '__main__':
    """
    In this demo script, we retrieve and build our IMDB data loader.
    Once built, we run the data loader to load the IMDB dataset and print it for visualization purposes.
    """
    directory = Path(__file__).parent.parent.resolve()
    Registry.setup(directory=directory)

    pipeline = SVCPipeline.build_component(name='pipeline',
                                           tags={'svc'},
                                           namespace='examples')
    pipeline.run()