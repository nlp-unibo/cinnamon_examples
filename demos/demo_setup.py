from pathlib import Path

from cinnamon_generic.api.commands import setup_registry

if __name__ == '__main__':
    directory = Path(__file__).parent.parent.resolve()

    setup_registry(directory=directory,
                   registrations_to_file=True)
