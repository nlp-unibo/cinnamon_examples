from pathlib import Path

from cinnamon_generic.api.commands import setup_registry

if __name__ == '__main__':
    """
    In this demo script, we test ``setup_registry`` command to check if all registration actions are performed correctly.
    If ``registrations_to_file=True`` you will see a ``dependencies.html`` file in ``demos/`` folder.
    You can open it in the browser to inspect all registered configurations and their dependencies.
    """

    setup_registry(directory=Path(__file__).parent.parent.resolve(),
                   registrations_to_file=True)
