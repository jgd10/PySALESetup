import pathlib
import random
from .objects import PySALEObject
from .constants import GRAIN_LIBRARY_PATH


class GrainLibrary:
    """Loads and deploys grains generated from file."""
    def __init__(self):
        self.grains = []

    def load_from_directory(self, directory: pathlib.Path) -> None:
        """Given a directory, loads all grain files it can find.

        Parameters
        ----------
        directory : pathlib.Path

        """
        for file in directory.iterdir():
            if file.is_file():
                try:
                    grain = PySALEObject.create_from_file(file)
                    self.grains.append(grain)
                except Exception as e:
                    print(e)

    def load_builtin_library(self) -> None:
        """Load all grains in the builtin library"""
        self.load_from_directory(GRAIN_LIBRARY_PATH)

    def random_grain(self):
        """Return a random grain from the list.

        Returns
        -------
        grain : PySALEObject
        """
        return random.choice(self.grains)
