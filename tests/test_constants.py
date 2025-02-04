from PySALESetup.constants import ASTEROID_TEMPLATE_PATH, \
    ADDITIONAL_TEMPLATE_PATH, \
    GRAIN_LIBRARY_PATH
import pathlib
import pytest


class TestPaths:
    @pytest.mark.parametrize('directory', [ASTEROID_TEMPLATE_PATH,
                                           ADDITIONAL_TEMPLATE_PATH,
                                           GRAIN_LIBRARY_PATH])
    def test_is_path(self, directory):
        assert isinstance(directory, pathlib.Path)

    @pytest.mark.parametrize('directory', [GRAIN_LIBRARY_PATH])
    def test_is_directory(self, directory):
        assert directory.is_dir()

    @pytest.mark.parametrize('file', [ASTEROID_TEMPLATE_PATH,
                                      ADDITIONAL_TEMPLATE_PATH])
    def test_is_directory(self, file):
        assert file.is_file()