import os
import pathlib

PACKAGE_ROOT_DIRECTORY = pathlib.Path(os.path.dirname(__file__))
ASTEROID_TEMPLATE_PATH = PACKAGE_ROOT_DIRECTORY / 'input_templates' \
                         / 'asteroid_template.inp'
ADDITIONAL_TEMPLATE_PATH = PACKAGE_ROOT_DIRECTORY / 'input_templates' \
                         / 'additional_template.inp'
GRAIN_LIBRARY_PATH = PACKAGE_ROOT_DIRECTORY / 'grain_library_directory'