from importlib.resources import files
from . import input_templates, grain_library_directory

ASTEROID_TEMPLATE_PATH = files(input_templates).joinpath(
    'asteroid_template.inp')
ADDITIONAL_TEMPLATE_PATH = files(input_templates).joinpath(
    'additional_template.inp')
GRAIN_LIBRARY_PATH = files(grain_library_directory)