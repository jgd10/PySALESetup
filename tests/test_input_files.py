from PySALESetup.input_files import AsteroidInput, AdditionalInput, \
    InputFile, TimeStep
import pathlib
from string import Template


class TestInputFile:
    def test_path(self):
        a = InputFile('asteroid')
        assert isinstance(a.template_path, pathlib.Path)

    def test_path_is_file(self):
        a = InputFile('asteroid')
        assert a.template_path.is_file()

    def test_template_is_template(self):
        a = InputFile('asteroid')
        assert isinstance(a.template, Template)


class TestAsteroidInput:
    def test_write_to_template(self, tmp_path, square_even_mesh):
        a = AsteroidInput('test', TimeStep(1, 1, 1, 1), square_even_mesh)
        a.write_to(tmp_path / 'test.inp')


class TestAdditionalInput:
    def test_write_to_template(self, tmp_path, square_even_mesh):
        a = AdditionalInput(square_even_mesh, {1: 'matter1'}, 1)
        a.write_to(tmp_path / 'test.inp')
