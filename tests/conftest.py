from PySALESetup.objects import PySALEObject
import pytest


@pytest.fixture()
def simple_object():
    object_ = PySALEObject([(0, 0), (0, 10), (10, 10), (10, 0)])
    return object_


@pytest.fixture()
def square_object():
    object_ = PySALEObject([(0, 0), (0, 10), (10, 10), (10, 0)])
    return object_
