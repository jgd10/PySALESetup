from PySALESetup import PySALEObject
from PySALESetup.creation import PySALENormalDistribution, \
    PySALEUniformDistribution, PySALEWeibull2Distribution, \
    PySALELogNormalDistribution
from math import pi
import pytest


@pytest.fixture()
def simple_object():
    object_ = PySALEObject([(0, 0), (0, 10), (10, 10), (10, 0)])
    return object_


@pytest.fixture()
def square_object():
    object_ = PySALEObject([(0, 0), (0, 10), (10, 10), (10, 0)])
    return object_


@pytest.fixture()
def object_with_uniform_distributions():
    object_ = PySALEObject.generate_ellipse([0, 0], 2, 1, 0)
    radii = PySALEUniformDistribution((.5, 2))
    areas = PySALEUniformDistribution((pi*.5, pi*2))
    angles = PySALEUniformDistribution((0, 360))
    return object_, radii, areas, angles


@pytest.fixture()
def object_with_normal_distributions():
    object_ = PySALEObject.generate_ellipse([0, 0], 2, 1, 0)
    radii = PySALENormalDistribution(1, 1)
    areas = PySALENormalDistribution(pi, pi)
    angles = PySALENormalDistribution(0, 180)
    return object_, radii, areas, angles


@pytest.fixture()
def object_with_lognormal_distributions():
    object_ = PySALEObject.generate_ellipse([0, 0], 2, 1, 0)
    radii = PySALELogNormalDistribution(1, 1)
    areas = PySALELogNormalDistribution(pi, pi)
    angles = PySALELogNormalDistribution(0, 180)
    return object_, radii, areas, angles


@pytest.fixture()
def object_with_weibull_distributions():
    object_ = PySALEObject.generate_ellipse([0, 0], 2, 1, 0)
    radii = PySALEWeibull2Distribution(1, 1)
    areas = PySALEWeibull2Distribution(pi, pi)
    angles = PySALEWeibull2Distribution(0, 180)
    return object_, radii, areas, angles


@pytest.fixture()
def uniform():
    return PySALEUniformDistribution((0., 1.))
