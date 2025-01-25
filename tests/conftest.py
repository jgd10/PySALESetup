from PySALESetup import PySALEObject, PySALEMesh, \
    ExtensionZoneFactor, ExtensionZone, Region
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
def circle():
    return PySALEObject.generate_ellipse([5., 5.], 1., 1., 0.)


@pytest.fixture()
def rectangle():
    return PySALEObject([(0, 0), (2, 0), (2, 1), (0, 1)])


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
def square_even_mesh() -> PySALEMesh:
    mesh = PySALEMesh.from_dimensions((10, 10), 0.5)
    return mesh


@pytest.fixture()
def populated_square_even_mesh(square_even_mesh):
    grain1 = PySALEObject.generate_ellipse([5., 3.], 2., 1.5, 90., 2)
    grain2 = PySALEObject.generate_ellipse([5., 7.], 2., 1.5, 90., 3)
    grain2.set_velocity(0., y=-500.)
    square_even_mesh.project_polygons_onto_mesh([grain1, grain2])
    return square_even_mesh


@pytest.fixture()
def simple_impactor_target(square_even_mesh):
    x = square_even_mesh.x_physical
    y = square_even_mesh.y_physical
    target = PySALEObject([(0, 0), (x, 0), (x, y/2), (0, y/2)])
    impactor = PySALEObject([(0, y/2), (x, y/2), (x, y), (0, y)])
    impactor.set_material(1)
    target.set_material(2)
    return target, impactor


@pytest.fixture()
def rectangular_even_mesh() -> PySALEMesh:
    mesh = PySALEMesh.from_dimensions((10, 20), 0.5)
    return mesh


@pytest.fixture()
def square_mesh_with_extension_zones() -> PySALEMesh:
    n = ExtensionZone(10, Region.NORTH, .5, ExtensionZoneFactor(1.01, 3))
    w = ExtensionZone(10, Region.WEST, .5, ExtensionZoneFactor(1.01, 3))
    e = ExtensionZone(10, Region.EAST, .5, ExtensionZoneFactor(1.01, 3))
    s = ExtensionZone(10, Region.SOUTH, .5, ExtensionZoneFactor(1.01, 3))
    mesh = PySALEMesh.from_dimensions((10, 10), 0.5, extensions=[n, e, w, s])
    return mesh
