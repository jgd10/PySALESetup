import numpy as np
from PySALESetup import PySALEMesh, \
    ExtensionZone, ExtensionZoneFactor, Region


class TestExtensionZone:
    def test_no_factor(self):
        factor = ExtensionZoneFactor(1, 1)
        zone = ExtensionZone(100, Region.NORTH, 1, factor)
        assert zone.length == 100


class TestPySALEMeshHighResolutionZoneOnly:
    def test_creation(self):
        mesh = PySALEMesh(100, 100, 1)
        assert all(mesh.x_range == np.arange(100)+.5)
        assert all(mesh.y_range == np.arange(100)+.5)

    def test_geometric_centre(self):
        mesh = PySALEMesh(100, 100, 1)
        assert mesh.get_geometric_centre() == (50., 50.)

    def test_apply_geometry(self, simple_object):
        simple_object.set_material(3)
        mesh = PySALEMesh(100, 100, 1)
        mesh.apply_geometry(simple_object)
        assert np.sum(mesh.material_meshes[3]) > 0

    def test_apply_geometries(self, simple_object):
        simple_object.set_material(3)
        ellipse = simple_object.spawn_ellipse_in_shape([5, 5], 5, 5, 0)
        ellipse.set_material(1)
        mesh = PySALEMesh(100, 100, 1)
        mesh.apply_geometry(simple_object)
        assert np.sum(mesh.material_meshes[3]) > 0
        assert np.sum(mesh.material_meshes[1]) > 0

    def test_apply_void_geometries(self, simple_object):
        simple_object.set_material(3)
        ellipse = simple_object.spawn_ellipse_in_shape([5, 5], 3, 3, 0)
        mesh = PySALEMesh(100, 100, 1)
        mesh.apply_geometry(simple_object)
        mesh.apply_polygon_as_void(ellipse)
        assert np.sum(mesh.material_meshes[3]) > 0
        assert np.sum(mesh.material_meshes[1]) == 0

