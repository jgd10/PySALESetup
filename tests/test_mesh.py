import matplotlib.pyplot as plt
import numpy as np
from PySALESetup import PySALEMesh, \
    ExtensionZone, ExtensionZoneFactor, Region, Cell, Velocity
from shapely.geometry import Point
from typing import Tuple
import math
import pytest


class TestRegion:
    def test_regions(self):
        counter = 0
        for r in Region:
            counter += 1
        assert counter == 4


class TestCell:
    def test_cell_is_mutable(self):
        c = Cell(Point(0, 0), 0, 0, 1, Velocity(0., 0.))
        c.material = 2


class TestExtensionZoneFactor:
    def test_is_immutable(self):
        factor = ExtensionZoneFactor(1, 1)
        with pytest.raises(AttributeError):
            factor.max_cell_size = 20.


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
        mesh.project_polygons_onto_mesh([simple_object])
        assert np.sum(mesh.material_meshes[3]) > 0

    def test_apply_geometries(self, simple_object):
        simple_object.set_material(3)
        ellipse = simple_object.spawn_ellipse_in_shape([5, 5], 5, 5, 0)
        ellipse.set_material(1)
        mesh = PySALEMesh(100, 100, 1)
        mesh.project_polygons_onto_mesh([simple_object])
        assert np.sum(mesh.material_meshes[3]) > 0
        assert np.sum(mesh.material_meshes[1]) > 0

    def test_apply_void_geometries(self, simple_object):
        simple_object.set_material(3)
        ellipse = simple_object.spawn_ellipse_in_shape([5, 5], 3, 3, 0)
        mesh = PySALEMesh(100, 100, 1)
        ellipse.set_as_void()
        mesh.project_polygons_onto_mesh([simple_object])
        assert np.sum(mesh.material_meshes[3]) > 0
        assert np.sum(mesh.material_meshes[1]) == 0

    def test_objresh(self, rectangular_even_mesh):
        assert rectangular_even_mesh.objresh == rectangular_even_mesh.y / 2

    def test_max_cell_size_no_ext_zones(self, square_even_mesh):
        assert square_even_mesh.max_cell_size == square_even_mesh.max_cell_size

    def test_collision_site(self, square_even_mesh):
        # default should be 0
        assert square_even_mesh.collision_site == 0

    @pytest.mark.parametrize('scale', [(1., 20, 0.5),
                                       (2., 40, 0.25),
                                       (0.5, 10, 1.),
                                       (0.9, 18, 0.5/0.9)])
    def test_spawn_copy(self, square_even_mesh, scale):
        factor, target_cells, target_cell_size = scale
        new = square_even_mesh.spawn_copy(factor)
        assert new.x == target_cells
        assert math.isclose(new.cell_size, target_cell_size)

    def test_collision_site_populated(self, populated_square_even_mesh):
        assert populated_square_even_mesh.collision_site > 0


    def test_x_range(self, rectangular_even_mesh):
        assert rectangular_even_mesh.x == rectangular_even_mesh.x_range.size

    def test_y_range(self, rectangular_even_mesh):
        assert rectangular_even_mesh.y == rectangular_even_mesh.y_range.size

    def test_geometric_centre_2(self, square_even_mesh):
        centre = square_even_mesh.get_geometric_centre()
        for val in centre:
            assert np.isclose(val, 5.)

    def test_material_meshes(self, square_even_mesh):
        assert len(square_even_mesh.material_meshes) == 9
        for i in range(9):
            assert i+1 in square_even_mesh.material_meshes

    def test_velocities(self, square_even_mesh):
        assert len(square_even_mesh.velocities) == 2
        for i in ['x', 'y']:
            assert i in square_even_mesh.velocities

    def test_cells(self, square_even_mesh):
        assert len(square_even_mesh.cells) == square_even_mesh.x * square_even_mesh.y

    def test_apply_geometry_2(self, square_even_mesh, simple_impactor_target):
        target, impactor = simple_impactor_target
        square_even_mesh.project_polygons_onto_mesh([target, impactor])
        assert square_even_mesh.material_numbers == [1, 2]

    def test_extension_zones_empty(self, square_even_mesh):
        assert not square_even_mesh.extension_zones

    def test_extension_zone_factor_default(self, square_even_mesh):
        assert square_even_mesh.extension_factor.max_cell_size == square_even_mesh.cell_size
        assert square_even_mesh.extension_factor.multiplier == 1.

    @pytest.mark.parametrize('x0', [10., -10.])
    def test_set_origin_x(self, x0):
        mesh = PySALEMesh.from_dimensions((1., 1.), 0.05, origin=(x0, 0.))
        assert all(mesh.x_range > x0)

    @pytest.mark.parametrize('y0', [10., -10.])
    def test_set_origin_y(self, y0):
        mesh = PySALEMesh.from_dimensions((1., 1.), 0.05, origin=(0., y0))
        assert all(mesh.y_range > y0)


class TestMeshWithExtensionZones:
    def test_build_mesh_with_zones(self, square_mesh_with_extension_zones):
        assert square_mesh_with_extension_zones.extension_zones

    def test_increased_physical_size(self, square_mesh_with_extension_zones):
        m = square_mesh_with_extension_zones
        assert m.x_physical > m.x * m.cell_size
        assert m.y_physical > m.y * m.cell_size


@pytest.mark.flaky
class TestMeshPlots:
    @pytest.mark.flaky
    def test_plot_materials(self, square_mesh_with_extension_zones):
        f, a = square_mesh_with_extension_zones.plot_materials()
        assert isinstance(f, plt.Figure)
        assert isinstance(a, plt.Axes)

    @pytest.mark.flaky
    def test_plot_velocities(self, square_mesh_with_extension_zones):
        f1, f2, a1, a2 = square_mesh_with_extension_zones.plot_velocities()
        for f, a in [(f1, a1), (f2, a2)]:
            assert isinstance(f, plt.Figure)
            assert isinstance(a, plt.Axes)

    @pytest.mark.flaky
    def test_plot_cells(self, square_mesh_with_extension_zones):
        f, a = square_mesh_with_extension_zones.plot_cells()
        assert isinstance(f, plt.Figure)
        assert isinstance(a, plt.Axes)


class TestSaveMesh:
    def test_simple_save(self, square_even_mesh, tmp_path):
        file_path = tmp_path / 'meso_m.iSALE'
        square_even_mesh.save(file_path, compress=False)
        assert file_path.is_file()

    def test_extensions_save(self,
                             square_mesh_with_extension_zones,
                             tmp_path):
        file_path = tmp_path / 'meso_m.iSALE'
        square_mesh_with_extension_zones.save(file_path, compress=False)
        assert file_path.is_file()

    def test_compression(self, square_even_mesh, tmp_path):
        file_path = tmp_path / 'meso_m.iSALE.gz'
        square_even_mesh.save(file_path, compress=True)
        assert file_path.is_file()

    def test_compression_no_gz_suffix(self, square_even_mesh, tmp_path):
        file_path = tmp_path / 'meso_m.iSALE'
        with pytest.raises(NameError):
            square_even_mesh.save(file_path, compress=True)
            assert file_path.is_file()
