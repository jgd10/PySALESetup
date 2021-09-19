import copy
from PySALESetup import PySALEObject, PySALEDomain, translate_polygon,\
    Velocity, rotate_polygon
from shapely.geometry import Point
import pytest


class TestObjectCreation:
    def test_object_creation(self, simple_object):
        assert simple_object.is_valid


class TestSpawningChildren:
    @staticmethod
    def spawn_new(domain: PySALEObject, coords):
        return domain.spawn_polygon_in_shape(coords)

    @staticmethod
    def spawn_new_ellipse(domain: PySALEObject, xy, major, minor, rotation):
        return domain.spawn_ellipse_in_shape(xy, major, minor, rotation)

    def test_spawn_outside(self, simple_object):
        with pytest.raises(AssertionError):
            self.spawn_new(simple_object, [(-10, -10), (-5, -10), (-10, -5)])

    def test_spawn_across_boundary(self, simple_object):
        with pytest.raises(AssertionError):
            self.spawn_new(simple_object, [(-5, 0), (5, 0), (0, 5)])

    def test_spawn_outside_but_overlaps(self, simple_object):
        with pytest.raises(AssertionError):
            self.spawn_new(simple_object, [(0, 0), (10, 0), (-5, 5)])

    def test_spawn_inside_but_overlaps_is_fine(self, simple_object):
        self.spawn_new(simple_object, [(0, 0), (10, 0), (5, 5)])

    def test_spawn_polygon_inside_another(self, simple_object):
        self.spawn_new(simple_object, [(0, 0), (10, 0), (5, 5)])
        self.spawn_new(simple_object, [(1, 1), (9, 1), (4, 4)])

    def test_spawn_overlapping_polygons(self, simple_object):
        n1 = self.spawn_new(simple_object, [(0, 0), (5, 0), (5, 5)])
        n2 = self.spawn_new(simple_object, [(10, 0), (10, 10), (3, 5)])
        assert not n1.contains(n2)

    def test_spawn_ellipse(self, simple_object):
        e = simple_object.spawn_ellipse_in_shape([5, 5], 2., 1., 0.)

    def test_rotate_ellipse(self, simple_object):
        e1 = simple_object.spawn_ellipse_in_shape([5, 5], 2., 1., 0.)
        e2 = simple_object.spawn_ellipse_in_shape([5, 5], 2., 1., 10.)
        assert e1 != e2


class TestTranslatePolygon:
    def test_move_nowhere(self, simple_object):
        new_object = translate_polygon(simple_object, simple_object.centroid.x, simple_object.centroid.y)
        assert new_object.centroid.x == simple_object.centroid.x
        assert new_object.centroid.y == simple_object.centroid.y

    def test_move_positive(self, simple_object):
        new_object = translate_polygon(simple_object, simple_object.centroid.x + 1., simple_object.centroid.y + 1.)
        assert new_object.centroid.x == simple_object.centroid.x + 1.
        assert new_object.centroid.y == simple_object.centroid.y + 1.

    def test_move_negative(self, simple_object):
        new_object = translate_polygon(simple_object,
                                       simple_object.centroid.x - 1.,
                                       simple_object.centroid.y - 1.)
        assert new_object.centroid.x == simple_object.centroid.x - 1.
        assert new_object.centroid.y == simple_object.centroid.y - 1.


class TestRotatePolygon:
    @staticmethod
    def assert_coords_equal(new_object, simple_object):
        all_coords = zip(new_object.exterior.coords,
                         simple_object.exterior.coords)
        for new_coords, simple_coords in all_coords:
            assert new_coords == simple_coords

    @staticmethod
    def assert_coords_equivalent(new_object, simple_object):
        new_coords = set(new_object.exterior.coords)
        old_coords = set(simple_object.exterior.coords)
        assert new_coords == old_coords

    def test_rotate_nowhere_about_center(self, square_object):
        new_object = rotate_polygon(square_object, 0., 'center')
        self.assert_coords_equal(new_object, square_object)

    def test_rotate_360_about_center(self, square_object):
        new_object = rotate_polygon(square_object, 360., 'center')
        self.assert_coords_equal(new_object, square_object)

    def test_rotate_nowhere_about_another_origin(self, square_object):
        new_object = rotate_polygon(square_object, 0.,
                                    Point([-99, -99]))
        self.assert_coords_equal(new_object, square_object)

    def test_rotate_360_about_another_origin(self, square_object):
        new_object = rotate_polygon(square_object, 360.,
                                    Point([-99, -99]))
        self.assert_coords_equal(new_object, square_object)

    @pytest.mark.parametrize('angle', [90, 180, 270])
    def test_rotate_90_degree_multiples(self, square_object, angle):
        new_object = rotate_polygon(square_object, angle,
                                    'center')
        self.assert_coords_equivalent(new_object, square_object)

    def test_rotate_about_origin(self, square_object):
        new_object = rotate_polygon(square_object, 180, Point([0., 0.]))
        mock_object = PySALEObject(([0, 0],
                                    [-10, 0],
                                    [-10, -10],
                                    [0, -10]))
        self.assert_coords_equivalent(new_object, mock_object)


class TestCopyPolygon:
    def test_traits_transferred(self, simple_object):
        simple_object.set_material(2)
        new_object = copy.copy(simple_object)
        assert new_object.material != simple_object.material
        simple_object.copy_properties_to_new_polygon(new_object)
        assert new_object.material == simple_object.material

    def test_traits_transferred_even_for_diff_geometries(self,
                                                         simple_object):
        new_object = PySALEObject(([0, 0], [0, 1], [.2, .2]))
        new_object.set_material(8)
        simple_object.set_material(7)
        simple_object.copy_properties_to_new_polygon(new_object)
        assert new_object.material == simple_object.material
        assert len(simple_object.exterior.coords) \
               != len(new_object.exterior.coords)

    def test_all_traits_transferred(self, simple_object):
        simple_object.spawn_polygon_in_shape(([0, 0], [0, 1], [.5, .5]))
        simple_object.set_material_colormap('magma')
        simple_object.set_material(5)
        simple_object.set_velocity(10., 10.)
        new_object = PySALEObject(([0, 0], [0.1, 0.1], [0.2, 0.7]))
        simple_object.copy_properties_to_new_polygon(new_object)

        properties = [(simple_object.material_colors,
                       new_object.material_colors),
                      (simple_object.material, new_object.material),
                      (simple_object.velocity, new_object.velocity),
                      (simple_object.children, new_object.children)]

        for simple_prop, new_prop in properties:
            assert simple_prop == new_prop


class TestPySALEObjectSetters:
    def test_set_velocity(self, simple_object):
        simple_object.set_velocity(-9, 100.)
        assert type(simple_object.velocity) == Velocity
        assert simple_object.velocity.x == -9.
        assert simple_object.velocity.y == 100.

    def test_set_as_void(self, simple_object):
        simple_object.set_material(5)
        simple_object.set_velocity(-9, 100.)
        simple_object.set_as_void()
        assert simple_object.material == 0
        assert simple_object.velocity == Velocity(0., 0.)

    def test_set_material_solid(self, simple_object):
        simple_object.set_material(5)
        assert simple_object.material == 5

    def test_set_material_void(self, simple_object):
        simple_object.set_velocity(10, 10)
        simple_object.set_material(0)
        assert simple_object.material == 0
        assert simple_object.is_void
        assert simple_object.velocity == Velocity(0, 0)

    def test_set_material_colormap(self, simple_object):
        simple_object.set_material_colormap('magma', void='green',
                                            unassigned='magenta')
        assert len(simple_object.material_colors) == 11
        assert simple_object.material_colors[0] == 'green'
        assert simple_object.material_colors[-1] == 'magenta'





