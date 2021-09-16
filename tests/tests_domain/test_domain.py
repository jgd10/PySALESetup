from PySALESetup import PySALEObject, PySALEDomain, translate_polygon
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


class TestPySALEDomain:
    def test_move_object_to_random_coords(self, simple_object):
        domain = PySALEDomain(simple_object)
        simple_object = domain._move_object_to_random_coordinate_in_domain(simple_object, 1., 1., 0., 0.)
        assert 0. < simple_object.centroid.x < 1.
        assert 0. < simple_object.centroid.y < 1.

    def test_fill_to_threshold(self, simple_object):
        domain = PySALEDomain(simple_object)
        grain = PySALEObject([(0, 0), (0.5, 0.5), (1, 0)])
        domain.fill_to_threshold_area(grain, 20)

    @pytest.mark.parametrize('threshold_', [10., 30., 60., 90.])
    def test_threshold_check(self, simple_object, threshold_):
        domain = PySALEDomain(simple_object)
        inserted_area, insertion_possible, threshold = domain._check_threshold_input(threshold_)
        assert inserted_area == 0.
        assert insertion_possible
        assert threshold == threshold_

    @pytest.mark.parametrize('threshold_', [30., 60., 90.])
    def test_threshold_check_already_populated(self, simple_object, threshold_):
        domain = PySALEDomain(simple_object)
        simple_object.spawn_polygon_in_shape([(0, 0), (0, 5), (5, 5), (5, 0)])
        inserted_area, insertion_possible, threshold = domain._check_threshold_input(threshold_)
        assert inserted_area == 25.
        assert insertion_possible
        assert threshold == threshold_

    @pytest.mark.parametrize('threshold_', [0., 10.])
    def test_object_already_over_threshold(self, simple_object, threshold_):
        domain = PySALEDomain(simple_object)
        simple_object.spawn_polygon_in_shape([(0, 0), (0, 5), (5, 5), (5, 0)])
        with pytest.raises(AssertionError):
            _, _, _ = domain._check_threshold_input(threshold_)

    @pytest.mark.parametrize('max_attempts', [1, 10, 100, 1000, 10000])
    def test_insert_randomly_maxes_out(self, simple_object, max_attempts):
        domain = PySALEDomain(simple_object)
        grain = PySALEObject([(0, 0), (11, 11), (12, 0)])
        with pytest.warns(UserWarning):
            domain.insert_randomly(grain, max_attempts=max_attempts)

    @pytest.mark.parametrize('max_attempts', [-1, 0, 6.5, '7'])
    def test_insert_randomly_invalid_max_attempts(self, simple_object, max_attempts):
        domain = PySALEDomain(simple_object)
        grain = PySALEObject([(0, 0), (11, 11), (12, 0)])
        with pytest.raises(AssertionError):
            domain.insert_randomly(grain, max_attempts=max_attempts)





