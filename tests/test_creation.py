from PySALESetup import PySALEDomain, PySALEObject
import pytest
from math import isclose


class TestPySALEDomain:
    def test_move_object_to_random_coords(self, simple_object):
        domain = PySALEDomain(simple_object)
        simple_object = \
            domain._move_object_to_random_coordinate_in_domain(simple_object, 1., 1., 0., 0.)
        assert 0. < simple_object.centroid.x < 1.
        assert 0. < simple_object.centroid.y < 1.

    def test_fill_to_threshold(self, simple_object):
        domain = PySALEDomain(simple_object)
        grain = PySALEObject([(0, 0), (0.5, 0.5), (1, 0)])
        domain.fill_to_threshold_area(grain, 20)
        frac = sum([c.area
                    for c in simple_object.children])/simple_object.area
        tolerance = (grain.area/simple_object.area)*100
        assert isclose(frac*100., 20., abs_tol=tolerance)

    @pytest.mark.parametrize('threshold_', [10., 30., 60., 90.])
    def test_threshold_check(self, simple_object, threshold_):
        domain = PySALEDomain(simple_object)
        inserted_area, insertion_possible, threshold = \
            domain._check_threshold_input(threshold_)
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