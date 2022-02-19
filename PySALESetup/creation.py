from PySALESetup.objects import PySALEObject
import warnings
from typing import Tuple, List, Callable
from shapely.geometry import MultiPolygon
import numpy as np
import scipy.special as scsp
import random


class PySALEDistributionBase:
    """Base class for distributions.

    In PySALESetup it is not uncommon to want to feed specific sizes,
    or rotations, or radii, to a setup routine. These distribution
    classes provide this capability. Everything random always comes
    from a distribution and if it's not specified, it's probably a
    uniform one.
    """
    def __init__(self):
        self.mean = None
        self.median = None
        self.mode = None
        self.variance = None
        self.skew = None
        self.name = None

    def details(self):
        details = "distribution has the following properties:\n"
        details += f"Name: {self.name}"
        details += f"mean = {self.mean:2.3f}\n"
        details += f"median = {self.median:2.3f}\n"
        details += f"mode = {self.mode:2.3f}\n"
        details += f"variance = {self.variance:2.3f}\n"
        details += f"skewness = {self.skew:2.3f}\n"
        return details

    def frequency(self, x: float, bounds: Tuple[float, float]) -> float:
        """Get the frequency from the PDF over a specified interval.

        Integrates over the probability density function of the
        chosen distribution over a specified interval
        to return an estimated frequency. Limits MUST be provided
        in the form of `bounds`, which allows for uneven limits and is
        always applied as + and - the given value of x.

        Returns the probability DENSITY! this must be converted to a
        useful value outside of the function.
        Parameters
        ----------
        x : float
        bounds : Tuple[float, float]

        Returns
        -------
        frequency_density : float
        """
        f = np.float64(abs(self.cdf(x + bounds[1]) - self.cdf(x - bounds[0])))
        return f

    def cdf(self, x: float):
        """Base method for calculating the CDF at a specified point.

        Parameters
        ----------
        x : float

        Returns
        -------

        """
        raise NotImplementedError

    def random_number(self):
        """Base method for getting a random number by the distribution.

        Returns
        -------

        """
        raise NotImplementedError


class PySALEUniformDistribution(PySALEDistributionBase):
    """The uniform distribution.

    All values are equally likely.
    """
    def __init__(self, limits: Tuple[float, float]):
        """Creates a uniform distribution given a pair of limits.

        Parameters
        ----------
        limits : Tuple[float, float]
        """
        super().__init__()
        assert limits is not None, \
            "ERROR: function must have size limits (ind var)"
        self.name = 'Uniform'
        self.limits = limits
        self.mean = .5 * (limits[0] + limits[1])
        self.median = self.mean
        self.variance = (1. / 12.) * (limits[1] - limits[0]) ** 2.
        self.skew = 0.

    def cdf(self, x: float) -> float:
        """CDF for a uniform probability density function.

        Parameters
        ----------
        x : float

        Returns
        -------
        probability : float
        """
        minx = self.limits[0]
        maxx = self.limits[1]
        f = (x - minx) / (maxx - minx)
        if x < minx:
            f = 0.
        elif x >= maxx:
            f = 1.
        return f

    def random_number(self):
        """Generates a random number from the uniform distribution.

        Returns
        -------
        number : float
        """
        return np.random.uniform(*self.limits)


class PySALENormalDistribution(PySALEDistributionBase):
    """Normal distribution.

    """
    def __init__(self, mu: float, sigma: float):
        """Constructs a normal distribution given a mean and std.

        Parameters
        ----------
        mu : float
            mean
        sigma : float
            standard deviation
        """
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.mean = mu
        self.median = mu
        self.mode = mu
        self.variance = sigma ** 2.
        self.skew = 0.
        self.name = 'Normal'

    def cdf(self, x) -> float:
        """CDF for a normal probability density function.

        Parameters
        ----------
        x : float

        Returns
        -------
        probability : float
        """
        mu = self.mu
        sigma = self.sigma
        f = .5 * (1. + scsp.erf((x - mu) / (sigma * np.sqrt(2.))))
        return f

    def random_number(self) -> float:
        """Generate a random number from the Normal distribution.

        Returns
        -------
        random_number : float
        """
        return np.random.normal(self.mean, np.sqrt(self.variance))


class PySALELogNormalDistribution(PySALEDistributionBase):
    """Lognormal distribution."""
    def __init__(self, mu: float, sigma: float):
        """Construct a Lognormal distribution from a mean and std.

        Parameters
        ----------
        mu : float
            mean
        sigma : float
            standard deviation
        """
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.mean = np.exp(mu + 0.5 * sigma ** 2.)
        self.median = np.exp(mu)
        self.mode = np.exp(mu - sigma ** 2.)
        self.variance = (np.exp(sigma ** 2.) - 1.) \
                        * np.exp(2. * mu + sigma ** 2.)
        self.skew = (np.exp(sigma ** 2.) + 2.) \
                    * np.sqrt(np.exp(sigma ** 2.) - 1.)
        self.name = 'Lognormal'

    def cdf(self, x: float) -> float:
        """CDF for the log-normal probability density function.

        Parameters
        ----------
        x : float

        Returns
        -------
        probability : float
        """
        mu = self.mu
        sigma = self.sigma
        f = .5 + .5 * scsp.erf((np.log(x) - mu) / (sigma * np.sqrt(2.)))
        return f

    def frequency(self, x: float, bounds: Tuple[float, float]) -> float:
        """Frequency of the Lognormal PDF over a given range.

        Parameters
        ----------
        x : float
        bounds : Tuple[float, float]

        Returns
        -------
        frequency : float
        """
        assert x >= 0., "ERROR: Lognormal distribution only works " \
                        "for input greater than 0"
        if type(bounds) != list:
            warnings.warn('dx must be a list, set to list this time')
            bounds = [bounds * .5, bounds * .5]
        f = np.float64(abs(self.cdf(x + bounds[1]) - self.cdf(x - bounds[0])))
        return f

    def random_number(self) -> float:
        """Generate a random number from the Lognormal distribution.

        Returns
        -------
        random_number : float
        """
        return np.random.lognormal(self.mean, np.sqrt(self.variance))


class PySALEWeibull2Distribution(PySALEDistributionBase):
    """Weibull 2-parameter distribution.

    This distribution is typically used for Particle Size Distributions
    generated by grinding, milling, and crushing operations.
    """
    def __init__(self, lambda_: float, k: float):
        """Construct a Weibull 2-parameter distribution.

        Parameters
        ----------
        lambda_ : float
            The 'scale' of the distribution
        k : float
            The 'shape' of the distribution.
        """
        super().__init__()
        if lambda_ < 0:
            warnings.warn("lambda must be >= 0, not {:2.2f}; "
                          "setting to zero this time".format(lambda_))
            lambda_ = 0.
        if k < 0.:
            warnings.warn("k must be >= 0, not {:2.2f}; "
                          "setting to zero this time".format(k))
            k = 0.
        self.lambda_ = lambda_
        self.k = k
        self.mean = lambda_ * scsp.gamma(1. + 1. / k)
        self.median = lambda_ * (np.log(2.)) ** (1. / k)
        if k > 1:
            self.mode = lambda_ * ((k - 1) / k) ** (1. / k)
        else:
            self.mode = 0
        self.variance = (lambda_ ** 2.) * (scsp.gamma(1. + 2. / k)
                                           - (scsp.gamma(1. + 1. / k))
                                           ** 2.)
        self.skew = (scsp.gamma(1. + 3. / k) * lambda_ ** 3.
                     - 3. * self.mean * self.variance - self.mean ** 3.)
        self.skew /= self.variance ** (3. / 2.)
        self.name = 'Weibull 2-parameter'

    def cdf(self, x: float) -> float:
        """CDF for a Weibull 2-parameter distribution

        Parameters
        ----------
        x : float

        Returns
        -------
        probability : float
        """
        if x >= 0:
            f = 1. - np.exp(-((x / self.lambda_) ** self.k))
        else:
            f = 0.
        return f

    def random_number(self) -> float:
        """Generate a random number from the Weibull2 distribution.

        Returns
        -------
        random_number : float
        """
        r = np.random.uniform()
        x = self.lambda_ * (-np.log(r+1)) ** (1./self.k)
        return x


class PySALECustomDistribution(PySALEDistributionBase):
    """A user-defined distribution."""
    def __init__(self, cdf_function: Callable,
                 random_number_function: Callable):
        """Construct a user-defined distribution from two functions.

        Parameters
        ----------
        cdf_function : Callable
        random_number_function : Callable
        """
        super().__init__()
        assert callable(cdf_function), \
            "cdf function provided is not a callable!"
        assert callable(random_number_function), \
            "random number function provided is not a callable!"
        self.cdf = cdf_function
        self.random_number = random_number_function
        self.name = 'User-defined'

    def cdf(self, x: float):
        """Overwritten by the constructor"""
        raise NotImplementedError

    def random_number(self):
        """Overwritten by the constructor"""
        raise NotImplementedError

    def details(self):
        return "Custom distribution in use."


class PySALEDomain:
    """Domain class providing insertion methods and statistics.

    Create an instance from an existing PySALEObject and then use the
    class to insert objects in more complex ways than available on the
    standard object class. In particular `optimise_materials` is
    available as a way to distribute materials in a host object such
    that as few objects of the same material are in contact as possible.

    Examples
    --------


    >>> from PySALESetup import PySALEObject, PySALEDomain
    >>> main = PySALEObject([(0, 0), (0, 30), (30, 30), (30, 0)])
    >>> main.set_material(0)
    >>> domain = PySALEDomain(main)
    >>> circle = PySALEObject.generate_ellipse([0., 0.], 1, 1, 0.)
    >>> domain.fill_to_threshold_area(circle, 40)
    >>> domain.optimise_materials()
    >>> fig, ax = main.plot()
    >>> ax.set_aspect(1)
    >>> fig.savefig('PySALEDomain_example.png')

    """
    def __init__(self, domain_object: PySALEObject):
        """

        Parameters
        ----------
        domain_object
        """
        self.object: PySALEObject = domain_object

    def fill_with_random_grains_to_threshold(self,
                                             grain_object: PySALEObject,
                                             threshold_fill_percent: float,
                                             rotation_distribution:
                                             PySALEDistributionBase
                                             = None,
                                             size_distribution:
                                             PySALEDistributionBase
                                             = None,
                                             max_retries: int = 10
                                             ) -> float:
        """Fill host object to threshold fill percent.
        
        Parameters
        ----------
        grain_object : PySALEObject
            The filler object
        threshold_fill_percent : float
        rotation_distribution : PySALEDistributionBase
            The distribution for the rotation angles.
        size_distribution : PySALEDistributionBase
            The distribution for the grain sizes.
        max_retries : int
            If a grain fails to be placed how many 
            retries with new grains before giving up? Does not reset
            between grains.

        Returns
        -------
        float
            Inserted area
        """
        inserted_area, insertion_possible, threshold \
            = self._check_threshold_input(threshold_fill_percent)
        retries = 0
        while inserted_area <= threshold and retries <= max_retries:
            if rotation_distribution is not None:
                grain_object = self.randomly_rotate_object(
                    grain_object, rotation_distribution
                )
            if size_distribution is not None:
                grain_object = self.randomly_resize_object(
                    grain_object, size_distribution
                )
            insertion_possible = self.insert_randomly(grain_object)
            if insertion_possible is not True:
                retries += 1
                continue
            inserted_area = sum([c.area for c in self.object.children])
        return inserted_area

    @staticmethod
    def randomly_rotate_object(duplicated_grain_object: PySALEObject,
                               rotation_distribution:
                               PySALEDistributionBase = None) \
            -> PySALEObject:
        """Rotate the supplied object by a random amount.
        
        Parameters
        ----------
        duplicated_grain_object : PySALEObject
        rotation_distribution : PySALEDistributionBase

        Returns
        ------- 
        rotated_object : PySALEObject
        """
        if rotation_distribution is not None:
            angle = rotation_distribution.random_number()
        else:
            angle = 0.
        duplicated_grain_object = duplicated_grain_object.rotate(angle)
        return duplicated_grain_object

    @staticmethod
    def randomly_resize_object(grain_object: PySALEObject,
                               size_distribution:
                               PySALEDistributionBase = None,
                               area: bool = False) \
            -> PySALEObject:
        """Resize the supplied object by a random amount.
        
        The supplied distribution should return either areas or radii.
        Areas and radii are taken to be "equivalent" radii. I.e. the
        radius of the equivalent circle to the polygon with the same
        area.
        
        Parameters
        ----------
        grain_object : PySALEObject
        size_distribution : PySALEDistributionBase
        area : bool
            True if the distribution is returning an area, False if its
            an equivalent radius.

        Returns
        ------- 
        rotated_object : PySALEObject
        """
        if size_distribution is not None:
            rand_val = size_distribution.random_number()
        else:
            rand_val = grain_object.area if area \
                else grain_object.calculate_equivalent_radius()

        if area is True:
            factor = rand_val/grain_object.area
        else:
            factor = rand_val/grain_object.calculate_equivalent_radius()
        return grain_object.scale_object(factor, area)

    def _check_threshold_input(self, threshold_fill_percent: float):
        target_fraction = threshold_fill_percent / 100.
        threshold = target_fraction * self.object.area
        assert 0. < target_fraction < 1., f"Target area percentage " \
                                          f"must be between 0 and 100, " \
                                          f"exclusive;" \
            f" not {threshold_fill_percent}"
        inserted_area = sum([c.area for c in self.object.children])
        current_fraction = inserted_area / self.object.area
        assert current_fraction < target_fraction, \
            f"Target fraction ({target_fraction}) " \
            f"less than current fraction ({current_fraction})."
        insertion_possible = True
        return inserted_area, insertion_possible, threshold

    def insert_randomly(self, grain_object: PySALEObject,
                        max_attempts: int = 100) -> bool:
        """Insert object into the host at random locations until it fits

        "fits" means - "is not intersecting with any other objects."

        Parameters
        ----------
        grain_object : PySALEObject
        max_attempts : int
            The number of attempts the algorithm will make
            before giving up

        Returns
        -------
        success : bool
        """
        assert type(max_attempts) == int and max_attempts > 0,\
            f'Maximum number of attempts ' \
            f'MUST be a positive integer and not {max_attempts}'
        minx, miny, maxx, maxy = self.object.bounds
        if not self.object.has_children:
            intersects_with_other_shapes = False
        else:
            intersects_with_other_shapes = True
        is_within_domain = False
        mp = MultiPolygon(self.object.children)
        counter = 0
        success = False
        while intersects_with_other_shapes \
                or is_within_domain is not True:
            grain_object = self._move_object_to_random_coordinate_in_domain(grain_object, maxx, maxy, minx, miny)
            is_within_domain = self.object.contains(grain_object)
            intersects_with_other_shapes = mp.intersects(grain_object)
            counter += 1
            if counter >= max_attempts:
                warnings.warn(f'Max insertion attempts '
                              f'reached ({max_attempts}). '
                              f'Object can not be placed.')
                success = False
                break
            else:
                success = True
        if is_within_domain and not intersects_with_other_shapes:
            self.object.add_child(grain_object)
        return success

    @staticmethod
    def _move_object_to_random_coordinate_in_domain(grain_object,
                                                    maxx, maxy,
                                                    minx, miny) \
            -> PySALEObject:
        centroid = np.random.uniform(low=minx, high=maxx), \
                   np.random.uniform(low=miny, high=maxy)
        grain_object = grain_object.translate(*centroid)
        return grain_object

    def optimise_materials(self,
                           allowed_materials: List[int]
                           = (1, 2, 3, 4, 5, 6, 7, 8, 9)) -> None:
        """Redistribute material numbers to not be in contact.

        This function has the greatest success and is based on that
        used in JP Borg's work with CTH.

        Function to assign material numbers to each particle
        This function tries to optimise the assignments such
        that as few particles of the same material are in contact as
        possible. It works by creating a list of the Mth
        closest polygons, where "M" is the number of
        different materials being used.

        Then they are sorted into the order closest -> furthest.

        If the list contains all the same elements, as the
        allowed materials then there are necessarily no repeats
        and all material numbers are used up. So,
        => use the number of the particle furthest away.

        If there is at least one repeat, then at least one
        material number has not been used. Select the
        first remaining material number that is unused
        and assign that.

        Continue until all the particles are assigned.

        Parameters
        ----------
        allowed_materials : List[int]
            defaults to all materials except for void.

        Returns
        -------
        None
        """
        num_materials = len(set(allowed_materials))
        self._reset_domain_object_materials()

        for child in self.object.children:
            ordered_neighbours = sorted(self.object.children[1:],
                                        key=child.centroid.distance)
            assigned_materials = set([o._material
                                      for o, m in zip(ordered_neighbours,
                                                      set(allowed_materials))
                                      if o._material is not None])
            if len(assigned_materials) == num_materials:
                # All material numbers have been used up and no repeats!
                # (if there were repeats not all would have
                # been used...)
                new_material = random.choice(allowed_materials)
            else:
                missing_materials = [item
                                     for item in allowed_materials
                                     if item not in assigned_materials]
                new_material = random.choice(missing_materials)
            child.set_material(new_material)

    def _reset_domain_object_materials(self):
        for child in self.object.children:
            child._material = None


