from abc import ABC
from math import pi, sqrt
from shapely.geometry import Polygon, Point
import shapely.affinity as affinity
import matplotlib.pyplot as plt
from typing import Union, Dict, List, Optional, Tuple
from pathlib import Path
from collections import namedtuple
import numpy as np


Velocity = namedtuple('Velocity', ['x', 'y'])


class PySALEObject(Polygon, ABC):
    """Base object for all objects in PySALESetup

    Based on `shapely.geometry.Polygon`.

    Examples
    --------
    This object forms the basis of everything we do in PySALESetup.
    First here's an example of a rectangular object 10 m x 15 m.

    >>> from PySALESetup import PySALEObject
    >>> import matplotlib.pyplot as plt
    >>> main = PySALEObject([(0, 0), (0, 15), (10, 15), (10, 0)])
    >>> main.plot()
    >>> plt.show()

    This example creates a triangular object.

    >>> from PySALESetup import PySALEObject
    >>> import matplotlib.pyplot as plt
    >>> main = PySALEObject([(0, 0), (5, 5), (10, 0)])
    >>> main.plot()
    >>> plt.show()

    This example creates an elliptical object with centroid (5, 5)
    major axis 1, minor axis 0.5 and rotated by 10 degrees.

    >>> from PySALESetup import PySALEObject
    >>> import matplotlib.pyplot as plt
    >>> main = PySALEObject.generate_ellipse([5., 5.], 1., .5, 10.)
    >>> main.plot()
    >>> plt.show()

    """
    def __init__(self, *args, **kwargs):
        """Construct a Polygon PySALEObject instance.

        Parameters
        ----------
        args
            See shapely.geometry.Polygon args
        kwargs
            See shapely.geometry.Polygon kwargs
        """
        super(PySALEObject, self).__init__(*args, **kwargs)
        self._children = None
        self._material_colors = None
        self._velocity = Velocity(x=0., y=0.)
        if 'material' in kwargs:
            self._material = kwargs['material']
        else:
            self._material = None

    def __repr__(self):
        coords = [(i, j) for i, j in zip(*self.exterior.xy)]
        return f"<PySALEObject({coords})>"

    def copy_properties_to_new_polygon(self, polygon: 'PySALEObject') \
            -> 'PySALEObject':
        """Copy properties to a another polygon.

        Properties copied are:
            - children
            - velocity
            - material colors
            - material number

        Parameters
        ----------
        polygon : PySALEObject

        Returns
        -------
        target : PySALEObject
        """
        polygon._children = self._children[:] \
            if self._children is not None else None
        polygon._velocity = self._velocity
        polygon._material_colors = self._material_colors
        polygon._material = self._material
        return polygon

    @classmethod
    def generate_ellipse(cls, xy: List[float],
                         major: float,
                         minor: float,
                         rotation: float,
                         material: int = 1) -> 'PySALEObject':
        """Create an "ellipse" version of the PySALEObject.

        ellipse PySALEObjects are actually just polygons with many
        points (65) and not true ellipses, however, given that these
        are to be applied to a discrete mesh, exact ellipses are not
        necessary.

        Parameters
        ----------
        xy : List[float]
            centroid of the ellipse
        major : float
            major radius/axis of the ellipse
        minor : float
            minor radius/axis of the ellipse
        rotation : float
            rotation of the ellipse in degrees from the horizontal
        material : int
            material number to be assigned to. Defaults to 1.

        Returns
        -------
        ellipse : PySALEObject
        """
        ellipse = Point(*xy).buffer(1)
        ellipse = affinity.scale(ellipse, float(major), float(minor))
        ellipse = affinity.rotate(ellipse, rotation) \
            if rotation > 0 else ellipse
        ellipse_coords = [(i, j) for i, j in zip(*ellipse.exterior.xy)]
        new = cls(ellipse_coords)
        new.set_material(material)
        return new

    @classmethod
    def create_from_file(cls, file_name: Path) -> 'PySALEObject':
        """Create a polygon object from coordinates in a csv file.

        It is assumed the text file is of the following format.

        > 0., 0.
        > 0., 1.
        > 1., 1.
        > 1., 0.

        With no headers or footers.

        This function is designed to be used with the grain library.

        Parameters
        ----------
        file_name : Path

        Returns
        -------
        new : PySALEObject
        """
        coords = np.genfromtxt(file_name, delimiter=',')
        new = cls(coords)
        return new

    @property
    def children(self) -> 'List[PySALEObject]':
        """PySALEObjects contained (and spawned from) the PySALEObject.

        Returns
        -------
        children : List[PySALEObject]
        """
        if self._children is None:
            self._initialise_children()
        return self._children

    @property
    def velocity(self) -> Velocity:
        """Named tuple Velocity assigned to the object.

        Use `PySALEObject.set_velocity()` to change this.

        Returns
        -------
        velocity_object : Velocity
        """
        return self._velocity

    @property
    def is_void(self) -> bool:
        """Return a bool indicating if the polygon is a void.

        Returns
        -------
        void_status : bool
        """
        return self._material == 0

    @property
    def material(self) -> int:
        """Returns the material assigned to the polygon.

        If no material has yet been assigned, it is assigned to material
        number 1.

        Returns
        -------
        material_number : int
        """
        if self._material is None:
            self.set_material(1)
        return self._material

    @property
    def material_colors(self) -> Dict[int, str]:
        """Returns the colors assigned to each material number.

        Returns
        -------
        material_colors : Dict[int, str]
        """
        if self._material_colors is None:
            self.set_material_colormap('viridis')
        return self._material_colors

    def set_velocity(self, x: float, y: float,
                     include_children: bool = True) -> None:
        """Set the velocity of the object.

        Parameters
        ----------
        x : float
        y : float
        include_children : bool
            Set velocity of all children as well? defaults to True.

        Returns
        -------
        None
        """
        self._velocity = Velocity(x=x, y=y)
        for child in self.children:
            child.set_velocity(x, y)

    def set_as_void(self) -> None:
        """Set the material to void.

        Additionally sets the velocities to 0.0.

        Returns
        -------
        None
        """
        self.set_velocity(0., 0.)
        self._material = 0

    def set_material(self, material: int) -> None:
        """Set the material number of the object.

        material must be between 0 and 9 inclusive.
        0 represents void.

        Parameters
        ----------
        material : int

        Returns
        -------
        None
        """
        assert type(material) == int and 0 <= material <= 9,\
            f"material number must be a number " \
            f"from 0 - 9. Not {material}."
        if material == 0:
            self.set_as_void()
        else:
            self._material = material

    def calculate_equivalent_radius(self):
        """Calculate the radius of the circle with the same area.

        Returns
        -------
        float
            Radius of the equivalent circle
        """

        return sqrt(self.area/pi)

    @property
    def has_children(self) -> bool:
        """Returns True if object has children.

        Returns
        -------
        has_children : bool
        """
        if self._children is not None:
            return True
        else:
            return False

    def _initialise_children(self):
        if self._children is None:
            self._children = []

    def add_child(self, child: 'PySALEObject') -> None:
        """Add existing object as a child to this object.

        Parameters
        ----------
        child : PySALEObject

        Returns
        -------
        None
        """
        self._initialise_children()
        self._children.append(child)

    def spawn_polygon_in_shape(self, *args, **kwargs) -> 'PySALEObject':
        """Create a child polygon for the object.

        Child must be contained within the host, otherwise an
        AssertionError is raised.

        Parameters
        ----------
        args
        kwargs

        Returns
        -------
        polygon : PySALEObject
        """
        child = PySALEObject(*args, **kwargs)
        self._process_new_child(child)
        return child

    def _process_new_child(self, child) -> None:
        assert self.contains(child), \
            "child objects must remain within the host"
        self._initialise_children()
        self._children.append(child)

    def spawn_ellipse_in_shape(self, xy: List[float],
                               major: float,
                               minor: float,
                               rotation: float,
                               material: int = 1) -> 'PySALEObject':
        """Create a child ellipse for the object.

        Parameters
        ----------
        xy : List[float]
            coordinates of the centroid
        major : float
            major axis
        minor : float
            minor axis
        rotation : float
            angle to the horizontal, anticlockwise, in degrees
        material : int
            material for the spawned object to have. Defaults to 1.

        Returns
        -------
        polygon : PySALEObject
        """
        ellipse = PySALEObject.generate_ellipse(xy, major, minor,
                                                rotation, material)
        self._process_new_child(ellipse)
        return ellipse

    def scale_object(self, factor: float,
                     area: bool = False) -> 'PySALEObject':
        """Create new scaled object from current object.

        Parameters
        ----------
        factor : float
        area : bool
            Does the factor refer to area or equivalent radius?

        Returns
        -------
        PySALEObject
            The new scaled object
        """
        if area is True:
            new_area = self.area * abs(factor)
            old_radius = self.calculate_equivalent_radius()
            new_radius = sqrt(new_area/pi)
            factor = new_radius/old_radius
        return self.resize(factor, factor)

    def plot(self,
             ax: Optional[plt.Axes] = None,
             include_children: bool = True,
             cmap: str = 'rainbow') -> Tuple[plt.Figure, plt.Axes]:
        """Plot the object on matplotlib axes.

        If no axes, the method will create some.

        Parameters
        ----------
        ax : Optional[plt.Axes]
        include_children : bool
            Recursively plot all children as well
        cmap : str
            The colormap used when plotting

        Returns
        -------
        fig, ax : plt.Figure, plt.Axes
            If axes were provided these are the same that were given.
        """
        self.set_material_colormap(cmap)
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()
        x, y = self.exterior.xy
        if self.material is not None:
            ax.plot(x, y, color=self.material_colors[self.material])
        else:
            ax.plot(x, y, color=self.material_colors[self.material])
        if include_children and self.has_children:
            for child in self._children:
                child.plot(ax, include_children, cmap)
        ax.set_aspect(1)
        return fig, ax

    def set_material_colormap(self,
                              colormap: str,
                              void: str = 'gray',
                              unassigned: str = 'brown') -> None:
        """Set the colors to be used when plotting.

        Parameters
        ----------
        colormap : str
        void : str
        unassigned : str

        Returns
        -------
        None
        """
        cm = plt.get_cmap(colormap)
        self._material_colors = {i: cm(1.*i/9.) for i in range(1, 9+1)}
        self._material_colors[0] = void
        self._material_colors[-1] = unassigned

    @property
    def krumbein_phi(self) -> float:
        """Return the Krumbein Phi of the object.

        A dimensionless, logarithmic, grain size metric commonly used in
        particle size distributions.

        >>> krumbein_phi = -log2(D/D0)

        Where D is the diameter of the sphere (circle in our case)
        that has the same volume (area) as the grain in question,
        and D0 is a reference length equal to 1 mm (so that the
        equation remains dimensionally consistent.

        See [Investigating the effect of mesostructure on
        the shock response of granular materials
        through numerical modelling
        - James G. Derrick 2018
        Section 6.1](https://doi.org/10.25560/66097)

        Returns
        -------
        krumbein_phi : float
        """
        diameter = np.sqrt(self.area/np.pi)*2.
        krumbein_phi = -np.log2(diameter)
        return krumbein_phi

    @property
    def elongation(self) -> float:
        length_bbox = max((self.bounds[2] - self.bounds[0]),
                          (self.bounds[3] - self.bounds[1]))
        width_bbox = min((self.bounds[2] - self.bounds[0]),
                         (self.bounds[3] - self.bounds[1]))
        elongation = length_bbox/width_bbox
        return elongation

    @property
    def area_ratio(self):
        length_bbox = max((self.bounds[2] - self.bounds[0]),
                          (self.bounds[3] - self.bounds[1]))
        width_bbox = min((self.bounds[2] - self.bounds[0]),
                         (self.bounds[3] - self.bounds[1]))
        area_ratio = (length_bbox*width_bbox)/self.area
        return area_ratio

    def translate(self, newx: float, newy: float):
        """Translate from one centroid to another.

        Returns a copy of the translated polygon.
        Does NOT change the polygon itself.

        Parameters
        ----------
        newx : float
        newy : float

        Returns
        -------
        translated_copy : PySALEObject
        """
        centroid = [self.centroid.x, self.centroid.y]
        diffs = [a - centroid[i] for i, a in enumerate([newx, newy])]
        new = self.copy_properties_to_new_polygon(
            affinity.translate(self, *diffs)
        )

        new._children = [child.translate(newx, newy)
                         for child in self.children]
        return new

    def rotate(self, angle: float,
               origin: Union[str, Tuple[float, float], Point] = 'center'):
        """Rotate by `angle` degrees and about the point `origin`.

        Returns a copy of the rotated polygon.
        Does NOT change the polygon itself.


        Parameters
        ----------
        angle : float
            rotation amount in degrees anticlockwise from the horizontal
        origin : Union[str, Point, Tuple[float, float]]
            can either be the string 'center', where the self origin
            is used or it can be a shapely.geometry.Point object.

        Returns
        -------
        rotated_copy : PySALEObject
        """
        new = self.copy_properties_to_new_polygon(
            affinity.rotate(self, angle, origin=origin)
        )
        if origin == 'center':
            new._children = [child.rotate(angle, origin=self.centroid)
                             for child in self.children]
        else:
            if isinstance(origin, tuple):
                origin = Point(*origin)
            new._children = [child.rotate(angle, origin=origin)
                             for child in self.children]
        return new

    def resize(self, xfactor: float = 1., yfactor: float = 1.):
        """Resize by xfactor and yfactor in the x and y directions.

        Returns a copy of the resized polygon.
        Does NOT change the polygon itself.

        Parameters
        ----------
        xfactor : float
        yfactor : float

        Returns
        -------
        resize_copy : PySALEObject
        """
        new = self.copy_properties_to_new_polygon(
            affinity.scale(self, xfactor, yfactor)
        )
        new._children = [child.resize(xfactor, yfactor)
                         for child in self.children]
        return new
