from PySALESetup.domain import PySALEObject, Velocity
from collections import namedtuple
import numpy as np
from dataclasses import dataclass
from shapely.geometry import Point
from typing import Iterable, Optional, Tuple, Union, List
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import gzip
from enum import Enum


class Region(Enum):
    NORTH = 1
    SOUTH = 2
    EAST = 3
    WEST = 4


ExtensionZoneFactor = namedtuple('ExtensionZoneFactors',
                                 ['multiplier', 'max_cell_size'])


@dataclass
class ExtensionZone:
    """iSALE Extension zone object.


    Extension zones can be bolted onto the main mesh (aka the
    high resolution zone). These can only have one material,
    a fixed velocity and a specific depth. Up to four can be
    added at a time and the order they appear in the PySALEMesh
    object is the order they will be applied.

            North
              |
              |
    West----MESH----East
              |
              |
            South
    """
    depth: int
    region: Region

    def calculate_zone_length(self,
                              cell_size: float,
                              extension_factor: ExtensionZoneFactor) \
            -> float:
        """Calculate physical length of the zone.

        Parameters
        ----------
        cell_size : float
        extension_factor : ExtensionZoneFactor

        Returns
        -------
        float
        """
        total_length = 0
        for i in range(self.depth):
            if cell_size < extension_factor.max_cell_size:
                cell_size *= extension_factor.multiplier
            total_length += i * cell_size
        return total_length


@dataclass
class Cell:
    """PySALEMesh cell dataclass object.

    Contains cell information including the physical centroid of a cell
    as a shapely.geometry.Point object; the indices of the cell in the
    mesh as integers; the material in the cell as an integer;
    the velocity of the cell as a pss Velocity object.
    """
    point: Point
    i: int
    j: int
    material: Optional[int]
    velocity: Velocity(0., 0.)


def get_figure_from_ax(ax: Optional[plt.Axes]) -> Tuple[plt.Axes,
                                                        plt.Figure]:
    """Get the matplotlib figure from an Axes or create some if None.

    Parameters
    ----------
    ax : plt.Axes

    Returns
    -------
    (ax, fig) : Tuple[plt.Axes, plt.figure]
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
    else:
        fig = ax.get_figure()
    return ax, fig


class PySALEMesh:
    """Mesh object of discrete cells which polygons can be projected on.

    Examples
    --------
    Once creating a polygon (or polygons) to represent your simulation
    it must then be projected onto/applied to this mesh object in order
    to save the resulting mesh to an iSALE input file.

    Here we create a void polygon that is 10 m x 15 m and populate
    it to 40% area fraction with circles of radius 1 m. Then we
    optimise the material distribution and apply the polygon to a
    mesh object. Then we can save the file.

    >>> from PySALESetup import PySALEObject, PySALEDomain, PySALEMesh
    >>> main = PySALEObject([(0, 0), (0, 15), (10, 15), (10, 0)])
    >>> main.set_material(0)
    >>> domain = PySALEDomain(main)
    >>> circle = PySALEObject.generate_ellipse([0., 0.], 1, 1, 0.)
    >>> domain.fill_to_threshold_area(circle, 40)
    >>> domain.optimise_materials()
    >>> mesh = PySALEMesh(100, 150, cell_size=.1)
    >>> mesh.apply_geometry(main)
    >>> mesh.save()
    """
    def __init__(self,
                 x_cells: int,
                 y_cells: int,
                 cell_size: float = 2.e-6,
                 extension_zones: List[ExtensionZone] = None,
                 extension_factor: ExtensionZoneFactor
                 = ExtensionZoneFactor(1., None)):
        """Discrete rectangular mesh construction.

        Parameters
        ----------
        x_cells : int
        y_cells : int
        cell_size : float
        """
        self.x = x_cells
        self.y = y_cells
        self.cell_size = cell_size
        self._x_range = None
        self._y_range = None
        self._cells = None
        self._material_meshes = None
        self._velocities = None
        self._extension_zones = extension_zones
        self._extension_factor = extension_factor
        self._y_physical_length = 0.
        self._x_physical_length = 0.

    @property
    def x_range(self) -> np.ndarray:
        if self._x_range is None:
            self._populate_x_range()
        return self._x_range

    @property
    def y_range(self) -> np.ndarray:
        if self._y_range is None:
            self._populate_y_range()
        return self._y_range

    def get_geometric_centre(self) -> Tuple[float, float]:
        x = np.ptp(self.x_range)*.5 + self.x_range[0]
        y = np.ptp(self.y_range)*.5 + self.y_range[0]
        return x, y

    @property
    def material_meshes(self):
        if self._material_meshes is None:
            self._populate_material_meshes()
        return self._material_meshes

    @property
    def velocities(self):
        if self._velocities is None:
            self._populate_velocities()
        return self._velocities

    def _populate_y_range(self):
        total_length = self.y
        self._y_physical_length = (self.y - 1) * self.cell_size
        zones = {zone.region: zone for zone in self.extension_zones}
        highres_start = 0
        highres_end = self.y + 1
        south_range = [0]
        if Region.SOUTH in zones:
            zone = zones[Region.SOUTH]
            total_length += zone.depth
            highres_start = zone.depth
            highres_end += zone.depth
            south_range = self._insert_south_zone(zone)
        if Region.NORTH in zones:
            zone = zones[Region.NORTH]
            total_length += zone.depth
            north_range = self._insert_north_zone(zone)
            highres_end = highres_start + self.y + 1

        self._y_range = np.zeros((total_length))
        highres_end_pos = (self.y+.5) * self.cell_size
        if Region.SOUTH in zones:
            self._y_range[:highres_start] = south_range
            highres_start_pos = np.amax(south_range)
            highres_end_pos += highres_start_pos

        self._y_range[highres_start:highres_end - 1] = \
            self._generate_highres_zone(highres_end,
                                        highres_start,
                                        south_range)

        if Region.NORTH in zones:
            self._y_range[highres_end-1:] = north_range

        return self._y_range

    def _generate_highres_zone(self,
                               highres_end,
                               highres_start,
                               range_):
        highres_zone = [np.amax(range_) + i * self.cell_size
                        for i in range(1, highres_end-highres_start)]
        return np.array(highres_zone)

    def _populate_x_range(self):
        total_length = self.x
        self._x_physical_length = (self.x - 1) * self.cell_size
        zones = {zone.region: zone for zone in self.extension_zones}
        highres_start = 0
        highres_end = self.x + 1
        west_range = [0]
        if Region.WEST in zones:
            zone = zones[Region.WEST]
            total_length += zone.depth
            highres_start = zone.depth
            highres_end += zone.depth
            west_range = self._insert_west_zone(zone)
        if Region.EAST in zones:
            zone = zones[Region.EAST]
            total_length += zone.depth
            east_range = self._insert_east_zone(zone)
            highres_end = highres_start + self.x + 1

        self._x_range = np.zeros((total_length))
        highres_end_pos = (self.x+.5) * self.cell_size
        if Region.WEST in zones:
            self._x_range[:highres_start] = west_range
            highres_start_pos = np.amax(west_range)
            highres_end_pos += highres_start_pos

        self._x_range[highres_start:highres_end - 1] = \
            self._generate_highres_zone(highres_end,
                                        highres_start,
                                        west_range)

        if Region.EAST in zones:
            self._x_range[highres_end-1:] = east_range

        return self._x_range

    def _populate_velocities(self):
        x_cells = self.x_range.size
        y_cells = self.y_range.size
        self._velocities = {r: np.zeros((x_cells, y_cells))
                           for r in ['x', 'y']}

    def _populate_material_meshes(self):
        x_cells = self.x_range.size
        y_cells = self.y_range.size
        self._material_meshes = {i: np.zeros((x_cells, y_cells))
                                 for i in range(1, 9 + 1)}

    @property
    def cells(self):
        if self._cells is None:
            self._populate_cells()
        return self._cells

    def _populate_cells(self):
        self._cells = [Cell(Point(x, y), i, j, None, Velocity(0., 0.))
                       for i, x in enumerate(self.x_range)
                       for j, y in enumerate(self.y_range)]

    def apply_geometry(self, geometry: PySALEObject) -> None:
        """Project a polygon (and all its children) onto the mesh.

        Method calls itself recursively on all children of the polygon.
        The children at the bottom of the hierachy get priority. Once
        a cell is populated with material, new material will NOT
        overwrite it. If a child is a void object it will inevitably be
        overwritten by its parent's material. To create voids use
        ``appy_geometry_as_void`` after this method. See that docstring
        for an example.

        Parameters
        ----------
        geometry : PySALEObject

        Returns
        -------
        None

        Examples
        --------
        Here we create a solid circle that is 5 m x 5 m and populate
        it to 40% area fraction with circles of radius 0.5 m. Then we
        optimise the material distribution and apply the polygon to a
        mesh object. Then we plot the result.

        >>> from PySALESetup import PySALEObject, PySALEDomain, PySALEMesh
        >>> import matplotlib.pyplot as plt
        >>> main = PySALEObject.generate_ellipse([5., 5.], 5., 5., 0.)
        >>> main.set_material(1)
        >>> domain = PySALEDomain(main)
        >>> circle = PySALEObject.generate_ellipse([0., 0.], .5, .5, 0.)
        >>> domain.fill_to_threshold_area(circle, 40)
        >>> domain.optimise_materials([2, 3, 4, 5])
        >>> mesh = PySALEMesh(100, 100, cell_size=.1)
        >>> mesh.apply_geometry(main)
        >>> mesh.plot_materials()
        >>> plt.show()
        """
        for child in geometry.children:
            self.apply_geometry(child)
        if geometry.material > 0:
            for cell in self.cells:
                if cell.point.within(geometry) and \
                        sum(self.material_meshes[i][cell.i, cell.j]
                            for i in range(1, 9+1)) < 1.:
                    self._fill_cell(cell, geometry)
        elif geometry.material == 0. and len(geometry.children) == 0:
            for cell in self.cells:
                if cell.point.within(geometry):
                    self._void_cell(cell)
        elif geometry.material == 0. and len(geometry.children) > 0:
            pass
        else:
            raise TypeError(f'Material "{geometry.material}" is not '
                            f'recognised')

    def _fill_cell(self, cell: Cell, geometry: PySALEObject):
        """Fill a mesh cell with the properties of a given polygon.

        Parameters
        ----------
        cell: Cell
        geometry : PySALEObject

        Returns
        -------
        None

        """
        self.material_meshes[geometry.material][cell.i, cell.j] = 1.
        self.velocities['x'][cell.i, cell.j] = geometry.velocity.x
        self.velocities['y'][cell.i, cell.j] = geometry.velocity.y
        cell.material = geometry.material
        cell.velocity = geometry.velocity

    def _void_cell(self, cell: Cell):
        """Fill a mesh cell with void.

        Parameters
        ----------
        cell : Cell

        Returns
        -------
        None
        """
        for number in self.material_meshes.keys():
            self.material_meshes[number][cell.i, cell.j] = 0.
        self.velocities['x'][cell.i, cell.j] = 0.
        self.velocities['y'][cell.i, cell.j] = 0.
        cell.material = 0.
        cell.velocity = Velocity(0., 0.)

    def apply_polygon_as_void(self, polygon: PySALEObject):
        """Apply polygon (not inc. children) as void to mesh.

        This is independent of the polygon's assigned material. if this
        method is called the polygon is treated as if it were void.

        Parameters
        ----------
        polygon: PySALEObject

        Returns
        -------
        None

        Examples
        --------

        Here we create a circle of material, then populate it with
        smaller circles of void. Once we have applied the parent object
        we can apply each child as void in a simple loop.

        >>> from PySALESetup import PySALEObject, PySALEDomain, PySALEMesh
        >>> import matplotlib.pyplot as plt
        >>> main = PySALEObject.generate_ellipse([5., 5.], 5., 5., 0.)
        >>> main.set_material(1)
        >>> domain = PySALEDomain(main)
        >>> circle = PySALEObject.generate_ellipse([0., 0.], .5, .5, 0.)
        >>> circle.set_as_void()
        >>> domain.fill_to_threshold_area(circle, 40)
        >>> mesh = PySALEMesh(100, 100, cell_size=.1)
        >>> mesh.apply_geometry(main)
        >>> for child in main.children:
        >>>     mesh.apply_polygon_as_void(child)
        >>> mesh.plot_materials()
        >>> plt.show()
        """
        for cell in self.cells:
            if cell.point.within(polygon):
                self._void_cell(cell)

    def plot_materials(self, ax: plt.Axes = None,
                       cmap: str = 'terrain'):
        """Plot the materials in the mesh using matplotlib.

        If no axes are provided, axes and a figure are made. Otherwise,
        the given axes are used and returned along with the associated
        figure object.

        Parameters
        ----------
        ax : plt.Axes
        cmap: str

        Returns
        -------
        fig, ax : Tuple[plt.Axes, plt.figure]

        Examples
        --------

        Here we construct a simple 2D meteorite impacting flat ground.
        Once our objects have been created and applied, we use
        plot_materials to view the mesh, although we need to use
        `plt.show()` to visualise the object you could just as easily
        save the figure instead.

        >>> from PySALESetup.domain import PySALEObject
        >>> from PySALESetup.mesh import PySALEMesh
        >>> import matplotlib.pyplot as plt
        >>> impactor = PySALEObject.generate_ellipse([5., 8.], 2., 2., 0.)
        >>> impactor.set_material(1)
        >>> impactor.set_velocity(0. -1000.)
        >>> target = PySALEObject([(0, 0), (0, 6), (10, 6), (10, 0)])
        >>> target.set_material(3)
        >>> mesh = PySALEMesh(100, 100, cell_size=.1)
        >>> mesh.apply_geometry(impactor)
        >>> mesh.apply_geometry(target)
        >>> mesh.plot_materials()
        >>> plt.show()
        """
        ax, fig = get_figure_from_ax(ax)
        xi, yi = np.meshgrid(self.x_range, self.y_range)
        for i in range(1, 9+1):
            matter = np.copy(self.material_meshes[i])*i
            matter = np.ma.masked_where(matter == 0., matter)
            ax.pcolormesh(xi, yi, matter.T, cmap=cmap, vmin=1, vmax=9,
                          shading='auto')
        self._set_plot_lims_and_labels(ax)
        ax.set_title('Materials')
        return fig, ax

    def _set_plot_lims_and_labels(self, ax):
        ax.set_xlim(np.amin(self.x_range), np.amax(self.x_range))
        ax.set_ylim(np.amin(self.y_range), np.amax(self.y_range))
        ax.set_xlabel('$x$ [m]')
        ax.set_ylabel('$y$ [m]')

    def plot_velocities(self,
                        ax1: Optional[plt.Axes] = None,
                        ax2: Optional[plt.Axes] = None,
                        cmap: str = 'viridis') -> Tuple[plt.Figure,
                                                        plt.Figure,
                                                        plt.Axes,
                                                        plt.Axes]:
        """Plot the velocities of cells.

        If axes are provided they are used. If any are not provided,
        they are created. Either way the axes and figures are returned.

        Parameters
        ----------
        ax1 : Optional[plt.Axes]
        ax2 : Optional[plt.Axes]
        cmap : str

        Returns
        -------
        fig1, fig2, ax1, ax2 : Tuple[plt.Figure, plt.Figure,
                                     plt.Axes, plt.Axes]
        Examples
        --------

        >>> from PySALESetup.domain import PySALEObject
        >>> from PySALESetup.mesh import PySALEMesh
        >>> import matplotlib.pyplot as plt
        >>> impactor = PySALEObject.generate_ellipse([5., 8.], 2., 2., 0.)
        >>> impactor.set_material(1)
        >>> impactor.set_velocity(0. -1000.)
        >>> target = PySALEObject([(0, 0), (0, 6), (10, 6), (10, 0)])
        >>> target.set_material(3)
        >>> mesh = PySALEMesh(100, 100, cell_size=.1)
        >>> mesh.apply_geometry(impactor)
        >>> mesh.apply_geometry(target)
        >>> mesh.plot_materials()
        >>> plt.show()

        """
        ax1, fig1 = get_figure_from_ax(ax1)
        ax2, fig2 = get_figure_from_ax(ax2)
        xi, yi = np.meshgrid(self.x_range, self.y_range)
        matter_sum = self.material_meshes[1]
        for i in range(2, 9+1):
            matter_sum += self.material_meshes[i]
        vx = np.ma.masked_where(matter_sum == 0., self.velocities['x'])
        vy = np.ma.masked_where(matter_sum == 0., self.velocities['y'])

        for ax, v in [(ax1, vx), (ax2, vy)]:
            ax.pcolormesh(xi, yi, v.T, cmap=cmap,
                          vmin=np.amin(v),
                          vmax=np.amax(v),
                          shading='auto')
            self._set_plot_lims_and_labels(ax)
        ax1.set_title('Velocity - x')
        ax2.set_title('Velocity - y')
        return fig1, fig2, ax1, ax2

    @staticmethod
    def _cell_to_row(cell: Cell, materials: Iterable[int]) -> str:
        material_columns = ['1.000' if cell.material == m
                            else '0.000' for m in materials]
        row = f'{cell.i} {cell.j} ' \
              + ' '.join(material_columns) \
              + f' {cell.velocity.x} {cell.velocity.y}\n'
        return row

    def save(self, file_name: Path = Path('./meso_m.iSALE'),
             compress: bool = False) -> None:
        """Save the current mesh to a meso_m.iSALE file.

        This compiles the integer indices of each cell,
        as well as the material in them. It saves all this to the file
        specified by the user, which defaults to ``meso_m.iSALE`` in
        the user's current directory.

        Parameters
        ----------
        file_name : Path
        compress : bool
            Compress the resulting text file using gunzip.
            Users are expected to add their own .gz file extension.
            If one is not present a UserWarning is raised.

        Returns
        -------
        None
        """
        cell_number = self.x * self.y
        material_numbers = [key for key, value
                            in self.material_meshes.items()
                            if np.sum(value) > 0.]
        if compress:
            if file_name.suffix != '.gz':
                warnings.warn(f'Mesh is being compressed but file '
                              f'name "{file_name}" does not have a '
                              f'.gz extension.')
            with gzip.open(file_name, 'w') as f:
                self._write_mesh_to_file(cell_number,
                                         f,
                                         material_numbers)
        else:
            with open(file_name, 'w') as f:
                self._write_mesh_to_file(cell_number,
                                         f,
                                         material_numbers)

    def _write_mesh_to_file(self,
                            cell_number: int,
                            file_object,
                            material_numbers: Union[List[int],
                                                    Tuple[int]]) \
            -> None:
        first_row = f'{cell_number}, {len(material_numbers)}\n'
        file_object.write(first_row)
        file_object.writelines([self._cell_to_row(cell,
                                                  material_numbers)
                                for cell in self.cells])

    def _insert_south_zone(self, zone: ExtensionZone):
        factor = self.extension_factor.multiplier
        max_size = self.extension_factor.max_cell_size
        varying_cell_size = self.cell_size
        position = -0.5*self.cell_size
        varying_cell_size, y_coord = self._create_extension_zone_coordinates(
            factor, max_size, [position], varying_cell_size, zone,
            'south/west')
        south_y_range = np.array(y_coord)
        south_y_range += abs(np.amin(south_y_range)) + varying_cell_size
        self._y_physical_length += np.amax(south_y_range)
        return south_y_range[::-1]

    def _insert_north_zone(self, zone: ExtensionZone):
        factor = self.extension_factor.multiplier
        max_size = self.extension_factor.max_cell_size
        varying_cell_size = self.cell_size
        position = self._y_physical_length + self.cell_size*2

        varying_cell_size, y_coord = self._create_extension_zone_coordinates(
            factor, max_size, [position], varying_cell_size, zone,
            'north/east')

        north_y_range = np.array(y_coord)
        self._y_physical_length += np.ptp(north_y_range)
        return north_y_range

    def _insert_west_zone(self, zone: ExtensionZone):
        factor = self.extension_factor.multiplier
        max_size = self.extension_factor.max_cell_size
        varying_cell_size = self.cell_size
        position = -0.5*self.cell_size

        varying_cell_size, x_coord = self._create_extension_zone_coordinates(
            factor, max_size, [position], varying_cell_size, zone,
            'south/west')

        west_x_range = np.array(x_coord)
        west_x_range += abs(np.amin(west_x_range)) + varying_cell_size
        self._x_physical_length += np.amax(west_x_range)
        return west_x_range[::-1]

    def _create_extension_zone_coordinates(self, factor: float,
                                           max_size: float,
                                           coord: List[float],
                                           varying_cell_size: float,
                                           zone: ExtensionZone,
                                           half: str):
        counter = 1
        position = coord[0]
        while counter < zone.depth:
            if varying_cell_size < max_size:
                varying_cell_size = counter * factor * self.cell_size
            else:
                varying_cell_size = max_size
            if half.lower() == 'south/west':
                position -= varying_cell_size
            elif half.lower() == 'north/east':
                position += varying_cell_size
            coord.append(position)
            counter += 1
        return varying_cell_size, coord

    def _insert_east_zone(self, zone: ExtensionZone):
        factor = self.extension_factor.multiplier
        max_size = self.extension_factor.max_cell_size
        varying_cell_size = self.cell_size
        position = self._x_physical_length + self.cell_size*2

        varying_cell_size, x_coord = self._create_extension_zone_coordinates(
            factor, max_size, [position], varying_cell_size, zone,
            'north/east')
        # x_coord = [position]
        # while counter < zone.depth:
        #     if varying_cell_size < max_size:
        #         varying_cell_size = counter*factor*self.cell_size
        #     else:
        #         varying_cell_size = max_size
        #     position += varying_cell_size
        #     x_coord.append(position)
        #     counter += 1
        #

        east_x_range = np.array(x_coord)
        self._x_physical_length += np.ptp(east_x_range)
        return east_x_range

    @property
    def extension_zones(self) -> List[ExtensionZone]:
        if self._extension_zones is None:
            self._extension_zones = {}
        return self._extension_zones

    @property
    def extension_factor(self) -> ExtensionZoneFactor:
        if self._extension_factor is None:
            self._extension_factor = \
                ExtensionZoneFactor(1., self.cell_size)
        return self._extension_factor

