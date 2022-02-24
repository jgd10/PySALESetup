from PySALESetup.objects import PySALEObject, Velocity
from PySALESetup.functions import get_figure_from_ax
from collections import namedtuple
import numpy as np
from dataclasses import dataclass
from shapely.geometry import Point
from typing import Iterable, Optional, Tuple, Union, List, Dict
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import gzip
from enum import Enum
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Region(Enum):
    NORTH = 1
    SOUTH = 2
    EAST = 3
    WEST = 4


ExtensionZoneFactor = namedtuple('ExtensionZoneFactors',
                                 ['multiplier', 'max_cell_size'])


class ExtensionZone:
    """iSALE Extension zone object.


    Extension zones can be bolted onto the main mesh (aka the
    high resolution zone). These can only have one material,
    a fixed velocity and a specific depth. Up to four can be
    added at a time and the order they appear in the PySALEMesh
    object is the order they will be applied.

    .. code-block::

                North
                  |
                  |
        West----MESH----East
                  |
                  |
                South
    """
    def __init__(self, depth: int, region: Region, cell_size: float,
                 factor: ExtensionZoneFactor = None):
        self.depth = depth
        self.region = region
        self.cell_size = cell_size
        if factor is None:
            factor = ExtensionZoneFactor(1, cell_size)
        self.factor = factor

    @property
    def length(self) -> float:
        """Physical length of the zone.

        Returns
        -------
        length : float
        """
        return self.calculate_zone_length()

    def calculate_zone_length(self) -> float:
        """Calculate physical length of the zone.

        Returns
        -------
        float
        """
        total_length = 0
        for i in range(self.depth):
            if self.cell_size < self.factor.max_cell_size:
                self.cell_size *= self.factor.multiplier
            total_length += self.cell_size
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
    material: int = None
    velocity: Velocity = Velocity(0., 0.)


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
    >>> domain.fill_with_random_grains_to_threshold(circle, 40)
    >>> domain.optimise_materials()
    >>> mesh = PySALEMesh(100, 150, cell_size=.1)
    >>> mesh.project_polygons_onto_mesh([main])
    >>> mesh.save()
    """
    def __init__(self,
                 x_cells: int,
                 y_cells: int,
                 cell_size: float = 2.e-6,
                 extension_zones: List[ExtensionZone] = None,
                 cylindrical_symmetry: bool = False,
                 collision_index: int = 0,
                 origin: Tuple[float, float] = (0., 0.)):
        """Discrete rectangular mesh construction.

        Parameters
        ----------
        x_cells : int
        y_cells : int
        cell_size : float
        extension_zones : Optional[List[ExtensionZone]]
        cylindrical_symmetry : bool
        collision_index : int
        origin : Tuple[float, float] - The origin for the
                 coordinate system.
        """
        self.x = x_cells
        self.y = y_cells
        self._origin = origin
        self.cell_size = cell_size
        self._x_range = None
        self._y_range = None
        self._cells = None
        self._material_meshes = None
        self._velocities = None
        self._extension_zones = extension_zones
        self._extension_factor = None
        self._y_physical_length = None
        self._x_physical_length = None
        self.cylindrical_symmetry = cylindrical_symmetry
        self._collision = collision_index

    def __str__(self):
        ezs = {Region.NORTH: 'N', Region.EAST: 'E',
               Region.WEST: 'W', Region.SOUTH: 'S'}
        extensions = [ezs[z.region] for z in self.extension_zones]
        s = '[' + ','.join(extensions) + ']'
        return f'PySALEMesh({self.x}, {self.y}, {self.cell_size}, ' \
               f'extension_zones={s}, origin={self.origin})'

    @property
    def origin(self) -> Tuple[int, int]:
        """The origin coordinate of the mesh.

        All dimensions are relative to this coordinate.
        Defaults to (0, 0). This coordinate is in the coordinate system
        of the mesh before the origin is applied. This has the 0, 0
        at the bottom-left corner of the *high-resolution* zone.

        Returns
        -------
        origin : Tuple[int, int]
        """
        return self._origin

    @classmethod
    def from_dimensions(cls, dimensions: Tuple[float, float],
                        cell_size: float,
                        extensions: Optional[List[ExtensionZone]] = None,
                        origin: Tuple[float, float] = (0., 0.)) \
            -> 'PySALEMesh':
        """Given high-res zone dimensions and cell size, return PySALEMesh.

        Parameters
        ----------
        dimensions : Tuple[float, float] X - Y Dimensions of the high-res
                     region in metres
        cell_size : float Dimension of a high-res cell in the mesh
        extensions : List[ExtensionZone] List of all the extension zones
                     that should be applied
        origin : Tuple[float, float] The coordinate to be considered the
                 origin. This coordinate is in the same coordinate
                 system as the default, where the origin is the bottom
                 left of the high-res zone.

        Returns
        -------
        PySALEMesh instance.
        """
        x_cells = round(dimensions[0] / cell_size)
        y_cells = round(dimensions[1] / cell_size)
        mesh = cls(x_cells, y_cells, cell_size,
                   extension_zones=extensions,
                   origin=origin)
        return mesh

    def _find_extension_factor(self):
        if self._extension_zones:
            assert all([e.cell_size == self.cell_size
                        for e in self._extension_zones]), \
                "All extension zones must have the same cell size!"
            self._extension_factor = self._extension_zones[0].factor
        else:
            self._extension_factor = \
                ExtensionZoneFactor(1., self.cell_size)

    @property
    def x_physical(self):
        """The physical x-length of the mesh.

        Returns
        -------
        length : float
        """
        if self._x_physical_length is None:
            self._populate_n_range()
        return self._x_physical_length

    @property
    def y_physical(self):
        """The physical y-length of the mesh.

        Returns
        -------
        length : float
        """
        if self._y_physical_length is None:
            self._populate_n_range()
        return self._y_physical_length

    @property
    def objresh(self) -> int:
        """iSALE input parameter; half the **height** of the mesh.

        Despite being the "horizontal" object resolution this refers to
        the height of the mesh. This is because there IS an OBJRESV
        but you only need to use it if your object does not cover
        the full width of the mesh. If no OBJRESV is present, its value
        defaults to OBJRESH. When using PySALESetup-created input files
        we never want anything less than the full width of the mesh so
        it is simpler to leave it out and use OBJRESH instead. In
        PySALESetup you can easily create objects of any size or shape
        you want!

        Notes
        -----
        If the number of cells is not divisible by 2, this property will
        guarantee that the returned value is rounded up, rather than
        down.

        E.g. a mesh width of 100 would return a value of 50.
        A mesh width of 99 would *also* return a value of 50.
        98 would return a value of 49, and so on.

        Returns
        -------
        objresh : int

        """
        if self.y % 2 == 0:
            objresh = int(self.y / 2)
        else:
            objresh = int((self.y // 2) + 1)
        return objresh

    @property
    def vertical_offset(self) -> int:
        """Half the vertical depth of the mesh, rounded down, in cells.

        Returns
        -------
        offset : int

        """
        if self.y % 2 == 0:
            offset = int(self.y / 2)
        else:
            offset = int((self.y-1) / 2)
        return offset

    @property
    def max_cell_size(self) -> float:
        """Return the maximum allowed cell size according to extensions.

        No extensions returns a max cell size identical to the
        mesh cell size.

        Returns
        -------
        max_cell_size : float
        """
        max_cell_size = self.cell_size
        if self.extension_zones:
            max_cell_size = self.extension_factor.max_cell_size
        return max_cell_size

    @property
    def collision_site(self) -> int:
        """The vertical collision location in the mesh, in cells.

        Defaults to 0.

        Returns
        -------
        collision_site : int

        """
        return self._collision

    @collision_site.setter
    def collision_site(self, value: int):
        self._collision = value

    @property
    def x_range(self) -> np.ndarray:
        """Array of the cell x-positions in the mesh.

        Returns
        -------
        x_range : float
        """
        if self._x_range is None:
            self._populate_n_range()
            self._set_origin()
        return self._x_range

    @property
    def y_range(self) -> np.ndarray:
        """Array of the cell y-positions in the mesh.

        Returns
        -------
        x_range : float
        """
        if self._y_range is None:
            self._populate_n_range()
            self._set_origin()
        return self._y_range

    def _set_origin(self):
        self._y_range -= self._origin[1]
        self._x_range -= self._origin[0]
        return

    def get_geometric_centre(self) -> Tuple[float, float]:
        """Return the geometric centre of the mesh in physical coords.

        Returns
        -------
        centre : Tuple[float, float]
        """
        x = np.ptp(self.x_range)*.5 + self.x_range[0]
        y = np.ptp(self.y_range)*.5 + self.y_range[0]
        return x, y

    @property
    def material_meshes(self) -> Dict[int, np.ndarray]:
        """Dictionary of numpy arrays representing material fill,
        indexed by material number.

        Returns
        -------
        meshes : Dict[int, np.ndarray]
        """
        if self._material_meshes is None:
            self._populate_material_meshes()
        return self._material_meshes

    @property
    def velocities(self) -> Dict[str, np.ndarray]:
        """Velocity arrays in the mesh in a dict indexed by axis.

        Returns
        -------
        velocities : Dict[str, np.ndarray]
        """
        if self._velocities is None:
            self._populate_velocities()
        return self._velocities

    def _populate_n_range(self):
        # funcs. they are a bit off. I think y-range works now
        # but x-range still broke. They should be combined probably
        y_length = self.y
        x_length = self.x
        self._y_physical_length = self.y * self.cell_size
        self._x_physical_length = self.x * self.cell_size
        zones = {zone.region: zone for zone in self.extension_zones}
        highres_xstart = 0
        highres_ystart = 0
        highres_xend = self.x
        highres_yend = self.y
        south_range = [-0.5 * self.cell_size]
        west_range = [-0.5 * self.cell_size]
        if Region.SOUTH in zones:
            zone = zones[Region.SOUTH]
            y_length += zone.depth
            highres_ystart = zone.depth
            highres_yend += zone.depth
            south_range = self._insert_extension_zone(zone)
        if Region.WEST in zones:
            zone = zones[Region.WEST]
            x_length += zone.depth
            highres_xstart = zone.depth
            highres_xend += zone.depth
            west_range = self._insert_extension_zone(zone)
        if Region.NORTH in zones:
            zone = zones[Region.NORTH]
            y_length += zone.depth
            highres_yend = highres_ystart + self.y + 1
            north_range = self._insert_extension_zone(zone)
        if Region.EAST in zones:
            zone = zones[Region.EAST]
            x_length += zone.depth
            highres_xend = highres_xstart + self.x + 1
            east_range = self._insert_extension_zone(zone)

        self._y_range = np.zeros((y_length))
        self._x_range = np.zeros((x_length))
        highres_yend_pos = (self.y+.5) * self.cell_size
        highres_xend_pos = (self.x+.5) * self.cell_size
        if Region.SOUTH in zones:
            self._y_range[:highres_ystart] = south_range
            highres_ystart_pos = np.amax(south_range)
            highres_yend_pos += highres_ystart_pos
        if Region.WEST in zones:
            self._x_range[:highres_xstart] = west_range
            highres_xstart_pos = np.amax(west_range)
            highres_xend_pos += highres_xstart_pos

        self._y_range[highres_ystart:highres_yend] = \
            self._generate_highres_zone(highres_yend,
                                        highres_ystart,
                                        south_range)
        self._x_range[highres_xstart:highres_xend] = \
            self._generate_highres_zone(highres_xend,
                                        highres_xstart,
                                        west_range)
        if Region.NORTH in zones:
            self._y_range[highres_yend-1:] = north_range
        if Region.EAST in zones:
            self._x_range[highres_xend-1:] = east_range
        return self._x_range, self._y_range

    def _generate_highres_zone(self,
                               highres_end,
                               highres_start,
                               range_):
        highres_zone = [np.amax(range_) + i * self.cell_size
                        for i in range(1, highres_end-highres_start+1)]
        return np.array(highres_zone)

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
    def cells(self) -> List[Cell]:
        """List of all Cell objects in the mesh.

        The mesh is represented by a collection of Cell objects,
        each of which represents a single cell in the mesh. These Cell
        objects are namedtuples containing all the information needed
        about that cell, including its indices, geometric centre,
        velocity, and material.

        Returns
        -------
        cells : List[Cell]
        """
        if self._cells is None:
            self._populate_cells()
        return self._cells

    def _populate_cells(self):
        self._cells = [Cell(Point(x, y), i, j, None, Velocity(0., 0.))
                       for i, x in enumerate(self.x_range)
                       for j, y in enumerate(self.y_range)]

    def project_polygons_onto_mesh(self,
                                   polygons: List[PySALEObject]) -> None:
        """Project a polygon (and all its children) onto the mesh.

        Method calls itself recursively on all children of the polygon.
        The children at the bottom of the hierachy get priority. Once
        a cell is populated with material, new material will NOT
        overwrite it.

        Parameters
        ----------
        polygons : List[PySALEObject]

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
        >>> domain.fill_with_random_grains_to_threshold(circle, 40)
        >>> domain.optimise_materials([2, 3, 4, 5])
        >>> mesh = PySALEMesh(100, 100, cell_size=.1)
        >>> mesh.project_polygons_onto_mesh([main])
        >>> mesh.plot_materials()
        >>> plt.show()
        """
        for i, cell in enumerate(self.cells):
            if cell.material is None:
                self._project_polygons_onto_cell(cell, polygons)

    def _project_polygons_onto_cell(self, cell: Cell, polygons):
        for polygon in polygons:
            if cell.point.within(polygon):
                if polygon.children:
                    self._project_polygons_onto_cell(
                        cell,
                        polygon.children
                    )
                if cell.material is None:
                    self._fill_cell(cell, polygon)
                    break

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
        if geometry.material == 0:
            self._void_cell(cell)
        else:
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
        cell.material = 0
        cell.velocity = Velocity(0., 0.)

    def plot_cells(self, ax: plt.Axes = None):
        """Plot the cell centres of the mesh.

        Parameters
        ----------
        ax : plt.Axes

        Returns
        -------
        fig, ax : Tuple[plt.Axes, plt.figure]

        """
        ax, fig = get_figure_from_ax(ax)
        xi, yi = np.meshgrid(self.x_range, self.y_range)
        ax.scatter(xi, yi, marker='.', color='k')
        self._set_plot_lims_and_labels(ax)
        ax.set_title('Cell centres')
        return fig, ax

    def plot_materials(self, ax: plt.Axes = None,
                       cmap: str = 'rainbow') -> Tuple[plt.Figure,
                                                       plt.Axes]:
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

        >>> from PySALESetup import PySALEObject
        >>> from PySALESetup import PySALEMesh
        >>> import matplotlib.pyplot as plt
        >>> impactor = PySALEObject.generate_ellipse([5., 8.], 2., 2., 0.)
        >>> impactor.set_material(1)
        >>> impactor.set_velocity(0. -1000.)
        >>> target = PySALEObject([(0, 0), (0, 6), (10, 6), (10, 0)])
        >>> target.set_material(3)
        >>> mesh = PySALEMesh(100, 100, cell_size=.1)
        >>> mesh.project_polygons_onto_mesh([impactor, target])
        >>> mesh.plot_materials()
        >>> plt.show()
        """
        ax, fig = get_figure_from_ax(ax)
        xi, yi = np.meshgrid(self.x_range, self.y_range)
        for i in range(1, 9+1):
            matter = np.copy(self.material_meshes[i])*i
            matter = np.ma.masked_where(matter == 0., matter)
            p = ax.pcolormesh(xi,
                              yi,
                              matter.T,
                              cmap=cmap,
                              vmin=1,
                              vmax=9,
                              shading='auto')
        self._set_plot_lims_and_labels(ax)
        self._add_colorbar(ax, p, 'Material No.')
        ax.set_title('Materials')
        return fig, ax

    @staticmethod
    def _add_colorbar(ax, graph_object, label):
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax, fig = get_figure_from_ax(ax)
        cb = fig.colorbar(graph_object, cax=cax)
        cb.set_label(label)
        return cb

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

        >>> from PySALESetup import PySALEObject
        >>> from PySALESetup.mesh import PySALEMesh
        >>> import matplotlib.pyplot as plt
        >>> impactor = PySALEObject.generate_ellipse([5., 8.], 2., 2., 0.)
        >>> impactor.set_material(1)
        >>> impactor.set_velocity(0. -1000.)
        >>> target = PySALEObject([(0, 0), (0, 6), (10, 6), (10, 0)])
        >>> target.set_material(3)
        >>> mesh = PySALEMesh(100, 100, cell_size=.1)
        >>> mesh.project_polygons_onto_mesh([impactor, target])
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
            p = ax.pcolormesh(xi, yi, v.T, cmap=cmap,
                              vmin=np.amin(v),
                              vmax=np.amax(v),
                              shading='auto')
            self._set_plot_lims_and_labels(ax)
            self._add_colorbar(ax, p, 'Velocity [m/s]')
        ax1.set_title('Velocity - x')
        ax2.set_title('Velocity - y')
        return fig1, fig2, ax1, ax2

    @staticmethod
    def _cell_to_row(cell: Cell, materials: Iterable[int]) -> str:
        material_columns = ['1.000' if cell.material == m
                            else '0.000' for m in materials]
        row = f'{cell.i} {cell.j} ' \
              + ' '.join(material_columns) \
              + f' {cell.velocity.x:.2f} {cell.velocity.y:.2f}\n'
        return row

    @property
    def material_numbers(self) -> List[int]:
        """List of non-zero materials in the mesh.

        Returns
        -------
        numbers : List[int]
        """
        return [key for key, value in self.material_meshes.items()
                if np.sum(value) > 0.]

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
        if compress:
            if file_name.suffix != '.gz':
                raise NameError(f'Mesh is being compressed but file '
                                f'name "{file_name}" does not have a '
                                f'.gz extension.')
            with gzip.open(file_name, 'wt') as f:
                self._write_mesh_to_file(cell_number,
                                         f,
                                         self.material_numbers)
        else:
            with open(file_name, 'w') as f:
                self._write_mesh_to_file(cell_number,
                                         f,
                                         self.material_numbers)

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

    def _insert_extension_zone(self, zone: ExtensionZone):
        factor = self.extension_factor.multiplier
        max_size = self.extension_factor.max_cell_size
        varying_cell_size = self.cell_size
        if zone.region in [Region.NORTH, Region.EAST]:
            half = 'north/east'
            if zone.region == Region.NORTH:
                position = self._y_physical_length + self.cell_size*.5
            else:
                position = self._x_physical_length + self.cell_size*.5
        else:
            half = 'south/west'
            position = -0.5*self.cell_size

        varying_cell_size, coord = self._create_extension_zone_coordinates(
            factor, max_size, [position], varying_cell_size, zone,
            half)

        if half == 'north/east':
            range_ = np.array(coord)
            if zone.region == Region.NORTH:
                self._y_physical_length += np.ptp(range_)
            else:
                self._x_physical_length += np.ptp(range_)
            return range_
        else:
            range_ = np.array(coord)
            return range_[::-1]

    def _insert_north_zone(self, zone: ExtensionZone):
        factor = self.extension_factor.multiplier
        max_size = self.extension_factor.max_cell_size
        varying_cell_size = self.cell_size
        position = self._y_physical_length + self.cell_size*.5

        varying_cell_size, y_coord = self._create_extension_zone_coordinates(
            factor, max_size, [position], varying_cell_size, zone,
            'north/east')

        north_y_range = np.array(y_coord)
        self._y_physical_length += np.ptp(north_y_range)
        return north_y_range

    def _insert_east_zone(self, zone: ExtensionZone):
        factor = self.extension_factor.multiplier
        max_size = self.extension_factor.max_cell_size
        varying_cell_size = self.cell_size
        position = self._x_physical_length + self.cell_size*.5

        varying_cell_size, x_coord = self._create_extension_zone_coordinates(
            factor, max_size, [position], varying_cell_size, zone,
            'north/east')

        east_x_range = np.array(x_coord)
        self._x_physical_length += np.ptp(east_x_range)
        return east_x_range

    def _insert_west_zone(self, zone: ExtensionZone):
        factor = self.extension_factor.multiplier
        max_size = self.extension_factor.max_cell_size
        varying_cell_size = self.cell_size
        position = -0.5*self.cell_size

        varying_cell_size, x_coord = self._create_extension_zone_coordinates(
            factor, max_size, [position], varying_cell_size, zone,
            'south/west')

        west_x_range = np.array(x_coord)
        return west_x_range[::-1]

    def _insert_south_zone(self, zone: ExtensionZone):
        factor = self.extension_factor.multiplier
        max_size = self.extension_factor.max_cell_size
        varying_cell_size = self.cell_size
        position = -0.5*self.cell_size

        varying_cell_size, y_coord = self._create_extension_zone_coordinates(
            factor, max_size, [position], varying_cell_size, zone,
            'south/west')
        south_y_range = np.array(y_coord)
        # south_y_range += abs(np.amin(south_y_range)) + varying_cell_size
        # self._y_physical_length += np.amax(south_y_range)
        return south_y_range[::-1]

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

    @property
    def extension_zones(self) -> List[ExtensionZone]:
        """The extension zones applied to the mesh.

        Returns
        -------
        zones : List[ExtensionZone]
        """
        if self._extension_zones is not None:
            self._find_extension_factor()
            return self._extension_zones
        else:
            return []

    @property
    def extension_factor(self) -> ExtensionZoneFactor:
        """The ExtensionZoneFactor associated with this mesh.

        There can only be one extension factor associated with a mesh.
        When this property is called it also checks that the given
        extension zones don't have clashing properties.

        Returns
        -------
        factor : ExtensionZoneFactor
        """
        if self._extension_factor is None:
            self._find_extension_factor()
        return self._extension_factor


