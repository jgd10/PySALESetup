import pathlib
from string import Template
from collections import namedtuple
from typing import Dict
from .constants import ASTEROID_TEMPLATE_PATH, ADDITIONAL_TEMPLATE_PATH
from .functions import _convert_input_to_fortran_strings
from .mesh import PySALEMesh


TimeStep = namedtuple('TimeStep', ['initial', 'max', 'end', 'save'])


class InputFile:
    """Base input file class.

    """
    def __init__(self, type_: str):
        """Input file type must be supplied to the constructor.

        Parameters
        ----------
        type_ : str
        """
        self.type = type_
        self._template = None
        self._template_path = None

    @property
    def template_path(self) -> pathlib.Path:
        """Path to the template

        Returns
        -------
        path : pathlib.Path
        """
        if self._template_path is None:
            if self.type == 'asteroid':
                self._template_path = ASTEROID_TEMPLATE_PATH
            elif self.type == 'additional':
                self._template_path = ADDITIONAL_TEMPLATE_PATH
            else:
                raise TypeError(f'Invalid input file type supplied. '
                                f'Must be "asteroid" or "additional", '
                                f'not {self.type}')
        return self._template_path

    @property
    def template(self) -> Template:
        """The template string for the input file.

        Returns
        -------
        template : str
        """
        if self._template is None:
            with open(pathlib.Path(self.template_path), 'r') as f:
                self._template = Template(f.read())
        return self._template


class AsteroidInput(InputFile):
    """asteroid.inp file creator class.

    Allows a user to build an asteroid.inp file and uses a simple
    template to write one to a new file of the user's choice.
    """
    def __init__(self, model_name: str, timestep: TimeStep, mesh: PySALEMesh):
        """Construct the AsteroidInput class.

        Only model_name, timestep and mesh, need to be provided.
        However, multiple other properties are also set. These are:

            * surface_temperature = 300.
            * east_bc = 'FREESLIP'
            * west_bc = 'FREESLIP'
            * south_bc = 'OUTFLOW'
            * north_bc = 'OUTFLOW'

        But they can all be changed after initialisation.

        Parameters
        ----------
        model_name : str
        timestep : TimeStep
        mesh : PySALEMesh
        """
        super().__init__('asteroid')
        self.model_name = model_name
        self.timestep = timestep
        self.mesh = mesh
        self._template = None
        self.surface_temperature = 300.
        self.east_bc = 'FREESLIP'
        self.west_bc = 'FREESLIP'
        self.south_bc = 'OUTFLOW'
        self.north_bc = 'OUTFLOW'

    def _parse_mesh(self):
        zones = {'east': 0, 'west': 0, 'north': 0, 'south': 0}
        for z in self.mesh.extension_zones:
            zones[z.region.name.lower()] = z.depth
        return zones['east'], zones['west'], zones['north'], zones['south']

    def _perform_substitutions(self):
        east, west, north, south = self._parse_mesh()
        output = self.template.substitute(
            _convert_input_to_fortran_strings(
                {
                    'simulation_name': self.model_name,
                    'east_cells': east,
                    'west_cells': west,
                    'north_cells': north,
                    'south_cells': south,
                    'cell_size': self.mesh.cell_size,
                    'high_res_x': self.mesh.x,
                    'high_res_y': self.mesh.y,
                    'cyl_geometry': int(self.mesh.cylindrical_symmetry),
                    'collision_index': self.mesh.collision_site,
                    'objresh': self.mesh.objresh,
                    'vert_offset': self.mesh.vertical_offset,
                    'timestep': self.timestep.initial,
                    'timestep_max': self.timestep.max,
                    'time_end': self.timestep.end,
                    'save_interval': self.timestep.save,
                    'east_boundary_condition': self.east_bc,
                    'west_boundary_condition': self.west_bc,
                    'north_boundary_condition': self.north_bc,
                    'south_boundary_condition': self.south_bc,
                    'max_cell_size': self.mesh.extension_factor.max_cell_size,
                    'ext_factor': self.mesh.extension_factor.multiplier,
                    'surface_temperature': self.surface_temperature
                }
            )
        )
        return output

    def write_to(self,
                 path: pathlib.Path =
                 pathlib.Path.cwd() / 'asteroid.inp') -> None:
        """Write the asteroid.inp file to given path.

        Defaults to asteroid.inp in the current working directory.

        Parameters
        ----------
        path : pathlib.Path
        """
        with open(path, 'w') as f:
            f.write(self._perform_substitutions())


class AdditionalInput(InputFile):
    """additional.inp file creator class.

    Allows a user to build an additional.inp file and uses a simple
    template to write one to a new file of the user's choice.
    """
    def __init__(self, mesh: PySALEMesh,
                 material_names: Dict[int, str],
                 host_object_number: int = 1):
        """Initialise the AdditionalInput class.

        Parameters
        ----------
        mesh : PySALEMesh
        material_names : Dict[int, str]
        host_object_number : int
        """
        super(AdditionalInput, self).__init__('additional')
        self.mesh = mesh
        self.material_names = material_names
        self.host_number = host_object_number

    def _perform_substitutions(self):
        material_names = ' : '.join(self.material_names[i]
                                    for i in self.mesh.material_numbers)
        host_numbers = ' : '.join(str(self.host_number)
                                  for i in self.mesh.material_numbers)
        output = self.template.substitute(
            _convert_input_to_fortran_strings(
                {
                    'material_number': len(self.mesh.material_numbers),
                    'material_names': material_names,
                    'host_object_numbers': host_numbers
                }
            )
        )
        return output

    def write_to(self,
                 path: pathlib.Path =
                 pathlib.Path.cwd() / 'asteroid.inp') -> None:
        """Write the asteroid.inp file to given path.

        Defaults to additional.inp in the current working directory.

        Parameters
        ----------
        path : pathlib.Path
        """
        with open(path, 'w') as f:
            f.write(self._perform_substitutions())
