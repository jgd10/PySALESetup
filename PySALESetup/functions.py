from typing import List, Optional, Tuple
from .mesh import ExtensionZone, PySALEMesh


def mesh_from_dimensions(dimensions: Tuple[float, float],
                         cell_size: float,
                         extensions: Optional[List[ExtensionZone]] = None) -> PySALEMesh:
    """Given high-res zone dimensions and cell size, return PySALEMesh.

    Parameters
    ----------
    dimensions : Tuple[float, float] X - Y Dimensions of the high-res
                 region in metres
    cell_size : float Dimension of a high-res cell in the mesh
    extensions : List[ExtensionZone] List of all the extension zones
                 that should be applied

    Returns
    -------
    PySALEMesh instance.
    """
    x_cells = int(dimensions[0] / cell_size)
    y_cells = int(dimensions[1] / cell_size)
    mesh = PySALEMesh(x_cells, y_cells, cell_size,
                      extension_zones=extensions)
    return mesh
