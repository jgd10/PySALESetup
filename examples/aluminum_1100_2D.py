"""
Aluminum 1100 2D
================

This example is based on the iSALE example Aluminum 1100 2D and uses
the same basic geometry, showing how you can use it for cylindrically
symmetric simulations.
"""
import PySALESetup as pss
import matplotlib.pyplot as plt

cell_size = 1.5875e-4
projectile = pss.PySALEObject.generate_ellipse([0., 169.*cell_size],
                                           20.*cell_size,
                                           20.*cell_size,
                                           0.)
target = pss.PySALEObject(([(0., 0.),
                        (200*cell_size, 0),
                        (200*cell_size, 149*cell_size),
                        (0., 149.*cell_size)]))

projectile.set_material(1)
target.set_material(2)

projectile.set_velocity(0., -7e3)

fig, ax = projectile.plot()
target.plot(ax=ax)
ax.set_aspect(1)
plt.show()

mesh = pss.PySALEMesh(200, 240, cell_size)
mesh.project_polygons_onto_mesh([projectile, target])
mesh.plot_materials()
mesh.plot_velocities()
plt.show()