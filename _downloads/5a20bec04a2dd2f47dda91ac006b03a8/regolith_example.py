"""
Regolith Sample Demo
====================

Lunar Regolith example simulation taken from JGD's 2018 Thesis.

Create two impactors with non-standard grain shapes, orientations, and sizes.
"""
import PySALESetup as pss
import matplotlib.pyplot as plt


regolith_top = pss.PySALEObject([(0, 0.003), (0.00125, 0.003),
                                 (0.00125, 0.006), (0, 0.006)])
regolith_bot = pss.PySALEObject([(0, 0), (0.00125, 0),
                                 (0.00125, 0.003), (0, 0.003)])

regolith_top.set_material(1)
regolith_bot.set_material(1)

top_domain = pss.PySALEDomain(regolith_top)
bot_domain = pss.PySALEDomain(regolith_bot)

####################################################
# Fill our two domains with grains, but let's get a random geometry
# from the grain library to use.
#

grain_library = pss.GrainLibrary()
grain_library.load_builtin_library()
grain = grain_library.random_grain()

###################################################
# It's even possible to set the distribution from which the grains are taken
# ensuring there is the right variation of grains available.
#

for domain in [top_domain, bot_domain]:
    domain.fill_with_random_grains_to_threshold(
        grain,
        50.,
        rotation_distribution=pss.PySALEUniformDistribution((0.,
                                                             360.)),
        size_distribution=pss.PySALENormalDistribution(mu=5e-5,
                                                       sigma=1e-5)
    )
    domain.optimise_materials([2, 3, 4, 5, 6, 7, 8, 9])


regolith_top.set_velocity(0, -1500.)
regolith_bot.set_velocity(0, +1500.)

###################################################
# Finally we can create the whole mesh and populate it
# The commands for this are simple but execution can take some time!
#

mesh = pss.PySALEMesh.from_dimensions((0.00125, 0.006),
                                      5.e-6)
print(mesh)
mesh.project_polygons_onto_mesh([regolith_top, regolith_bot])
f, a = mesh.plot_materials()

plt.show()

###################################################
# Let's also plot the velocities for completion's sake.
#
f1, f2, a1, a2 = mesh.plot_velocities()

plt.show()
