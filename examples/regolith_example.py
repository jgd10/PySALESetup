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

grain_library = pss.GrainLibrary()
grain_library.load_builtin_library()
grain = grain_library.random_grain()

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

mesh = pss.PySALEMesh.from_dimensions((0.00125, 0.006), 2.5e-6)
print(mesh)
mesh.project_polygons_onto_mesh([regolith_top, regolith_bot])

f, a = mesh.plot_materials()
#mesh.plot_velocities()

f.savefig('regolith_example.png', dpi=300)