"""
Object Hierarchy Demo
=====================

The geometry of the simulation is built like a tree graph. There is a root (
or roots) at the top, a host object, and then child objects can be added to
the host, and those children can have children and so on. The top level is
considered to be at depth = 0. The grandchild of a root object would be at
depth = 2.

Objects at the lowest depth are placed into the mesh first.
Objects are inserted in the order their parents were added when they are at
the same depth. Then they are inserted in the order *they* were added when
these objects have the same parent at the same depth.
"""
import PySALESetup as pss
import matplotlib.pyplot as plt


####################################################
# In this example we create two identical circles and give them an offset to
# demonstrate how this works.
#
top_circle = pss.PySALEObject.generate_ellipse((0,0), 100, 100, rotation=0.,
                                               material=1)
top_circle.add_child(pss.PySALEObject.generate_ellipse((0,0), 70, 70,
                                                       rotation=0.,
                                                       material=2))
top_circle.add_child(pss.PySALEObject.generate_ellipse((50,50), 70, 70,
                                                       rotation=0.,
                                                       material=3))

for child in top_circle.children:
    child.add_child(
        pss.PySALEObject.generate_ellipse(child.polygon.centroid,
                                          major=10,
                                          minor=10,
                                          rotation=0.,
                                          material=4)
    )


for child in top_circle.children:
    for child2 in child.children:
        child2.add_child(pss.PySALEObject.generate_ellipse(
            child2.polygon.centroid, major=5, minor=5, rotation=0, material=5))

####################################################
# Finally lets plot this situation.
#

fig, ax = top_circle.plot()
plt.show()

####################################################
# Create our mesh and fill in the objects. See how the mesh is populated.
#

mesh = pss.PySALEMesh.from_dimensions((200, 200),
                                      cell_size=1.,
                                      origin=(-100., -100.))
mesh.project_polygons_onto_mesh([top_circle])
fig2, ax2 = mesh.plot_materials()

plt.show()