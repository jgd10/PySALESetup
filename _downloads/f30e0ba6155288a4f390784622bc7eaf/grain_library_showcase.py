"""
Grain Library Demo
==================

Demos selection of Grain geometries available within the library.
"""

from PySALESetup import GrainLibrary
import matplotlib.pyplot as plt


grain_library = GrainLibrary()
grain_library.load_builtin_library()
fig = plt.figure()

for i in range(1, 25+1):
    ax = fig.add_subplot(5, 5, i)
    grain = grain_library.random_grain()
    grain.plot(ax)

fig.tight_layout()
plt.show()
