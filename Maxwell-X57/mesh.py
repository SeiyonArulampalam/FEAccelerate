"""A asimple model to experiemnt with splitting the airgap for an electric motor"""

import gmsh
import numpy as np
import sys

"""Define the rotation angle for the rotor"""
rotor_span = np.deg2rad(45)  #
rot = np.deg2rad(0)

lc = 1e-3
airgap = 1e-3
r_rotor = 50e-3
gmsh.initialize(sys.argv)
gmsh.option.setNumber("General.Terminal", 1)  # 0 = no print, 1 = print
gmsh.model.add("motor")

# Points that define the rotor
gmsh.model.geo.addPoint(0, 0, 0, lc, 0)
gmsh.model.geo.addPoint(r_rotor, 0, 0, lc, 1)  # top boundary
gmsh.model.geo.addPoint(0, r_rotor, 0, lc, 2)  # bottom boundary

# Points that define the stator

# Draw the arcs of the rotor
line1 = gmsh.model.geo.addLine(0, 1)
line2 = gmsh.model.geo.addLine(0, 2)
arc1 = gmsh.model.geo.addCircleArc(1, 0, 2)

# Required to call synchronize in order to be meshed
gmsh.model.geo.synchronize()

# Generate the mesh
gmsh.model.mesh.generate(2)

# Launch the GUI to see the results:
if "-nopopup" not in sys.argv:
    gmsh.fltk.run()
