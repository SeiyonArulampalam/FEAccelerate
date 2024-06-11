import gmsh
import sys

gmsh.initialize()
gmsh.model.add("model")


"""
 --------------------------------------------------------
|                                  (magnet 2)            |                  
|        (magnet 1)                  -------             |
|           ----      d1       d2   |       |            |
|       l1 |    | <--------|------->|       | l2         | H
|           ----                     -------             |
|             l1                       l2                |
|                                                        |
 --------------------------------------------------------
                            B
"""


def create_mesh(B=200, H=100, l1=20, l2=30, d1=50, d2=30, lc=5):
    """
    _summary_

    Parameters
    ----------
    B : float, optional
        base of model boundary, by default 200
    H : float, optional
        Height of model boundary, by default 100
    l1 : float, optional
        dimension of cube magnet 1, by default 20
    l2 : float, optional
        dimension of cube magnet 2, by default 30
    d1 : float, optional
         distance magnet 1 is to the left of x=0, y=0, z=0, by default 50
    d2 : float, optional
         distance magnet 1 is to the left of x=0, y=0, z=0, by default 30
    lc : float, optional
         mesh refinement parameter, by default 1
    """
    gmsh.initialize()
    gmsh.model.add("model")
    prob = gmsh.model.geo

    # Define the points for the main model boundary
    prob.addPoint(-B / 2, -H / 2, 0, lc, 1)
    prob.addPoint(B / 2, -H / 2, 0, lc, 2)
    prob.addPoint(B / 2, H / 2, 0, lc, 3)
    prob.addPoint(-B / 2, H / 2, 0, lc, 4)

    # Define points for magnet 1
    prob.addPoint(-d1 - l1, -l1 / 2, 0, lc, 5)
    prob.addPoint(-d1, -l1 / 2, 0, lc, 6)
    prob.addPoint(-d1, l1 / 2, 0, lc, 7)
    prob.addPoint(-d1 - l1, l1 / 2, 0, lc, 8)

    # Define points for magnet 2
    prob.addPoint(d2, -l2 / 2, 0, lc, 9)
    prob.addPoint(d2 + l2, -l2 / 2, 0, lc, 10)
    prob.addPoint(d2 + l2, l2 / 2, 0, lc, 11)
    prob.addPoint(d2, l2 / 2, 0, lc, 12)

    # Draw lines that define the outter boundary
    prob.addLine(1, 2, 1)
    prob.addLine(2, 3, 2)
    prob.addLine(3, 4, 3)
    prob.addLine(4, 1, 4)

    # Draw lines of magnet 1
    prob.addLine(5, 6, 5)
    prob.addLine(6, 7, 6)
    prob.addLine(7, 8, 7)
    prob.addLine(8, 5, 8)

    # Draw line of magnet 2
    prob.addLine(9, 10, 9)
    prob.addLine(10, 11, 10)
    prob.addLine(11, 12, 11)
    prob.addLine(12, 9, 12)

    # Define the curve loop for the model boundary
    prob.addCurveLoop([1, 2, 3, 4], tag=1, reorient=True)

    # Define the curve loop for the 1st magnet
    prob.addCurveLoop([5, 6, 7, 8], tag=2, reorient=True)

    # Define the curev loop for the 2nd magnet
    prob.addCurveLoop([9, 10, 11, 12], tag=3, reorient=True)

    # Define the model surface with holes for the magnets
    prob.addPlaneSurface([1, 2, 3], tag=1)
    prob.addPlaneSurface([2], tag=2)
    prob.addPlaneSurface([3], tag=3)

    # Required to call synchronize in order to be meshed
    gmsh.model.geo.synchronize()

    # Generate the mesh
    gmsh.model.mesh.generate(2)

    # Launch the GUI to see the results:
    if "-nopopup" not in sys.argv:
        gmsh.fltk.run()

    gmsh.finalize()


if __name__ == "__main__":
    create_mesh()
