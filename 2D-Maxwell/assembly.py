from tabulate import tabulate
import numpy as np
import scipy as sp
from numba import njit, cuda
import time
import geometry as gm
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib as mpl
from rich.console import Console

np.set_printoptions(precision=4, linewidth=200)


def contour_mpl(xyz_nodeCoords, z, title="fig", fname="contour.jpg", flag=False):
    """
    Create a contour plot of the solution.
    Inputs:
        xyz_nodeCoords = [x,y,z] node positions
        z = solution field vector

    Parameters
    ----------
    xyz_nodeCoords : 3d np array
        [x,y,z]
    z : 1d array
        solution vector to plot at each xyz position
    title : str, optional
        figure title, by default "fig"
    fname : str, optional
        name of figure make sure to include extension (.jpg), by default "contour.jpg"
    flag : bool, optional
        save figure, by default False
    """
    min_level = min(z)
    max_level = max(z)
    levels = np.linspace(min_level, max_level, 30)
    x = xyz_nodeCoords[:, 0]
    y = xyz_nodeCoords[:, 1]

    # create a Delaunay triangultion
    tri = mtri.Triangulation(x, y)
    # ntri = tri.triangles.shape[0]

    # Refine the data
    refiner = mtri.UniformTriRefiner(tri)
    # tri_refi, z_refi = refiner.refine_field(z, subdiv=3)

    # Defin colormap
    # cmap = cmasher.guppy_r
    cmap = "coolwarm"
    cmap = "jet"

    # Plot solution
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    # plot = ax.tricontour(tri_refi, z_refi, levels=levels, cmap=cmap)
    plot = ax.tricontour(tri, z, levels=levels, cmap=cmap)
    plot = ax.tricontourf(tri, z, levels=levels, cmap=cmap)
    # ax.set_aspect("equal", adjustable="box")
    # ax.set_title(title, fontsize=10)

    norm = mpl.colors.Normalize(vmin=min_level, vmax=max_level)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, location="right"
    )

    cbar.set_ticks([min(z), (min(z) + max(z)) / 2.0, max(z)])
    # cbar.set_ticklabels([mn,md,mx])

    fig.tight_layout()

    if flag == True:
        plt.savefig(fname, dpi=800, edgecolor="none")

    return


@njit
def get_element_props(element, connectivity_array, xyz_nodeCoords):
    # Extract the material coefficients
    alpha = connectivity_array[element, 5]

    # Extract the magnetization value for the element
    magnetization = connectivity_array[element, 6]

    # i, j, m are the global node tags for each element
    i = int(connectivity_array[element, 2])
    j = int(connectivity_array[element, 3])
    m = int(connectivity_array[element, 4])

    # Global location of n(1,element)
    xi = xyz_nodeCoords[i - 1][0]
    yi = xyz_nodeCoords[i - 1][1]

    # Global location of n(1,element)
    xj = xyz_nodeCoords[j - 1][0]
    yj = xyz_nodeCoords[j - 1][1]

    # Global location of n(1,element)
    xm = xyz_nodeCoords[m - 1][0]
    ym = xyz_nodeCoords[m - 1][1]

    # Element coefficients
    b1_e = yj - ym
    b2_e = ym - yi
    b3_e = yi - yj
    b_e = np.array([b1_e, b2_e, b3_e])

    c1_e = xm - xj
    c2_e = xi - xm
    c3_e = xj - xi
    c_e = np.array([c1_e, c2_e, c3_e])

    # area of element
    area_e = (b1_e * c2_e - b2_e * c1_e) / 2.0

    return (alpha, magnetization, area_e, b_e, c_e)


@njit
def assemble_K_and_b_standard(K, b, connectivity_array, xyz_nodeCoords, numElems):
    for e in range(numElems):
        (
            alpha,
            magnetization_x,
            area_e,
            b_e,
            c_e,
        ) = get_element_props(
            e,
            connectivity_array,
            xyz_nodeCoords,
        )

        # Update the global K matrix
        for row in range(3):
            for col in range(3):
                K_row_col = (
                    alpha * b_e[row] * b_e[col] + alpha * c_e[row] * c_e[col]
                ) / (4 * area_e)
                n_row_e = int(connectivity_array[e, row + 2])  # global row index
                n_col_e = int(connectivity_array[e, col + 2])

                K[n_row_e - 1, n_col_e - 1] += K_row_col

        # Update the global b vector
        for row in range(3):
            if magnetization == 0:
                b_local = 0
            else:
                b_local = (
                    (magnetization_x / (2.0)) * (c_e[row]) * (4 * np.pi * 1e-7) * alpha
                )
            n_row_e = int(connectivity_array[e, row + 2])
            b[n_row_e - 1] += b_local
    return K, b


@njit
def get_BC_nodes(nodeTags, xyz_nodeCoords, B, H):
    nodes_BC = []
    for i in range(len(nodeTags)):
        tag = nodeTags[i]
        x = xyz_nodeCoords[i, 0]
        y = xyz_nodeCoords[i, 1]
        if (-0.5 * B <= x <= 0.5 * B) and y == -H * 0.5:
            # bottom edge boundary
            nodes_BC.append(tag)

        elif -0.5 * B <= x <= 0.5 * B and y == H * 0.5:
            # top edge of boundary
            nodes_BC.append(tag)

        elif x == B * 0.5 and -0.5 * H <= y <= 0.5 * H:
            # right edge of boundary
            nodes_BC.append(tag)

        elif x == -B * 0.5 and -0.5 * H <= y <= 0.5 * H:
            # left edge of boundary
            nodes_BC.append(tag)

    return nodes_BC


@njit
def applyDirichletBC(nodes_BC, K, b):
    for i in range(len(nodes_BC)):
        nd_i = int(nodes_BC[i] - 1)
        for j in range(len(b)):
            if nd_i == j:
                K[j, j] = 1.0
                b[j] = 0.0

            else:
                b[j] -= K[j, nd_i] * 0.0
                K[nd_i, j] = 0.0
                K[j, nd_i] = 0.0
    return K, b


if __name__ == "__main__":
    # Start timer
    t0 = time.time()

    # Define geometry of the model
    B = 200
    H = 100
    l1 = 20
    l2 = 20
    d1 = 30
    d2 = 30
    lc = 8
    lc1 = 0.5
    mu_r_mag = 1.04
    magnetization = 1e6

    mesh_time_arr = []
    prop_time_arr = []
    sys_time_arr = []
    getBC_time_arr = []
    applyBC_time_arr = []
    solve_time_arr = []
    for i in range(6):
        """Generate the mesh and create the connectivity arrays"""
        # Generate the mesh using GMSH
        t0_mesh = time.perf_counter()
        nodeTags, xyz_nodeCoords, elemTags, elemNodeTags = gm.create_mesh(
            B=B,
            H=H,
            l1=l1,
            l2=l2,
            d1=d1,
            d2=d2,
            lc=lc,
            lc1=lc1,
            # flag=True,
        )
        tf_mesh = time.perf_counter()

        numNodes = len(nodeTags)
        numElems = len(elemTags)

        # Create the connectivity array
        t0_mesh_prop = time.perf_counter()
        connectivity_array = gm.get_mesh_props(
            nodeTags=nodeTags,
            xyz_nodeCoords=xyz_nodeCoords,
            elemTags=elemTags,
            elemNodeTags=elemNodeTags,
            l1=l1,
            l2=l2,
            d1=d1,
            d2=d2,
        )
        tf_mesh_prop = time.perf_counter()

        """Assemble K and b"""
        t0_sys = time.perf_counter()
        K = np.zeros((numNodes, numNodes))
        b = np.zeros(numNodes)
        K, b = assemble_K_and_b_standard(
            K,
            b,
            connectivity_array,
            xyz_nodeCoords,
            numElems,
        )
        tf_sys = time.perf_counter()

        """Get nodes on edge of the boundary"""
        t0_getBC = time.perf_counter()
        nodes_BC = get_BC_nodes(nodeTags, xyz_nodeCoords, B, H)
        tf_getBC = time.perf_counter()

        """Apply Dirichlet BC"""
        t0_applyBC = time.perf_counter()
        K, b = applyDirichletBC(nodes_BC, K, b)
        tf_applyBC = time.perf_counter()

        """Solve System of Equation"""
        t0_solve = time.perf_counter()
        x = np.linalg.solve(K, b)
        tf_solve = time.perf_counter()

        """Plot solution field"""
        # contour_mpl(xyz_nodeCoords, x)
        # plt.show()
        # exit()

        # Save times to list
        time_gmsh = tf_mesh - t0_mesh
        time_mesh_prop = tf_mesh_prop - t0_mesh_prop
        time_sys = tf_sys - t0_sys
        time_getBC = tf_getBC - t0_getBC
        time_applyBC = tf_applyBC - t0_applyBC
        time_solve = tf_solve - t0_solve

        mesh_time_arr.append(time_gmsh)
        prop_time_arr.append(time_mesh_prop)
        sys_time_arr.append(time_sys)
        getBC_time_arr.append(time_getBC)
        applyBC_time_arr.append(time_applyBC)
        solve_time_arr.append(time_solve)

    tf = time.time()

    print(f"Total Number of Nodes : {len(nodeTags)}")
    print(f"Total Number of Elements : {len(elemTags)}")
    print(f"Total Time = {tf-t0:.4f} s")
    headers = ["Iteration", "1", "2", "3", "4", "5", "6"]
    table = [
        ["GMSH"] + mesh_time_arr,
        ["Mesh Prop."] + prop_time_arr,
        ["Kx = b"] + sys_time_arr,
        ["Get BCs Nodes"] + getBC_time_arr,
        ["Apply BC"] + applyBC_time_arr,
        ["Solve Time"] + solve_time_arr,
    ]

    console = Console()
    console.print(
        tabulate(table, headers, tablefmt="fancy_outline", floatfmt="3e"),
        style="bold yellow",
    )
