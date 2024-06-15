from tabulate import tabulate
import numpy as np
from scipy import sparse
from numba import njit, cuda
import time
import geometry as gm
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib as mpl
from rich.console import Console
import cupy as cp
import torch
import cupyx.scipy.sparse.linalg as cpssl
from cupyx.scipy.sparse import csr_matrix, csc_matrix

np.set_printoptions(precision=4, linewidth=800)


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
    ax.set_title(title, fontsize=10)

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

                # if e == 0:
                #     print(n_row_e - 1, n_col_e - 1, K_row_col)

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


@cuda.jit
def _assemble_K_and_b(
    K_rows, K_cols, K_data, b_gpu, connectivity_array, xyz_nodeCoords
):
    # get the thread Id
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    # get the stride length (cuda.blockIdx.x * cuda.gridDim.x)
    stride = cuda.blockDim.x * cuda.gridDim.x  # cuda.grid_size(1)

    # print(stride)
    # exit()

    for e in range(idx, connectivity_array.shape[0], stride):
        # Extract the material coefficients
        alpha = connectivity_array[e, 5]

        # Extract the magnetization value for the element
        magnetization = connectivity_array[e, 6]

        # i, j, m are the global node tags for each element
        i = int(connectivity_array[e, 2])
        j = int(connectivity_array[e, 3])
        m = int(connectivity_array[e, 4])

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

        c1_e = xm - xj
        c2_e = xi - xm
        c3_e = xj - xi

        # area of element
        area_e = (b1_e * c2_e - b2_e * c1_e) * 0.5

        # Update the global K matrix
        index = e * 9  # starting index for rows, cols, and data
        for row in range(3):
            if row == 0:
                b_e_row = b1_e
                c_e_row = c1_e
            elif row == 1:
                b_e_row = b2_e
                c_e_row = c2_e
            elif row == 2:
                b_e_row = b3_e
                c_e_row = c3_e

            for col in range(3):
                if col == 0:
                    b_e_col = b1_e
                    c_e_col = c1_e
                elif col == 1:
                    b_e_col = b2_e
                    c_e_col = c2_e
                elif col == 2:
                    b_e_col = b3_e
                    c_e_col = c3_e

                K_row_col = (alpha * b_e_row * b_e_col + alpha * c_e_row * c_e_col) / (
                    4 * area_e
                )
                n_row_e = int(connectivity_array[e, row + 2] - 1)  # global row index
                n_col_e = int(connectivity_array[e, col + 2] - 1)

                # K_gpu[int(n_row_e - 1), int(n_col_e - 1)] += K_row_col
                K_rows[index] = n_row_e
                K_cols[index] = n_col_e
                K_data[index] = K_row_col

                # update the index counter
                index += 1

                # if e == 0:
                #     print(int(n_row_e - 1), int(n_col_e - 1), K_row_col)

        cuda.syncthreads()

        # Update the global b vector
        for row in range(3):
            if magnetization == 0:
                b_local = 0
            else:
                if row == 0:
                    c_e = c1_e
                elif row == 1:
                    c_e = c2_e
                elif row == 2:
                    c_e = c3_e

                b_local = (magnetization / (2.0)) * (c_e) * (4 * np.pi * 1e-7) * alpha
            n_row_e = int(connectivity_array[e, row + 2])
            # b_gpu[int(n_row_e - 1)] += b_local
            cuda.atomic.add(b_gpu, int(n_row_e - 1), b_local)

        cuda.syncthreads()


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
    # Set FEA settings
    assembly = "cpu"
    solver = "numpy-solve"
    
    # Print Model Settings Out
    print(assembly, "\n")

    # Start timer
    t0 = time.time()

    # Define geometry of the model
    B = 200
    H = 100
    l1 = 20
    l2 = 20
    d1 = 30
    d2 = 30
    lc = 5
    lc1 = 0.5
    mu_r_mag = 1.04
    magnetization = 1e6

    mesh_time_arr = []
    prop_time_arr = []
    sys_time_arr = []
    getBC_time_arr = []
    applyBC_time_arr = []
    solve_time_arr = []
    for i in range(20):
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
        if assembly == "cpu":
            """ CPU implementation of Kx = b """
            K = np.zeros((numNodes, numNodes))
            b = np.zeros(numNodes)
            t0_sys = time.perf_counter() 
            K_sys, b_sys = assemble_K_and_b_standard(
                K,
                b,
                connectivity_array,
                xyz_nodeCoords,
                numElems,
            )
            tf_sys = time.perf_counter()

        
        elif assembly == "gpu":
            """GPU implementation of Kx = b"""
            K_rows = np.zeros(numElems * 9)
            K_cols = np.zeros(numElems * 9)
            K_data = np.zeros(numElems * 9)
            # K = np.zeros((numNodes, numNodes))
            b = np.zeros(numNodes)
            # K_gpu = cuda.to_device(K)
            K_rows_gpu = cuda.to_device(K_rows)
            K_cols_gpu = cuda.to_device(K_cols)
            K_data_gpu = cuda.to_device(K_data)

            b_gpu = cuda.to_device(b)
            connectivity_array_gpu = cuda.to_device(connectivity_array)
            xyz_nodeCoords_gpu = cuda.to_device(xyz_nodeCoords)

            # Define kernel executino parameters
            threadsperblock = 32
            blockspergrid = (numNodes + (threadsperblock - 1)) // threadsperblock

            t0_sys = time.perf_counter()
            _assemble_K_and_b[blockspergrid, threadsperblock](
                K_rows_gpu,
                K_cols_gpu,
                K_data_gpu,
                b_gpu,
                connectivity_array_gpu,
                xyz_nodeCoords_gpu,
            )
            tf_sys = time.perf_counter()
            
            cuda.synchronize()

            # Send back the matrix from GPU to CPU
            K_rows = K_rows_gpu.copy_to_host()
            K_cols = K_cols_gpu.copy_to_host()
            K_data = K_data_gpu.copy_to_host()

            # Convert system to sparse matrix format
            K_sys = sparse.coo_matrix(
                (K_data, (K_rows, K_cols)),
                shape=(numNodes, numNodes),
            ).toarray()
            b_sys = b_gpu.copy_to_host()

        """Get nodes on edge of the boundary"""
        t0_getBC = time.perf_counter()
        nodes_BC = get_BC_nodes(nodeTags, xyz_nodeCoords, B, H)
        tf_getBC = time.perf_counter()

        """Apply Dirichlet BC"""
        t0_applyBC = time.perf_counter()
        K_sys, b_sys = applyDirichletBC(nodes_BC, K_sys, b_sys)
        tf_applyBC = time.perf_counter()

        """Solve System of Equation"""
        t0_solve = time.perf_counter()

        if solver == "numpy-solve":
            """ Numpy solve """
            x = np.linalg.solve(K_sys, b_sys)

        elif solver == "scipy-gmres":
            """ Scipy solve gmres """
            K_csc = sparse.csc_array(K_sys)
            ilu = sparse.linalg.spilu(K_csc)
            Mx = lambda x: ilu.solve(x)
            M = sparse.linalg.LinearOperator((len(b_sys), len(b_sys)), Mx)
            x, info = sparse.linalg.gmres(
                A=K_csc,
                b=b_sys,
                M=M,
                rtol=1e-10,
                maxiter=200,
            )
            if info == 1:
                raise ValueError("gmres failed to converge in 200 iterations")

        elif solver == "torch":
            """ Torch """
            x = torch.linalg.solve(torch.from_numpy(K_sys), torch.from_numpy(b_sys))
            
        elif solver == "cupy-solve":
            """Solve system of equation using cupy"""
            K_gpu = cp.asarray(K_sys)  # send K_global to gpu
            b_gpu = cp.asarray(b_sys)  # send F_global to gpu
            soln_gpu = cp.linalg.solve(K_gpu, b_gpu)  # solve using cupy kernel
            x = cp.asnumpy(soln_gpu) # send soln back to the host
            
        elif solver == "cupyx-spsolve":
            """Solve system of equations using cupyx spsolve"""
            K_gpu = cp.asarray(K_sys)  # send K_global to gpu
            K_csc_gpu = csr_matrix(K_gpu)  # convert K to a csr matrix
            b_gpu = cp.asarray(b_sys)  # send F_global to gpu
            soln = cpssl.spsolve(K_csc_gpu, b_gpu)  # solve using spsolve
            x = soln.get()  # send soln back to host

        tf_solve = time.perf_counter()

        """Plot solution field"""
        # contour_mpl(xyz_nodeCoords, x, title = assembly + f" {i}", fname = assembly+".jpg", flag=True)
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
    # headers = ["Iteration", "1", "2", "3", "4", "5", "6"]
    # table = [
    #     ["GMSH"] + mesh_time_arr,
    #     ["Mesh Prop."] + prop_time_arr,
    #     ["Kx = b"] + sys_time_arr,
    #     ["Get BCs Nodes"] + getBC_time_arr,
    #     ["Apply BC"] + applyBC_time_arr,
    #     ["Solve Time"] + solve_time_arr,
    # ]
    # console = Console()
    # console.print(
    #     tabulate(table, headers, tablefmt="fancy_outline", floatfmt="3e"),
    #     style="bold yellow",
    # ) 
      
    stats = {
        "GMSH" : mesh_time_arr,
        "Mesh Prop." : prop_time_arr,
        "Kx = b" : sys_time_arr,
        "Get BC Nodes" : getBC_time_arr,
        "Apply BC" : applyBC_time_arr,
        "Solver" : solve_time_arr,
    }

    console = Console()
    console.print(
        tabulate(stats, 
                 headers = "keys",
                 tablefmt="fancy_outline", 
                 floatfmt="3e",
                 showindex="always"
        ),
        style="grey74",
    )
