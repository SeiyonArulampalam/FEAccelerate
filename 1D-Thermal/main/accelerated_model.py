import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit, cuda
import torch
from scipy import sparse
import cupy
import cupyx.scipy.sparse.linalg as cpssl
from cupyx.scipy.sparse import csr_matrix


np.set_printoptions(precision=4)

"""
PDE: - ∂/∂x(a*∂u/∂x) + c0*u = g(x,t)

Insulated BC: k*A*∂u/∂t = 0 

Convection BC: k*A*∂u/∂t + β*A*u = 0

Solution: u = T - T_ambient

a = k*A
c0 = P*β

rho = mass density
β = heat transfer coeff
k = thermal conductivity
A = cross-section area
P = perimeter of rod end (circumprence of rod)
cp =specific heat at constant pressure
g = internal excitation


The FEM discretization of a beam leads to:

Element (M)      (0)         (1)         (2)
            o-----------o-----------o-----------o-----> (x-axis)
Node (N)    0           1           2           3
"""


def plot_result(L, num_nodes, num_elems, steady_state_soln, filename="soln.jpg"):
    # Plot Results
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(np.linspace(0, L, num_nodes), steady_state_soln, "m-")
    ax.set_xlabel("Location (m)")
    ax.set_ylabel("Temperature (C)")
    ax.set_title(f"Steady-State Heat transfer simulation with {num_elems} elements")
    plt.grid()
    # plt.show()
    plt.savefig(filename, dpi=800)


@njit
def generate_mesh(L, num_nodes, num_elems):
    # Generate the mesh
    xloc = np.linspace(0, L, num_nodes)  # location of x nodes

    # Create an array that maps node tag to node position: | node tag | node location
    node_array = np.zeros((num_nodes, 2))
    for i in range(num_nodes):
        node_array[i][0] = i
        node_array[i][1] = xloc[i]

    # Create element array: | element # | node 1 position | node 2 position |
    element_array = np.zeros((num_elems, 3))
    for i in range(num_elems):
        element_array[i][0] = i
        element_array[i][1] = node_array[i][1]
        element_array[i][2] = node_array[i + 1][1]

    # Create element node tag array: | element # | node 1 tag | node 2 tag |
    element_node_tag_array = np.zeros((num_elems, 3))
    for i in range(num_elems):
        element_node_tag_array[i][0] = int(i)
        element_node_tag_array[i][1] = int(node_array[i][0])
        element_node_tag_array[i][2] = int(node_array[i + 1][0])

    return element_array, element_node_tag_array


@njit
def apply_dirichlet_BC(K_global, F_global, u_root=300.0):
    # Apply Dirichlet B.C.
    # modify K_global to be a pivot
    K_global[0, 0] = 1.0
    K_global[0, 1:] = 0.0  # zero the 1st row of the K matrix

    # Modify the F_global due to dirichlet BC
    F_global[0] = u_root
    F_global[1:] = F_global[1:] - K_global[1:, 0] * u_root

    # Zero the 1st col of the K matrix
    K_global[1:, 0] = 0.0

    return K_global, F_global


@njit
def compute_local_K(a, c0, wts, xi_pts, jacobian, i, j):
    # compute the local K matrix
    I = 0.0
    for k in range(len(wts)):
        # gauss points and weight
        weight = wts[k]
        xi = xi_pts[k]

        # shape func evaluate at gauss point
        N = np.array(
            [
                1 - xi,
                xi,
            ]
        )

        # shape func derivative wrt global frame
        dN_dx = np.array(
            [
                -1 * (1.0 / jacobian),
                1 * (1.0 / jacobian),
            ]
        )

        # integrand components
        comp1 = a * dN_dx[i] * dN_dx[j]
        comp2 = c0 * N[i] * N[j]

        # update integral approximation
        I += weight * (comp1 + comp2) * jacobian

    return I


@njit
def compute_local_F(wts, xi_pts, jacobian, g_e, i):
    # gauss points and weight
    I = 0.0
    for k in range(len(wts)):
        # gauss points and weight
        weight = wts[k]
        xi = xi_pts[k]

        # shape func evaluate at gauss point
        N = np.array(
            [
                1 - xi,
                xi,
            ]
        )

        # integrand component
        comp1 = N[i] * g_e

        # update the integral approximation
        I += weight * comp1 * jacobian
    return I


@njit
def assemble_K_and_F(
    num_elems,
    a,
    c0,
    wts,
    xi_pts,
    g,
    element_node_tag_array,
    element_array,
    K_global,
    F_global,
):
    for e in range(num_elems):
        # loop through each element and update the global matrix
        x_left = element_array[e][1]
        x_right = element_array[e][2]
        jacobian = x_right - x_left

        for i in range(2):
            for j in range(2):
                # compute the local K matrix
                I = compute_local_K(a, c0, wts, xi_pts, jacobian, i, j)

                n_i_e = element_node_tag_array[e][i + 1]
                n_j_e = element_node_tag_array[e][j + 1]
                K_global[int(n_i_e), int(n_j_e)] += I

        for i in range(2):
            # compute the local f vector
            I = compute_local_F(wts, xi_pts, jacobian, g[e], i)
            n_i_e = element_node_tag_array[e][i + 1]
            F_global[int(n_i_e)] += I

    return K_global, F_global


@cuda.jit
def assemble_K_and_F_kernel(
    wts,
    xi_pts,
    num_elems,
    a,
    c0,
    g,
    element_node_tag_array,
    element_array,
    K_global,
    F_global,
):
    # get the thread Id
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    # get the stride length (cuda.blockIdx.x * cuda.gridDim.x)
    stride = cuda.gridsize(1)

    # if e < num_elems:
    for e in range(idx, num_elems, stride):
        # loop through each element and update the global matrix
        x_left = element_array[e][1]
        x_right = element_array[e][2]
        jacobian = x_right - x_left

        for i in range(2):
            for j in range(2):
                # compute the local K matrix
                I = 0.0
                for k in range(len(wts)):
                    # gauss points and weight
                    weight = wts[k]
                    xi = xi_pts[k]

                    # shape func evaluate at gauss point
                    if i == 0 and j == 0:
                        N = (1 - xi) * (1 - xi)
                        dN_dx = (-1 * (1.0 / jacobian)) * (-1 * (1.0 / jacobian))
                    elif (i == 1 and j == 0) or (i == 0 and j == 1):
                        N = (xi) * (1 - xi)
                        dN_dx = (1 * (1.0 / jacobian)) * (-1 * (1.0 / jacobian))
                    elif i == 1 and j == 1:
                        N = xi * xi
                        dN_dx = (1 * (1.0 / jacobian)) * (1 * (1.0 / jacobian))

                    # integrand components
                    comp1 = a * dN_dx
                    comp2 = c0 * N

                    # update integral approximation
                    I += weight * (comp1 + comp2) * jacobian

                n_i_e = element_node_tag_array[e][i + 1]
                n_j_e = element_node_tag_array[e][j + 1]
                K_global[int(n_i_e), int(n_j_e)] += I

        for i in range(2):
            # compute the local f vector
            I = 0.0
            for k in range(len(wts)):
                # gauss points and weight
                weight = wts[k]
                xi = xi_pts[k]

                # shape func evaluate at gauss point
                if i == 0:
                    N = 1 - xi
                elif i == 1:
                    N = xi

                # integrand component
                comp1 = N * g[e]

                # update the integral approximation
                I += weight * comp1 * jacobian

            n_i_e = element_node_tag_array[e][i + 1]
            F_global[int(n_i_e)] += I

    cuda.syncthreads()


if __name__ == "__main__":
    # Setup the argument parser for command line interface
    parser = argparse.ArgumentParser(
        description="Select solver settings for the 1D heat trasnfer model."
    )
    parser.add_argument(
        "-a",
        "--assemble",
        type=str,
        help="Choose between jit and cuda to assemble Kx = F.",
        default="cuda",
    )
    parser.add_argument(
        "-n",
        "--num_elems",
        type=int,
        help="Insert total number of element for analysis.",
        default=1000,
    )
    parser.add_argument(
        "-s",
        "--solver",
        type=str,
        help="Select cupy, cupyx, torch, or numpy to solve system of equations.",
        default="cupyx",
    )
    args = parser.parse_args()
    assemble_parsed = args.assemble
    num_elems_parsed = args.num_elems
    solver_parsed = args.solver

    print("\n-----------------------------------")
    print("FEA Setting:")
    print("   # of Elements :", num_elems_parsed)
    print("   Assembly Method :", assemble_parsed)
    print("   Solver Method :", solver_parsed)
    print("-----------------------------------\n")

    # Flags
    apply_convection = False  # apply forced convection at tip of beam

    # Establish the total number of elements and nodes and beam length
    num_elems = num_elems_parsed
    num_nodes = num_elems + 1
    L = 0.05  # length of beam [m]
    D = 0.02  # diameter of rod [m]

    # Define the physical coefficients of the simulation
    rho = 7700  # material density [kg/m3]
    beta = 100.0  # heat transfer coefficient [W/(m2.K)]
    k = 50.0  # thermal conductivity [W/(m.K)]
    A = np.pi * (D / 2) ** 2  # cross sectional area [m2]
    P = 2 * np.pi * (D / 2)  # perimeter [m]
    cp = 0.452  # specific heat at constant pressure [J/(kg.K)]

    # Define the root temperature of the beam
    T_ambient = 20.0  # ambient temperature [C]
    T_root = 320.0  # temperature at the root [C]
    u_root = T_root - T_ambient  # temperature solution at the root [C]

    # Define coefficienct found in the heat unsteady 1d heat equation
    a = k * A
    c0 = P * beta

    # Define the gauss quadrature points and weights
    # Weights and quad points shifted s.t. xi in [0,1] coordinates
    wts = np.array([0.568889, 0.478629, 0.478629, 0.236927, 0.236927]) * 0.5
    xi_pts = np.array([0.0, 0.538469, -0.538469, 0.90618, -0.90618]) * 0.5 + 0.5

    print("\n-------------------------------------------------------------")
    start = time.perf_counter()  # start timer
    for i in range(6):
        print(f"Iteration : {i}:")
        """Generate mesh and connectivity array"""
        start_mesh = time.perf_counter()  # start timer
        element_array, element_node_tag_array = generate_mesh(
            L=L,
            num_nodes=num_nodes,
            num_elems=num_elems,
        )
        end_mesh = time.perf_counter()  # end timer
        total_time_mesh = end_mesh - start_mesh
        print(f"    Time to generate mesh : {total_time_mesh:.6e} s")

        """Initialize K and F and beam excitations"""
        # Initialize the vectors and matrixes for the finite element analysis
        # Create device-side copies of the arrays
        K_global = np.zeros((num_nodes, num_nodes))
        F_global = np.zeros(num_nodes)

        # Excitation of the beam
        g = np.zeros(num_elems)  # initialize excitation vector

        if assemble_parsed == "cuda":
            """Compute K matrix and F vector using cuda kernel"""
            # send copy to gpu
            K_gpu = cuda.to_device(K_global)
            F_gpu = cuda.to_device(F_global)
            g_gpu = cuda.to_device(g)
            wts_gpu = cuda.to_device(wts)
            xi_pts_gpu = cuda.to_device(xi_pts)
            element_array_gpu = cuda.to_device(element_array)
            element_node_tag_array_gpu = cuda.to_device(element_node_tag_array)

            # define kernel execution parameters
            threadsperblock = 16
            blockspergrid = (num_nodes + (threadsperblock - 1)) // threadsperblock

            start_sys = time.perf_counter()  # start timer
            assemble_K_and_F_kernel[blockspergrid, threadsperblock](
                wts_gpu,
                xi_pts_gpu,
                num_elems,
                a,
                c0,
                g_gpu,
                element_node_tag_array_gpu,
                element_array_gpu,
                K_gpu,
                F_gpu,
            )
            end_sys = time.perf_counter()  # end timer
            total_time_sys = end_sys - start_sys
            print(f"    Time to assemble K and F : {total_time_sys:.6e} s")

            # send back the K matrix and F vector
            K_global = K_gpu.copy_to_host()
            F_global = F_gpu.copy_to_host()

            cuda.synchronize()

        elif assemble_parsed == "jit":
            """Compute K matrix anf F vector Using Jit"""
            start_sys = time.perf_counter()  # start timer
            K_global, F_global = assemble_K_and_F(
                num_elems=num_elems,
                a=a,
                c0=c0,
                wts=wts,
                xi_pts=xi_pts,
                g=g,
                element_node_tag_array=element_node_tag_array,
                element_array=element_array,
                K_global=K_global,
                F_global=F_global,
            )
            end_sys = time.perf_counter()  # end timer
            total_time_sys = end_sys - start_sys
            print(f"    Time to assemble K and F : {total_time_sys:.6e} s")

        """Apply BCs to model"""
        start_bc = time.perf_counter()  # start timer
        K_global, F_global = apply_dirichlet_BC(
            K_global=K_global,
            F_global=F_global,
            u_root=u_root,
        )

        # Apply convection BC at beam tip
        if apply_convection == True:
            K_global[-1, -1] += beta * A

        end_bc = time.perf_counter()  # end timer
        total_time_bc = end_bc - start_bc
        print(f"    Time to apply BC : {total_time_bc:.6e} s")

        """Solve system of Equations"""
        start_solve = time.perf_counter()  # start timer

        if solver_parsed == "cupy":
            """Solve system of equation using cupy"""
            K_gpu = cupy.asarray(K_global)  # send K_global to gpu
            F_gpu = cupy.asarray(F_global)  # send F_global to gpu
            soln_gpu = cupy.linalg.solve(K_gpu, F_gpu)  # solve using cupy kernel

            # send back the GPU solution to the CPU
            steady_state_soln = cupy.asnumpy(soln_gpu)

        elif solver_parsed == "cupyx":
            """Solve system of equations using cupyx"""
            K_gpu = cupy.asarray(K_global)  # send K_global to gpu
            K_csc_gpu = csr_matrix(K_gpu)  # convert K to a csc matrix
            F_gpu = cupy.asarray(F_global)  # send F_global to gpu
            soln = cpssl.spsolve(K_csc_gpu, F_gpu)  # solve using spsolve
            steady_state_soln = soln.get()  # send soln back to host

        elif solver_parsed == "torch":
            """Solve system of equation using torch"""
            steady_state_soln = torch.linalg.solve(
                torch.from_numpy(K_global), torch.from_numpy(F_global)
            )

        elif solver_parsed == "numpy":
            """Solve system of equation using numpy"""
            steady_state_soln = np.linalg.solve(K_global, F_global)

        end_solve = time.perf_counter()  # end timer
        total_time_solve = end_solve - start_solve
        print(f"    Time to solve Kx = F : {total_time_solve:.6e} s\n")

        # plot result
        plot_result(L, num_nodes, num_elems, steady_state_soln, filename=f"gpu_{i}.jpg")

    end = time.perf_counter()  # end timer
    total_time = end - start
    print("\n-------------------------------------")
    print(f"Total Time  : {total_time:.6e} s")
    print("-------------------------------------")
