import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit, prange
from pyamg.aggregation import adaptive_sa_solver
from pyamg.krylov import fgmres
from concurrent.futures import ThreadPoolExecutor

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
P = perimeter of rod end (curcumprence of rod)
cp =specific heat at constant pressure
g = internal excitation


The FEM discretization of a beam leads to:

Element (M)      (0)         (1)         (2)
            o-----------o-----------o-----------o-----> (x-axis)
Node (N)    0           1           2           3
"""


@njit
def generate_mesh(L, num_nodes, num_elems):
    # Generate the mesh
    xloc = np.linspace(0, L, num_nodes)  # location of x nodes

    # Create an array that maps node tag to node position: | node tag | node location
    node_array = np.zeros((num_nodes, 2))
    for i in range(num_nodes):
        node_array[i][0] = i
        node_array[i][1] = xloc[i]

    # print("Mapping node tag -> node location:")
    # print(node_array)

    # Create element array: | element # | node 1 position | node 2 position |
    element_array = np.zeros((num_elems, 3))
    for i in range(num_elems):
        element_array[i][0] = i
        element_array[i][1] = node_array[i][1]
        element_array[i][2] = node_array[i + 1][1]

    # print("\nMapping element tag -> node position:")
    # print(element_array)

    # Create element node tag array: | element # | node 1 tag | node 2 tag |
    element_node_tag_array = np.zeros((num_elems, 3))
    for i in range(num_elems):
        element_node_tag_array[i][0] = int(i)
        element_node_tag_array[i][1] = int(node_array[i][0])
        element_node_tag_array[i][2] = int(node_array[i + 1][0])

    # print("\nMapping element tag -> node tag:")
    # print(element_node_tag_array)

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
        N = [
            1 - xi,
            xi,
        ]

        # shape func derivative wrt global frame
        dN_dx = [
            -1 * (1.0 / jacobian),
            1 * (1.0 / jacobian),
        ]

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
        N = [
            1 - xi,
            xi,
        ]

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


if __name__ == "__main__":
    # Flags
    apply_convection = False  # apply forced convection at tip of beam

    # Establish the total number of elements and nodes and beam length
    num_elems = 100_000
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

    K_list = []
    F_list = []
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

        """Compute K matrix anf F vector"""
        # Excitation of the beam
        g = np.zeros(num_elems)  # initialize excitation vector

        # Initialize the vectors and matrixes for the finite element analysis
        # Create device-side copies of the arrays
        K_global = np.zeros((num_nodes, num_nodes))
        F_global = np.zeros(num_nodes)

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

        K_list.append(K_global)
        F_list.append(F_global)

        # """Solve system of Equations using numpy"""
        # start = time.perf_counter()  # start timer
        # steady_state_soln = np.linalg.solve(K_global, F_global)
        # end = time.perf_counter()  # end timer
        # total_time = end - start
        # print(f"    Time to solve Kx = F : {total_time:.6e} s\n")
    end = time.perf_counter()  # end timer
    total_time = end - start
    print("\n-------------------------------------")
    print(f"Total Time  : {total_time:.6e} s")
    print(f"Total Elems : {num_elems}")
    print("-------------------------------------")

    # # Plot Results
    # fig, ax = plt.subplots(nrows=1, ncols=1)
    # ax.plot(np.linspace(0, L, num_nodes), steady_state_soln, "m-")
    # ax.set_xlabel("Location (m)")
    # ax.set_ylabel("Temperature (C)")
    # ax.set_title(f"Steady-State Heat transfer simulation with {num_elems} elements")
    # plt.grid()
    # plt.show()
    # # plt.savefig("soln_jit.jpg", dpi=800)
