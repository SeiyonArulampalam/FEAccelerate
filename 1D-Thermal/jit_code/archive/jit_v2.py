import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit
from pyamg.aggregation import adaptive_sa_solver
from pyamg.krylov import fgmres

np.set_printoptions(precision=4)
# plt.style.use("dark_background")

"""
PDE: c1*∂u/∂t - ∂/∂x(a*∂u/∂x) + c0*u = g(x,t)

Insulated BC: k*A*∂u/∂t = 0 

Convection BC: k*A*∂u/∂t + β*A*u = 0

Solution: u = T - T_ambient

c1 = rho*cp*A
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


@njit()
def compute_local_K(wts, xi_pts, jacobian, i, j):
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


@njit()
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


@njit()
def assemble_K_and_F(
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
                I = compute_local_K(wts, xi_pts, jacobian, i, j)

                n_i_e = element_node_tag_array[e][i + 1]
                n_j_e = element_node_tag_array[e][j + 1]
                K_global[int(n_i_e), int(n_j_e)] += I

        for i in range(2):
            # compute the local f vector
            I = compute_local_F(wts, xi_pts, jacobian, g[e], i)
            n_i_e = element_node_tag_array[e][i + 1]
            F_global[int(n_i_e)] += I

    return K_global, F_global


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
c1 = rho * cp * A
a = k * A
c0 = P * beta

"""Generate mesh and connectivity array"""
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

"""Compute K matrix anf F vector"""
# Define the gauss quadrature points and weights
# Weights and quad points shifted s.t. xi in [0,1] coordinates
wts = np.array([0.568889, 0.478629, 0.478629, 0.236927, 0.236927]) * 0.5
xi_pts = np.array([0.0, 0.538469, -0.538469, 0.90618, -0.90618]) * 0.5 + 0.5

# Excitation of the beam
g = np.zeros(num_elems)  # initialize excitation vector

# Initialize the vectors and matrixes for the finite element analysis
# Create device-side copies of the arrays
K_global = np.zeros((num_nodes, num_nodes))
F_global = np.zeros(num_nodes)

start = time.perf_counter()  # start timer
for i in range(6):
    K_global, F_global = assemble_K_and_F(
        wts,
        xi_pts,
        g,
        element_node_tag_array,
        element_array,
        K_global,
        F_global,
    )
end = time.perf_counter()  # end timer
total_time = end - start
print(f"\n Time to assemble K and F : {total_time:.6e} s")
print(f" Total number of elements : {num_elems} \n")

exit()

"""Apply BCs to model"""
# Apply Dirichlet B.C.
# modify K_global to be a pivot
K_global[0, 0] = 1.0
K_global[0, 1:] = 0.0  # zero the 1st row of the K matrix

# Modify the F_global due to dirichlet BC
F_global[0] = u_root
F_global[1:] = F_global[1:] - K_global[1:, 0] * u_root

# Zero the 1st col of the K matrix
K_global[1:, 0] = 0.0

# Apply convection BC at beam tip
if apply_convection == True:
    K_global[-1, -1] += beta * A

# print("K matrix Modified:")
# print(K_global)
# print("\nF vector Modified:")
# print(F_global)


start = time.perf_counter()  # start timer
"""Solve system of Equations using numpy"""
steady_state_soln = np.linalg.solve(K_global, F_global)

"""Solve system using pyAMG"""
# [asa, work] = adaptive_sa_solver(
#     K_global,
#     num_candidates=5,
#     candidate_iters=10,
#     prepostsmoother="gauss_seidel",
#     strength="symmetric",
#     aggregate="standard",
#     smooth="jacobi",
#     coarse_solver="splu",
# )
# M = asa.aspreconditioner(cycle="W")
# steady_state_soln, _ = fgmres(K_global, F_global, tol=1e-16, maxiter=30, M=M)

end = time.perf_counter()  # end timer
total_time = end - start
print(f"\n Time to solve Kx = F : {total_time:.6e} s\n")

# print("\nSolution Steady State:")
# print(steady_state_soln)


# # Plot Results
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(xloc, steady_state_soln, "m-")
ax.set_xlabel("Location (m)")
ax.set_ylabel("Temperature (C)")
ax.set_title(f"Steady-State Heat transfer simulation with {num_elems} elements")
plt.grid()
plt.show()
# plt.savefig("soln_jit.jpg", dpi=800)
