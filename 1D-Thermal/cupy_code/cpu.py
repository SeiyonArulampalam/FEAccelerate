import numpy as np
import matplotlib.pyplot as plt
from pyinstrument import Profiler
import time

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


def Order1ShapeFunc(xi):
    """
    Definition of the shape functions for a linear element

    Parameters
    ----------
    xi : float
        quadrature point in [0,1] frame

    Returns
    -------
    arrays
        f : the value of the shape functions at xi
        df: derivative of shape functions at xi
    """
    # shape functions
    f1 = 1 - xi
    f2 = xi
    f = np.array([f1, f2])

    # shape function derivatives
    df1 = -1
    df2 = 1
    df = np.array([df1, df2])

    return f, df


def Order1Map(xi, x_vec):
    """
    Map the local reference frame to the global reference fram

    Parameters
    ----------
    xi : float
        quadrature point of interest [0,1]
    x_vec : array
        first entry is the left node position,
        and the second entry is the 2nd node of the element

    Returns
    -------
    float
        the mapped global coordinate of the quadrature point
    """
    x1 = x_vec[0]
    x2 = x_vec[1]

    f, _ = Order1ShapeFunc(xi)
    N1 = f[0]
    N2 = f[1]

    return x1 * N1 + x2 * N2


def Order1Jacobian(x_vec):
    """
    Compute the jacobian of the element

    Parameters
    ----------
    x_vec : array
        first entry is the left node position,
        and the second entry is the 2nd node of the element

    Returns
    -------
    float
        Jacobian is the scaling factor for a change of frame for integration.
        In our case we want to change our integration fram from the global coordiantes
        to the [0,1] reference frame. The jacobian acts as a scaling term.

    """
    x1 = x_vec[0]
    x2 = x_vec[1]
    return x2 - x1


def Order1_dNdx(xi, x_vec):
    """
    Compute the shape function derivatives w.r.t the global frame

    Parameters
    ----------
    xi : float
        quadrature point of interest [0,1]
    x_vec : array
        first entry is the left node position,
        and the second entry is the 2nd node of the element


    Returns
    -------
    array
        First element is the derivative of the first shape function w.r.t. x.
        Second element is the derivative of the second shape function w.r.t. x.
    """
    jac = Order1Jacobian(x_vec)

    _, df = Order1ShapeFunc(xi)
    dN1_dxi = df[0]
    dN2_dxi = df[1]

    dN1_dx = dN1_dxi / jac
    dN2_dx = dN2_dxi / jac

    dNdx = np.array([dN1_dx, dN2_dx])
    return dNdx


def GaussOrder4():
    """
    Define the Gaussian Quadrature points and weights exact up to order 4. The
    weights are multiplied by 0.5 to shift the weights from wiki that are from the [-1,1] frame.
    Similarly, the quadrature points are shifted as well.

    var_hat -> [0,1] frame. While w and xi are in the [-1,1] frame
    w_hat = 0.5*w
    xi_hat = 0.5*xi + 0.5

    Returns
    -------
    arrays
        wts: quadrature weight in [0,1] frame
        xi : quadrature pts in the [0,1] frame
    """
    # Weights and quad points shifted s.t. xi in [0,1] coordinates
    wts = np.array([0.568889, 0.478629, 0.478629, 0.236927, 0.236927]) * 0.5
    xi = np.array([0.0, 0.538469, -0.538469, 0.90618, -0.90618]) * 0.5 + 0.5
    return wts, xi


def Integrate_K(
    wts,
    xi_pts,
    jac_func,
    shape_func_derivatives,
    shape_func_vals,
    x_vec,
    a,
    c0,
    i,
    j,
):
    """
    Evaluate the integral that defines the elemental stiffness matrix input.

    Parameters
    ----------
    wts : array
        quadrature weights
    xi_pts : array
        quadrature points
    jac_func : function
        jacobian function handle
    shape_func_derivatives : function
        shape function derivatives handle that returns the derivative at quad point
    shape_func_vals : function
        shape function handle that returns the derivative at quad point
    x_vec : array
        first entry is the left node position,
        and the second entry is the 2nd node of the element
    a : float
        elemental coeff
    c0 : float
        elemental coeff
    i : int
        index for the local stiffness matrix (row)
    j : int
        index for the local stiffness matrix (col)

    Returns
    -------
    float
        Gaussian quadrature integration result
    """
    I = 0.0
    for k in range(len(wts)):
        weight = wts[k]
        xi = xi_pts[k]
        jac = jac_func(x_vec)
        dN_dx = shape_func_derivatives(xi, x_vec)
        N, _ = shape_func_vals(xi)

        comp1 = a * dN_dx[i] * dN_dx[j]
        comp2 = c0 * N[i] * N[j]

        I += weight * (comp1 + comp2) * jac

    return I


def Integrate_f(
    wts,
    xi_pts,
    jac_func,
    shape_func_vals,
    x_vec,
    g,
    i,
):
    """
    Evaluate the elemental f values to build the global F vector.

    Parameters
    ----------
    wts : array
        quadrature weights
    xi_pts : array
        quadrature points
    jac_func : function
        jacobian function handle
    shape_func_vals : function
        shape function handle that returns the derivative at quad point
    x_vec : array
        first entry is the left node position,
        and the second entry is the 2nd node of the element
    g : float
        excitation value for the element
    i : int
        local node number

    Returns
    -------
    float
        Gaussian quadrature integration result
    """
    I = 0.0
    for k in range(len(wts)):
        weight = wts[k]
        xi = xi_pts[k]
        jac = jac_func(x_vec)
        N, _ = shape_func_vals(xi)

        comp1 = N[i] * g

        I += weight * comp1 * jac

    return I


# Flags
apply_convection = False  # apply forced convection at tip of beam

# Establish the total number of elements and nodes and beam length
num_elems = 20_000
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

start = time.perf_counter()  # start timer
for i in range(6):
    print(f"Iteration : {i}:")
    start_mesh = time.perf_counter()  # start timer
    # Generate the mesh
    xloc = np.linspace(0, L, num_nodes)  # location of x nodes

    # Create an array that maps node tag to node position: | node tag | node location
    node_array = np.zeros((num_nodes, 2))
    for i in range(num_nodes):
        node_array[i][0] = i
        node_array[i][1] = xloc[i]

    # print("Mapping node tag -> node location:")
    # print(node_array)
    # print()

    # Create element array: | element # | node 1 position | node 2 position |
    element_array = np.zeros((num_elems, 3))
    for i in range(num_elems):
        element_array[i][0] = i
        element_array[i][1] = node_array[i][1]
        element_array[i][2] = node_array[i + 1][1]

    # print("Mapping element tag -> node position:")
    # print(element_array)
    # print()

    # Create element node tag array: | element # | node 1 tag | node 2 tag |
    element_node_tag_array = np.zeros((num_elems, 3))
    for i in range(num_elems):
        element_node_tag_array[i][0] = int(i)
        element_node_tag_array[i][1] = int(node_array[i][0])
        element_node_tag_array[i][2] = int(node_array[i + 1][0])

    # print("Mapping element tag -> node tag:")
    # print(element_node_tag_array)
    # print()
    end_mesh = time.perf_counter()  # end timer
    total_time_mesh = end_mesh - start_mesh
    print(f"    Time to generate mesh : {total_time_mesh:.6e} s")

    # Assemble K matrix, F vector, C Matrix
    # FEA: K*u + C*u_dot = F
    start_sys = time.perf_counter()  # start timer

    # Excitation of the beam
    g = np.zeros(num_elems)  # initialize excitation vector

    # Initialize the vectors and matrixes for the finite element analysis
    K_global = np.zeros((num_nodes, num_nodes))
    F_global = np.zeros(num_nodes)
    C_global = np.zeros((num_nodes, num_nodes))

    # define key function handles for gaussian integration
    wts, xi_pts = GaussOrder4()
    jac_func = Order1Jacobian
    shape_func_derivatives = Order1_dNdx
    shape_func_vals = Order1ShapeFunc

    for e in range(num_elems):
        # loop through each element and update the global matrix
        x_left = element_array[e][1]
        x_right = element_array[e][2]
        x_vec = np.array([x_left, x_right])

        K_local = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                K_local[i, j] = Integrate_K(
                    wts,
                    xi_pts,
                    jac_func,
                    shape_func_derivatives,
                    shape_func_vals,
                    x_vec,
                    a,
                    c0,
                    i,
                    j,
                )
                n_i_e = element_node_tag_array[e][i + 1]
                n_j_e = element_node_tag_array[e][j + 1]
                K_global[int(n_i_e), int(n_j_e)] += K_local[i, j]

        f_local = np.zeros(2)
        for i in range(2):
            f_local[i] = Integrate_f(
                wts,
                xi_pts,
                jac_func,
                shape_func_vals,
                x_vec,
                g[e],
                i,
            )
            n_i_e = element_node_tag_array[e][i + 1]
            F_global[int(n_i_e)] += f_local[i]

    end_sys = time.perf_counter()  # end timer
    total_time_sys = end_sys - start_sys
    print(f"    Time to assemble K and F : {total_time_sys:.6e} s")

    start_bc = time.perf_counter()  # start timer
    # Apply Dirichlet B.C.
    # modify K_global to be a pivot
    K_global[0, 0] = 1.0
    K_global[0, 1:] = 0.0  # zero the 1st row of the K matrix

    # modify the F_global due to dirichlet BC
    F_global[0] = u_root
    F_global[1:] = F_global[1:] - K_global[1:, 0] * u_root

    # Zero the 1st col of the K matrix
    K_global[1:, 0] = 0.0

    # # Apply convection BC at beam tip
    if apply_convection == True:
        K_global[-1, -1] += beta * A

    end_bc = time.perf_counter()  # end timer
    total_time_bc = end_bc - start_bc
    print(f"    Time to apply BC : {total_time_bc:.6e} s")

    # print("K matrix Modified:")
    # print(K_global)
    # print()
    # print("F vector Modified:")
    # print(F_global)
    # print()

    # Solve system fo Equations
    start_solve = time.perf_counter()  # start timer
    steady_state_soln = np.linalg.solve(K_global, F_global)
    end_solve = time.perf_counter()  # end timer
    total_time_solve = end_solve - start_solve
    print(f"    Time to solve Kx = F : {total_time_solve:.6e} s\n")

end = time.perf_counter()  # end timer
total_time = end - start
print("\n-------------------------------------")
print(f"Total Time  : {total_time:.6e} s")
print(f"Total Elems : {num_elems}")
print("-------------------------------------")

# print("Solution Steady State:")
# print(steady_state_soln)
# print()

# Plot Results
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(xloc, steady_state_soln, "m-")
ax.set_xlabel("Location (m)")
ax.set_ylabel("Temperature (C)")
ax.set_title(f"Steady-State Heat transfer simulation with {num_elems} elements")
plt.grid()
# plt.show()
plt.savefig("soln_cpu.jpg", dpi=800)
