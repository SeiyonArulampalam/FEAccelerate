from tabulate import tabulate
import numpy as np
from scipy import sparse
from numba import njit, cuda
import time
import geometry
import assembly 
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib as mpl
from rich.console import Console
import cupy as cp
import torch
import cupyx.scipy.sparse.linalg as cpssl
from cupyx.scipy.sparse import csr_matrix, csc_matrix, coo_matrix

processor = "cpu"


# Start timer
t0 = time.time()

# Define geometry of the model
B = 200
H = 100
l1 = 20
l2 = 20
d1 = 30
d2 = 30
lc = 50
lc1 = 10
mu_r_mag = 1.04
magnetization = 1e6

mesh_time_arr = []
prop_time_arr = []
sys_time_arr = []
getBC_time_arr = []
applyBC_time_arr = []
solve_time_arr = []

for i in range(6):
    """Generate the mesh using GMSH """
    t0_mesh = time.perf_counter()
    nodeTags, xyz_nodeCoords, elemTags, elemNodeTags = geometry.create_mesh(
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
    
    """Create the connectivity array"""
    t0_mesh_prop = time.perf_counter()
    connectivity_array = geometry.get_mesh_props(
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
    
    """Mesh properties"""
    numNodes = len(nodeTags)
    numElems = len(elemTags)
    
    """Get nodes on edge of the boundary"""
    t0_getBC = time.perf_counter()
    nodes_BC = assembly.get_BC_nodes(nodeTags, xyz_nodeCoords, B, H)
    tf_getBC = time.perf_counter()

    if processor == "cpu":
        """CPU implementation of Kx = b """
        K = np.zeros((numNodes, numNodes))
        b = np.zeros(numNodes)
        t0_sys = time.perf_counter() 
        K_sys, b_sys = assembly.assemble_K_and_b_standard(
            K,
            b,
            connectivity_array,
            xyz_nodeCoords,
            numElems,
        )
        tf_sys = time.perf_counter()
        
        """CPU implementation of applying the BC"""
        t0_applyBC = time.perf_counter()
        K_sys, b_sys = assembly.applyDirichletBC(nodes_BC, K_sys, b_sys)
        tf_applyBC = time.perf_counter()
        
        """Solve the system of equations on the CPU"""
        t0_solve = time.perf_counter() # start timer
        x = np.linalg.solve(K_sys, b_sys) # solve system
        tf_solve = time.perf_counter() # stop timer
        

    elif processor == "gpu":
        """GPU implementation of Kx = b"""
        K_rows = np.zeros(numElems * 9)
        K_cols = np.zeros(numElems * 9)
        K_data = np.zeros(numElems * 9)
        b = np.zeros(numNodes)
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
        assembly._assemble_K_and_b[blockspergrid, threadsperblock](
            K_rows_gpu,
            K_cols_gpu,
            K_data_gpu,
            b_gpu,
            connectivity_array_gpu,
            xyz_nodeCoords_gpu,
        )
        tf_sys = time.perf_counter()
        
        K_data_gpu = cp.array(K_data_gpu)
        K_rows_gpu = cp.array(K_rows_gpu)
        K_cols_gpu = cp.array(K_cols_gpu) 
        K_coo_gpu = coo_matrix((K_data_gpu, (K_rows_gpu, K_cols_gpu)), shape = (numNodes, numNodes)).toarray()
        
        cuda.synchronize()
        
        """GPU implementation of applying the BC"""
        nodes_BC_gpu = cuda.to_device(nodes_BC)
        
        # Define kernel executino parameters
        threadsperblock = 1
        blockspergrid = (len(nodes_BC) + (threadsperblock - 1)) // threadsperblock
        
        t0_applyBC = time.perf_counter()
        assembly._applyDirichletBC[blockspergrid, threadsperblock](K_coo_gpu, b_gpu, nodes_BC_gpu)
        tf_applyBC = time.perf_counter()  
    
        """Solve the system of equations on the GPU"""
        K_sparse_gpu = csr_matrix(K_coo_gpu)  # convert K to a csr matrix
        b_gpu = cp.asarray(b_gpu)  # send F_global to gpu
        t0_solve = time.perf_counter() # start timer 
        soln = cpssl.spsolve(K_sparse_gpu, b_gpu)  # solve using spsolve
        tf_solve = time.perf_counter() # stop timer
        x = soln.get()  # send soln back to host
    
    """Plot solution field"""
    assembly.contour_mpl(xyz_nodeCoords, 
                x, 
                test_type = processor, 
                title = processor + f" {i}", 
                fname = processor+".jpg", 
                flag=True,)
    # plt.show()
    exit()
        
    """Save times to list"""
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

# Stop timer
tf = time.time()

"""Print out statistics of the simulation"""
print(f"Total Number of Nodes : {len(nodeTags)}")
print(f"Total Number of Elements : {len(elemTags)}")
print(f"Total Time = {tf-t0:.4f} s")

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
