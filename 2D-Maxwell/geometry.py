import gmsh
import sys
import numpy as np
import time
from tabulate import tabulate
from numba import njit, cuda

"""
+ ------------------------------------------------------ +
|                                  (magnet 2)            |                  
|        (magnet 1)                  -------             |
|           ----      d1       d2   |       |            |
|       l1 |    | <--------|------->|       | l2         | H
|           ----                     -------             |
|             l1                       l2                |
|                                                        |
+ ------------------------------------------------------ +
                            B
"""


def create_mesh(
    B=200,
    H=100,
    l1=20,
    l2=30,
    d1=50,
    d2=30,
    lc=5,
    flag=False,
):
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
    gmsh.initialize(sys.argv)
    gmsh.option.setNumber("General.Terminal", 0)  # 0 = no print, 1 = print
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
    if flag == True:
        if "-nopopup" not in sys.argv:
            gmsh.fltk.run()

    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes(
        dim=-1,
        tag=-1,
    )

    get_element_types = gmsh.model.mesh.getElementTypes()
    linear_tri_elem_type = get_element_types[1]  # 3 node triangle

    elemTags, elemNodeTags = gmsh.model.mesh.getElementsByType(
        linear_tri_elem_type,
        tag=-1,
    )

    xyz_nodeCoords = nodeCoords.reshape((-1, 3))
    elemNodeTags = elemNodeTags.reshape((-1, 3))

    gmsh.finalize()

    return nodeTags, xyz_nodeCoords, elemTags, elemNodeTags


@njit
def get_mesh_props(
    nodeTags,
    xyz_nodeCoords,
    elemTags,
    elemNodeTags,
    l1,
    l2,
    d1,
    d2,
    mu_r_mag=1.04,
    magnetization=1e6,
):
    """
    Create the conectivity array for the mesh
    python element # | gmsh element tag | n1 | n2 | n3 | Material | Magnetization
    """
    # Initialize numpy array to hold [node tag, x, y, x]
    # Each row correponds to a node from the mesh
    # Note: you must subtract 1 when using this mapping
    # because gmsh ordering is human indexed.
    map_node_tag_to_xyz = np.zeros((len(nodeTags), 4))
    for i in range(len(nodeTags)):
        tag_i = nodeTags[i]
        x_i = xyz_nodeCoords[i, 0]  # x coord
        y_i = xyz_nodeCoords[i, 1]  # y coord
        z_i = xyz_nodeCoords[i, 2]  # z coord

        # Update the array that maps node tag to xyz coord
        map_node_tag_to_xyz[i, 0] = tag_i
        map_node_tag_to_xyz[i, 1] = x_i
        map_node_tag_to_xyz[i, 2] = y_i
        map_node_tag_to_xyz[i, 3] = z_i

    # Initialize numpy array that maps the gmsh node tag to start at 1.
    # Note that the new ordering starts at value of 1
    # (human indexed for the tags)
    connectivity_array = np.zeros((len(elemTags), 7))
    for i in range(len(elemTags)):
        tag_i = int(elemTags[i])  # extract the gmsh node tag
        n1 = int(elemNodeTags[i, 0])  # gmsh node 1 tag
        n2 = int(elemNodeTags[i, 1])  # gmsh node 2 tag
        n3 = int(elemNodeTags[i, 2])  # gmsh node 3 tag

        # node 1 x and y coords
        n1_x = map_node_tag_to_xyz[n1 - 1][1]
        n1_y = map_node_tag_to_xyz[n1 - 1][2]

        # node 2 x and y coords
        n2_x = map_node_tag_to_xyz[n2 - 1][1]
        n2_y = map_node_tag_to_xyz[n2 - 1][2]

        # node 3 and y coords
        n3_x = map_node_tag_to_xyz[n3 - 1][1]
        n3_y = map_node_tag_to_xyz[n3 - 1][2]

        # compute the area of each element
        area_e = 0.5 * (
            n1_x * (n2_y - n3_y) + n2_x * (n3_y - n1_y) + n3_x * (n1_y - n2_y)
        )

        # compute the centroid of each element
        x_center = (n1_x + n2_x + n3_x) * (1 / 3.0)
        y_center = (n1_y + n2_y + n3_y) * (1 / 3.0)

        # Check if element is located inside region of magnet of magnet 1
        if -l1 - d1 <= x_center <= -d1 and -0.5 * l1 <= y_center <= 0.5 * l1:
            # Inside permanent magnet 1
            connectivity_array[i, 5] = mu_r_mag
            connectivity_array[i, 6] = magnetization

        # Check if element is located inside region of magnet of magnet 1
        elif d2 <= x_center <= d2 + l2 and -0.5 * l2 <= y_center <= 0.5 * l2:
            # Inside permanent magnet 2
            connectivity_array[i, 5] = mu_r_mag
            connectivity_array[i, 6] = -magnetization

        # Ensure all elements are correctly oriented
        if area_e < 0:
            connectivity_array[i, 0] = i
            connectivity_array[i, 1] = tag_i
            connectivity_array[i, 4] = n1
            connectivity_array[i, 3] = n2
            connectivity_array[i, 2] = n3
        else:
            connectivity_array[i, 0] = i
            connectivity_array[i, 1] = tag_i
            connectivity_array[i, 2] = n1
            connectivity_array[i, 3] = n2
            connectivity_array[i, 4] = n3

    return connectivity_array


@njit
def _map_node_to_xyz(nodeTags, xyz_nodeCoords, map_node_tag_to_xyz):
    for i in range(len(nodeTags)):
        tag_i = nodeTags[i]
        x_i = xyz_nodeCoords[i, 0]  # x coord
        y_i = xyz_nodeCoords[i, 1]  # y coord
        z_i = xyz_nodeCoords[i, 2]  # z coord

        # Update the array that maps node tag to xyz coord
        map_node_tag_to_xyz[i, 0] = tag_i
        map_node_tag_to_xyz[i, 1] = x_i
        map_node_tag_to_xyz[i, 2] = y_i
        map_node_tag_to_xyz[i, 3] = z_i
    return map_node_tag_to_xyz


@cuda.jit
def _generate_connectivity_array(
    map_node_tag_to_xyz,
    connectivity_array,
    elemTags,
    elemNodeTags,
    numElems,
    l1,
    l2,
    d1,
    d2,
    mu_r_mag,
    magnetization,
):
    # get the thread Id
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    # get the stride length (cuda.blockIdx.x * cuda.gridDim.x)
    stride = cuda.gridsize(1)

    for i in range(idx, numElems, stride):
        tag_i = int(elemTags[i])  # extract the gmsh node tag
        n1 = int(elemNodeTags[i, 0])  # gmsh node 1 tag
        n2 = int(elemNodeTags[i, 1])  # gmsh node 2 tag
        n3 = int(elemNodeTags[i, 2])  # gmsh node 3 tag

        # node 1 x and y coords
        n1_x = map_node_tag_to_xyz[n1 - 1][1]
        n1_y = map_node_tag_to_xyz[n1 - 1][2]

        # node 2 x and y coords
        n2_x = map_node_tag_to_xyz[n2 - 1][1]
        n2_y = map_node_tag_to_xyz[n2 - 1][2]

        # node 3 and y coords
        n3_x = map_node_tag_to_xyz[n3 - 1][1]
        n3_y = map_node_tag_to_xyz[n3 - 1][2]

        # compute the area of each element
        area_e = 0.5 * (
            n1_x * (n2_y - n3_y) + n2_x * (n3_y - n1_y) + n3_x * (n1_y - n2_y)
        )

        # compute the centroid of each element
        x_center = (n1_x + n2_x + n3_x) * (1 / 3.0)
        y_center = (n1_y + n2_y + n3_y) * (1 / 3.0)

        # Check if element is located inside region of magnet of magnet 1
        if -l1 - d1 <= x_center <= -d1 and -0.5 * l1 <= y_center <= 0.5 * l1:
            # Inside permanent magnet 1
            connectivity_array[i, 5] = mu_r_mag
            connectivity_array[i, 6] = magnetization

        # Check if element is located inside region of magnet of magnet 1
        elif d2 <= x_center <= d2 + l2 and -0.5 * l2 <= y_center <= 0.5 * l2:
            # Inside permanent magnet 2
            connectivity_array[i, 5] = mu_r_mag
            connectivity_array[i, 6] = -magnetization

        # Ensure all elements are correctly oriented
        if area_e < 0:
            connectivity_array[i, 0] = i
            connectivity_array[i, 1] = tag_i
            connectivity_array[i, 4] = n1
            connectivity_array[i, 3] = n2
            connectivity_array[i, 2] = n3
        else:
            connectivity_array[i, 0] = i
            connectivity_array[i, 1] = tag_i
            connectivity_array[i, 2] = n1
            connectivity_array[i, 3] = n2
            connectivity_array[i, 4] = n3

    cuda.syncthreads()


if __name__ == "__main__":
    # Start timer
    t0 = time.time()

    # Define computation method: cpu or gpu.
    #   -> cpu runs jit
    #   -> gpu runs jit.cuda
    compute_method = "cpu"

    # Define geometry of the model
    B = 200
    H = 100
    l1 = 20
    l2 = 30
    d1 = 50
    d2 = 30
    lc = 10
    mu_r_mag = 1.04
    magnetization = 1e6

    mesh_time_arr = []
    prop_time_arr = []
    for i in range(10):
        # Generate the mesh using GMSH
        t0_mesh = time.perf_counter()
        nodeTags, xyz_nodeCoords, elemTags, elemNodeTags = create_mesh(
            B + B,
            H + H,
            l1=l1,
            l2=l2,
            d1=d1,
            d2=d2,
            lc=lc,
            flag=False,
        )
        tf_mesh = time.perf_counter()

        numNodes = len(nodeTags)
        numElems = len(elemTags)

        if compute_method == "cpu":
            """Compute the connectivity array using jit on CPU"""
            # Create the connectivity array
            t0_mesh_prop = time.perf_counter()
            connectivity_array = get_mesh_props(
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

        elif compute_method == "gpu":
            """Compute the connectivity array using the GPU"""
            t0_mesh_prop = time.perf_counter()

            # Initialize variables for connectivity array
            map_node_tag_to_xyz = np.zeros((len(nodeTags), 4))
            connectivity_array = np.zeros((len(elemTags), 7))

            # Get the node mapping array
            map_node_tag_to_xyz = _map_node_to_xyz(
                nodeTags,
                xyz_nodeCoords,
                map_node_tag_to_xyz,
            )

            # Send variables to gpu (device memory)
            map_node_tag_to_xyz_gpu = cuda.to_device(map_node_tag_to_xyz)
            connectivity_array_gpu = cuda.to_device(connectivity_array)
            elemTags_gpu = cuda.to_device(elemTags)
            elemNodeTags_gpu = cuda.to_device(elemNodeTags)

            # Define kernel execution parameters
            threadsperblock = 16
            blockspergrid = (len(elemTags) + (threadsperblock - 1)) // threadsperblock

            # Execute cuda kernel
            _generate_connectivity_array[blockspergrid, threadsperblock](
                map_node_tag_to_xyz_gpu,
                connectivity_array_gpu,
                elemTags_gpu,
                elemNodeTags_gpu,
                numElems,
                l1,
                l2,
                d1,
                d2,
                mu_r_mag,
                magnetization,
            )
            # Send back solution to the cpu (host)
            connectivity_array = connectivity_array_gpu.copy_to_host()

            tf_mesh_prop = time.perf_counter()

        # Save times to list
        time_gmsh = tf_mesh - t0_mesh
        time_mesh_prop = tf_mesh_prop - t0_mesh_prop
        mesh_time_arr.append(time_gmsh)
        prop_time_arr.append(time_mesh_prop)

    tf = time.time()

    print(f"Total Number of Nodes : {len(nodeTags)}")
    print(f"Total Number of Elements : {len(elemTags)}")
    print(f"Total Time = {tf-t0:.4f} s")
    headers = ["Section", "1", "2", "3", "4", "5", "6"]
    table = [
        ["GMSH"] + mesh_time_arr,
        ["Mesh Prop."] + prop_time_arr,
    ]
    print(tabulate(table, headers, tablefmt="fancy_outline"))
