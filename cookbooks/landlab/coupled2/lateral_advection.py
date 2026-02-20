
from mpi4py import MPI
import json
import os
import numpy as np
import landlab
from landlab.components import AdvectionSolverTVD

from landlab.io.legacy_vtk import write_legacy_vtk

current_time = 0

comm = None

model_grid = None
elevation = None
surface_advector = None
horizontal_velocity = None
ux = None
uy = None

timestep = 0
vtks = []

def initialize(comm_handle):
    if not comm_handle is None:
        # Convert the handle back to an MPI communicator
        global comm
        comm = MPI.Comm.f2py(comm_handle)

        rank = comm.Get_rank()
        size = comm.Get_size()

        print(f"Python: Hello from Rank {rank} of {size}")

        data = 1
        globalsum = comm.allreduce(data, op=MPI.SUM)
        if comm.rank == 0:
            print(f"\tPython: testing communication; sum {globalsum}")
    else:
        print("Python: running sequentially!")

def finalize():
    pass


# Run the Landlab simulation from the current time to end_time and return
# the new topographic elevation (in m) at each local node.
# dict_variable_name_to_value_in_nodes is a dictionary mapping variables
# (x velocity, y velocity, temperature, etc.) to an array of values in each
# node.
def update_until(end_time, dict_variable_name_to_value_in_nodes):
    global current_time, elevation, surface_advector, horizontal_velocity, ux, uy, timestep
    dt = end_time - current_time
    timestep += 1

    deposition_erosion = np.zeros(model_grid.number_of_nodes)

    x_velocity = dict_variable_name_to_value_in_nodes["x velocity"]
    y_velocity = dict_variable_name_to_value_in_nodes["y velocity"]
    z_velocity = dict_variable_name_to_value_in_nodes["z velocity"]

    # The x, y, z velocities passed in from ASPECT are defined at the node points of the LandLab mesh.
    # However, lateral advection of the LandLab surface requires that the velocity is defined at the links.
    # To do this, we determine the average value of the x,y velocity at the link using the node points that 
    # are connected by the link. Then, we can use AdvectionSolverTVD to advect the surface.

    x_vel_at_links = model_grid.map_mean_of_link_nodes_to_link(x_velocity)
    y_vel_at_links = model_grid.map_mean_of_link_nodes_to_link(y_velocity)

    horizontal_velocity[model_grid.horizontal_links] = x_vel_at_links[model_grid.horizontal_links]
    horizontal_velocity[model_grid.vertical_links]   = y_vel_at_links[model_grid.vertical_links]
    surface_advector = AdvectionSolverTVD(model_grid, fields_to_advect=elevation)

    # Substepping for surface processes
    if dt>0:
        n_substeps = 10
        sub_dt = dt / n_substeps
        for _ in range(n_substeps):

          # TODO:
          elevation_before = elevation.copy()

          # Uplift the topography using the ASPECT vertical velocity.
          elevation[model_grid.core_nodes] += z_velocity[model_grid.core_nodes] * sub_dt

          # Advect the grid with the horizontal velocity from ASPECT. The ASPECT velocities are already being passed
          # correctly at each NODE point of the landlab grid. However, the TVDAdvector requires the velocity at the LINKS.
          # There must be a relationship between the two that can be leveraged, but at the moment I'm not sure how.
          surface_advector.update(sub_dt)

          deposition_erosion += elevation - elevation_before

    current_time = end_time

    # filename = f"./output/landlab_{str(timestep).zfill(3)}.vtk"
    # print("Writing output VTK file...", filename)
    # vtk_file = write_legacy_vtk(path=filename, grid=model_grid, clobber=True)
    # vtks.append((current_time, filename))

    # if True:
    #     # write vtk.series file (ParaView supports legacy VTK in this format)
    #     with open("./output/landlab.vtk.series", "w") as f:
    #         series = {
    #             "file-series-version": "1.0",
    #             "files": [
    #                 {"name": os.path.basename(filename), "time": time}
    #                 for time, filename in vtks
    #             ]
    #         }
    #         json.dump(series, f, indent=2)
    
    # print ("deposition/erosion:", np.linalg.norm(deposition_erosion))
    print("Max elevation:", np.max(elevation), "Min elevation:", np.min(elevation))
    # print("Max elevation:", np.max(deposition_erosion), "Min elevation:", np.min(deposition_erosion))

    return deposition_erosion
    # return elevation

def set_mesh_information(dict_grid_information):
    global model_grid, elevation, horizontal_velocity, ux, uy

    if not model_grid:
        print("* Creating RasterModelGrid ...")
        x_extent = dict_grid_information["Mesh X extent"]
        y_extent = dict_grid_information["Mesh Y extent"]
        spacing = dict_grid_information["Mesh Spacing"]

        nrows = int(y_extent / spacing)  # number of node rows
        ncols = int(x_extent / spacing)  # number of node columns

        model_grid = landlab.RasterModelGrid((nrows, ncols), xy_spacing=(spacing, spacing))
        model_grid.set_closed_boundaries_at_grid_edges(True, True, True, True)

        print("* Creating topographic elevation ...")
        # Initialize topography array with zeros
        elevation = model_grid.add_zeros("topographic__elevation", at="node")

        # create a Gaussian hill in the center of the domain
        gaussian_height = 5e3
        x_extent = model_grid.x_of_node.max() - model_grid.x_of_node.min()
        y_extent = model_grid.y_of_node.max() - model_grid.y_of_node.min()
        elevation += gaussian_height * np.exp(-((model_grid.node_x - x_extent/2)**2 + (model_grid.node_y - y_extent/2)**2) / (2 * (10e3)**2))

        # Initialize the horizontal velocity so that we can pass the ASPECT velocities to it later.
        horizontal_velocity = model_grid.add_zeros("advection__velocity", at="link")

        ux = model_grid.add_zeros("x_velocity", at="node")
        uy = model_grid.add_zeros("y_velocity", at="node")
        print("\tnumber of nodes:", model_grid.number_of_nodes)

        print("* Done")

# Return the x coordinates of the locally owned nodes on this
# MPI rank. grid_id is always 0.
def get_grid_x(grid_id):
    global model_grid
    return model_grid.node_x

# Return the y coordinates of the locally owned nodes on this
# MPI rank. grid_id is always 0.
def get_grid_y(grid_id):
    global model_grid
    return model_grid.node_y

def initalize_landlab_components(landlab_component_parameters):
    global model_grid, elevation

    pass

# Return the initial topography at the start of the simulation
# in each node.
def get_initial_topography(grid_id):
    global elevation
    return elevation


def write_output(timestep):
    # Write the grid to vtk
    print("Writing output VTK file...")
    vtk_file = write_legacy_vtk(path=f"./output_{str(timestep).zfill(3)}.vtk", grid=model_grid, clobber=True)
