
from mpi4py import MPI
import json
import os
import numpy as np
import landlab
from landlab.components import LinearDiffuser

from landlab.io.native_landlab import save_grid, load_grid

from landlab.io.legacy_vtk import write_legacy_vtk

current_time = 0
comm = None

model_grid = None
elevation  = None

linear_diffuser    = None

s2yr = 60 * 60 * 24 * 365.25
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
    global elevation, linear_diffuser, timestep, current_time

    dt = end_time - current_time
    timestep += 1

    deposition_erosion = np.zeros(model_grid.number_of_nodes)

    # Extract the velocity along the ASPECT model, which is located at y=0 on the landlab mesh.
    slice_x_velocity = dict_variable_name_to_value_in_nodes["x velocity"]
    slice_y_velocity = dict_variable_name_to_value_in_nodes["y velocity"]

    # Create empty arrays to project the velocity and composition values out from y=0 to all
    # nodes on the landlab mesh.
    vertical_velocity = np.zeros(model_grid.number_of_nodes)

    unique_x_values   = np.unique(model_grid.x_of_node)
    for x in unique_x_values:
        vertical_velocity[model_grid.x_of_node == x]  = slice_y_velocity[unique_x_values == x]

    # Substepping for surface processes
    if dt>0:
        n_substeps = 10
        sub_dt = dt / n_substeps
        for _ in range(n_substeps):

          elevation_before = elevation.copy()
          linear_diffuser.run_one_step(sub_dt)

          elevation[model_grid.core_nodes] += vertical_velocity[model_grid.core_nodes] * sub_dt

          deposition_erosion += elevation - elevation_before
        pass

    current_time = end_time
    print("Min elevation", np.min(elevation), "Max elevation:", np.max(elevation))

    deposition_erosion_2d = np.zeros(len(np.unique(model_grid.x_of_node)))
    for x in unique_x_values:
        deposition_erosion_2d[unique_x_values == x] = np.average(deposition_erosion[model_grid.x_of_node == x])
    
    return deposition_erosion_2d

def set_mesh_information(dict_grid_information):
    global model_grid, elevation

    if not model_grid:
        print("* Creating RasterModelGrid ...")
        x_extent = 100e3
        y_extent = 100e3
        spacing  = 1000.0

        nrows = int(y_extent / spacing) + 3 # number of node rows
        ncols = int(x_extent / spacing) + 3 # number of node columns

        model_grid = landlab.RasterModelGrid((nrows, ncols), xy_spacing=(spacing, spacing), xy_of_lower_left=(-spacing, -y_extent / 2 - spacing))

        print("* Creating topographic elevation ...")
        # Initialize topography array with zeros
        elevation = model_grid.add_zeros("topographic__elevation", at="node")
        elevation += np.sin(np.pi * model_grid.x_of_node / x_extent) * 10000.0

        # Close all boundaries
        model_grid.set_closed_boundaries_at_grid_edges(right_is_closed=True, 
                                                       left_is_closed=True, 
                                                       top_is_closed=True, 
                                                       bottom_is_closed=True)
                                                       
        print("\tnumber of nodes:", model_grid.number_of_nodes)

        initialize_landlab_components(None)

        print("* Done")

# If the ASPECT mesh is 2D, then we only want to return the unique values of x on the LandLab mesh,
# the logic here is probably different if the LandLab mesh is not a raster.
def get_grid_x(grid_dictionary):
    global model_grid
    return np.unique(model_grid.x_of_node)

# If the ASPECT mesh is 2D, then we only return an array of 0s for the y values, since this is where
# the ASPECT surface is located.
def get_grid_y(grid_dictionary):
    global model_grid
    return np.zeros(np.unique(model_grid.x_of_node).shape)

def initialize_landlab_components(landlab_component_parameters):
    global model_grid, elevation, linear_diffuser

    D_val = 1000 / s2yr

    print("* Initializing Landlab components ...")
    linear_diffuser     = LinearDiffuser(model_grid, linear_diffusivity=D_val)
    print("* Done initializing Landlab components")

    pass

def checkpoint_model_grid():
    global model_grid

    print("Checkpointing the LandLab model grid...")
    filename = "./landlab_model_grid_checkpoint.grid"
    save_grid(model_grid, filename, clobber=True)
    pass

def load_model_grid():
    global model_grid, elevation

    print("Loading the LandLab model grid...")
    filename = "./landlab_model_grid_checkpoint.grid"
    model_grid = load_grid(filename)
    elevation = model_grid.at_node["topographic__elevation"]

    # We need to initialize the components after loading the LandLab grid, since these
    # are not stored.
    initialize_landlab_components(None)
    pass

# Return the initial topography along y=0, where the ASPECT surface is located.
def get_initial_topography(grid_dictionary):
    global elevation
    return elevation[model_grid.y_of_node == 0]

def write_output(postprocess_dictionary):
    global model_grid

    timestep = postprocess_dictionary["ASPECT timestep"]
    current_time = postprocess_dictionary["ASPECT time"]
    output_directory = postprocess_dictionary["ASPECT output directory"]
    landlab_output_directory = os.path.join(output_directory, "landlab")

    LandLab_output_frequency = 10
    if timestep % LandLab_output_frequency != 0:
        return

    if not os.path.isdir(landlab_output_directory):
        os.makedirs(landlab_output_directory, exist_ok=True)

    # Write the grid to vtk
    filename = os.path.join(landlab_output_directory, f"landlab_{str(timestep).zfill(3)}.vtk")
    print("Writing output VTK file...", filename)
    vtk_file = write_legacy_vtk(path=filename, grid=model_grid, clobber=True)
    vtks.append((current_time, filename))

    if True:
        # write vtk.series file (ParaView supports legacy VTK in this format)
        with open(f"{output_directory}/landlab.vtk.series", "w") as f:
            series = {
                "file-series-version": "1.0",
                "files": [
                    {"name": os.path.basename(filename), "time": time}
                    for time, filename in vtks
                ]
            }
            json.dump(series, f, indent=2)