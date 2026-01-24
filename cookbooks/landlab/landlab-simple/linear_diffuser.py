#from mpi4py import MPI

import numpy as np
import landlab
from landlab.components import LinearDiffuser

comm = None

current_time = 0.

mg = None
elevation = None
linear_diffuser = None

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


def finalize():
    pass

# Run the Landlab simulation from the current time to end_time and return
# the new topographic elevation (in m) at each local node.
# dict_variable_name_to_value_in_nodes is a dictionary mapping variables
# (x velocity, y velocity, temperature, etc.) to an array of values in each
# node.
def update_until(end_time, dict_variable_name_to_value_in_nodes):
    global current_time
    dt = end_time - current_time
    x_velocity = dict_variable_name_to_value_in_nodes["x velocity"]
    y_velocity = dict_variable_name_to_value_in_nodes["y velocity"]
    z_velocity = dict_variable_name_to_value_in_nodes["z velocity"]

    deposition_erosion = np.zeros(mg.number_of_nodes)

    # Set up the 'linear diffuser' landlab component. This takes the "topographic__elevation" 
    # stored in mg, and diffuses it based on diffusivity D and the time step.
    if dt>0:
        n_substeps = 10
        sub_dt = dt / n_substeps
        for _ in range(n_substeps):
          
          # TODO: add uplift(z_velocity * sub_dt)
          # TODO: advect(x_velocity * sub_dt, y_velocity * sub_dt)
          elevation_before = elevation

          # Actually run the diffusion of the surface using the linear diffuser. This 
          # applies the diffusion to the "topographic__elevation" field in mg.
          linear_diffuser.run_one_step(sub_dt)

          deposition_erosion += elevation - elevation_before
        pass
    
    current_time = end_time
    
    return deposition_erosion

# Notify the code to create the parallel grid. Done
# once at the beginning of the simulation and later,
# if the ASPECT volume mesh changes. This can be
# ignored if no adaptivity3 is supported on the 
# Landlab side.
def set_mesh_information(dict_grid_information):
    global mg, elevation, linear_diffuser

    if not mg:
        print("* Creating HexModelGrid ...")
        mg = landlab.HexModelGrid((3, 3))
        print("* Creating topographic elevation ...")
        elevation = mg.add_zeros("topographic__elevation", at="node")
        print("\tnumber of nodes:", mg.number_of_nodes)

        D = 0.01 # m2
        print("* Creating LinearDiffuser ... with D =", D)
        linear_diffuser = LinearDiffuser(mg, linear_diffusivity=D)

        print("* Done")
        # if rank==0:
        #     global_grid = HexGrid()
        #     partition_grid()
        #     broadcast_pieces()
        # else
        #     mg = receive_my_piece()

# Return the x coordinates of the locally owned nodes on this
# MPI rank. grid_id is always 0.
def get_grid_x(grid_id):
    return mg.node_x

# Return the y coordinates of the locally owned nodes on this
# MPI rank. grid_id is always 0.
def get_grid_y(grid_id):
    return mg.node_y

# Return the initial topography at the start of the simulation
# in each node.
def get_initial_topography(grid_id):
  global elevation
  return elevation



# old:

def get_grid_nodes(dict_grid_information):
    #numpoints = 5
    #xcoords = np.array([np.random.uniform(0,1) for i in range(numpoints)])
    #ycoords = np.array([np.random.uniform(0,1) for i in range(numpoints)])
    return get_grid_x(0), get_grid_y(0)

def do_timestep():
    print("hello")

def start_update(t, vec):
    print(f"update at time {t}, {vec}")

def write_output():
    pass    

def update_single(x,y,old):
    print(f"{x} {y}")
    n = old + np.random.uniform(-1,1)*0.1
    return n

if __name__ == "__main__":
    print("Running Landlab separately:")


    from matplotlib.pyplot import figure, legend, plot, title, xlabel, ylabel, ylim
    from landlab.plot.imshow import imshow_grid

    #comm = MPI.COMM_WORLD
    initialize(None)

    set_mesh_information({})
    print("grid coordinates:", get_grid_x(0), get_grid_y(0))

    dt = 0.1
    for n in range(10):
        data = {}
        data["x velocity"] = np.zeros(mg.number_of_nodes)
        data["y velocity"] = np.zeros(mg.number_of_nodes)
        data["z velocity"] = np.zeros(mg.number_of_nodes)
        update_until(n*dt, data)
        write_output()

