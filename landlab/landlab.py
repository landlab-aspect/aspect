from mpi4py import MPI
import numpy as np

print("loading landlab.py...")

current_time = 0

comm = None

def initialize(comm_handle):
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
    dt = end_time-current_time
    x_velocity = dict_variable_name_to_value_in_nodes["x velocity"]
    y_velocity = dict_variable_name_to_value_in_nodes["y velocity"]
    z_velocity = dict_variable_name_to_value_in_nodes["z velocity"]

    if dt>0:
        n_substeps = 10
        sub_dt = dt / n_substeps
        for _ in range(n_substeps):
          
          uplift(z_velocity * sub_dt)
          advect(x_velocity * sub_dt, y_velocity * sub_dt)
          elevation_before = "topographic__elevation"
          run all components
          deposition_erosion += "topographic__elevation" - elevation_before
        pass
    
    current_time = end_time
    
    return deposition_erosion


mg = None

# Notify the code to create the parallel grid. Done
# once at the beginning of the simulation and later,
# if the ASPECT volume mesh changes. This can be
# ignored if no adaptivity is supported on the 
# Landlab side.
def set_mesh_information(dict_grid_information):
    if not mg:
        if rank==0:
            global_grid = HexGrid()
            partition_grid()
            broadcast_pieces()
        else
            mg = receive_my_piece()

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
  return topographic__elevation



# old:

def get_grid_nodes(dict_grid_information):
    numpoints = 5
    xcoords = np.array([np.random.uniform(0,1) for i in range(numpoints)])
    ycoords = np.array([np.random.uniform(0,1) for i in range(numpoints)])
    return xcoords, ycoords

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
    comm = MPI.COMM_WORLD
    initialize(MPI.comm.py2f(comm))

    set_mesh_information({})

    dt = 0.1
    for n in range(10):
        update_until(n*dt)
        write_output()
