from mpi4py import MPI
import numpy as np

print("loading landlab.py...")

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



def define_mesh():
    numpoints = 5
    xcoords = np.array([np.random.uniform(0,1) for i in range(numpoints)])
    ycoords = np.array([np.random.uniform(0,1) for i in range(numpoints)])
    return xcoords, ycoords



def start_update(t, vec):
    print(f"update at time {t}, {vec}")

    

def update_single(x,y,old):
    print(f"{x} {y}")
    n = old + np.random.uniform(-1,1)*0.1
    return n
