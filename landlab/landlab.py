from mpi4py import MPI

print("loading landlab.py...")

def initialize(comm_handle):
    # Convert the handle back to an MPI communicator
    comm = MPI.Comm.f2py(comm_handle)

    rank = comm.Get_rank()
    size = comm.Get_size()

    print(f"Python: Hello from Rank {rank} of {size}")

    data = 1
    globalsum = comm.allreduce(data, op=MPI.SUM)
    if comm.rank == 0:
        print(f"\tPython: testing communication; sum {globalsum}")
    
