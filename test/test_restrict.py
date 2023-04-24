from firedrake import *
from firedrake.petsc import PETSc

import os
import signal
from fdutils import PointCloud
from fdutils.mg import get_lump_mass_matrix, get_mass_inv, get_mass_mult


def sync_printf(msg):
    rank, size = COMM_WORLD.rank, COMM_WORLD.size
    print(f'[{rank}/{size}] ' + msg, flush=True)

def printf(msg):
    rank, size = COMM_WORLD.rank, COMM_WORLD.size
    if rank == 0:
        print(msg, flush=True)

# for debug with gdb
PID = os.getpid()
sync_printf(f'PID = {PID}')
signal.signal(signal.SIGUSR1, lambda *args: None)
os.kill(PID, signal.SIGUSR1)


opts = PETSc.Options()
p = opts.getInt('p', default=1)
dim = opts.getInt('dim', default=2)
Ns_str = opts.getString('ns', default='10,20')
Ns = [int(_) for _ in Ns_str.split(',')]

if dim == 2:
    meshes  = [UnitSquareMesh(_, _) for _ in Ns]
elif dim == 3:
    meshes = [UnitCubeMesh(_, _, _) for _ in Ns]
else:
    raise

def make_pn_fun(mesh):
    V = FunctionSpace(mesh, 'CG', p)
    f = Function(V)
    return f

mesh1, mesh2 = meshes[0], meshes[1]
f1 = make_pn_fun(mesh1)
f2 = make_pn_fun(mesh2)

f2.assign(1)

pc = PointCloud(mesh1, mesh2.coordinates.dat.data_ro, tolerance=1)

pvs = f2.dat.data_ro
pc.restrict(pvs, f1)
# TODO: how to check the results?





