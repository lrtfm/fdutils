from firedrake import *
from firedrake.petsc import PETSc

import os
import signal
from fdutils import PointCloud
from fdutils.mg import get_lump_mass_matrix


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
dim = opts.getInt('dim', default=2)
Ns_str = opts.getString('ns', default='10,20')
Ns = [int(_) for _ in Ns_str.split(',')]

if dim == 2:
    meshes  = [UnitSquareMesh(_, _) for _ in Ns]
elif dim == 3:
    meshes = [UnitCubeMesh(_, _, _) for _ in Ns]
else:
    raise

def make_p1_fun(mesh):
    V = FunctionSpace(mesh, 'CG', 1)
    f = Function(V)
    return f

mesh1, mesh2 = meshes[0], meshes[1]
f1 = make_p1_fun(mesh1)
f2 = make_p1_fun(mesh2)

f2.assign(1)

M2 = get_lump_mass_matrix(f2.function_space())
M1 = get_lump_mass_matrix(f1.function_space())


pc = PointCloud(mesh1, mesh2.coordinates.dat.data_ro, tolerance=1)

pvs = M2.dat.data_ro * f2.dat.data_ro
pc.restrict(pvs, f1)

s = sum(pvs)
s1 = COMM_WORLD.reduce(s)
with f1.dat.vec_ro as vec:
    s2 = vec.sum()
printf(f'{s1} == {s2}')





