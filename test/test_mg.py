from firedrake import *
from firedrake.petsc import PETSc

import os
import signal
from fdutils import PointCloud
from fdutils.mg import NonnestedTransferManager


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
N = opts.getInt('n', default=4)

nest = opts.getBool('nest', default=False)

if dim == 2:
    mesh_builder = lambda N: UnitSquareMesh(N, N)
elif dim == 3:
    mesh_builder = lambda N: UnitCubeMesh(N, N, N)
else:
    raise

if nest:
    mesh_init = mesh_builder(N)
    hierarchy = MeshHierarchy(mesh_init, 1)
else:
    meshes  = [mesh_builder(N*(2**i)) for i in range(2)]
    hierarchy = NonNestedHierarchy(*meshes)

mesh = hierarchy[-1]

x = SpatialCoordinate(mesh)
if dim == 2:
    u_exact = sin(pi*x[0])*sin(pi*x[1])
elif dim == 3:
    u_exact = sin(pi*x[0])*sin(pi*x[1])*sin(pi*x[2])
else:
    raise

V = FunctionSpace(mesh, 'CG', 1)
u, v = TrialFunction(V), TestFunction(V)

f = - div(grad(u_exact))

a = inner(grad(u), grad(v))*dx - inner(f, v)*dx

bcs = DirichletBC(V, 0, 'on_boundary')

sol = Function(V, name='sol')
problem = LinearVariationalProblem(lhs(a), rhs(a), sol, bcs=bcs)
solver_parameters = {
    'ksp_type': 'gmres',
    # 'ksp_view': None,
    # 'ksp_error_if_not_converged': None,
    'ksp_monitor_true_residual': None,
    'ksp_converged_reason': None,
    'pc_type': 'mg',
}
if not nest:
    solver_parameters['mg_transfer_manager'] = 'fdutils.NonnestedTransferManager'

solver = LinearVariationalSolver(problem, options_prefix='', solver_parameters=solver_parameters)

solver.solve()

err = errornorm(u_exact, sol, 'H1')
H1 = norm(u_exact, 'H1')
printf(f'Error: {err} ({err/H1*100}%)')




