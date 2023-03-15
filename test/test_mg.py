from firedrake import *
from firedrake.petsc import PETSc

import os
import signal
from fdutils import PointCloud
from fdutils.mg import NonnestedTransferManager


def printf(msg):
    rank, size = COMM_WORLD.rank, COMM_WORLD.size
    print(f'[{rank}/{size}] ' + msg, flush=True)


# for debug with gdb
PID = os.getpid()
printf(f'PID = {PID}')
signal.signal(signal.SIGUSR1, lambda *args: None)
os.kill(PID, signal.SIGUSR1)


opts = PETSc.Options()
dim = opts.getInt('dim', default=2)
Ns_str = opts.getString('ns', default='20,40')
Ns = [int(_) for _ in Ns_str.split(',')]

if dim == 2:
    meshes  = [UnitSquareMesh(_, _) for _ in Ns]
elif dim == 3:
    meshes = [UnitCubeMesh(_, _, _) for _ in Ns]
else:
    raise

hierarchy = NonNestedHierarchy(*meshes)
mesh = hierarchy[-1]

x = SpatialCoordinate(mesh)
if dim == 2:
    u_exact = sin(x[0])*sin(x[1])
elif dim == 3:
    u_exact = sin(x[0])*sin(x[1])*sin(x[2])
else:
    raise

V = FunctionSpace(mesh, 'CG', 1)
u, v = TrialFunction(V), TestFunction(V)

f = div(grad(u_exact))

a = inner(grad(u), grad(v))*dx - inner(f, v)*dx

bcs = DirichletBC(V, 0, 'on_boundary')

sol = Function(V, name='sol')
problem = LinearVariationalProblem(lhs(a), rhs(a), sol)
solver_parameters = {
    'ksp_type': 'gmres',
    # 'ksp_view': None,
    'ksp_monitor_true_residual': None,
    'pc_type': 'mg',
    'mg_transfer_manager': 'fdutils.NonnestedTransferManager'
}
solver = LinearVariationalSolver(problem, options_prefix='', solver_parameters=solver_parameters)

solver.solve()

err = errornorm(u_exact, sol, 'H1')
H1 = norm(u_exact, 'H1')
print(f'Error: {err} ({err/H1})')




