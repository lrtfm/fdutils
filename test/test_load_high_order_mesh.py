from firedrake import *
from fdutils import *

from firedrake.petsc import PETSc
import numpy as np
import os

opts = PETSc.Options()
bbox_relax_factor = opts.getReal('bbox_relax_factor', default=0.01)
tolerance = opts.getReal('tolerance', default=1e-12)

mesh = Mesh(os.path.join(os.path.dirname(__file__), "sphere-2nd-order.msh"))
mesh.bbox_relax_factor = bbox_relax_factor

V = FunctionSpace(mesh, 'CG', 3)
u = Function(V, name='u').interpolate(SpatialCoordinate(mesh)[0])

# File('sphere-2nd-order.pvd').write(u)

x = np.array([[0.5, 0, 0]])

pc = PointCloud(mesh, x, tolerance=tolerance)
flag = np.allclose(pc.evaluate(u), 0.5)
# flag = np.allclose(u.at(x), 1)

if not flag:
    PETSc.Sys.Print(f'Test for high order mesh: Failed:')
else:
    PETSc.Sys.Print(f'Test for high order mesh: OK')

