from firedrake import *
from fdutils import *

from firedrake.petsc import PETSc
import numpy as np
import os


def points2bdy(points, r=1.0):
    _r = np.linalg.norm(points, axis=1).reshape([-1, 1]) / r
    return points/_r


@PETSc.Log.EventDecorator()
def make_high_order_mesh_simple(m, p, tag2r=None): 
    if tag2r is None:
        tag2r = {'on_boundary': 1}

    if p == 1:
        return m
    coords_1 = m.coordinates 
    coords_i = coords_1
    for i in range(2, p+1):
        coords_im1 = coords_i
        V_i = VectorFunctionSpace(m, 'CG', i)
        coords_i = Function(V_i, name=f'coords_p{i}').interpolate(coords_im1)
        
        for tag, r in tag2r.items():
            bc = DirichletBC(V_i, 0, tag)
            coords_i.dat.data_with_halos[bc.nodes] = \
                points2bdy(coords_i.dat.data_ro_with_halos[bc.nodes], r=r)

    return Mesh(coords_i)


opts = PETSc.Options()
bbox_relax_factor = opts.getReal('bbox_relax_factor', default=0.01)

mesh_orig = Mesh(os.path.join(os.path.dirname(__file__), "disk.msh"))
mesh = make_high_order_mesh_simple(mesh_orig, 3, tag2r={1: 1, 2: 1.3})
mesh.bbox_relax_factor = bbox_relax_factor

V = FunctionSpace(mesh, 'CG', 3)
u = Function(V).interpolate(SpatialCoordinate(mesh)[0])

x = np.array([[1, 0, 0]])

pc = PointCloud(mesh, x, tolerance=1e-10)
flag = np.allclose(pc.evaluate(u), 1)
# flag = np.allclose(u.at(x), 1)

if not flag:
    PETSc.Sys.Print(f'Test for high order mesh: Failed:')
else:
    PETSc.Sys.Print(f'Test for high order mesh: OK')

