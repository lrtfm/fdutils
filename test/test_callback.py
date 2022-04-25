from firedrake import *
import fdutils
from fdutils import PointCloud
import numpy as np

fdutils.__file__

m = RectangleMesh(10, 10, 1, 1)
m2 = RectangleMesh(10, 10, 2, 2)

V = FunctionSpace(m, 'CG', 1)
V_v = VectorFunctionSpace(m, 'CG', 1)

x = SpatialCoordinate(m)

f = Function(V).interpolate(x**2)
f_v = Function(V_v).interpolate(x + as_vector([sin(_) for _ in x]))

pc = PointCloud(m, m2.coordinates.dat.data_ro_with_halos, tolerance=None)

def callback(x):
    return np.sum(x*x, axis=1)

def callback_v(x):
    return x + np.sin(x)

ret = pc.evaluate(f, callback=callback)

ret_v = pc.evaluate(f_v, callback=callback_v)

