import pytest
import numpy as np
from firedrake import *
from peval.pointcloud import PointCloud

import matplotlib.pyplot as plt

# logging.set_level(logging.DEBUG)

m = UnitSquareMesh(8, 8, quadrilateral=True)
x = SpatialCoordinate(m)

V = FunctionSpace(m, 'DG', 0)
f = Function(V)
f.interpolate(3*x[0] + 9*x[1] - 1)


Vv = VectorFunctionSpace(m, 'CG', 1)
fv = Function(Vv)
fv.interpolate(as_vector([x[0]**2, x[1]**2]))


# m.topology_dm.view()
m_ref = UnitSquareMesh(100, 100, quadrilateral=True)
V_ref = FunctionSpace(m_ref, 'CG', 1)
f_ref = Function(V_ref)
Vv_ref = VectorFunctionSpace(m_ref, 'CG', 1)
fv_ref = Function(Vv_ref)

coords = m_ref.coordinates.dat.data

pc = PointCloud(m, coords)
v = pc.evaluate(f)
f_ref.dat.data[:] = v
tricontourf(f)
tricontourf(f_ref)